package com.fons.cloud.ai.agent.observability.autoconfigure.instrumentation;

import com.alibaba.fastjson2.JSON;
import com.fons.cloud.ai.agent.observability.api.AgentTraceRecorder;
import com.fons.cloud.ai.agent.observability.api.AgentTraceRun;
import com.fons.cloud.ai.agent.observability.api.ObservationConvention;
import com.fons.cloud.ai.agent.observability.core.NativeTraceContext;
import com.fons.cloud.ai.agent.observability.model.TraceEvent;
import com.fons.cloud.ai.agent.observability.model.TraceEventType;
import com.fons.cloud.ai.agent.observability.model.TraceRunContext;
import com.fons.cloud.ai.agent.observability.model.TraceStatus;
import com.fons.cloud.reactor.api.ReactiveTaskInterceptor;
import com.fons.cloud.reactor.api.ReactiveTaskRun;
import com.fons.cloud.reactor.api.ReactiveTaskScope;
import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.context.Context;
import io.opentelemetry.context.ContextKey;
import io.opentelemetry.context.Scope;
import io.opentelemetry.instrumentation.reactor.v3_1.ContextPropagationOperator;
import lombok.extern.slf4j.Slf4j;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.publisher.SignalType;

import java.time.Instant;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * 声明式观测的通用响应式生命周期支持。
 *
 * <p>该类承担注解接入后的全部通用采集职责：根据 Convention 提供的稳定身份创建 Run
 * 观测作用域、管理根与步骤 Span、通过 OTel Context 保存共享 Recorder 权柄、协调根与
 * 子节点的关闭门禁，并提供快照脱敏与序列化失败降级。业务状态经 Convention 映射为
 * {@link TraceStatus}；SDK 模型聚合不进入本类。</p>
 *
 * <p>本类同时是 common-reactor 的 {@link ReactiveTaskInterceptor} 扩展：RUN 方法执行
 * 期间，注解拦截器把创建描述压入线程本地的创建期绑定栈；工厂创建 Run 时通过
 * {@link #capture(String)} 取走栈顶描述并返回本次 Run 专属的观测实例，真实启动时在
 * 唯一根订阅前建立根。创建期绑定栈只服务于同步创建，绝不作为跨线程执行上下文。</p>
 *
 * <p>全部观测异常均按 fail-open 处理：采集初始化、约定映射、记录或导出失败只降级
 * 自身，不改变业务执行次数、结果或取消语义。</p>
 *
 * @author hongqy
 */
@Slf4j
public final class ReactiveObservationSupport implements ReactiveTaskInterceptor {

    /**
     * 公共 Run 作用域只沿 OTel/Reactor 上下文传播，不能通过用户参数伪造。
     */
    public static final ContextKey<RunScope> RUN_SCOPE = ContextKey.named("fons.pipeline.trace");

    /**
     * 创建期绑定栈：仅保存同步创建描述，空栈时移除 ThreadLocal 条目。
     */
    private static final ThreadLocal<Deque<RunCreation>> CREATION_STACK = new ThreadLocal<>();

    /**
     * 与注解入口和 SDK 适配层共享的 Recorder；noop 时所有采集入口低成本透传。
     */
    private final AgentTraceRecorder recorder;

    /**
     * 使用指定 Recorder 构建通用观测支持。
     *
     * @param recorder Trace Recorder 权柄；可为 noop 实现
     */
    public ReactiveObservationSupport(AgentTraceRecorder recorder) {
        this.recorder = Objects.requireNonNull(recorder, "recorder cannot be null");
        if (recorder != AgentTraceRecorder.noop()) {
            // 在执行链组装前统一安装；不在每次 Run 启停时修改 JVM 全局 Hook。
            ObservationContextBridge.install();
        }
    }

    // ------------------------------------------------------------------
    // 创建期绑定栈与工厂扩展
    // ------------------------------------------------------------------

    /**
     * 工厂创建 Run 时捕获栈顶的创建描述。
     *
     * <p>只有注解拦截器正在执行的 RUN 方法内部创建的 Run 才能取到描述；独立业务创建
     * 的 Run 返回 null，不参与观测拦截。同一描述被第二个 Run 捕获时按冲突处理并放弃
     * 本次采集，避免把观测误绑给某个内部 Run。</p>
     *
     * @param runId 本次运行标识
     * @return 本次 Run 专属的观测拦截实例；不适用时返回 null
     */
    @Override
    public ReactiveTaskInterceptor capture(String runId) {
        RunCreation creation = currentCreation();
        if (creation == null) {
            return null;
        }
        if (!creation.tryCapture(runId)) {
            log.warn("AgentObserved RUN creation bound to multiple runs, observation disabled, runId:{}",
                    runId);
            return null;
        }
        return new RunObservation(creation);
    }

    /**
     * 本实例只承担创建期捕获；真实执行增强由 {@link RunObservation} 专属实例完成。
     *
     * @param scope 本次运行的受控作用域
     * @param execution 原始执行链
     * @param <R> 本次运行的结构化收口结果类型
     * @return 原执行链
     */
    @Override
    public <R> Mono<R> intercept(ReactiveTaskScope<?> scope, Mono<R> execution) {
        return execution;
    }

    /**
     * 压入一次 RUN 方法的创建描述。
     *
     * <p>由注解拦截器在进入 RUN 方法前调用；描述只保存 Invocation 与匹配约定，
     * 不开始 Span、不准备 Recorder、不读取业务状态。</p>
     *
     * @param invocation 当前拦截的调用声明
     * @param convention 匹配的观测约定
     * @return 本次创建描述
     */
    RunCreation beginCreation(ObservationConvention.Invocation invocation, ObservationConvention convention) {
        RunCreation creation = new RunCreation(invocation, convention);
        Deque<RunCreation> stack = CREATION_STACK.get();
        if (stack == null) {
            stack = new ArrayDeque<>();
            CREATION_STACK.set(stack);
        }
        stack.push(creation);
        return creation;
    }

    /**
     * 弹出并冻结一次 RUN 方法的创建描述。
     *
     * <p>必须在 RUN 方法返回或抛错的 finally 中调用，保证严格 LIFO。只有方法同步返回
     * 尚未启动的 Run，且其运行标识与捕获结果一致时描述才有效；缺少捕获、标识不匹配、
     * 多个候选或方法抛错都会使描述失效并记录诊断，后续订阅按无观测透传。</p>
     *
     * @param creation 本次创建描述
     * @param result RUN 方法返回的结果；抛错时为 null
     */
    void endCreation(RunCreation creation, Object result) {
        Deque<RunCreation> stack = CREATION_STACK.get();
        if (stack == null || !stack.remove(creation)) {
            log.warn("AgentObserved RUN creation stack corrupted, observation disabled, target:{}",
                    creation.invocation.target().getClass().getName());
        } else if (stack.isEmpty()) {
            CREATION_STACK.remove();
        }
        if (result instanceof ReactiveTaskRun<?, ?> run
                && creation.capturedRunId != null
                && !creation.conflict
                && creation.capturedRunId.equals(run.runId())) {
            creation.valid = true;
            return;
        }
        log.warn("AgentObserved RUN creation binding invalid, observation disabled, "
                        + "target:{}, method:{}, captured:{}, conflict:{}, result:{}",
                creation.invocation.target().getClass().getName(),
                creation.invocation.method().getName(),
                creation.capturedRunId, creation.conflict,
                result == null ? "none" : result.getClass().getName());
    }

    /**
     * 读取当前线程创建期绑定栈的栈顶描述。
     *
     * @return 栈顶创建描述；栈空时返回 null
     */
    static RunCreation currentCreation() {
        Deque<RunCreation> stack = CREATION_STACK.get();
        return stack == null ? null : stack.peek();
    }

    // ------------------------------------------------------------------
    // RUN 观测
    // ------------------------------------------------------------------

    /**
     * 建立一次受管运行的根观测。
     *
     * <p>真实启动时调用：按稳定身份创建根 Span 和 Recorder 权柄，记录 RUN_STARTED，
     * 在原执行链（任务＋结果处理器）外传播根上下文。根终态来自 outcome 提供的业务
     * 权威结果；根流结束后仍等待已登记子节点收尾，防止 completion 先发射导致父节点
     * 提前结束。Recorder 为 noop 时直接透传原执行链。</p>
     *
     * @param identity 稳定观测身份
     * @param input 已选择的输入快照
     * @param execution 原始执行链
     * @param outcome 业务结果到观测终态的映射
     * @param <T> 执行链结果类型
     * @return 增强后的执行链
     */
    public <T> Mono<T> observeRun(TraceRunContext identity, Object input, Mono<T> execution,
                                  Function<T, ObservationConvention.Outcome> outcome) {
        if (recorder == AgentTraceRecorder.noop()) {
            return execution;
        }
        return Mono.deferContextual(view -> {
            Context parent = ContextPropagationOperator.getOpenTelemetryContextFromContextView(view, Context.current());
            Span span = null;
            RunScope runScope;
            try {
                span = startSpan(identity.agentName(), parent);
                Map<String, Object> attributes = new LinkedHashMap<>(identity.attributes());
                attributes.put("otel.traceId", span.getSpanContext().getTraceId());
                attributes.put("otel.spanId", span.getSpanContext().getSpanId());
                TraceRunContext stableIdentity = new TraceRunContext(identity.runId(), identity.conversationId(),
                        identity.messageId(), identity.userId(), identity.agentName(), identity.agentType(),
                        identity.originRunId(), attributes);
                runScope = new RunScope(recorder.prepare(stableIdentity), span, parent);
            } catch (Throwable error) {
                Span unused = span;
                safely(() -> { if (unused != null) unused.end(); });
                logFailure(error);
                return execution;
            }
            safely(() -> {
                try (Scope ignored = runScope.context().makeCurrent()) {
                    runScope.run().start(snapshot(input));
                }
            });
            // 显式把根 Context 写入 Reactor Context：runWithContext 的 scalar 优化在部分链形下
            // 不写入订阅期 Context，必须依赖本调用保证子节点在无全局 Hook 时仍能继承根。
            return execution.contextWrite(reactorContext ->
                            ContextPropagationOperator.storeOpenTelemetryContext(reactorContext, runScope.context()))
                    .doOnSuccess(value -> safely(() -> runScope.result(outcome.apply(value))))
                    .doOnError(runScope::failure)
                    .doFinally(runScope::close);
        });
    }

    // ------------------------------------------------------------------
    // STEP 观测
    // ------------------------------------------------------------------

    /**
     * 观测一次响应式步骤（Mono 形态）。
     *
     * <p>调用方已在真实订阅时判定需要采集。只有存在 Run 作用域时才建立步骤 Span；
     * 步骤开始时 retain，正常结束读取 output 提供的快照，错误、取消按信号收尾，
     * 结束 Span 后幂等 release。</p>
     *
     * @param name 步骤节点名称
     * @param execution 原始步骤执行链
     * @param output 步骤返回值到输出快照的映射
     * @param <T> 步骤结果类型
     * @return 增强后的执行链
     */
    public <T> Mono<T> observeStep(String name, Mono<T> execution, Function<T, Object> output) {
        return Mono.deferContextual(view -> {
            Context parent = ContextPropagationOperator.getOpenTelemetryContextFromContextView(view, Context.current());
            RunScope runScope = scope(parent);
            if (runScope == null) {
                return execution;
            }
            Runnable release = runScope.retain();
            if (release == null) {
                return execution;
            }
            Span span;
            try {
                span = startSpan(name, parent);
            } catch (Throwable error) {
                release.run();
                logFailure(error);
                return execution;
            }
            Context stepContext = NativeTraceContext.withMode(span.storeInContext(parent), NativeTraceContext.Mode.ENRICH_RUN);
            String nodeId = "step:" + UUID.randomUUID();
            record(runScope.run(), stepContext, TraceEventType.STEP_STARTED, nodeId, null, Map.of("status", "RUNNING"));
            return execution.contextWrite(reactorContext ->
                            ContextPropagationOperator.storeOpenTelemetryContext(reactorContext, stepContext))
                    .doOnSuccess(value -> safely(() -> record(runScope.run(), stepContext, TraceEventType.STEP_FINISHED,
                            nodeId, snapshot(output.apply(value)), Map.of("status", "SUCCEEDED"))))
                    .doOnError(error -> {
                        safely(() -> span.setStatus(StatusCode.ERROR));
                        record(runScope.run(), stepContext, TraceEventType.STEP_FINISHED,
                                nodeId, errorData(error), Map.of("status", "FAILED"));
                    })
                    .doFinally(signal -> {
                        if (signal == SignalType.CANCEL) {
                            record(runScope.run(), stepContext, TraceEventType.STEP_FINISHED, nodeId, null,
                                    Map.of("status", "CANCELLED"));
                        }
                        endSpan(span, release);
                    });
        });
    }

    /**
     * 观测一次响应式步骤（Flux 形态）。
     *
     * <p>与 Mono 形态共享同一套 Span 与关闭协调语义；流正常结束时以完成信号读取
     * output 提供的快照，单个元素不单独记录节点。</p>
     *
     * @param name 步骤节点名称
     * @param execution 原始步骤执行链
     * @param output 完成时输出快照的提供器
     * @param <T> 步骤元素类型
     * @return 增强后的执行链
     */
    public <T> Flux<T> observeStep(String name, Flux<T> execution, Supplier<Object> output) {
        return Flux.deferContextual(view -> {
            Context parent = ContextPropagationOperator.getOpenTelemetryContextFromContextView(view, Context.current());
            RunScope runScope = scope(parent);
            if (runScope == null) {
                return execution;
            }
            Runnable release = runScope.retain();
            if (release == null) {
                return execution;
            }
            Span span;
            try {
                span = startSpan(name, parent);
            } catch (Throwable error) {
                release.run();
                logFailure(error);
                return execution;
            }
            Context stepContext = NativeTraceContext.withMode(span.storeInContext(parent), NativeTraceContext.Mode.ENRICH_RUN);
            String nodeId = "step:" + UUID.randomUUID();
            record(runScope.run(), stepContext, TraceEventType.STEP_STARTED, nodeId, null, Map.of("status", "RUNNING"));
            return execution.contextWrite(reactorContext ->
                            ContextPropagationOperator.storeOpenTelemetryContext(reactorContext, stepContext))
                    .doOnComplete(() -> safely(() -> record(runScope.run(), stepContext, TraceEventType.STEP_FINISHED,
                            nodeId, snapshot(output.get()), Map.of("status", "SUCCEEDED"))))
                    .doOnError(error -> {
                        safely(() -> span.setStatus(StatusCode.ERROR));
                        record(runScope.run(), stepContext, TraceEventType.STEP_FINISHED,
                                nodeId, errorData(error), Map.of("status", "FAILED"));
                    })
                    .doFinally(signal -> {
                        if (signal == SignalType.CANCEL) {
                            record(runScope.run(), stepContext, TraceEventType.STEP_FINISHED, nodeId, null,
                                    Map.of("status", "CANCELLED"));
                        }
                        endSpan(span, release);
                    });
        });
    }

    // ------------------------------------------------------------------
    // 共享静态工具
    // ------------------------------------------------------------------

    /**
     * 读取当前 OTel Context 中的 Run 作用域。
     *
     * @param context 当前 OTel Context
     * @return Run 作用域；不在受管运行内时返回 null
     */
    public static RunScope scope(Context context) {
        return context == null ? null : context.get(RUN_SCOPE);
    }

    /**
     * 序列化并脱敏任意观测数据；序列化失败时返回降级描述，不抛出异常。
     *
     * @param value 原始数据
     * @return 已脱敏的快照
     */
    public static Object snapshot(Object value) {
        if (value == null) {
            return null;
        }
        try {
            return redact(JSON.parse(JSON.toJSONString(value)));
        } catch (Throwable error) {
            logFailure(error);
            return Map.of("valueType", value.getClass().getName(), "serializationError", error.getClass().getName());
        }
    }

    /**
     * 递归移除常见敏感字段，规则与既有 Pipeline 采集保持一致。
     *
     * @param value 已序列化的数据
     * @return 脱敏后的数据
     */
    private static Object redact(Object value) {
        if (value instanceof Map<?, ?> map) {
            Map<String, Object> result = new LinkedHashMap<>();
            map.forEach((key, item) -> {
                String name = String.valueOf(key).toLowerCase(Locale.ROOT).replace("_", "").replace("-", "");
                boolean sensitive = name.contains("apikey") || name.contains("secret") || name.contains("password")
                        || name.contains("authorization") || name.contains("credential") || name.contains("accesstoken")
                        || name.contains("refreshtoken") || name.equals("cookie");
                result.put(String.valueOf(key), sensitive ? "[REDACTED]" : redact(item));
            });
            return result;
        }
        if (value instanceof Collection<?> items) {
            List<Object> result = new ArrayList<>(items.size());
            items.forEach(item -> result.add(redact(item)));
            return result;
        }
        return value;
    }

    /**
     * 在指定 OTel Context 中记录一条 Trace 事件，异常被隔离。
     *
     * @param run 当前 Recorder 权柄
     * @param context 事件所属上下文
     * @param type 事件类型
     * @param nodeId 节点标识
     * @param data 事件数据快照
     * @param attributes 事件属性
     */
    public static void record(AgentTraceRun run, Context context, TraceEventType type, String nodeId,
                              Object data, Map<String, Object> attributes) {
        safely(() -> {
            try (Scope ignored = context.makeCurrent()) {
                run.record(new TraceEvent(type, nodeId, null, Instant.now(), data, attributes));
            }
        });
    }

    /**
     * 构造异常的观测描述。
     *
     * @param error 业务或观测异常
     * @return 异常类型与消息
     */
    public static Map<String, Object> errorData(Throwable error) {
        return Map.of("errorType", error.getClass().getName(),
                "errorMessage", Objects.toString(error.getMessage(), ""));
    }

    /**
     * 以指定父上下文创建观测 Span。
     *
     * @param name Span 名称
     * @param parent 父上下文
     * @return 已开始的 Span
     */
    public static Span startSpan(String name, Context parent) {
        return GlobalOpenTelemetry.getTracer("com.fons.cloud.ai.agent.trace")
                .spanBuilder(name).setParent(parent).startSpan();
    }

    /**
     * 结束 Span 并归还子节点关闭门禁计数。
     *
     * @param span 已开始的 Span，可为 null
     * @param release retain 返回的幂等释放动作
     */
    public static void endSpan(Span span, Runnable release) {
        try {
            safely(() -> { if (span != null) span.end(); });
        } finally {
            release.run();
        }
    }

    private static void safely(Runnable action) {
        try {
            action.run();
        } catch (Throwable error) {
            logFailure(error);
        }
    }

    private static void logFailure(Throwable error) {
        try {
            log.warn("Reactive observation collection failed, business execution continues", error);
        } catch (Throwable ignored) {
            /* 观测和日志异常不能改变业务执行。 */
        }
    }

    // ------------------------------------------------------------------
    // 嵌套类型
    // ------------------------------------------------------------------

    /**
     * 单次受管运行的观测数据与关闭门禁；不持有业务执行权，也不控制运行取消。
     *
     * <p>根上下文包含本作用域，供分析步骤、SDK 节点等子节点继承；Recorder 权柄被
     * 整条 Run 共享，子节点只能记录，不能替根结束。</p>
     */
    public static final class RunScope {

        /**
         * 整条 Run 共享的 Recorder 权柄与固定身份。
         */
        private final AgentTraceRun run;

        /**
         * 根上下文包含本作用域，供分析节点和 Master 继承。
         */
        private final Context context;

        /**
         * Run 根 Span 及其子 Span 关闭协调锁。
         */
        private final Span span;

        private final ReentrantLock lock = new ReentrantLock();

        /**
         * 已开始且尚未完成观测收尾的节点数。
         */
        private int children;

        /**
         * 根执行是否已结束，以及根 Span 是否已经关闭。
         */
        private boolean closing;

        private boolean ended;

        /**
         * 根终态必须由业务结果或异常确定，未获得有效结果时不能默认成功。
         */
        private TraceStatus status = TraceStatus.FAILED;

        /**
         * 已脱敏的 Run 结果快照。
         */
        private Object output;

        /**
         * Run 执行异常；业务结构化失败经 Convention 映射为终态，不进入本字段。
         */
        private Throwable failure;

        RunScope(AgentTraceRun run, Span span, Context parent) {
            this.run = Objects.requireNonNull(run, "trace run cannot be null");
            this.span = span;
            this.context = NativeTraceContext.withMode(span.storeInContext(parent).with(RUN_SCOPE, this),
                    NativeTraceContext.Mode.ENRICH_RUN);
        }

        /**
         * 获取共享的 Recorder 权柄。
         *
         * @return 本 Run 的 Trace 权柄
         */
        public AgentTraceRun run() {
            return run;
        }

        /**
         * 获取根上下文；子节点从这里分支继承，不能沿上一孩子的 Context 继续。
         *
         * @return 包含本作用域的根上下文
         */
        public Context context() {
            return context;
        }

        /**
         * 登记一个已开始的子节点；返回幂等的释放动作。
         *
         * <p>根已进入收口时不再接受新子节点。每个已启动节点恰好释放一次；
         * 不在锁中执行 Recorder 或 Span 动作。</p>
         *
         * @return 释放动作；根已收口时返回 null
         */
        public Runnable retain() {
            lock.lock();
            try {
                if (closing || ended) {
                    return null;
                }
                children++;
            } finally {
                lock.unlock();
            }
            AtomicBoolean released = new AtomicBoolean();
            return () -> {
                if (!released.compareAndSet(false, true)) {
                    return;
                }
                lock.lock();
                try {
                    children--;
                } finally {
                    lock.unlock();
                }
                finishIfReady();
            };
        }

        /**
         * 记录业务权威结果映射的终态。
         *
         * @param outcome Convention 提供的观测终态与输出
         */
        void result(ObservationConvention.Outcome outcome) {
            Object safeOutput = snapshot(outcome == null ? null : outcome.output());
            lock.lock();
            try {
                // 取消已进入收口时，迟到的成功回调不能覆盖 CANCELLED。
                if (closing || ended) {
                    return;
                }
                output = safeOutput;
                TraceStatus outcomeStatus = outcome == null ? null : outcome.status();
                status = outcomeStatus == null ? TraceStatus.FAILED : outcomeStatus;
            } finally {
                lock.unlock();
            }
        }

        /**
         * 记录执行异常；取消异常映射为 CANCELLED。
         *
         * @param error 执行链异常
         */
        void failure(Throwable error) {
            lock.lock();
            try {
                if (closing || ended) {
                    return;
                }
                failure = error;
                status = error instanceof CancellationException ? TraceStatus.CANCELLED : TraceStatus.FAILED;
            } finally {
                lock.unlock();
            }
        }

        /**
         * 根执行流结束，进入关闭等待；取消信号先锁定 CANCELLED 终态。
         *
         * @param signal 根执行流结束信号
         */
        void close(SignalType signal) {
            lock.lock();
            try {
                closing = true;
                if (signal == SignalType.CANCEL) {
                    status = TraceStatus.CANCELLED;
                }
            } finally {
                lock.unlock();
            }
            finishIfReady();
        }

        private void finishIfReady() {
            lock.lock();
            try {
                if (!closing || children != 0 || ended) {
                    return;
                }
                ended = true;
            } finally {
                lock.unlock();
            }
            try {
                safely(() -> {
                    try (Scope ignored = context.makeCurrent()) {
                        run.finish(status, output, failure);
                        span.setAttribute("fons.run.status", status.name());
                        if (status == TraceStatus.FAILED || status == TraceStatus.TIMED_OUT) {
                            span.setStatus(StatusCode.ERROR);
                        }
                    }
                });
            } finally {
                safely(span::end);
            }
        }
    }

    /**
     * 一次 RUN 方法执行期的创建描述；只在创建线程访问，有效性在方法返回时冻结。
     */
    static final class RunCreation {

        /**
         * 当前拦截的调用声明。
         */
        final ObservationConvention.Invocation invocation;

        /**
         * 匹配的观测约定。
         */
        final ObservationConvention convention;

        /**
         * 工厂捕获的本次 Run 标识；同一描述的第二个 Run 捕获按冲突处理。
         */
        private String capturedRunId;

        private boolean conflict;

        /**
         * 创建期判断结果在真实执行前固定；真实启动时读取。
         */
        private volatile boolean valid;

        RunCreation(ObservationConvention.Invocation invocation, ObservationConvention convention) {
            this.invocation = Objects.requireNonNull(invocation, "invocation cannot be null");
            this.convention = Objects.requireNonNull(convention, "convention cannot be null");
        }

        /**
         * 尝试把本描述绑定给指定 Run。
         *
         * <p>创建期在调用线程同步执行，无并发竞争；不需要锁。</p>
         *
         * @param runId 工厂生成的运行标识
         * @return 绑定成功返回 true；已被其他 Run 捕获返回 false
         */
        boolean tryCapture(String runId) {
            if (conflict) {
                return false;
            }
            if (capturedRunId == null) {
                capturedRunId = runId;
                return true;
            }
            conflict = true;
            return false;
        }
    }

    /**
     * 一次 Run 专属的观测拦截实例；真实启动时按冻结的创建描述建立根观测。
     */
    private final class RunObservation implements ReactiveTaskInterceptor {

        /**
         * 本次 Run 绑定的创建描述。
         */
        private final RunCreation creation;

        RunObservation(RunCreation creation) {
            this.creation = creation;
        }

        @Override
        public <R> Mono<R> intercept(ReactiveTaskScope<?> runScope, Mono<R> execution) {
            if (!creation.valid) {
                return execution;
            }
            try {
                ObservationConvention.Invocation invocation = creation.invocation;
                TraceRunContext identity = creation.convention.runContext(invocation, runScope.runId());
                Object input = creation.convention.runInput(invocation);
                return observeRun(identity, input, execution,
                        value -> creation.convention.runOutcome(invocation, value));
            } catch (Throwable error) {
                // 约定映射失败只放弃本次观测，业务执行链原样返回。
                logFailure(error);
                return execution;
            }
        }
    }
}
