package com.fons.cloud.ai.agent.infrastructure.observability;

import com.fons.cloud.ai.agent.observability.api.AgentTraceRecorder;
import com.fons.cloud.ai.agent.observability.autoconfigure.instrumentation.ReactiveObservationSupport;
import com.fons.cloud.ai.agent.observability.core.NativeTraceContext;
import com.fons.cloud.ai.agent.observability.model.TraceEventType;
import com.fons.cloud.ai.agent.observability.model.TraceStatus;
import io.agentscope.core.agent.AgentBase;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.TextBlock;
import io.agentscope.core.model.ChatResponse;
import io.agentscope.core.model.ChatUsage;
import io.agentscope.core.model.ChatModelBase;
import io.agentscope.core.model.GenerateOptions;
import io.agentscope.core.model.ToolSchema;
import io.agentscope.core.tracing.NoopTracer;
import io.agentscope.core.tracing.Tracer;
import io.agentscope.core.tracing.TracerRegistry;
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
import reactor.util.context.ContextView;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Supplier;

/**
 * AgentScope 原生 Tracer 桥接：自动采集普通 AgentBase 分析 Agent 与模型调用。
 *
 * <p>接入范围是注解观测作用域内的普通 Agent：AgentBase.call 进入
 * {@link TracerRegistry#callAgent} 生成 invoke_agent 节点，ChatModelBase.stream 进入
 * {@link TracerRegistry#callModel} 生成 chat 节点，二者共用外层 Run 的 Recorder 权柄。
 * 工具明细继续由既有 Harness Middleware 负责，本 Tracer 对工具调用直接透传。</p>
 *
 * <p>采集所有权通过进程内 OTel ContextKey 标记划分：由 {@link AgentScopeTraceLifecycle}
 * 管理的 Harness 执行链携带所有权标记，本 Tracer 在这些分支只透传，避免与
 * Middleware 重复记录 Agent 或 Generation；没有观测作用域的普通 SDK 调用同样透传，
 * 不制造孤立新根。标记不能通过用户参数或持久化状态伪造。</p>
 *
 * <p>JVM 全局注册是单实例边界：仅在当前为 Noop 或本框架管理时安装，销毁时只在
 * 仍持有注册时恢复 previous；遇到未知第三方 Tracer 保留其注册并降级本桥接，
 * 不调用 resetToNoop，也不对复用的 previous 或全局 OTel SDK 执行 shutdown。</p>
 *
 * @author hongqy
 */
@Slf4j
public final class AgentScopeTracer implements Tracer {

    /**
     * Harness 受管执行链的采集所有权标记；进程内传播，不进入用户参数或持久化状态。
     */
    private static final ContextKey<Boolean> HARNESS_OWNERSHIP = ContextKey.named("fons.harness.ownership");

    /**
     * 与注解观测和原生 Middleware 共用的 Recorder；noop 时全部入口透传。
     */
    private final AgentTraceRecorder recorder;

    /**
     * 只协调本框架自身的注册与卸载，不保护第三方注册。
     */
    private final ReentrantLock registrationLock = new ReentrantLock();

    /**
     * 是否已把本 Tracer 安装到全局注册点。
     */
    private final AtomicBoolean registered = new AtomicBoolean();

    /**
     * 安装前保存的 Tracer；销毁时仅在本 Tracer 仍持有时恢复。
     */
    private Tracer previous;

    /**
     * 创建 SDK Tracer 桥接。
     *
     * @param recorder 共享的 Trace Recorder，可为 noop
     */
    public AgentScopeTracer(AgentTraceRecorder recorder) {
        this.recorder = Objects.requireNonNull(recorder, "recorder cannot be null");
    }

    // ------------------------------------------------------------------
    // 采集所有权标记
    // ------------------------------------------------------------------

    /**
     * 为 Harness 受管执行链标记采集所有权。
     *
     * <p>由 {@link AgentScopeTraceLifecycle} 在执行原生节点之前调用；标记跟随整条
     * 受管分支传播，SDK Tracer 在这些分支只透传，由原有 Middleware 采集。</p>
     *
     * @param context 受管分支的根上下文
     * @return 携带所有权标记的上下文
     */
    public static Context markHarnessOwnership(Context context) {
        return context == null ? null : context.with(HARNESS_OWNERSHIP, Boolean.TRUE);
    }

    /**
     * 判断当前上下文是否属于 Harness 采集所有权范围。
     *
     * @param context 当前 OTel Context
     * @return 受管分支返回 true
     */
    static boolean harnessOwned(Context context) {
        return context != null && Boolean.TRUE.equals(context.get(HARNESS_OWNERSHIP));
    }

    // ------------------------------------------------------------------
    // Agent 与模型节点
    // ------------------------------------------------------------------

    /**
     * 为普通 AgentBase 调用建立 invoke_agent 节点。
     *
     * <p>没有观测作用域、Harness 已持有所有权或关闭采集时直接透传原执行。
     * 节点使用外层 Run 的共享 Recorder，retain/release 与根关闭门禁协调。</p>
     */
    @Override
    public Mono<Msg> callAgent(AgentBase agent, List<Msg> messages, Supplier<Mono<Msg>> execution) {
        return Mono.deferContextual(view -> {
            Context parent = ContextPropagationOperator.getOpenTelemetryContextFromContextView(view, Context.current());
            ReactiveObservationSupport.RunScope runScope = ReactiveObservationSupport.scope(parent);
            if (runScope == null || harnessOwned(parent) || recorder == AgentTraceRecorder.noop()) {
                return execution.get();
            }
            Mono<Msg> executionMono = execution.get();
            if (executionMono == null) {
                return Mono.error(new IllegalStateException("callAgent execution returned null"));
            }
            Runnable release = runScope.retain();
            if (release == null) {
                return executionMono;
            }
            Span agentSpan;
            Context agentContext;
            String nodeId;
            Map<String, Object> actor;
            try {
                agentSpan = ReactiveObservationSupport.startSpan("invoke_agent " + agent.getName(), parent);
                agentSpan.setAttribute("gen_ai.operation.name", "invoke_agent");
                agentSpan.setAttribute("gen_ai.agent.name", agent.getName());
                agentSpan.setAttribute("gen_ai.agent.id", agent.getAgentId());
                agentContext = NativeTraceContext.withMode(agentSpan.storeInContext(parent),
                        NativeTraceContext.Mode.ENRICH_AGENT);
                nodeId = "analysis:" + UUID.randomUUID();
                actor = Map.of("agentName", agent.getName(), "agentId", agent.getAgentId(),
                        "agentSessionId", runScope.run().context().conversationId(), "status", "RUNNING");
            } catch (Throwable error) {
                release.run();
                logFailure("create invoke_agent node", error);
                return executionMono;
            }
            Map<String, Object> input = Map.of("messages", ReactiveObservationSupport.snapshot(messages));
            ReactiveObservationSupport.record(runScope.run(), agentContext, TraceEventType.STEP_STARTED,
                    nodeId, input, actor);
            return executionMono
                    .contextWrite(reactorContext ->
                            ContextPropagationOperator.storeOpenTelemetryContext(reactorContext, agentContext))
                    .doOnSuccess(value -> ReactiveObservationSupport.record(runScope.run(), agentContext,
                            TraceEventType.STEP_FINISHED, nodeId,
                            value == null ? null : Map.of("text", ReactiveObservationSupport.snapshot(value)),
                            finishedAttributes(actor, TraceStatus.SUCCEEDED)))
                    .doOnError(error -> {
                        safely(() -> agentSpan.setStatus(StatusCode.ERROR));
                        ReactiveObservationSupport.record(runScope.run(), agentContext, TraceEventType.STEP_FINISHED,
                                nodeId, ReactiveObservationSupport.errorData(error),
                                finishedAttributes(actor, TraceStatus.FAILED));
                    })
                    .doFinally(signal -> {
                        if (signal == SignalType.CANCEL) {
                            ReactiveObservationSupport.record(runScope.run(), agentContext, TraceEventType.STEP_FINISHED,
                                    nodeId, null, finishedAttributes(actor, TraceStatus.CANCELLED));
                        }
                        ReactiveObservationSupport.endSpan(agentSpan, release);
                    });
        });
    }

    /**
     * 为真实模型调用建立 chat 节点并聚合用量。
     *
     * <p>透传所有 ChatResponse，不改变流式拼接与工具、结构化输出参数；Usage 取最后一个
     * 非空快照，不累计多个累计用量。已有模型观测上下文的嵌套调用直接透传，
     * 避免重复 Generation。正常、错误、取消路径只结束一次。</p>
     */
    @Override
    public Flux<ChatResponse> callModel(ChatModelBase model, List<Msg> messages, List<ToolSchema> tools,
                                        GenerateOptions options, Supplier<Flux<ChatResponse>> execution) {
        return Flux.deferContextual(view -> {
            Context parent = ContextPropagationOperator.getOpenTelemetryContextFromContextView(view, Context.current());
            ReactiveObservationSupport.RunScope runScope = ReactiveObservationSupport.scope(parent);
            if (runScope == null || harnessOwned(parent) || recorder == AgentTraceRecorder.noop()) {
                return execution.get();
            }
            if (NativeTraceContext.currentMode(parent) == NativeTraceContext.Mode.ENRICH_MODEL) {
                // 外层模型观测已建立 chat 节点，嵌套调用透传以保持采集所有权。
                // ENRICH_AGENT 是本 Tracer 正常的父节点传播模式，不构成防重信号。
                return execution.get();
            }
            Flux<ChatResponse> executionFlux = execution.get();
            if (executionFlux == null) {
                return Flux.error(new IllegalStateException("callModel execution returned null"));
            }
            Runnable release = runScope.retain();
            if (release == null) {
                return executionFlux;
            }
            Span generation;
            Context modelContext;
            String nodeId;
            Map<String, Object> actor;
            try {
                generation = ReactiveObservationSupport.startSpan("chat " + model.getModelName(), parent);
                generation.setAttribute("gen_ai.operation.name", "chat");
                generation.setAttribute("gen_ai.request.model", model.getModelName());
                modelContext = NativeTraceContext.withMode(generation.storeInContext(parent),
                        NativeTraceContext.Mode.ENRICH_MODEL);
                nodeId = "analysis:" + UUID.randomUUID();
                actor = Map.of("status", "RUNNING");
            } catch (Throwable error) {
                release.run();
                logFailure("create chat node", error);
                return executionFlux;
            }
            Map<String, Object> input = Map.of("messages", ReactiveObservationSupport.snapshot(messages));
            ReactiveObservationSupport.record(runScope.run(), modelContext, TraceEventType.MODEL_CALL_STARTED,
                    nodeId + ":model", input, actor);
            StringBuilder text = new StringBuilder();
            ReentrantLock contentLock = new ReentrantLock();
            AtomicReference<ChatUsage> latestUsage = new AtomicReference<>();
            AtomicReference<Throwable> failure = new AtomicReference<>();
            return executionFlux
                    .contextWrite(reactorContext ->
                            ContextPropagationOperator.storeOpenTelemetryContext(reactorContext, modelContext))
                    .doOnNext(response -> safely(() -> {
                        if (response.getUsage() != null) {
                            latestUsage.set(response.getUsage());
                        }
                        // 取消收尾可能来自另一线程，短锁只保护正文快照，不包住模型或Recorder。
                        contentLock.lock();
                        try {
                            if (response.getContent() != null) {
                                response.getContent().forEach(block -> {
                                    if (block instanceof TextBlock value) {
                                        text.append(value.getText());
                                    }
                                });
                            }
                        } finally {
                            contentLock.unlock();
                        }
                    }))
                    .doOnError(failure::set)
                    .doFinally(signal -> {
                        TraceStatus status = signal == SignalType.CANCEL ? TraceStatus.CANCELLED
                                : failure.get() == null ? TraceStatus.SUCCEEDED : TraceStatus.FAILED;
                        Map<String, Object> attributes = new LinkedHashMap<>();
                        attributes.put("status", status.name());
                        safely(() -> {
                            ChatUsage usage = latestUsage.get();
                            if (usage != null) {
                                generation.setAttribute("gen_ai.usage.input_tokens", usage.getInputTokens());
                                generation.setAttribute("gen_ai.usage.output_tokens", usage.getOutputTokens());
                                attributes.put("inputTokens", usage.getInputTokens());
                                attributes.put("outputTokens", usage.getOutputTokens());
                                attributes.put("cachedTokens", usage.getCachedTokens());
                                attributes.put("totalTokens", usage.getTotalTokens());
                            }
                            if (failure.get() != null) {
                                generation.setStatus(StatusCode.ERROR);
                            }
                            String responseText;
                            contentLock.lock();
                            try {
                                responseText = text.toString();
                            } finally {
                                contentLock.unlock();
                            }
                            Object output = failure.get() == null
                                    ? Map.of("text", responseText)
                                    : ReactiveObservationSupport.errorData(failure.get());
                            ReactiveObservationSupport.record(runScope.run(), modelContext,
                                    TraceEventType.MODEL_CALL_FINISHED, nodeId + ":model", output, attributes);
                        });
                        ReactiveObservationSupport.endSpan(generation, release);
                    });
        });
    }

    /**
     * 按 SDK 契约恢复 ContextView 对应的 OTel Context 后执行 Supplier。
     *
     * <p>Supplier 恰好执行一次；自身上下文解析失败时降级为当前线程上下文透传，
     * 业务 Supplier 抛出的异常原样传播。</p>
     */
    @Override
    public <TResp> TResp runWithContext(ContextView contextView, Supplier<TResp> execution) {
        Context context;
        try {
            context = ContextPropagationOperator.getOpenTelemetryContextFromContextView(
                    contextView, Context.current());
        } catch (Throwable error) {
            logFailure("resolve context for runWithContext", error);
            return execution.get();
        }
        try (Scope ignored = context.makeCurrent()) {
            return execution.get();
        }
    }

    // ------------------------------------------------------------------
    // JVM 全局注册与卸载
    // ------------------------------------------------------------------

    /**
     * 安装到全局 TracerRegistry。
     *
     * <p>仅在当前为 Noop 或本 Tracer 自身时安装；遇到未知第三方 Tracer 保留其注册，
     * 记录诊断并保持本桥接降级（不采集分析 Agent），不覆盖、不叠加。</p>
     */
    public void install() {
        if (registered.get()) {
            return;
        }
        registrationLock.lock();
        try {
            if (registered.get()) {
                return;
            }
            Tracer current = TracerRegistry.get();
            if (current == this) {
                registered.set(true);
                return;
            }
            if (!(current instanceof NoopTracer)) {
                log.warn("Existing non-noop Tracer detected ({}), AgentScope analysis auto-capture degraded",
                        current.getClass().getName());
                return;
            }
            previous = current;
            TracerRegistry.register(this);
            registered.set(true);
        } finally {
            registrationLock.unlock();
        }
    }

    /**
     * 卸载并恢复安装前的 Tracer。
     *
     * <p>只有本 Tracer 仍持有注册时才恢复 previous；不调用 resetToNoop 关闭其他组件，
     * 不对复用的 previous 或全局 OTel SDK 执行 shutdown。</p>
     */
    @Override
    public void shutdown() {
        if (!registered.compareAndSet(true, false)) {
            return;
        }
        registrationLock.lock();
        try {
            if (TracerRegistry.get() == this && previous != null) {
                TracerRegistry.register(previous);
            }
            previous = null;
        } finally {
            registrationLock.unlock();
        }
    }

    // ------------------------------------------------------------------
    // 内部工具
    // ------------------------------------------------------------------

    private static Map<String, Object> finishedAttributes(Map<String, Object> actor, TraceStatus status) {
        Map<String, Object> attributes = new LinkedHashMap<>(actor);
        attributes.put("status", status.name());
        return attributes;
    }

    private static void safely(Runnable action) {
        try {
            action.run();
        } catch (Throwable error) {
            logFailure("collect AgentScope tracer event", error);
        }
    }

    private static void logFailure(String action, Throwable error) {
        try {
            log.warn("AgentScope tracer failed: {}, business execution continues", action, error);
        } catch (Throwable ignored) {
            /* 观测和日志异常不能改变业务执行。 */
        }
    }
}
