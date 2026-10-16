package com.fons.cloud.ai.agent.infrastructure.observability;

import com.fons.cloud.ai.agent.model.runtime.AgentRunState;
import com.fons.cloud.ai.agent.model.runtime.AgentRunStateChangedEvent;
import com.fons.cloud.ai.agent.model.runtime.AgentRunStateEventPublisher;
import com.fons.cloud.ai.agent.observability.api.AgentTraceRun;
import com.fons.cloud.ai.agent.observability.autoconfigure.instrumentation.ReactiveObservationSupport;
import com.fons.cloud.ai.agent.observability.core.NativeTraceContext;
import com.fons.cloud.ai.agent.observability.model.TraceEvent;
import com.fons.cloud.ai.agent.observability.model.TraceEventType;
import com.fons.cloud.ai.agent.observability.model.TraceStatus;
import io.agentscope.core.event.AgentEvent;
import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.context.Context;
import io.opentelemetry.context.Scope;
import io.opentelemetry.instrumentation.reactor.v3_1.ContextPropagationOperator;
import lombok.extern.slf4j.Slf4j;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Instant;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * 管理一次AgentScope适配Run的Fons Trace生命周期。
 *
 * <p>框架在执行任何父子节点前建立 Run Span，Middleware记录模型、工具和HITL明细；
 * common状态机发布的状态事件负责决定整个Run的开始与结束状态。原生事件流自然结束
 * 不会提前结束Trace，避免其状态与common最终状态不一致。</p>
 *
 * <p>该对象按Run创建，并保存稳定的Run级OpenTelemetry上下文，使取消等发生在
 * 其他线程上的状态事件仍可关联原生AgentScope Trace。全部观测异常均按fail-open处理。</p>
 *
 * @author hongqy
 */
@Slf4j
public final class AgentScopeTraceLifecycle implements AgentRunStateEventPublisher {

    /** Middleware绑定的单次Trace权柄。 */
    private final AtomicReference<AgentTraceRun> traceRun = new AtomicReference<>();

    /** Run的公共父上下文，不随某个子Agent进入或完成而改写。 */
    private final AtomicReference<Context> telemetryContext = new AtomicReference<>();

    /** 框架持有的Run Span，不能由原生子Agent结束。 */
    private final AtomicReference<Span> runSpan = new AtomicReference<>();

    /** 在独立AgentRun创建时捕获上游Context，避免内部subscribeOn丢失Pipeline父节点。 */
    private final Context parentContext;

    /** 可选的外层Pipeline作用域；独立调用Harness时仍由本类管理Recorder生命周期。 */
    private ReactiveObservationSupport.RunScope pipelineScope;

    /** 原生Run Span结束后释放外层根的关闭门禁。 */
    private Runnable releasePipeline;

    /** 受管Run在公共Recorder中的分段节点标识。 */
    private final String runNodeId = "run:" + UUID.randomUUID();

    /** 原生执行流是否自然结束；取消状态可能早于子节点保存完成。 */
    private final AtomicBoolean executionFinished = new AtomicBoolean();

    /** 只保护一次性绑定，不在锁内调用Recorder或执行Agent。 */
    private final ReentrantLock bindingLock = new ReentrantLock();

    /** Run开始事件携带的输入快照。 */
    private final AtomicReference<Object> input = new AtomicReference<>();

    /** 原生Agent最终输出快照。 */
    private final AtomicReference<Object> output = new AtomicReference<>();

    /** 原生执行过程中观察到的异常。 */
    private final AtomicReference<Throwable> failure = new AtomicReference<>();

    /** common状态机是否已经推进到RUNNING。 */
    private final AtomicBoolean startRequested = new AtomicBoolean();

    /** 是否已经调用Trace开始入口。 */
    private final AtomicBoolean started = new AtomicBoolean();

    /** common状态机决定的Trace结束状态。 */
    private final AtomicReference<TraceStatus> terminalStatus = new AtomicReference<>();

    /** 是否已经调用Trace结束入口。 */
    private final AtomicBoolean finished = new AtomicBoolean();

    public AgentScopeTraceLifecycle() {
        this(Context.current());
    }

    public AgentScopeTraceLifecycle(Context parentContext) {
        this.parentContext = parentContext;
    }

    /**
     * 在子恢复之前建立公共Run Span，并通过SDK相同的Reactor桥接传播上下文。
     * execution包含固定身份的绑定及完整执行链；构造和订阅均处于Run上下文中。
     */
    public Flux<AgentEvent> traceRun(String agentName, Supplier<Flux<AgentEvent>> execution) {
        return Flux.deferContextual(subscriberContext -> {
            Context parent = ContextPropagationOperator.getOpenTelemetryContextFromContextView(
                    subscriberContext, parentContext);
            pipelineScope = ReactiveObservationSupport.scope(parent);
            if (pipelineScope != null) {
                releasePipeline = pipelineScope.retain();
            }
            Context runContext = parent;
            try {
                Span span = GlobalOpenTelemetry.getTracer("com.fons.cloud.ai.agent.trace")
                        .spanBuilder("agent.run " + agentName).setParent(parent).startSpan();
                runSpan.set(span);
                // Harness受管分支携带采集所有权标记：SDK Tracer在这些分支只透传，由Middleware采集。
                runContext = AgentScopeTracer.markHarnessOwnership(
                        NativeTraceContext.withMode(span.storeInContext(parent), NativeTraceContext.Mode.ENRICH_RUN));
            } catch (Throwable error) {
                log.warn("Failed to create AgentScope Run span, trace failure ignored", error);
                // Span创建失败时只保留审计，不能把上游HTTP Span误当成Run补充。
                runContext = AgentScopeTracer.markHarnessOwnership(
                        NativeTraceContext.withMode(parent, NativeTraceContext.Mode.AUDIT_ONLY));
            }
            telemetryContext.set(runContext);
            Flux<AgentEvent> events;
            try (Scope ignored = runContext.makeCurrent()) {
                events = execution.get();
            } catch (Throwable error) {
                events = Flux.error(error);
            }
            return ContextPropagationOperator.runWithContext(events, runContext)
                    .doFinally(signal -> {
                        executionFinished.set(true);
                        finishIfNecessary();
                    });
        });
    }

    /** 根层恢复分支及Master汇总必须显式使用此上下文，不能继承上一兄弟的回调上下文。 */
    public Context runContext() {
        Context context = telemetryContext.get();
        return context == null ? Context.current() : context;
    }

    /** Pipeline已绑定固定身份时，Middleware复用它，不再准备第二个Recorder权柄。 */
    public AgentTraceRun inheritedTraceRun() {
        return pipelineScope == null ? null : pipelineScope.run();
    }

    /**
     * 嵌套父节点在恢复孩子之前还没有原生invoke_agent Span，用恢复分段表达真实父子关系。
     * 只有存在挂起孩子的中间节点使用此方法；叶节点直接保留SDK原生Span。
     */
    public <T> Mono<T> traceRecovery(String agentName, String sessionId, Context parent,
                                    Function<Context, Mono<T>> execution) {
        return Mono.defer(() -> {
            if (runSpan.get() == null || traceRun.get() == null) {
                return execution.apply(parent);
            }
            Span span;
            try {
                span = GlobalOpenTelemetry.getTracer("com.fons.cloud.ai.agent.trace")
                        .spanBuilder("agent.restore " + agentName).setParent(parent).startSpan();
            } catch (Throwable error) {
                log.warn("Failed to create AgentScope recovery span, trace failure ignored", error);
                return execution.apply(parent);
            }
            Context branchContext = NativeTraceContext.withMode(span.storeInContext(parent),
                    NativeTraceContext.Mode.ENRICH_RUN);
            String nodeId = "restore:" + UUID.randomUUID();
            recordRecovery(branchContext, nodeId, agentName, sessionId, TraceEventType.STEP_STARTED,
                    TraceStatus.RUNNING);
            return ContextPropagationOperator.runWithContext(Mono.defer(() -> execution.apply(branchContext)),
                            branchContext)
                    .doOnSuccess(value -> recordRecovery(branchContext, nodeId, agentName, sessionId,
                            TraceEventType.STEP_FINISHED, value == null ? TraceStatus.SUSPENDED : TraceStatus.SUCCEEDED))
                    .doOnError(error -> {
                        recordRecovery(branchContext, nodeId, agentName, sessionId,
                                TraceEventType.STEP_FINISHED, TraceStatus.FAILED);
                        safelyExecute("mark AgentScope recovery failure", branchContext,
                                () -> span.setStatus(StatusCode.ERROR));
                    })
                    .doFinally(signal -> safelyExecute("end AgentScope recovery span", branchContext, span::end));
        });
    }

    /** 恢复分段只记录执行者和状态，模型及工具正文仍由原生节点负责。 */
    private void recordRecovery(Context context, String nodeId, String agentName, String sessionId,
                                TraceEventType type, TraceStatus status) {
        safelyExecute("record AgentScope recovery span", context, () -> record(new TraceEvent(
                type, nodeId, null, Instant.now(), Map.of("status", status.name()),
                Map.of("agentName", agentName, "agentSessionId", sessionId))));
    }

    /**
     * 绑定Middleware在框架根入口创建的Trace权柄。
     *
     * <p>common状态可能先于原生Middleware推进到RUNNING，因此绑定完成后会补充执行
     * 尚未处理的开始或结束动作。重复绑定不会覆盖首个有效权柄。</p>
     *
     * @param currentTraceRun 当前Run的Trace权柄
     * @param inputSnapshot 原生输入快照
     */
    public void bind(AgentTraceRun currentTraceRun, Object inputSnapshot) {
        if (currentTraceRun == null) {
            return;
        }
        bindingLock.lock();
        try {
            if (traceRun.get() != null) {
                return;
            }
            input.set(inputSnapshot);
            telemetryContext.compareAndSet(null, Context.current());
            // 最后发布Trace权柄，避免状态线程读取到尚未初始化完成的输入和OTel上下文。
            traceRun.set(currentTraceRun);
        } finally {
            bindingLock.unlock();
        }
        startIfNecessary();
        finishIfNecessary();
    }

    /**
     * 记录AgentScope原生轨迹事件。
     *
     * @param event 原生执行转换后的Fons Trace事件
     */
    public void record(TraceEvent event) {
        AgentTraceRun currentTraceRun = traceRun.get();
        if (currentTraceRun == null || event == null || finished.get()) {
            return;
        }
        // 明细调用方已经位于对应 Agent / Model / Acting 上下文中，不能切回根 Agent Span。
        safelyExecute("record AgentScope trace event", Context.current(),
                () -> currentTraceRun.record(event));
    }

    /**
     * 获取已绑定的根 Run 权柄，供后续原生执行段复用稳定的业务关联信息。
     *
     * <p>子 Agent 共享该权柄，但不能用自己的输出或状态结束根 Run。</p>
     *
     * @return 已绑定权柄；框架尚未进入Run观测范围时为空
     */
    public AgentTraceRun currentTraceRun() {
        return traceRun.get();
    }

    /**
     * 保存原生Agent最终输出，供common状态机结束Trace时使用。
     *
     * @param result 原生Agent输出快照
     */
    public void recordOutput(Object result) {
        output.set(result);
    }

    /**
     * 保存原生执行异常，供common状态机结束Trace时使用。
     *
     * @param cause 原生执行异常
     */
    public void recordFailure(Throwable cause) {
        if (cause != null) {
            failure.compareAndSet(null, cause);
        }
    }

    /**
     * 根据common权威状态推进Trace生命周期。
     *
     * @param event common状态机成功推进后发布的状态事件
     */
    @Override
    public void publish(AgentRunStateChangedEvent event) {
        if (event == null || event.getCurrentState() == null) {
            return;
        }
        if (event.getCurrentState() == AgentRunState.RUNNING) {
            startRequested.set(true);
            startIfNecessary();
            return;
        }

        TraceStatus status = toTraceStatus(event.getCurrentState());
        if (status != null) {
            terminalStatus.compareAndSet(null, status);
            finishIfNecessary();
        }
    }

    /**
     * 在状态与原生Trace权柄都就绪后开始Trace。
     */
    private void startIfNecessary() {
        AgentTraceRun currentTraceRun = traceRun.get();
        if (!startRequested.get()
                || currentTraceRun == null
                || !started.compareAndSet(false, true)) {
            return;
        }
        safelyExecute("start AgentScope trace", telemetryContext.get(), () -> {
            if (pipelineScope == null) {
                currentTraceRun.start(input.get());
            } else {
                // 外层拥有唯一RUN_STARTED；Master只记录自己的执行分段。
                currentTraceRun.record(new TraceEvent(TraceEventType.STEP_STARTED, runNodeId, null,
                        Instant.now(), input.get(), Map.of("status", TraceStatus.RUNNING.name())));
            }
        });
    }

    /**
     * 在common结束状态与原生Trace权柄都就绪后结束Trace。
     */
    private void finishIfNecessary() {
        AgentTraceRun currentTraceRun = traceRun.get();
        TraceStatus status = terminalStatus.get();
        if (status == null
                || (currentTraceRun == null && runSpan.get() == null && releasePipeline == null)
                || ((runSpan.get() != null || releasePipeline != null) && !executionFinished.get())
                || !finished.compareAndSet(false, true)) {
            return;
        }
        Context context = telemetryContext.get();
        // 受管Run Span持续到执行流收口；旧入口仍保留独立终态事件的兼容行为。
        Span span = runSpan.get();
        if (span == null && context != null && NativeTraceContext.currentMode(context)
                != NativeTraceContext.Mode.EVENT) {
            context = NativeTraceContext.withMode(context, NativeTraceContext.Mode.RUN_TERMINAL);
        }
        try {
            if (currentTraceRun != null) {
                safelyExecute("finish AgentScope trace", context, () -> {
                    if (pipelineScope == null) {
                        currentTraceRun.finish(status, output.get(), failure.get());
                    } else {
                        Map<String, Object> result = new java.util.LinkedHashMap<>();
                        result.put("status", status.name());
                        if (output.get() != null) result.put("output", output.get());
                        if (failure.get() != null) result.put("error",
                                ReactiveObservationSupport.errorData(failure.get()));
                        currentTraceRun.record(new TraceEvent(TraceEventType.STEP_FINISHED, runNodeId, null,
                                Instant.now(), result, Map.of("status", status.name())));
                    }
                });
            }
        } finally {
            try {
                if (span != null) {
                    safelyExecute("end AgentScope Run span", context, () -> {
                        try {
                            span.setAttribute("fons.run.status", status.name());
                            if (status == TraceStatus.FAILED || status == TraceStatus.TIMED_OUT) {
                                span.setStatus(StatusCode.ERROR);
                            }
                        } finally {
                            span.end();
                        }
                    });
                }
            } finally {
                if (releasePipeline != null) releasePipeline.run();
            }
        }
    }

    /**
     * 将common运行状态转换为Trace结束状态。
     *
     * @param state common运行状态
     * @return 对应Trace状态；非结束状态返回null
     */
    static TraceStatus toTraceStatus(AgentRunState state) {
        if (state == null) return null;
        return switch (state) {
            case WAITING_APPROVAL -> TraceStatus.SUSPENDED;
            case COMPLETED -> TraceStatus.SUCCEEDED;
            case FAILED -> TraceStatus.FAILED;
            case CANCELLED -> TraceStatus.CANCELLED;
            case REJECTED, APPROVAL_REJECTED -> TraceStatus.REJECTED;
            case TIMED_OUT -> TraceStatus.TIMED_OUT;
            case CREATED, RUNNING -> null;
        };
    }

    /**
     * 在捕获的OpenTelemetry上下文中执行Trace动作并隔离异常。
     *
     * @param action 动作说明
     * @param context 当前动作对应的上下文；生命周期动作使用根上下文，明细使用节点上下文
     * @param runnable Trace动作
     */
    private void safelyExecute(String action, Context context, Runnable runnable) {
        try {
            if (context == null) {
                runnable.run();
                return;
            }
            try (Scope ignored = context.makeCurrent()) {
                runnable.run();
            }
        } catch (Throwable error) {
            try {
                log.warn("Failed to {}, trace failure ignored", action, error);
            } catch (RuntimeException ignored) {
                // 可观测性链路必须保持fail-open，日志实现异常也不能影响Agent主链。
            }
        }
    }
}
