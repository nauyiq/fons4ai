package com.fons.cloud.ai.agent.infrastructure.observability;

import com.fons.cloud.ai.agent.model.runtime.AgentRunState;
import com.fons.cloud.ai.agent.model.runtime.AgentRunStateChangedEvent;
import com.fons.cloud.ai.agent.model.runtime.AgentRunStateEventPublisher;
import com.fons.cloud.ai.agent.observability.api.AgentTraceRun;
import com.fons.cloud.ai.agent.observability.model.TraceEvent;
import com.fons.cloud.ai.agent.observability.model.TraceStatus;
import io.opentelemetry.context.Context;
import io.opentelemetry.context.Scope;
import lombok.extern.slf4j.Slf4j;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * 管理一次AgentScope适配Run的Fons Trace生命周期。
 *
 * <p>AgentScope Middleware负责绑定原生轨迹权柄并记录模型、工具和HITL明细；
 * common状态机发布的状态事件负责决定整个Run的开始与结束状态。原生事件流自然结束
 * 不会提前结束Trace，避免其状态与common最终状态不一致。</p>
 *
 * <p>该对象按Run创建，并保存Middleware绑定时的OpenTelemetry上下文，使取消等发生在
 * 其他线程上的状态事件仍可关联原生AgentScope Trace。全部观测异常均按fail-open处理。</p>
 *
 * @author hongqy
 */
@Slf4j
public final class AgentScopeTraceLifecycle implements AgentRunStateEventPublisher {

    /** Middleware绑定的单次Trace权柄。 */
    private final AtomicReference<AgentTraceRun> traceRun = new AtomicReference<>();

    /** Middleware进入原生调用链时捕获的OpenTelemetry上下文。 */
    private final AtomicReference<Context> telemetryContext = new AtomicReference<>();

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

    /**
     * 绑定Middleware在原生调用链中创建的Trace权柄。
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
        synchronized (this) {
            if (traceRun.get() != null) {
                return;
            }
            input.set(inputSnapshot);
            telemetryContext.set(Context.current());
            // 最后发布Trace权柄，避免状态线程读取到尚未初始化完成的输入和OTel上下文。
            traceRun.set(currentTraceRun);
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
        safelyExecute("record AgentScope trace event", () -> currentTraceRun.record(event));
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
        safelyExecute("start AgentScope trace", () -> currentTraceRun.start(input.get()));
    }

    /**
     * 在common结束状态与原生Trace权柄都就绪后结束Trace。
     */
    private void finishIfNecessary() {
        AgentTraceRun currentTraceRun = traceRun.get();
        TraceStatus status = terminalStatus.get();
        if (currentTraceRun == null
                || status == null
                || !finished.compareAndSet(false, true)) {
            return;
        }
        safelyExecute("finish AgentScope trace",
                () -> currentTraceRun.finish(status, output.get(), failure.get()));
    }

    /**
     * 将common运行状态转换为Trace结束状态。
     *
     * @param state common运行状态
     * @return 对应Trace状态；非结束状态返回null
     */
    private static TraceStatus toTraceStatus(AgentRunState state) {
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
     * @param runnable Trace动作
     */
    private void safelyExecute(String action, Runnable runnable) {
        try {
            Context context = telemetryContext.get();
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
