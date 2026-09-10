package com.fons.cloud.ai.agent.core;

import com.fons.cloud.ai.agent.model.runtime.RuntimeActions;
import com.fons.cloud.common.base.exception.SystemIntervalException;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.harness.agent.HarnessAgent;
import lombok.Builder;
import lombok.NonNull;
import lombok.experimental.SuperBuilder;
import lombok.extern.slf4j.Slf4j;
import reactor.core.Disposable;
import reactor.core.scheduler.Schedulers;

import java.io.Serial;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/**
 * AgentScope单次Run的行为权柄。
 *
 * <p>取消时先触发AgentScope原生中断，使其有机会整理未完成工具调用并保存当前
 * AgentState。原生流在宽限时间内没有结束时，再强制释放Reactor订阅。</p>
 *
 * @author hongqy
 */
@Slf4j
@SuperBuilder
public class AgentScopeRuntimeActions extends RuntimeActions {

    @Serial
    private static final long serialVersionUID = 1L;

    /**
     * AgentScope原生中断的默认宽限时间。
     */
    protected static final long DEFAULT_INTERRUPT_GRACE_PERIOD_MILLIS = 3000L;

    /**
     * 当前Run使用的AgentScope委托Agent。
     */
    @NonNull
    protected final HarnessAgent delegate;

    /**
     * 当前Run对应的AgentScope运行上下文。
     */
    @NonNull
    protected final RuntimeContext runtimeContext;

    /**
     * 原生中断转为强制终止前的宽限时间。
     */
    @Builder.Default
    protected final long interruptGracePeriodMillis = DEFAULT_INTERRUPT_GRACE_PERIOD_MILLIS;

    /**
     * 原生中断超时后的强制终止任务。
     */
    private final transient AtomicReference<Disposable> forceCancellationTask =
            new AtomicReference<>();

    /**
     * 触发AgentScope原生中断，并登记超时强制终止任务。
     */
    @Override
    protected void doCancelExecution() {
        try {
            delegate.interrupt(runtimeContext);
        } catch (SystemIntervalException exception) {
            throw exception;
        } catch (RuntimeException exception) {
            log.error("Failed to interrupt AgentScope execution, runId:{}",
                    getAgentRunContext().getRunId(), exception);
            throw SystemIntervalException.of("Failed to interrupt AgentScope execution");
        }

        try {
            scheduleForceCancellation();
        } catch (RuntimeException exception) {
            // 原生中断已经成功发出，兜底任务登记失败时直接降级为common强制终止。
            log.warn("Failed to schedule AgentScope force cancellation, runId:{}",
                    getAgentRunContext().getRunId(), exception);
            disposePrimaryExecution();
        }
    }

    /**
     * AgentScope需要等待原生流处理中断信号，不能在interrupt返回后立即释放订阅。
     *
     * @return false，由原生流结束或超时任务完成资源释放
     */
    @Override
    protected boolean shouldReleaseManagedDisposablesAfterCancel() {
        return false;
    }

    /**
     * 登记原生中断超时后的强制终止任务。
     */
    private void scheduleForceCancellation() {
        if (interruptGracePeriodMillis <= 0) {
            disposePrimaryExecution();
            return;
        }

        Disposable task = Schedulers.parallel().schedule(() -> {
            if (isReleased()) {
                return;
            }
            log.warn("AgentScope interrupt grace period elapsed, force cancel execution, runId:{}",
                    getAgentRunContext().getRunId());
            try {
                disposePrimaryExecution();
            } catch (RuntimeException exception) {
                log.warn("Failed to force cancel AgentScope execution, runId:{}",
                        getAgentRunContext().getRunId(), exception);
            }
        }, interruptGracePeriodMillis, TimeUnit.MILLISECONDS);

        if (!forceCancellationTask.compareAndSet(null, task)) {
            task.dispose();
            return;
        }
        if (isReleased()) {
            disposeForceCancellationTask();
        }
    }

    /**
     * 释放AgentScope取消链路持有的超时任务。
     */
    @Override
    protected void releaseExtensionResources() {
        disposeForceCancellationTask();
    }

    /**
     * 幂等释放强制终止任务。
     */
    private void disposeForceCancellationTask() {
        Disposable task = forceCancellationTask.getAndSet(null);
        if (task != null && !task.isDisposed()) {
            task.dispose();
        }
    }

}
