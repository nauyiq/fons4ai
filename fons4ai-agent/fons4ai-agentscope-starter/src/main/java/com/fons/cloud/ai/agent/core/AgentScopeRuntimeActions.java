package com.fons.cloud.ai.agent.core;

import com.fons.cloud.ai.agent.model.runtime.RuntimeActions;
import com.fons.cloud.common.base.exception.SystemIntervalException;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.harness.agent.HarnessAgent;
import lombok.NonNull;
import lombok.experimental.SuperBuilder;
import lombok.extern.slf4j.Slf4j;

import java.io.Serial;

/**
 * AgentScope单次Run的行为权柄。
 *
 * <p>取消时先触发顶层HarnessAgent原生中断，使其有机会整理未完成工具调用并保存当前
 * AgentState。等待原生流自然结束后释放订阅，不提供超时强制释放策略。</p>
 *
 * @author hongqy
 */
@Slf4j
@SuperBuilder
public class AgentScopeRuntimeActions extends RuntimeActions {

    @Serial
    private static final long serialVersionUID = 1L;

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
     * 中断当前顶层HarnessAgent执行。
     */
    public void interruptNativeExecution() {
        delegate.interrupt(runtimeContext);
    }

    /**
     * 触发AgentScope原生中断，保留订阅供原生状态持久化与自然收口。
     */
    @Override
    protected void doCancelExecution() {
        try {
            interruptNativeExecution();
        } catch (SystemIntervalException exception) {
            throw exception;
        } catch (RuntimeException exception) {
            log.error("Failed to interrupt AgentScope execution, runId:{}",
                    getAgentRunContext().getRunId(), exception);
            throw SystemIntervalException.of("Failed to interrupt AgentScope execution");
        }

    }

    /**
     * AgentScope需要等待原生流处理中断信号，不能在interrupt返回后立即释放订阅。
     *
     * @return false，由原生流结束完成资源释放
     */
    @Override
    protected boolean shouldReleaseManagedDisposablesAfterCancel() {
        return false;
    }

}
