package com.fons.cloud.ai.agent.core;

import com.fons.cloud.ai.agent.model.runtime.RuntimeActions;
import com.fons.cloud.common.base.exception.SystemIntervalException;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.harness.agent.HarnessAgent;
import lombok.NonNull;
import lombok.experimental.SuperBuilder;
import lombok.extern.slf4j.Slf4j;

import java.io.Serial;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

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

    /** 提供根节点状态存储给审批处理器，生命周期权柄仍由本Run持有。 */
    public HarnessAgent getDelegateForApproval() {
        return delegate;
    }

    /** 当前正在恢复的本地子节点；取消时也必须通知它们保存原生中断状态。 */
    private final Map<RuntimeContext, HarnessAgent> recoveringChildren = new ConcurrentHashMap<>();

    /** 将子节点的原生执行纳入本Run取消范围。 */
    public void registerRecoveringChild(HarnessAgent child, RuntimeContext context) {
        // 工厂可能返回共享Agent，按本次会话上下文登记，避免前一个子流清理后一个子流的关联。
        recoveringChildren.put(context, child);
        // 覆盖注册与取消同时发生的竞争，不能让晚注册的子节点继续执行。
        if (isCancellationRequested()) {
            child.interrupt(context);
        }
    }

    /** 子节点流自然结束后释放取消关联，不关闭可能由下游共享的Agent实例。 */
    public void releaseRecoveringChild(RuntimeContext context) {
        recoveringChildren.remove(context);
    }

    /**
     * 中断当前顶层HarnessAgent执行。
     */
    public void interruptNativeExecution() {
        delegate.interrupt(runtimeContext);
        recoveringChildren.forEach((context, child) -> child.interrupt(context));
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
