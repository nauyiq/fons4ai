package com.fons.cloud.ai.agent.model.runtime;

import com.fons.cloud.ai.agent.model.hitl.HumanInTheLoopInfo;
import com.fons.cloud.ai.agent.model.response.AgentResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import lombok.extern.slf4j.Slf4j;

import java.util.List;
import java.util.Objects;

/**
 * Agent运行状态机。
 *
 * <p>状态机持有当前Run上下文，只提供受约束的语义推进方法。状态成功推进后发布
 * 状态事件；事件发布失败不会回滚状态，也不会破坏Agent主执行链路。</p>
 *
 * @author hongqy
 */
@Slf4j
public final class AgentRunStateMachine {

    /**
     * 当前Run上下文。
     */
    private final AgentRunContext context;

    /**
     * 状态事件发布端口。
     */
    private final AgentRunStateEventPublisher eventPublisher;

    /**
     * 创建使用空事件发布器的状态机。
     *
     * @param context 当前Run上下文
     */
    public AgentRunStateMachine(AgentRunContext context) {
        this(context, AgentRunStateEventPublisher.noop());
    }

    /**
     * 创建Agent运行状态机。
     *
     * @param context 当前Run上下文
     * @param eventPublisher 状态事件发布器；为空时使用空实现
     */
    public AgentRunStateMachine(AgentRunContext context,
                                AgentRunStateEventPublisher eventPublisher) {
        if (context == null) {
            throw BusinessRuntimeException.of(AgentResultCode.FAILED_EXECUTE_AGENT);
        }
        this.context = context;
        this.eventPublisher = eventPublisher == null
                ? AgentRunStateEventPublisher.noop() : eventPublisher;
    }

    /**
     * 判断状态机是否绑定指定Run上下文。
     *
     * @param context 待检查的Run上下文
     * @return true表示绑定同一上下文实例
     */
    public boolean isBoundTo(AgentRunContext context) {
        return this.context == context;
    }

    /**
     * 尝试启动当前Run。
     *
     * @return 是否首次成功进入RUNNING
     */
    public boolean tryStart() {
        AgentRunState previousState = AgentRunState.CREATED;
        long occurredAt;
        synchronized (this) {
            if (!context.compareAndSetState(previousState, AgentRunState.RUNNING)) {
                return false;
            }
            occurredAt = System.currentTimeMillis();
            context.markStarted(occurredAt);
        }
        publish(previousState, AgentRunState.RUNNING, occurredAt);
        return true;
    }

    /**
     * 尝试暂停当前执行分段并保存未解决的HITL信息快照。
     *
     * @param hitlInfos HITL信息列表
     * @return 是否首次成功进入WAITING_APPROVAL
     */
    public boolean tryPauseForApprovals(List<HumanInTheLoopInfo> hitlInfos) {
        if (hitlInfos == null || hitlInfos.isEmpty()
                || hitlInfos.stream().anyMatch(Objects::isNull)) {
            return false;
        }
        AgentRunState previousState = AgentRunState.RUNNING;
        long occurredAt;
        synchronized (this) {
            if (!context.compareAndSetState(previousState, AgentRunState.WAITING_APPROVAL)) {
                return false;
            }
            context.replaceHumanInTheLoopInfos(hitlInfos);
            occurredAt = System.currentTimeMillis();
        }
        publish(previousState, AgentRunState.WAITING_APPROVAL, occurredAt);
        return true;
    }

    /**
     * 尝试将当前Run推进到不可逆终态。
     *
     * <p>WAITING_APPROVAL只允许被取消、超时或明确的顶层审批拒绝终结。</p>
     *
     * @param terminalState 目标终态
     * @return 是否首次成功进入目标终态
     */
    public boolean tryFinalize(AgentRunState terminalState) {
        if (terminalState == null || !terminalState.isTerminal()) {
            return false;
        }

        AgentRunState previousState;
        long occurredAt;
        synchronized (this) {
            previousState = context.getState();
            if (previousState.isTerminal()) {
                return false;
            }
            if (previousState == AgentRunState.WAITING_APPROVAL
                    && terminalState != AgentRunState.CANCELLED
                    && terminalState != AgentRunState.TIMED_OUT
                    && terminalState != AgentRunState.APPROVAL_REJECTED) {
                return false;
            }
            if (!context.compareAndSetState(previousState, terminalState)) {
                return false;
            }

            occurredAt = System.currentTimeMillis();
            context.markFinished(occurredAt);
            context.clearHumanInTheLoopInfos();
        }
        publish(previousState, terminalState, occurredAt);
        return true;
    }

    private void publish(AgentRunState previousState,
                         AgentRunState currentState,
                         long occurredAt) {
        AgentRunStateChangedEvent event = AgentRunStateChangedEvent.builder()
                .runId(context.getRunId())
                .conversationId(context.getConversationId())
                .previousState(previousState)
                .currentState(currentState)
                .occurredAt(occurredAt)
                .build();
        try {
            eventPublisher.publish(event);
        } catch (RuntimeException exception) {
            log.warn("Failed to publish Agent run state event, runId:{}, previousState:{}, currentState:{}",
                    context.getRunId(), previousState, currentState, exception);
        }
    }
}
