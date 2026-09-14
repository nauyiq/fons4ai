package com.fons.cloud.ai.agent.model.runtime;

import lombok.Builder;
import lombok.Getter;

/**
 * Agent运行状态变更事件。
 *
 * <p>该事件只描述一次成功的状态推进，不携带业务数据。监听器异常不会回滚已经完成的
 * 状态变更。</p>
 *
 * @author hongqy
 */
@Getter
@Builder
public final class AgentRunStateChangedEvent {

    /**
     * 执行唯一标识。
     */
    private final String runId;

    /**
     * 会话标识。
     */
    private final String conversationId;

    /**
     * 变更前状态。
     */
    private final AgentRunState previousState;

    /**
     * 变更后状态。
     */
    private final AgentRunState currentState;

    /**
     * 状态变更时间戳。
     */
    private final long occurredAt;
}
