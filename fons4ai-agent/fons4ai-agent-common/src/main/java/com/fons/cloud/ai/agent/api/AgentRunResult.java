package com.fons.cloud.ai.agent.api;

import com.fons.cloud.ai.agent.chat.AgentChatFinalContext;
import lombok.Builder;
import lombok.Getter;

import java.util.Objects;

/**
 * 一次智能体执行的结构化终态结果。
 */
@Getter
public final class AgentRunResult {
    /** 执行唯一标识。 */
    private final String runId;
    /** 会话标识。 */
    private final String conversationId;
    /** 不可逆的终态。 */
    private final AgentRunState state;
    /** 聚合后的最终上下文，不包含供应商对象或异常堆栈。 */
    private final AgentChatFinalContext finalContext;
    /** 可选的安全错误码。 */
    private final String errorCode;
    /** 可选的安全错误信息。 */
    private final String errorMessage;

    @Builder
    public AgentRunResult(String runId, String conversationId, AgentRunState state,
                          AgentChatFinalContext finalContext, String errorCode, String errorMessage) {
        this.runId = Objects.requireNonNull(runId, "runId cannot be null");
        this.conversationId = Objects.requireNonNull(conversationId, "conversationId cannot be null");
        this.state = Objects.requireNonNull(state, "state cannot be null");
        if (!state.isTerminal()) {
            throw new IllegalArgumentException("AgentRunResult requires a terminal state");
        }
        this.finalContext = Objects.requireNonNullElseGet(finalContext,
                () -> AgentChatFinalContext.builder().build());
        this.errorCode = errorCode;
        this.errorMessage = errorMessage;
    }
}
