package com.fons.cloud.ai.agent.api;

import com.fons.cloud.ai.agent.chat.AgentChatFinalContext;
import lombok.Builder;
import lombok.Getter;

import java.util.Objects;

/**
 * 一次智能体执行分段的结构化结果，可表示不可逆终态或 checkpoint 审批等待快照。
 * @author hongqy
 */
@Getter
public final class AgentRunResult {

    /** 可选的消息标识。 */
    private final String messageId;
    /** 执行唯一标识。 */
    private final String runId;
    /** 会话标识。 */
    private final String conversationId;
    /** 不可逆终态或 WAITING_APPROVAL。 */
    private final AgentRunState state;
    /** 聚合后的最终上下文，不包含供应商对象或异常堆栈。 */
    private final AgentChatFinalContext finalContext;
    /** 可选的安全错误码。 */
    private final String errorCode;
    /** 可选的安全错误信息。 */
    private final String errorMessage;
    /** WAITING_APPROVAL 时可选的待审批请求标识。 */
    private final String pendingApprovalId;

    @Builder
    public AgentRunResult(String messageId, String runId, String conversationId, AgentRunState state,
                          AgentChatFinalContext finalContext, String errorCode, String errorMessage,
                          String pendingApprovalId) {
        this.messageId = messageId;
        this.runId = Objects.requireNonNull(runId, "runId cannot be null");
        this.conversationId = Objects.requireNonNull(conversationId, "conversationId cannot be null");
        this.state = Objects.requireNonNull(state, "state cannot be null");
        if (!state.isTerminal() && state != AgentRunState.WAITING_APPROVAL) {
            throw new IllegalArgumentException("AgentRunResult requires a terminal or waiting state");
        }
        this.finalContext = Objects.requireNonNullElseGet(finalContext,
                () -> AgentChatFinalContext.builder().build());
        this.errorCode = errorCode;
        this.errorMessage = errorMessage;
        this.pendingApprovalId = pendingApprovalId;
    }
}
