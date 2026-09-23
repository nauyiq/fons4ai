package com.fons.cloud.ai.agent.model.response;

import com.fons.cloud.ai.agent.model.hitl.HumanInTheLoopInfo;
import com.fons.cloud.ai.agent.model.runtime.AgentRunState;
import lombok.Builder;
import lombok.Getter;

import java.util.List;

/**
 * 一次智能体执行分段的结构化结果，可表示不可逆终态或 checkpoint 审批等待快照。
 *
 * @author hongqy
 */
@Getter
public final class AgentRunResult {

    /**
     * 执行唯一标识。
     */
    private final String runId;

    /**
     * 会话标识。
     */
    private final String conversationId;

    /**
     * 可选的消息标识。
     */
    private final String messageId;

    /**
     * 当前执行分段的收口状态；WAITING_APPROVAL 表示审批暂停，其余终态按状态机解释。
     */
    private final AgentRunState state;

    /**
     * 可选的安全错误码。
     */
    private final String errorCode;

    /**
     * 可选的安全错误信息。
     */
    private final String errorMessage;

    /**
     * 可选的完成信息。
     */
    private final AgentCompleteInfo completeInfo;

    /**
     * 当前执行分段交给用户的结构化交互信息。
     *
     * <p>WAITING_APPROVAL 时为待审批快照；COMPLETED 时可以包含 INPUT_REQUIRED，
     * 表示本轮正常结束、用户通过同一会话的新请求补充信息。不得仅凭列表非空判断为审批等待。</p>
     */
    private final List<HumanInTheLoopInfo> humanInTheLoopInfos;

    /**
     * 创建一次Agent执行分段结果。
     *
     * @param runId 执行唯一标识
     * @param conversationId 会话标识
     * @param messageId 消息标识
     * @param state 执行状态
     * @param errorCode 错误码
     * @param errorMessage 错误信息
     * @param completeInfo 完成信息
     * @param humanInTheLoopInfos 本分段对用户发布的HITL信息
     */
    @Builder
    private AgentRunResult(String runId,
                           String conversationId,
                           String messageId,
                           AgentRunState state,
                           String errorCode,
                           String errorMessage,
                           AgentCompleteInfo completeInfo,
                           List<HumanInTheLoopInfo> humanInTheLoopInfos) {
        this.runId = runId;
        this.conversationId = conversationId;
        this.messageId = messageId;
        this.state = state;
        this.errorCode = errorCode;
        this.errorMessage = errorMessage;
        this.completeInfo = completeInfo;
        this.humanInTheLoopInfos = humanInTheLoopInfos == null
                ? List.of() : List.copyOf(humanInTheLoopInfos);
    }

}
