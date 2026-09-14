package com.fons.cloud.ai.agent.infrastructure.handler;

import com.fons.cloud.ai.agent.api.HumanInTheLoopDataConverter;
import com.fons.cloud.ai.agent.infrastructure.middleware.AgentScopeApprovalRejectMiddleware;
import com.fons.cloud.ai.agent.core.AgentScopeRunContext;
import com.fons.cloud.ai.agent.model.hitl.HumanInTheLoopInfo;
import com.fons.cloud.ai.agent.model.hitl.HumanInTheLoopKind;
import com.fons.cloud.ai.agent.model.request.AgentApprovalAction;
import com.fons.cloud.ai.agent.model.request.HitlRequestInfo;
import com.fons.cloud.common.base.exception.SystemIntervalException;
import io.agentscope.core.agent.RuntimeContext;
import io.agentscope.core.event.RequireUserConfirmEvent;
import io.agentscope.core.message.Msg;
import io.agentscope.core.message.MsgRole;
import io.agentscope.core.message.ToolCallState;
import io.agentscope.core.message.ToolUseBlock;
import io.agentscope.core.message.UserMessage;
import io.agentscope.core.state.AgentState;
import io.agentscope.harness.agent.HarnessAgent;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;

import java.util.List;

/**
 * AgentScope顶层人工审批处理器。
 *
 * <p>负责顶层审批请求校验、AgentState读取和原生审批消息转换，不持久化业务审批单，
 * 不推进common状态，也不处理子Agent审批恢复。</p>
 *
 * @author hongqy
 */
@Slf4j
public class AgentScopeApprovalHandler {

    private static final AgentScopeApprovalHandler INSTANCE = new AgentScopeApprovalHandler();

    private AgentScopeApprovalHandler() {
    }

    /**
     * 获取人工审批处理器单例。
     *
     * @return 人工审批处理器
     */
    public static AgentScopeApprovalHandler getInstance() {
        return INSTANCE;
    }

    /**
     * 校验AgentScope顶层审批恢复请求。
     *
     * @param requestInfo 审批恢复请求
     * @param delegate    AgentScope委托Agent
     */
    public void validateResume(HitlRequestInfo requestInfo,
                               HarnessAgent delegate) {
        if (requestInfo == null) {
            throw new SystemIntervalException("hitlRequestInfo cannot be null");
        }
        if (StringUtils.isBlank(requestInfo.getHitlId())) {
            throw new SystemIntervalException("hitlRequestInfo.hitlId cannot be blank");
        }
        if (StringUtils.isBlank(requestInfo.getOriginRunId())) {
            throw new SystemIntervalException("hitlRequestInfo.originRunId cannot be blank");
        }
        if (StringUtils.isBlank(requestInfo.getCheckpointId())) {
            throw new SystemIntervalException("hitlRequestInfo.checkpointId cannot be blank");
        }
        if (requestInfo.getHumanInTheLoopKind() != HumanInTheLoopKind.APPROVAL) {
            throw new SystemIntervalException("Only approval can resume AgentScope HITL");
        }
        if (requestInfo.getDecision() != AgentApprovalAction.APPROVE
                && requestInfo.getDecision() != AgentApprovalAction.EDIT
                && requestInfo.getDecision() != AgentApprovalAction.REJECT) {
            throw new SystemIntervalException(
                    "Only APPROVE/EDIT/REJECT can resume AgentScope HITL");
        }
        if (requestInfo.getDecision() == AgentApprovalAction.REJECT) {
            validateApprovalRejectMiddleware(delegate);
        }
    }

    /**
     * 创建顶层Agent审批恢复消息。
     *
     * @param context   当前Run上下文
     * @param delegate  AgentScope委托Agent
     * @param converter HITL数据转换器
     * @return AgentScope审批恢复消息
     */
    public UserMessage createResumeMessage(
            AgentScopeRunContext context,
            HarnessAgent delegate,
            HumanInTheLoopDataConverter converter) {
        HitlRequestInfo requestInfo = context.getRequest().getHitlRequestInfo();
        validateResume(requestInfo, delegate);
        List<ToolUseBlock> toolCalls = loadPendingApprovalToolCalls(
                delegate, context.getRuntimeContext(), requestInfo.getCheckpointId());
        UserMessage resumeMessage = converter.toResumeMessage(
                requestInfo.getDecision(), requestInfo.getParams(), toolCalls);
        if (resumeMessage == null) {
            throw new SystemIntervalException("AgentScope resume message cannot be null");
        }
        log.info("Resume AgentScope approval, hitlId:{}, replyId:{}, toolCount:{}",
                requestInfo.getHitlId(), requestInfo.getCheckpointId(), toolCalls.size());
        return resumeMessage;
    }

    /**
     * 将顶层Agent审批事件转换为common HITL信息。
     *
     * @param sourceAgent 顶层Agent逻辑标识
     * @param context     当前Run上下文
     * @param event       AgentScope审批事件
     * @param converter   HITL数据转换器
     * @return common HITL信息
     */
    public HumanInTheLoopInfo createHitlInfo(
            String sourceAgent,
            AgentScopeRunContext context,
            RequireUserConfirmEvent event,
            HumanInTheLoopDataConverter converter) {
        return converter.toHitlInfo(sourceAgent, context, event);
    }

    /**
     * 判断当前Run是否为顶层审批拒绝恢复请求。
     *
     * @param context 当前Run上下文
     * @return true表示当前Run恢复的是顶层审批拒绝
     */
    public boolean isApprovalReject(AgentScopeRunContext context) {
        return isApprovalReject(context.getRequest().getHitlRequestInfo());
    }

    /**
     * 判断审批请求是否为顶层审批拒绝。
     *
     * @param requestInfo 审批恢复请求
     * @return true表示审批请求为顶层审批拒绝
     */
    public boolean isApprovalReject(HitlRequestInfo requestInfo) {
        return requestInfo != null
                && requestInfo.getDecision() == AgentApprovalAction.REJECT;
    }

    /**
     * 从顶层Agent最新状态读取待审批工具调用。
     *
     * @param delegate       AgentScope委托Agent
     * @param runtimeContext AgentScope单次调用上下文
     * @param replyId        审批绑定的原生replyId
     * @return 待审批工具调用
     */
    private List<ToolUseBlock> loadPendingApprovalToolCalls(
            HarnessAgent delegate,
            RuntimeContext runtimeContext,
            String replyId) {
        if (delegate.getStateStore() != null) {
            delegate.clearStateCache(runtimeContext);
        }
        AgentState agentState = delegate.getDelegate().getAgentState(runtimeContext);
        return findPendingApprovalToolCalls(agentState, replyId);
    }

    /**
     * 校验原生Agent已安装审批拒绝停止中间件。
     *
     * @param delegate AgentScope委托Agent
     */
    private void validateApprovalRejectMiddleware(HarnessAgent delegate) {
        boolean installed = delegate.getDelegate().getMiddlewares().stream()
                .anyMatch(AgentScopeApprovalRejectMiddleware.class::isInstance);
        if (!installed) {
            throw new SystemIntervalException(
                    "AgentScope approval rejection requires AgentScopeApprovalRejectMiddleware");
        }
    }

    /**
     * 从AgentState中查找指定回复对应的待审批工具调用。
     *
     * @param agentState AgentScope Agent状态
     * @param replyId    审批绑定的原生replyId
     * @return 待审批工具调用
     */
    private List<ToolUseBlock> findPendingApprovalToolCalls(
            AgentState agentState,
            String replyId) {
        List<Msg> messages = agentState.getContext();
        boolean pendingApprovalFound = false;
        for (int index = messages.size() - 1; index >= 0; index--) {
            Msg message = messages.get(index);
            if (message.getRole() != MsgRole.ASSISTANT) {
                continue;
            }

            List<ToolUseBlock> askingToolCalls = message
                    .getContentBlocks(ToolUseBlock.class)
                    .stream()
                    .filter(toolCall -> toolCall.getState() == ToolCallState.ASKING)
                    .toList();
            if (askingToolCalls.isEmpty()) {
                continue;
            }
            pendingApprovalFound = true;

            Object pendingReplyId = message.getMetadata() == null
                    ? null
                    : message.getMetadata().get(Msg.METADATA_CONFIRM_REQUEST_REPLY_ID);
            if (StringUtils.equals(
                    replyId, pendingReplyId instanceof String value ? value : null)) {
                return List.copyOf(askingToolCalls);
            }
            // 当前审批不匹配时继续查找更早的待审批消息。
        }
        if (pendingApprovalFound) {
            throw new SystemIntervalException(
                    "AgentScope HITL replyId does not match any pending approval");
        }
        throw new SystemIntervalException(
                "AgentScope state contains no pending approval tool call");
    }

}
