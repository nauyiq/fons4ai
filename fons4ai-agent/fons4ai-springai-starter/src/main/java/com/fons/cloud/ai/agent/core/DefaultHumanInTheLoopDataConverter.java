package com.fons.cloud.ai.agent.core;

import cn.hutool.core.lang.Assert;
import cn.hutool.core.util.IdUtil;
import com.alibaba.cloud.ai.graph.OverAllState;
import com.alibaba.cloud.ai.graph.action.InterruptionMetadata;
import com.alibaba.cloud.ai.graph.action.InterruptionMetadata.ToolFeedback;
import com.alibaba.cloud.ai.graph.action.InterruptionMetadata.ToolFeedback.FeedbackResult;
import com.alibaba.cloud.ai.graph.state.StateSnapshot;
import com.alibaba.fastjson2.JSON;
import com.fons.cloud.ai.agent.api.HumanInTheLoopDataConverter;
import com.fons.cloud.ai.agent.model.hitl.HumanInTheLoopInfo;
import com.fons.cloud.ai.agent.model.hitl.HumanInTheLoopKind;
import com.fons.cloud.ai.agent.model.hitl.HitlToolsInfo;
import com.fons.cloud.ai.agent.model.request.AgentApprovalAction;
import com.fons.cloud.ai.agent.model.request.HitlRequestInfo;
import com.fons.cloud.ai.agent.model.runtime.AgentRunContext;
import com.fons.cloud.common.base.exception.SystemIntervalException;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.ToolResponseMessage;

import java.util.List;
import java.util.Map;

/**
 * @author hongqy
 */
@Slf4j
public class DefaultHumanInTheLoopDataConverter implements HumanInTheLoopDataConverter {
    private DefaultHumanInTheLoopDataConverter() {
    }

    private static final DefaultHumanInTheLoopDataConverter INSTANCE = new DefaultHumanInTheLoopDataConverter();

    public static DefaultHumanInTheLoopDataConverter getInstance() {
        return INSTANCE;
    }

    @Override
    public InterruptionMetadata toToolFeedback(HitlRequestInfo requestInfo,
                                               StateSnapshot checkpoint) {
        Assert.notNull(requestInfo, () -> new SystemIntervalException("requestInfo cannot be null"));
        Assert.notNull(checkpoint, () -> new SystemIntervalException("checkpoint cannot be null"));
        if (requestInfo.getHumanInTheLoopKind() != HumanInTheLoopKind.APPROVAL) {
            throw new SystemIntervalException("Only approval can be converted to tool feedback");
        }

        if (requestInfo.getDecision() != AgentApprovalAction.APPROVE
                && requestInfo.getDecision() != AgentApprovalAction.EDIT) {
            throw new SystemIntervalException("Only APPROVE/EDIT can be converted to tool feedback");
        }

        AssistantMessage pendingMessage = findPendingAssistantMessage(checkpoint);
        List<AssistantMessage.ToolCall> toolCalls = pendingMessage.getToolCalls();
        if (requestInfo.getDecision() == AgentApprovalAction.EDIT && toolCalls.size() != 1) {
            throw new SystemIntervalException("Default HITL converter only supports EDIT for one pending tool call");
        }

        String editedArguments = null;
        if (requestInfo.getDecision() == AgentApprovalAction.EDIT) {
            if (requestInfo.getParams() == null) {
                throw new SystemIntervalException("hitlRequestInfo.params is required for EDIT");
            }
            editedArguments = JSON.toJSONString(requestInfo.getParams());
        }

        InterruptionMetadata.Builder feedback = InterruptionMetadata.builder(checkpoint.node(), checkpoint.state());
        for (AssistantMessage.ToolCall toolCall : toolCalls) {
            feedback.addToolFeedback(ToolFeedback.builder()
                    .id(toolCall.id())
                    .name(toolCall.name())
                    .arguments(editedArguments == null ? toolCall.arguments() : editedArguments)
                    .result(editedArguments == null ? FeedbackResult.APPROVED : FeedbackResult.EDITED)
                    .build());
        }
        return feedback.build();
    }

    @Override
    public HumanInTheLoopInfo toHitlInfo(String checkpointId, String sourceAgent,
                                         AgentRunContext context,
                                         InterruptionMetadata interruptionMetadata) {
        Assert.notNull(context, () -> new SystemIntervalException("context cannot be null"));
        Assert.notNull(interruptionMetadata,
                () -> new SystemIntervalException("interruptionMetadata cannot be null"));
        if (StringUtils.isBlank(checkpointId)) {
            throw new SystemIntervalException("checkpointId cannot be blank for tool approval");
        }
        if (StringUtils.isBlank(sourceAgent)) {
            throw new SystemIntervalException("sourceAgent cannot be blank for tool approval");
        }

        List<ToolFeedback> feedbacks = interruptionMetadata.toolFeedbacks();
        if (feedbacks == null || feedbacks.isEmpty()) {
            throw new SystemIntervalException("Default HITL converter requires pending tool feedback");
        }

        List<HitlToolsInfo> tools = feedbacks.stream()
                .map(feedback -> new HitlToolsInfo(
                        feedback.getId(),
                        feedback.getName(),
                        feedback.getDescription(),
                        feedback.getArguments()))
                .toList();

        HumanInTheLoopInfo humanInTheLoopInfo = HumanInTheLoopInfo.builder()
                .id(checkpointId)
                .kind(HumanInTheLoopKind.APPROVAL)
                .originRunId(resolveOriginRunId(context))
                .sourceAgent(sourceAgent)
                .data(Map.of("tools", List.copyOf(tools)))
                .build();
        log.debug("Converted tool approval HITL, id:{}, checkpointId:{}, toolCount:{}",
                humanInTheLoopInfo.getId(), checkpointId, tools.size());
        return humanInTheLoopInfo;
    }

    private String resolveOriginRunId(AgentRunContext context) {
        if (context instanceof DefaultAgentRunContext runContext) {
            HitlRequestInfo previous = runContext.getRequest().getHitlRequestInfo();
            if (previous != null && StringUtils.isNotBlank(previous.getOriginRunId())) {
                return previous.getOriginRunId();
            }
        }
        return context.getRunId();
    }

    private AssistantMessage findPendingAssistantMessage(StateSnapshot checkpoint) {
        OverAllState state = checkpoint.state();
        List<Message> messages = state.<List<Message>>value("messages").orElse(List.of());
        for (int index = messages.size() - 1; index >= 0; index--) {
            Message message = messages.get(index);
            if (!(message instanceof AssistantMessage assistantMessage)) {
                continue;
            }
            if (index + 1 < messages.size() && messages.get(index + 1) instanceof ToolResponseMessage) {
                break;
            }
            if (assistantMessage.hasToolCalls()) {
                return assistantMessage;
            }
            break;
        }
        throw new SystemIntervalException("Checkpoint contains no pending tool call for HITL resume");
    }
}
