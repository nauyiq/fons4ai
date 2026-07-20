package com.fons.cloud.ai.agent.standard.adaptor;

import com.alibaba.cloud.ai.graph.OverAllState;
import com.alibaba.cloud.ai.graph.action.InterruptionMetadata;
import com.alibaba.cloud.ai.graph.checkpoint.Checkpoint;
import com.fons.cloud.ai.agent.approval.AgentApprovalAction;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.chat.messages.AssistantMessage;

import java.util.List;
import java.util.Map;

/**
 * Alibaba HITL 元数据转换工具；不保存审批状态或 checkpoint。
 * @author hongqy
 */
public final class HumanFeedbacks {
    private HumanFeedbacks() {
    }

    /** 从 Saver 的 checkpoint 重建 HumanInTheLoopHook 校验决定所需的工具信息。 */
    public static InterruptionMetadata fromCheckpoint(Checkpoint checkpoint) {
        OverAllState state = new OverAllState(checkpoint.getState());
        Object value = state.value("messages").orElse(List.of());
        if (!(value instanceof List<?> messages)) {
            throw new IllegalStateException("checkpoint messages are unavailable");
        }
        AssistantMessage assistant = null;
        for (int index = messages.size() - 1; index >= 0; index--) {
            if (messages.get(index) instanceof AssistantMessage candidate
                    && candidate.hasToolCalls()) {
                assistant = candidate;
                break;
            }
        }
        if (assistant == null) {
            throw new IllegalStateException("checkpoint contains no pending tool call");
        }
        InterruptionMetadata.Builder builder = InterruptionMetadata.builder(
                checkpoint.getNextNodeId(), state);
        for (AssistantMessage.ToolCall call : assistant.getToolCalls()) {
            builder.addToolFeedback(InterruptionMetadata.ToolFeedback.builder()
                    .id(call.id()).name(call.name()).arguments(call.arguments())
                    .description("Pending tool approval").build());
        }
        return builder.build();
    }

    /** 把公共决定转换为 Alibaba HumanInTheLoopHook 可消费的原生反馈。 */
    public static InterruptionMetadata apply(
            InterruptionMetadata source, AgentApprovalAction action,
            String comment, Map<String, String> editedArguments) {
        if (action == AgentApprovalAction.EDIT
                && (source.toolFeedbacks().size() != 1
                || editedArguments.size() != 1
                || !editedArguments.containsKey("arguments"))) {
            throw new IllegalArgumentException(
                    "EDIT supports exactly one tool and the 'arguments' field");
        }
        InterruptionMetadata.Builder builder = InterruptionMetadata.builder()
                .nodeId(source.node()).state(source.state())
                .toolsAutomaticallyApproved(source.getToolsAutomaticallyApproved());
        for (InterruptionMetadata.ToolFeedback original : source.toolFeedbacks()) {
            InterruptionMetadata.ToolFeedback.Builder item =
                    InterruptionMetadata.ToolFeedback.builder(original);
            switch (action) {
                case APPROVE -> item.result(
                        InterruptionMetadata.ToolFeedback.FeedbackResult.APPROVED);
                case EDIT -> item.arguments(editedArguments.get("arguments")).result(
                        InterruptionMetadata.ToolFeedback.FeedbackResult.EDITED);
                case REJECT -> item.description(StringUtils.defaultIfBlank(comment,
                                "Rejected by human reviewer"))
                        .result(InterruptionMetadata.ToolFeedback.FeedbackResult.REJECTED);
            }
            builder.addToolFeedback(item.build());
        }
        return builder.build();
    }
}
