package com.fons.cloud.ai.agent.standard.adaptor;

import com.fons.cloud.ai.agent.api.AgentRunOptions;
import com.fons.cloud.ai.agent.approval.AgentApprovalAction;
import com.fons.cloud.ai.agent.approval.ApprovalRejectionMode;
import com.fons.cloud.ai.agent.chat.AgentChatRequest;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * 恢复 Spring AI Alibaba checkpoint 所需的最小命令。
 *
 * <p>前端审批期间不保持原 SSE 连接。下游保存中断事件中的 runId、threadId、checkpointId，
 * 完成鉴权、审批聚合和幂等控制后提交本命令；框架从配置的 BaseCheckpointSaver 取回 Graph
 * 状态，并把决定转换为 Alibaba Human Feedback。</p>
 *
 * @param request 原始会话请求快照；用于 Fons4AI 外层会话、Hook 和最终上下文
 * @param options 原 Run 的编排参数；后续再次产生工具调用时仍按该配置决定是否中断
 * @param runId 原逻辑 Run 标识
 * @param threadId 中断事件返回的 Alibaba threadId
 * @param checkpointId 中断事件返回的 Alibaba checkpointId
 * @param action 最终批准、编辑或拒绝决定
 * @param comment 可选人工意见
 * @param editedArguments EDIT 时只允许提供 {@code arguments} 字段
 * @param rejectionMode 拒绝后终止或把意见作为原生 observation 恢复；默认安全终止
 */
public record AgentResumeRequest(
        AgentChatRequest request,
        AgentRunOptions options,
        String runId,
        String threadId,
        String checkpointId,
        AgentApprovalAction action,
        String comment,
        Map<String, String> editedArguments,
        ApprovalRejectionMode rejectionMode) {

    public AgentResumeRequest {
        request = Objects.requireNonNull(request, "request cannot be null").snapshot();
        options = Objects.requireNonNullElseGet(options, AgentRunOptions::defaults);
        runId = requireText(runId, "runId");
        threadId = requireText(threadId, "threadId");
        checkpointId = requireText(checkpointId, "checkpointId");
        action = Objects.requireNonNull(action, "action cannot be null");
        rejectionMode = rejectionMode == null
                ? ApprovalRejectionMode.TERMINATE : rejectionMode;
        editedArguments = editedArguments == null ? Map.of()
                : Map.copyOf(new LinkedHashMap<>(editedArguments));
        if (action == AgentApprovalAction.EDIT
                && (editedArguments.size() != 1
                || !editedArguments.containsKey("arguments"))) {
            throw new IllegalArgumentException(
                    "EDIT requires exactly the 'arguments' field");
        }
        if (action != AgentApprovalAction.EDIT && !editedArguments.isEmpty()) {
            throw new IllegalArgumentException(
                    "editedArguments are only allowed for EDIT");
        }
    }

    private static String requireText(String value, String name) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(name + " cannot be blank");
        }
        return value.trim();
    }
}
