package com.fons.cloud.ai.agent.observability.model;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * 一次 Agent 执行的稳定上下文，用于关联同一条完整轨迹中的所有事件。
 *
 * @param runId 本次 Agent 执行的唯一标识
 * @param conversationId 所属会话的唯一标识
 * @param messageId 触发本次执行的消息标识，可为空
 * @param userId 发起本次执行的用户标识，可为空
 * @param agentName Agent 名称
 * @param agentType Agent 类型或实现框架标识，可为空
 * @param originRunId 续跑、重试或人工恢复前的原始 Run 标识，可为空
 * @param attributes 框架或业务扩展属性，只做浅层不可变副本
 * @author hongqy
 */
public record TraceRunContext(
        String runId,
        String conversationId,
        String messageId,
        String userId,
        String agentName,
        String agentType,
        String originRunId,
        Map<String, Object> attributes) {

    public TraceRunContext {
        Objects.requireNonNull(runId, "runId cannot be null");
        Objects.requireNonNull(conversationId, "conversationId cannot be null");
        Objects.requireNonNull(agentName, "agentName cannot be null");
        attributes = attributes == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(attributes));
    }
}
