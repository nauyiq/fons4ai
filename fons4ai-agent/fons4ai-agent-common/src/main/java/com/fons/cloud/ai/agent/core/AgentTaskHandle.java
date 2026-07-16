package com.fons.cloud.ai.agent.core;

import org.apache.commons.lang3.StringUtils;

/**
 * 精确标识一次运行任务，防止旧运行的迟到终态影响同会话的新运行。
 *
 * @param conversationId 会话标识
 * @param runId 执行唯一标识
 */
public record AgentTaskHandle(String conversationId, String runId) {
    public AgentTaskHandle {
        if (StringUtils.isBlank(conversationId) || StringUtils.isBlank(runId)) {
            throw new IllegalArgumentException("conversationId and runId are required");
        }
    }
}
