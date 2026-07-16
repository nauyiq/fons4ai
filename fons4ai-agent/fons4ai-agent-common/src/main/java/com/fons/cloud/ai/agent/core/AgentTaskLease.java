package com.fons.cloud.ai.agent.core;

import com.fons.cloud.ai.agent.constants.AgentType;

/**
 * Redis 中保存的临时任务租约，不包含请求正文或业务数据。
 *
 * @param version 租约格式版本
 * @param instanceId 持有任务的应用实例
 * @param runId 执行唯一标识
 * @param agentType 智能体类型
 */
public record AgentTaskLease(int version, String instanceId, String runId, AgentType agentType) {
    public static final int CURRENT_VERSION = 1;
}
