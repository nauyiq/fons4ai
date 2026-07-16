package com.fons.cloud.ai.agent.core;

/**
 * 跨实例主动停止命令。
 *
 * @param version 消息格式版本
 * @param conversationId 会话标识
 * @param runId 目标执行唯一标识
 */
public record AgentStopCommand(int version, String conversationId, String runId) {
    public static final int CURRENT_VERSION = 1;
}
