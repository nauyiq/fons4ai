package com.fons.cloud.ai.agent.model.runtime;

import com.fons.cloud.ai.agent.model.hitl.InputRequiredRequest;

/**
 * 本次委派结果识别出的输入请求与真实调用来源。
 *
 * <p>适配器将本对象序列化到原生工具结果消息的 metadata，随 AgentState 保存。
 * 不包含 Java Agent 实例或 common 运行权柄，不承担子 Agent 恢复编排。</p>
 *
 * @author hongqy
 */
public record AgentScopeInputRequired(String id,
                                     String toolCallId,
                                     String sourceAgent,
                                     String agentKey,
                                     InputRequiredRequest request) {
}
