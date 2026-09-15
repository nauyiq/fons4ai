package com.fons.cloud.ai.agent.api;

/**
 * Agent注册表
 * @author hongqy
 */
public interface AgentRegistry {

    /**
     * 根据agentId获取Agent
     * @param agentId
     * @return
     */
    Agent getAgent(String agentId);

    /**
     * 将Agent注册到注册表中
     * @param agentId
     * @param agent
     */
    void register(String agentId, Agent agent);
}
