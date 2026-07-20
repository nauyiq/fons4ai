package com.fons.cloud.ai.agent.standard.skill;

import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.constants.AgentType;
import com.fons.cloud.ai.agent.standard.adaptor.FrameAgentRunContext;
import lombok.Getter;

/**
 * Skills Agent 单次执行的技能授权和 delegate 状态。
 * 通用 Graph 流、代次、checkpoint 和中断状态由 AlibabaAgentRunContext 统一提供。
 */
@Getter
final class SkillsAgentRunContext extends FrameAgentRunContext {
    private final GuardedSkillRegistry skillRegistry;
    /** 本 Run 固定的资源视图；恢复重建 delegate 时必须复用，防止资源版本漂移。 */
    private final SkillResourceResolver resourceResolver;
    /** 当前一代 Alibaba delegate；原生中断恢复时会替换，但永不跨 Run 共享。 */
    private volatile com.alibaba.cloud.ai.graph.agent.ReactAgent delegate;
    SkillsAgentRunContext(AgentType agentType, AgentChatRequest request, String runId,
                          GuardedSkillRegistry skillRegistry,
                          SkillResourceResolver resourceResolver) {
        super(agentType, request, runId);
        this.skillRegistry = skillRegistry;
        this.resourceResolver = resourceResolver;
    }

    void replaceDelegate(com.alibaba.cloud.ai.graph.agent.ReactAgent delegate) {
        this.delegate = java.util.Objects.requireNonNull(delegate, "delegate cannot be null");
    }
}
