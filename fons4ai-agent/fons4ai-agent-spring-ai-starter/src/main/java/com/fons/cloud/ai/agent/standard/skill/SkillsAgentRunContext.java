package com.fons.cloud.ai.agent.standard.skill;

import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.constants.AgentType;
import com.fons.cloud.ai.agent.standard.runtime.AgentRunContext;
import lombok.Getter;

import java.util.concurrent.atomic.AtomicBoolean;

/** Skills Agent 单次执行的 delegate、技能授权和流式聚合状态。 */
@Getter
final class SkillsAgentRunContext extends AgentRunContext {
    private final GuardedSkillRegistry skillRegistry;
    private final com.alibaba.cloud.ai.graph.agent.ReactAgent delegate;
    private final AtomicBoolean nativeTerminated = new AtomicBoolean();
    private final StringBuilder currentModelText = new StringBuilder();
    private final StringBuilder nativeThinking = new StringBuilder();
    private volatile boolean currentTurnStreamed;
    private volatile boolean currentTurnReasoningStreamed;
    private volatile String nativeFinalAnswer = "";

    SkillsAgentRunContext(AgentType agentType, AgentChatRequest request, String runId,
                          GuardedSkillRegistry skillRegistry,
                          com.alibaba.cloud.ai.graph.agent.ReactAgent delegate) {
        super(agentType, request, runId);
        this.skillRegistry = skillRegistry;
        this.delegate = delegate;
    }

    void markCurrentTurnStreamed() {
        currentTurnStreamed = true;
    }

    void markCurrentTurnReasoningStreamed() {
        currentTurnReasoningStreamed = true;
    }

    void setNativeFinalAnswer(String nativeFinalAnswer) {
        this.nativeFinalAnswer = nativeFinalAnswer;
    }

    void resetCurrentTurn() {
        currentModelText.setLength(0);
        currentTurnStreamed = false;
        currentTurnReasoningStreamed = false;
    }
}
