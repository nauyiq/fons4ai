package com.fons.cloud.ai.agent.chat;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

/**
 * 一次执行聚合后的最终上下文。
 */
@Getter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class AgentChatFinalContext {
    private String finalAnswer;
    private String thinking;
    private String recommendations;
    private String tools;
    private String skills;
    private String references;
    private long firstResponseTime;
    private long totalResponseTime;
}
