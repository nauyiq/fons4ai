package com.fons.cloud.ai.agent.api;

/**
 * 智能体执行状态。终态为 COMPLETED、FAILED、CANCELLED 或 REJECTED，终态不可逆。
 */
public enum AgentRunState {
    CREATED,
    RUNNING,
    COMPLETED,
    FAILED,
    CANCELLED,
    REJECTED;

    /** @return 当前状态是否已经结束 */
    public boolean isTerminal() {
        return this == COMPLETED || this == FAILED || this == CANCELLED || this == REJECTED;
    }
}
