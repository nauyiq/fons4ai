package com.fons.cloud.ai.agent.model.runtime;

import lombok.Getter;

/**
 * AgentScope已经完成聚合的文本工具结果。
 *
 * <p>该对象只承载从原生工具事件中确认完成的数据，不负责common工具结果通知。</p>
 *
 * @author hongqy
 */
@Getter
public final class AgentScopeCompletedToolResult {

    /**
     * 工具调用ID。
     */
    private final String toolCallId;

    /**
     * 工具名称。
     */
    private final String toolName;

    /**
     * 工具文本结果。
     */
    private final String result;

    /**
     * 创建已经完成聚合的工具结果。
     *
     * @param toolCallId 工具调用ID
     * @param toolName 工具名称
     * @param result 工具文本结果
     */
    public AgentScopeCompletedToolResult(String toolCallId,
                                         String toolName,
                                         String result) {
        this.toolCallId = toolCallId;
        this.toolName = toolName;
        this.result = result;
    }

}
