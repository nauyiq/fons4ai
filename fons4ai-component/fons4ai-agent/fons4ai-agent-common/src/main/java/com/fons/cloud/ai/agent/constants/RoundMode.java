package com.fons.cloud.ai.agent.constants;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * Agent 轮次模式枚举
 * @author hongqy
 */
@Getter
@AllArgsConstructor
public enum RoundMode {

    /**
     * 未知模式
     */
    UNKNOWN("未知"),

    /**
     * 最终答案模式
     */
    FINAL_ANSWER("最终答案"),

    /**
     * 工具调用模式
     */
    TOOL_CALL("工具调用");

    private final String desc;

}
