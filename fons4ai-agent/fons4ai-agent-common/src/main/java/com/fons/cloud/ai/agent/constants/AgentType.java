package com.fons.cloud.ai.agent.constants;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * @author hongqy
 */
@Getter
@AllArgsConstructor
public enum AgentType {

    /**
     * React模式的智能体
     */
    REACT("react"),

    /**
     * 计划执行模式的智能体
     */
    PLAN_EXECUTOR("plan-execute"),

    /**
     * 自定义模式的智能体
     */
    CUSTOM("custom"),

    ;

    private final String type;



}
