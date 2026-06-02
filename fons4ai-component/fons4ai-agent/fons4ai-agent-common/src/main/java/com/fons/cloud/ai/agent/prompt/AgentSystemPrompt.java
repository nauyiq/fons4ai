package com.fons.cloud.ai.agent.prompt;

import lombok.Getter;
import lombok.Setter;

/**
 * 智能体系统提示词
 * <pre>
 *     agent系统提示词模板：
 *     ## 角色
 *     ## 工具调用规则
 *     ## 通用最终答案规则
 *     ## 通用输出规范
 *     ## 通用强制要求
 * </pre>
 * @author hongqy
 */
@Getter
@Setter
public class AgentSystemPrompt {

    /**
     * 角色定义
     */
    private String roleDefinition;

    /**
     * 工具调用规则
     */
    private String toolCallingRules;

    /**
     * 通用最终答案规则
     */
    private String finalAnswerRules;

    /**
     * 通用输出规范
     */
    private String outputSpecifications;

    /**
     * 通用强制要求
     */
    private String mandatoryRequirements;



    @Override
    public String toString() {
        return roleDefinition + "\n\n" +
                getSystemTimePrompt() + "\n\n" +
                REACT_TOOL_CALLING_RULES + "\n\n" +
                REACT_FINAL_ANSWER_RULES + "\n\n" +
                OUTPUT_SPECIFICATIONS + "\n\n" +
                REACT_MANDATORY_REQUIREMENTS;
    }


}
