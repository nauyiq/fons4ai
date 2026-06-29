package com.fons.cloud.ai.agent.infrastructure.prompt;

import com.fons.cloud.ai.agent.constants.AgentPrompts;
import lombok.*;

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
@Builder
@NoArgsConstructor
@AllArgsConstructor
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

    public String getPrompt() {
        return this.toString();
    }

    @Override
    public String toString() {
        return roleDefinition + "\n\n" +
                AgentPrompts.SYSTEM_TIME_PROMPT + "\n\n" +
                toolCallingRules + "\n\n" +
                finalAnswerRules + "\n\n" +
                outputSpecifications + "\n\n" +
                mandatoryRequirements;
    }


}
