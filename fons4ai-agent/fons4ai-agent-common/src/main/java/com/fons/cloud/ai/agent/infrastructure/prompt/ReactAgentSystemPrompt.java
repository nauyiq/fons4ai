package com.fons.cloud.ai.agent.infrastructure.prompt;

import com.fons.cloud.ai.agent.constants.AgentPrompts;
import lombok.Getter;
import lombok.Setter;
import lombok.experimental.SuperBuilder;

/**
 * react智能体的结构化系统提示词
 * @author hongqy
 */
@Getter
@Setter
@SuperBuilder
public class ReactAgentSystemPrompt implements ConstructSystemPrompt {

    /**
     * 角色
     */
    private String role;

    /**
     * 目标;
     */
    private String goal;

    /**
     * 工作流
     */
    private String workflow;

    /**
     * 工具使用规则
     */
    private String toolUsageRule;

    /**
     * 约束
     */
    private String constraints;

    /**
     * 异常处理
     */
    private String errorHandling;

    /**
     * 输出规范
     */
    private String format;


    public String getPrompt() {
        return this.toString();
    }

    @Override
    public String toString() {
        return role + "\n\n" +
                goal + "\n\n" +
                AgentPrompts.SYSTEM_TIME_PROMPT_EN + "\n\n" +
                workflow + "\n\n" +
                toolUsageRule + "\n\n" +
                constraints + "\n\n" +
                errorHandling + "\n\n" +
                format;
    }


    @Override
    public String getSystemPrompt() {
        return this.toString();
    }

    public static void main(String[] args) {
        ReactAgentSystemPrompt prompt = com.fons.cloud.ai.agent.infrastructure.prompt.builder.ReactAgentSystemPromptBuilder.build();
        System.out.println(prompt);
    }

}
