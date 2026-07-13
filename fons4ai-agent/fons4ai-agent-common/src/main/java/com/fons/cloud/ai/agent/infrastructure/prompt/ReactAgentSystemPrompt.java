package com.fons.cloud.ai.agent.infrastructure.prompt;

import com.fons.cloud.ai.agent.constants.prompt.AgentPrompts;
import lombok.Getter;
import lombok.Setter;
import lombok.experimental.SuperBuilder;

import static com.fons.cloud.ai.agent.constants.prompt.ReactAgentSystemPromptConstants.*;

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

    public static ReactAgentSystemPrompt defaultPrompt() {
        return ReactAgentSystemPrompt.builder()
                .role(DEFAULT_ROLE)
                .goal(DEFAULT_GOAL)
                .workflow(DEFAULT_WORKFLOW)
                .toolUsageRule(DEFAULT_TOOL_USAGE_RULE)
                .constraints(DEFAULT_CONSTRAINTS)
                .errorHandling(DEFAULT_ERROR_HANDLING)
                .format(DEFAULT_OUTPUT_FORMAT)
                .build();
    }


    @Override
    public String toString() {
        return role + "\n\n" +
                goal + "\n\n" +
                AgentPrompts.getSystemTimePromptEn() + "\n\n" +
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


}
