package com.fons.cloud.ai.doudou.infrastructure.prompt;

import com.fons.cloud.ai.agent.constants.AgentPrompts;
import com.fons.cloud.ai.agent.infrastructure.prompt.ReactAgentSystemPrompt;
import lombok.Getter;
import lombok.Setter;
import lombok.experimental.SuperBuilder;

/**
 * 文件Agent提示词
 * @author hongqy
 */
@Getter
@Setter
@SuperBuilder
public class FileAgentPrompt extends ReactAgentSystemPrompt {

    /**
     * 文件读取策略
     */
    private String contentStrategy;

    /**
     * 回答原则
     */
    private String evidencePolicy;


    @Override
    public String toString() {
        return getRole() + "\n\n" +
                getGoal() + "\n\n" +
                AgentPrompts.SYSTEM_TIME_PROMPT_EN + "\n\n" +
                getContentStrategy() + "\n\n" +
                getWorkflow() + "\n\n" +
                getToolUsageRule() + "\n\n" +
                getConstraints() + "\n\n" +
                getErrorHandling() + "\n\n" +
                getEvidencePolicy() + "\n\n" +
                getFormat();
    }

    @Override
    public String getSystemPrompt() {
        return toString();
    }

    public static void main(String[] args) {
        System.out.println(DouDouAgentPrompt.getFileAgentSystemPrompt());
    }

}
