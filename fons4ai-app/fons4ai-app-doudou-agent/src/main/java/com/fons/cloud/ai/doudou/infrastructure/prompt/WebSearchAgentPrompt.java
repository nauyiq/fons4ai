package com.fons.cloud.ai.doudou.infrastructure.prompt;

import com.fons.cloud.ai.agent.constants.AgentPrompts;
import com.fons.cloud.ai.agent.infrastructure.prompt.ReactAgentSystemPrompt;
import lombok.Getter;
import lombok.Setter;
import lombok.experimental.SuperBuilder;

/**
 * 网络搜索AGENT提示词
 * @author hongqy
 */
@Getter
@Setter
@SuperBuilder
public class WebSearchAgentPrompt extends ReactAgentSystemPrompt {

    /**
     * 搜索决策
     */
    private String searchDecision;

    /**
     * 搜索策略
     */
    private String searchStrategy;

    /**
     * 信息源选择
     */
    private String sourcePolicy;

    /**
     * 回答原则
     */
    private String evidencePolicy;

    @Override
    public String toString() {
        return getRole() + "\n\n" +
                getGoal() + "\n\n" +
                AgentPrompts.SYSTEM_TIME_PROMPT_EN + "\n\n" +
                getSearchDecision() + "\n\n" +
                getSearchStrategy() + "\n\n" +
                getWorkflow() + "\n\n" +
                getToolUsageRule() + "\n\n" +
                getSourcePolicy() + "\n\n" +
                getConstraints() + "\n\n" +
                getErrorHandling() + "\n\n" +
                getEvidencePolicy() + "\n\n" +
                getFormat();
    }

    @Override
    public String getSystemPrompt() {
        return this.toString();
    }

    public static void main(String[] args) {
        System.out.println(DouDouAgentPrompt.getWebSearchAgentSystemPrompt());
    }
}
