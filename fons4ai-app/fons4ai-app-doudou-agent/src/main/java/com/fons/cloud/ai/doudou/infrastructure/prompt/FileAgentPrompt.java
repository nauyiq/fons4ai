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

    // ----------- FILE AGENT --------------
    private static final String FILE_AGENT_ROLE =
            """
            ## Role
            你是一个遵循 ReAct（Reason → Act → Observation）工作模式的智能体问答助手，名字叫做：豆豆，英文名叫dodo。
            擅长通过文件内容获取信息，并结合自身知识进行分析、归纳和总结，而不仅仅是复述文件内容。
            """;

    private static final String FILE_AGENT_GOAL =
            """
            ## Goal
            你的目标是帮助用户准确理解、分析和总结文件内容。
            """;

    private static final String FILE_AGENT_CONTENT_STRATEGY =
            """
            ## Content Strategy
            1. 优先读取与当前问题最相关的章节、段落或页面。
            2. 若信息不足，可继续读取其他相关内容。
            3. 已获得足够信息后立即停止读取。
            4. 避免重复读取相同内容。
            """;

    private static final String FILE_AGENT_EVIDENCE_POLICY =
            """
            ## Evidence Policy
            - 基于文件内容进行总结，而不是复制全文；
            - 可以引用关键内容支持结论；
            - 不确定的信息应明确说明；
            - 不得推测文件未包含的信息。
            """;

    /**
     * 基于文件进行文档生成以及RAG检索的AGENT系统提示词
     */
    private static final FileAgentPrompt FILE_AGENT_PROMPT = FileAgentPrompt.builder()
            .role(FILE_AGENT_ROLE)
            .goal(FILE_AGENT_GOAL)
            .workflow(com.fons.cloud.ai.agent.infrastructure.prompt.builder.ReactAgentSystemPromptBuilder.DEFAULT_WORKFLOW)
            .toolUsageRule(com.fons.cloud.ai.agent.infrastructure.prompt.builder.ReactAgentSystemPromptBuilder.DEFAULT_TOOL_USAGE_RULE)
            .constraints(com.fons.cloud.ai.agent.infrastructure.prompt.builder.ReactAgentSystemPromptBuilder.DEFAULT_CONSTRAINTS)
            .errorHandling(com.fons.cloud.ai.agent.infrastructure.prompt.builder.ReactAgentSystemPromptBuilder.DEFAULT_ERROR_HANDLING)
            .format(com.fons.cloud.ai.agent.infrastructure.prompt.builder.ReactAgentSystemPromptBuilder.DEFAULT_OUTPUT_FORMAT)
            .contentStrategy(FILE_AGENT_CONTENT_STRATEGY)
            .evidencePolicy(FILE_AGENT_EVIDENCE_POLICY)
            .build();

    public static FileAgentPrompt defaultPrompt() {
        return FILE_AGENT_PROMPT;
    }

    public static void main(String[] args) {
        System.out.println(FILE_AGENT_PROMPT);
    }

}
