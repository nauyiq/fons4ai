package com.fons.cloud.ai.agent.infrastructure.prompt;

import lombok.*;

import static com.fons.cloud.ai.agent.constants.prompt.PlanExecutorSystemPromptConstants.*;

/**
 * @author hongqy
 */
@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PlanExecuteSystemPrompt {

    /**
     * 生成执行计划提示词
     */
    private String planPrompt;

    /**
     * 执行工具提示词（React 执行器）
     */
    private String executePrompt;

    /**
     * 任务批判提示词
     */
    private String critiquePrompt;

    /**
     * 上下文压缩提示词
     */
    private String compressPrompt;

    /**
     * 最终总结提示词
     */
    private String summarizePrompt;

    public static PlanExecuteSystemPrompt defaultPrompt() {
        return PlanExecuteSystemPrompt.builder()
                .planPrompt(GENERATION_PLAN_PROMPT)
                .executePrompt(EXECUTE)
                .critiquePrompt(CRITIQUE)
                .compressPrompt(COMPRESS)
                .summarizePrompt(SUMMARIZE)
                .build();
    }


}
