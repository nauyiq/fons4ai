package com.fons.cloud.ai.agent.infrastructure.prompt;

import com.fons.cloud.ai.agent.constants.prompt.AgentPrompts;
import lombok.*;

/**
 * RTF（Role–Task–Format）
 * RTF 是最基础的提示词模板。它主要强调三个核心要素：角色设定（Role）、任务说明（Task） 与 输出格式（Format）。这种结构简洁高效，适合通用类任务或者简单任务。
 * @author hongqy
 */
@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class RTFSystemPrompt implements ConstructSystemPrompt {

    /**
     * 角色设定
     */
    private String role;

    /**
     * 任务说明
     */
    private String task;

    /**
     * 输出格式
     */
    private String format;

    @Override
    public String toString() {
        return role + "\n\n" +
                AgentPrompts.getSystemTimePrompt() + "\n\n" +
                task + "\n\n" +
                format;
    }

    @Override
    public String getSystemPrompt() {
        return this.toString();
    }

    public static void main(String[] args) {
        RTFSystemPrompt prompt = RTFSystemPrompt.builder()
                .role(
                        """
                        # Role
                        你是一名资深市场分析师。
                        """)
                .task(
                        """
                        # Task
                        请分析当前中国新能源汽车市场的发展趋势，指出未来三年的主要增长点。
                        """)
                .format(
                        """
                        # Format
                        请以 Markdown 表格形式输出，包含“趋势”、“原因”、“预测增长率”三列。
                        """)
                .build();
        System.out.println(prompt);
    }


}
