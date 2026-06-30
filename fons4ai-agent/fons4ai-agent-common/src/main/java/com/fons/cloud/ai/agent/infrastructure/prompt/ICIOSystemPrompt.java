package com.fons.cloud.ai.agent.infrastructure.prompt;

import lombok.*;

/**
 * ICIO（Instruction–Context–Input–Output）
 * 强调通过清晰分隔任务指令、背景信息、输入与输出要求，使模型在复杂场景中仍能高效执行。适合需要上下文理解或多阶段推理的任务。
 * <pre>
 *     Instruction（指令）：核心任务说明
 *     Context（上下文）：背景与辅助信息
 *     Input（输入）：具体需要处理的文本或数据
 *     Output（输出）：结果格式或表达要求
 * </pre>
 * @author hongqy
 */
@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ICIOSystemPrompt implements ConstructSystemPrompt {

    /**
     * Instruction（指令）：核心任务说明
     */
    private String instruction;

    /**
     * Context（上下文）：背景与辅助信息
     */
    private String context;

    /**
     * Input（输入）：具体需要处理的文本或数据
     */
    private String input;

    /**
     * Output（输出）：结果格式或表达要求
     */
    private String output;

    @Override
    public String toString() {
        return instruction + "\n\n" +
                context + "\n\n" +
                input + "\n\n" +
                output;
    }

    @Override
    public String getSystemPrompt() {
        return this.toString();
    }

    public static void main(String[] args) {
        ICIOSystemPrompt prompt = ICIOSystemPrompt.builder()
                .instruction(
                        """
                                # Instruction
                                请为指定产品生成一段简短、有吸引力的营销文案。
                                """)
                .context(
                        """
                                # Context
                                你是一名资深品牌营销文案策划，擅长为电商产品撰写高转化率的广告语。目标受众是年轻、注重品质的都市白领。       
                                """)
                .input(
                        """
                                # Input
                                产品名称：AI健康监测手环
                                产品特点：实时监测、可测量血压血氧、监测睡眠和呼吸暂停、外观时尚简约、佩戴轻盈舒适
                                """)
                .output(
                        """
                                # Output
                                请输出一段不超过100字的营销文案，请使用markdown格式，语言简洁、有节奏感，突出健康的生活方式。
                                """
                )
                .build();
        System.out.println(prompt);
    }

}
