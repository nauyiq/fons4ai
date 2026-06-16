package com.fons.cloud.ai.agent.prompt;

/**
 * 联网搜索ReactAgent系统提示语构建器
 * @author hongqy
 */
public final class WebSearchReactAgentSystemPromptBuilder {

    /**
     * 联网搜索 React 模式角色定义
     * <pre>
     *     Reasoning 仅用于模型内部决策，Observation 仅由系统注入，避免模型输出内容干扰 ToolCall 解析。
     * </pre>
     */
    public static final String DEFAULT_ROLE_DEFINITION =
            """
            你是一个严格遵循 ReAct 模式的智能 AI 助手，并且擅长使用联网搜索工具来获取信息。
            ## ReAct 执行模式
            每一轮必须按照以下流程执行：
            1. 在内部判断当前信息是否足够，禁止输出内部思考过程或推理轨迹。
            2. 如果需要使用工具，只能输出 ToolCall，不得混杂任何文本。
            3. 系统会将工具执行结果作为 ToolResponseMessage 注入上下文，该结果即为 Observation。
            4. 读取 Observation 后，继续在内部判断是否需要调用工具。
            5. 信息足够时，停止调用工具并输出最终自然语言答案。
            
            ## 联网搜索规则
            1. 回答用户问题前，必须使用联网搜索工具核验事实，不要仅依赖模型自身知识。
            2. 分析问题中的主体、时间维度和核心事件，使用准确关键词搜索。
            3. 优先筛选与用户问题时间范围一致的信息，过滤无关、过期或广告内容。
            4. 搜索结果不足时，调整关键词继续搜索；已有足够信息时，停止搜索并给出最终答案。
            """;


    public static AgentSystemPrompt build() {
        return AgentSystemPrompt.builder()
                .roleDefinition(DEFAULT_ROLE_DEFINITION)
                .toolCallingRules(ReactAgentSystemPromptBuilder.DEFAULT_TOOL_CALL_DEFINITION)
                .finalAnswerRules(ReactAgentSystemPromptBuilder.DEFAULT_FINAL_ANSWER_RULES)
                .outputSpecifications(ReactAgentSystemPromptBuilder.DEFAULT_OUTPUT_SPECIFICATIONS)
                .mandatoryRequirements(ReactAgentSystemPromptBuilder.DEFAULT_MANDATORY_REQUIREMENTS)
                .build();
    }

}
