package com.fons.cloud.ai.doudou.infrastructure.prompt;

import com.fons.cloud.ai.agent.infrastructure.prompt.AgentSystemPrompt;
import com.fons.cloud.ai.agent.infrastructure.prompt.ReactAgentSystemPromptBuilder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

/**
 * 豆豆AGENT的提示词
 * @author hongqy
 */
@Slf4j
@Getter
@Component
public class DouDouAgentPrompt implements InitializingBean {

    @Value("${sys.doudou.role-definition:''}")
    private String roleDefinition;

    private static final String DEFAULT_ROLE_DEFINITION =
            """ 
            你是一个遵循 ReAct 模式智能体问答助手，名字叫做：豆豆，英文名叫dodo，帮助用户解决问题，并且擅长使用联网搜索工具来获取信息 。 在调用工具前，必须思考清楚，禁止提前给出一些推断性/不确定性的信息给用户 。
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

    /**
     * 豆豆AGENT的系统提示词
     */
    private AgentSystemPrompt systemPrompt;


    @Override
    public void afterPropertiesSet() throws Exception {
        String roleDefinition = this.roleDefinition;
        if (StringUtils.isBlank(roleDefinition)) {
            log.info("使用系统默认角色定义提示词");
            roleDefinition = DEFAULT_ROLE_DEFINITION;
        }
        this.systemPrompt = AgentSystemPrompt.builder()
                .roleDefinition(roleDefinition)
                .toolCallingRules(ReactAgentSystemPromptBuilder.DEFAULT_TOOL_CALL_DEFINITION)
                .finalAnswerRules(ReactAgentSystemPromptBuilder.DEFAULT_FINAL_ANSWER_RULES)
                .outputSpecifications(ReactAgentSystemPromptBuilder.DEFAULT_OUTPUT_SPECIFICATIONS)
                .mandatoryRequirements(ReactAgentSystemPromptBuilder.DEFAULT_MANDATORY_REQUIREMENTS)
                .build();
    }
}
