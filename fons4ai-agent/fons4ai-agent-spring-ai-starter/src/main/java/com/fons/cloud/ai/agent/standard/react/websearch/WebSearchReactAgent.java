package com.fons.cloud.ai.agent.standard.react.websearch;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONObject;
import com.fons.cloud.ai.agent.chat.AgentExecutionContext;
import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.infrastructure.prompt.ReactAgentSystemPrompt;
import com.fons.cloud.ai.agent.infrastructure.hook.AgentChatHook;
import com.fons.cloud.ai.agent.standard.react.ReactAgent;
import com.fons.cloud.ai.agent.standard.react.ReactAgentRunContext;
import com.fons.cloud.ai.agent.standard.runtime.AgentRunContext;
import com.fons.cloud.ai.tool.model.ToolMeta;
import com.fons.cloud.ai.tool.model.WebExtractResult;
import com.fons.cloud.ai.tool.model.WebSearchResult;
import com.fons.cloud.ai.tool.registry.ToolRegistry;
import com.fons.cloud.ai.tool.spi.ToolProvider;
import com.fons.cloud.ai.tool.spi.ToolResultParser;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.tool.ToolCallback;

import java.util.List;

/**
 * 在通用 {@link ReactAgent} 上聚合搜索/抓取结果的 Web Agent。
 *
 * <p>问题 → 模型选择搜索或抓取工具 → 复用 {@code react.before-tool} 可选 HITL → 工具执行 →
 * 结果解析与引用聚合 → 下一轮模型 → 最终回答。Web 层只增加进度提示和引用解析，不定义专属审批点、
 * 恢复协议或 checkpoint；下游只需启用原生工具审批并处理 checkpoint 恢复。搜索和提取结果保存在每个
 * {@link WebSearchAgentRunContext}，共享 Agent 不保存请求数据。</p>
 *
 * @author hongqy
 */
@Slf4j
public class WebSearchReactAgent extends ReactAgent {
    /** 共享工具元数据注册表；每次调用结果仍写入当前 WebSearchAgentRunContext。 */
    private final ToolRegistry toolsRegistry;

    protected WebSearchReactAgent(List<ToolCallback> tools, ChatModel chatModel, AgentTaskManager agentTaskManager, ToolRegistry toolsRegistry) {
        super(tools, chatModel, agentTaskManager);
        this.toolsRegistry = toolsRegistry;
    }

    /**
     * Web 工具调用前输出脱敏进度提示，不改变 React 的统一审批边界。
     * @param context 当前 Run 上下文
     * @param toolCall 即将执行的工具调用
     */
    @Override
    protected void beforeToolCall(ReactAgentRunContext context, AssistantMessage.ToolCall toolCall) {
        // 工具名
        String name = toolCall.name();
        // 工具参数
        String arguments = toolCall.arguments();
        JSONObject args = JSON.parseObject(arguments);
        ToolMeta toolMeta = toolsRegistry.getToolMeta(name);
        if (toolMeta == null) {
            log.warn("未找到工具元数据, 工具名：{}", name);
            return;
        }
        if (toolMeta.isSearch()) {
            // 确认是搜索工具， 则输出一些实时的提示语
            String query = args.getString("query");
            String message = StringUtils.isBlank(query) ? "🔍 正在搜索相关信息\n" : "🔍 正在搜索信息:" + query + "\n";
            emit(context, message, com.fons.cloud.ai.agent.constants.AgentMessageType.THINKING);
        }
    }

    /**
     * 工具成功后解析搜索/抓取结果，并写入当前 Run 的引用集合。
     * @param context 当前 Run 上下文
     * @param toolCall 已执行的工具调用
     * @param result 工具原始返回值
     */
    @Override
    protected void afterToolCall(ReactAgentRunContext context,
                                 AssistantMessage.ToolCall toolCall, String result) {
        WebSearchAgentRunContext webContext = (WebSearchAgentRunContext) context;
        // 工具名
        String name = toolCall.name();
        // 获取工具解析器
        ToolProvider toolProvider = toolsRegistry.getToolProvider(name);
        if (toolProvider == null) {
            log.warn("未找到工具提供者, 工具名：{}", name);
        } else {
            ToolMeta toolMeta = toolsRegistry.getToolMeta(name);
            if (toolMeta == null) {
                log.warn("未找到工具元数据, 工具名：{}", name);
                return;
            }
            if (toolMeta.isSearch()) {
                // 搜索相关的工具
                ToolResultParser<WebSearchResult> resultParser = toolProvider.getResultParser(toolMeta.category());
                List<WebSearchResult> results = resultParser.parse(result);
                webContext.getSearchResults().addAll(results);
            } else if (toolMeta.isExtract()) {
                // 提取相关的工具
                ToolResultParser<WebExtractResult> resultParser = toolProvider.getResultParser(toolMeta.category());
                List<WebExtractResult> results = resultParser.parse(result);
                webContext.getExtractResults().addAll(results);
            }
        }
    }

    /**
     * 最终正文后追加当前 Run 收集到的引用事件。
     * @param context 当前 Run 上下文
     * @param finalText 已生成的最终正文
     */
    @Override
    protected void emitAdditionalFinalResponses(ReactAgentRunContext context, String finalText) {
        WebSearchAgentRunContext webContext = (WebSearchAgentRunContext) context;
        if (CollectionUtils.isNotEmpty(webContext.getSearchResults())) {
            // TODO 暂时只输出搜索结果
            String references = JSON.toJSONString(webContext.getSearchResults());
            context.setReferences(references);
            emit(context, references, com.fons.cloud.ai.agent.constants.AgentMessageType.REFERENCE);
        }
    }

    @Override
    protected AgentRunContext createRunContext(AgentChatRequest request, String runId) {
        return new WebSearchAgentRunContext(request, runId, new AgentExecutionContext());
    }


    /** WebSearchReactAgent 构建器；所有字段在 build 后作为共享只读配置使用。 */
    public static class Builder {
        private final List<ToolCallback> tools;
        private final ChatModel chatModel;
        private final AgentTaskManager agentTaskManager;
        private final ToolRegistry toolsRegistry;

        private List<Advisor> advisors;
        private ReactAgentSystemPrompt systemPrompt;
        private int maxRounds = 5;
        private boolean useChatMemory;
        private int maxMemoryMessages;
        private boolean enableRecommendations = true;
        private AgentChatHook hook;

        public Builder(List<ToolCallback> tools, ChatModel chatModel, AgentTaskManager agentTaskManager, ToolRegistry toolsRegistry) {
            this.tools = tools;
            this.chatModel = chatModel;
            this.agentTaskManager = agentTaskManager;
            this.toolsRegistry = toolsRegistry;

        }

        /** 配置共享 Spring AI Advisors。 */
        public Builder advisors(List<Advisor> advisors) {
            this.advisors = advisors;
            return this;
        }

        /** 覆盖默认 ReAct 系统提示词。 */
        public Builder systemPrompt(ReactAgentSystemPrompt systemPrompt) {
            this.systemPrompt = systemPrompt;
            return this;
        }

        /** 设置单个 Run 的最大 ReAct 轮数。 */
        public Builder maxRounds(int maxRounds) {
            this.maxRounds = maxRounds;
            return this;
        }

        /** 是否启用按 conversationId 隔离的消息记忆。 */
        public Builder useChatMemory(boolean useChatMemory) {
            this.useChatMemory = useChatMemory;
            return this;
        }

        /** 设置启用记忆时的窗口上限。 */
        public Builder maxMemoryMessages(int maxMemoryMessages) {
            this.maxMemoryMessages = maxMemoryMessages;
            return this;
        }

        /** 配置共享生命周期 Hook。 */
        public Builder hook(AgentChatHook hook) {
            this.hook = hook;
            return this;
        }

        /** 是否在完成后生成推荐问题。 */
        public Builder enableRecommendations(boolean enableRecommendations) {
            this.enableRecommendations = enableRecommendations;
            return this;
        }

        /** 校验并创建可共享 Agent；请求态将在 start 时创建。 */
        public WebSearchReactAgent build() {
            WebSearchReactAgent reactAgent = new WebSearchReactAgent(tools, chatModel, agentTaskManager, toolsRegistry);
            reactAgent.systemPrompt = this.systemPrompt;
            reactAgent.hook = this.hook;
            reactAgent.advisors = this.advisors == null ? List.of() : List.copyOf(this.advisors);
            reactAgent.maxRounds = this.maxRounds;
            reactAgent.maxMemoryMessages = this.maxMemoryMessages;
            reactAgent.enableRecommendations = this.enableRecommendations;
            // 初始化
            reactAgent.init(this.useChatMemory);
            return reactAgent;
        }
    }

}
