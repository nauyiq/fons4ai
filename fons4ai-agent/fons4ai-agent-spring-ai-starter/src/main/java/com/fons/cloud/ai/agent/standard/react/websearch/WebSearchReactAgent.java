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

import java.util.ArrayList;
import java.util.List;

/**
 * 基于网络搜索的ReactAgent
 * @author hongqy
 */
@Slf4j
public class WebSearchReactAgent extends ReactAgent {
    private final ToolRegistry toolsRegistry;

    protected WebSearchReactAgent(List<ToolCallback> tools, ChatModel chatModel, AgentTaskManager agentTaskManager, ToolRegistry toolsRegistry) {
        super(tools, chatModel, agentTaskManager);
        this.toolsRegistry = toolsRegistry;
    }

    /**
     * 新增工具输出, websearch工具调用之前可以输出一些提示语
     * @param sink
     * @param toolCall
     * @param context
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
     * 工具调用之后 进行逻辑增强
     * @param toolCall
     * @param result
     * @param context
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
     * 发送额外的消息， 输出网络搜索的结果
     * @param sink
     * @param finalText
     * @param context
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

        public Builder advisors(List<Advisor> advisors) {
            this.advisors = advisors;
            return this;
        }

        public Builder systemPrompt(ReactAgentSystemPrompt systemPrompt) {
            this.systemPrompt = systemPrompt;
            return this;
        }

        public Builder maxRounds(int maxRounds) {
            this.maxRounds = maxRounds;
            return this;
        }

        public Builder useChatMemory(boolean useChatMemory) {
            this.useChatMemory = useChatMemory;
            return this;
        }

        public Builder maxMemoryMessages(int maxMemoryMessages) {
            this.maxMemoryMessages = maxMemoryMessages;
            return this;
        }

        public Builder hook(AgentChatHook hook) {
            this.hook = hook;
            return this;
        }

        public Builder enableRecommendations(boolean enableRecommendations) {
            this.enableRecommendations = enableRecommendations;
            return this;
        }

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
