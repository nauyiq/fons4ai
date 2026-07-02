package com.fons.cloud.ai.agent.standard.react.websearch;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONObject;
import com.fons.cloud.ai.agent.chat.AgentExecutionContext;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.infrastructure.prompt.ReactAgentSystemPrompt;
import com.fons.cloud.ai.agent.infrastructure.tool.ToolMeta;
import com.fons.cloud.ai.agent.infrastructure.tool.ToolProvider;
import com.fons.cloud.ai.agent.infrastructure.tool.ToolResultParser;
import com.fons.cloud.ai.agent.infrastructure.tools.ToolsRegistry;
import com.fons.cloud.ai.agent.response.WebExtractResult;
import com.fons.cloud.ai.agent.response.WebSearchResult;
import com.fons.cloud.ai.agent.standard.hook.AgentChatHook;
import com.fons.cloud.ai.agent.standard.react.ReactAgent;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.chat.client.advisor.api.Advisor;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.tool.ToolCallback;
import reactor.core.publisher.Sinks;

import java.util.ArrayList;
import java.util.List;

/**
 * 基于网络搜索的ReactAgent
 * @author hongqy
 */
@Slf4j
public class WebSearchReactAgent extends ReactAgent {
    private final ToolsRegistry toolsRegistry;

    protected WebSearchReactAgent(List<ToolCallback> tools, ChatModel chatModel, AgentTaskManager agentTaskManager, ToolsRegistry toolsRegistry) {
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
    protected void beforeToolCall(Sinks.Many<String> sink, AssistantMessage.ToolCall toolCall, AgentExecutionContext context) {
        // 工具名
        String name = toolCall.name();
        // 工具参数
        String arguments = toolCall.arguments();
        JSONObject args = JSON.parseObject(arguments);
        ToolMeta toolMeta = toolsRegistry.getToolMeta(name);
        if (toolMeta.isSearch()) {
            // 确认是搜索工具， 则输出一些实时的提示语
            String query = args.getString("query");
            String message = StringUtils.isBlank(query) ? "🔍 正在搜索相关信息\n" : "🔍 正在搜索信息:" + query + "\n";
            sink.tryEmitNext(createThinkingResponse(message));
        }
    }

    /**
     * 工具调用之后 进行逻辑增强
     * @param toolCall
     * @param result
     * @param context
     */
    @Override
    protected void afterToolCall(AssistantMessage.ToolCall toolCall, String result, AgentExecutionContext context) {
        // 跨轮次上下文
        WebSearchAgentExecutionContext webContext = (WebSearchAgentExecutionContext) context;
        // 工具名
        String name = toolCall.name();
        // 获取工具解析器
        ToolProvider toolProvider = toolsRegistry.getToolProvider(name);
        if (toolProvider == null) {
            log.warn("未找到工具提供者, 工具名：{}", name);
        } else {
            ToolMeta toolMeta = toolsRegistry.getToolMeta(name);
            if (toolMeta.isSearch()) {
                // 搜索相关的工具
                ToolResultParser<WebSearchResult> resultParser = toolProvider.getResultParser(toolMeta.category());
                List<WebSearchResult> results = resultParser.parse(result);
                webContext.searchResults.addAll(results);
            } else if (toolMeta.isExtract()) {
                // 提取相关的工具
                ToolResultParser<WebExtractResult> resultParser = toolProvider.getResultParser(toolMeta.category());
                List<WebExtractResult> results = resultParser.parse(result);
                webContext.extractResults.addAll(results);
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
    protected void emitAdditionalFinalResponses(Sinks.Many<String> sink, String finalText, AgentExecutionContext context) {
        WebSearchAgentExecutionContext webContext = (WebSearchAgentExecutionContext) context;
        if (webContext.hasSearchResult()) {
            // TODO 暂时只输出搜索结果
            this.referenceJson = createReferenceResponse(JSON.toJSONString(webContext.searchResults));
            sink.tryEmitNext(this.referenceJson);
        }
    }

    @Override
    protected AgentExecutionContext createReactExecutionContext() {
        return new WebSearchAgentExecutionContext();
    }

    /**
     * websearchagent 跨轮次执行上下文。
     */
    private static class WebSearchAgentExecutionContext extends AgentExecutionContext {
        /** 搜索结果列表（对应 tavily-search 等搜索工具） */
        List<WebSearchResult> searchResults = new ArrayList<>();
        /** 提取结果列表（对应 tavily-extract 等内容提取工具） */
        List<WebExtractResult> extractResults = new ArrayList<>();

        public boolean hasSearchResult() {
            return CollectionUtils.isNotEmpty(searchResults);
        }

        public boolean hasResult() {
            return CollectionUtils.isNotEmpty(searchResults) || CollectionUtils.isNotEmpty(extractResults);
        }
    }


    public static class Builder {
        private final List<ToolCallback> tools;
        private final ChatModel chatModel;
        private final AgentTaskManager agentTaskManager;
        private final ToolsRegistry toolsRegistry;

        private List<Advisor> advisors;
        private ReactAgentSystemPrompt systemPrompt;
        private int maxRounds = 5;
        private boolean useChatMemory;
        private int maxMemoryMessages;
        private AgentChatHook hook;

        public Builder(List<ToolCallback> tools, ChatModel chatModel, AgentTaskManager agentTaskManager, ToolsRegistry toolsRegistry) {
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

        public WebSearchReactAgent build() {
            WebSearchReactAgent reactAgent = new WebSearchReactAgent(tools, chatModel, agentTaskManager, toolsRegistry);
            reactAgent.systemPrompt = this.systemPrompt;
            reactAgent.hook = this.hook;
            reactAgent.advisors = this.advisors;
            reactAgent.maxRounds = this.maxRounds;
            reactAgent.maxMemoryMessages = this.maxMemoryMessages;
            // 初始化
            reactAgent.init(this.useChatMemory);
            return reactAgent;
        }
    }

}
