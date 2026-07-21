package com.fons.cloud.ai.agent.standard.react.websearch;

import com.alibaba.cloud.ai.graph.agent.interceptor.Interceptor;
import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONObject;
import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.infrastructure.prompt.ReactAgentSystemPrompt;
import com.fons.cloud.ai.agent.standard.BaseAgentBuilder;
import com.fons.cloud.ai.agent.standard.react.ReactAgent;
import com.fons.cloud.ai.agent.standard.react.ReactAgentRunContext;
import com.fons.cloud.ai.agent.standard.runtime.AgentRunContext;
import com.fons.cloud.ai.tool.model.ToolMeta;
import com.fons.cloud.ai.tool.model.WebExtractResult;
import com.fons.cloud.ai.tool.model.WebSearchResult;
import com.fons.cloud.ai.tool.model.WebToolResult;
import com.fons.cloud.ai.tool.registry.ToolRegistry;
import com.fons.cloud.ai.tool.spi.ToolProvider;
import com.fons.cloud.ai.tool.spi.ToolResultParser;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.tool.ToolCallback;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

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
        String name = toolCall.name();
        ToolMeta toolMeta = toolsRegistry.getToolMeta(name);
        if (toolMeta == null) {
            log.warn("未找到工具元数据, 工具名：{}", name);
            return;
        }
        if (toolMeta.isSearch()) {
            try {
                JSONObject args = JSON.parseObject(toolCall.arguments());
                String query = args == null ? null : args.getString("query");
                String message = StringUtils.isBlank(query)
                        ? "🔍 正在搜索相关信息\n" : "🔍 正在搜索信息:" + query + "\n";
                emit(context, message, com.fons.cloud.ai.agent.constants.AgentMessageType.THINKING);
            } catch (RuntimeException parseError) {
                // 参数可能包含用户敏感数据，日志只保留工具名。
                log.warn("Web工具参数解析失败, 工具名：{}", name);
            }
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
        String name = toolCall.name();
        ToolProvider toolProvider = toolsRegistry.getToolProvider(name);
        if (toolProvider == null) {
            log.warn("未找到工具提供者, 工具名：{}", name);
            return;
        }
        ToolMeta toolMeta = toolsRegistry.getToolMeta(name);
        if (toolMeta == null) {
            log.warn("未找到工具元数据, 工具名：{}", name);
            return;
        }
        try {
            if (toolMeta.isSearch()) {
                ToolResultParser<WebSearchResult> resultParser = toolProvider.getResultParser(toolMeta.category());
                addParsedResults(webContext.getSearchResults(), resultParser, result);
            } else if (toolMeta.isExtract()) {
                ToolResultParser<WebExtractResult> resultParser = toolProvider.getResultParser(toolMeta.category());
                addParsedResults(webContext.getExtractResults(), resultParser, result);
            }
        } catch (RuntimeException parseError) {
            // 工具结果可能包含网页正文，禁止写入普通日志。
            log.warn("Web工具结果解析失败, 工具名：{}", name);
        }
    }

    private <T> void addParsedResults(List<T> target, ToolResultParser<T> parser, String result) {
        if (parser == null) {
            return;
        }
        List<T> parsed = parser.parse(result);
        if (CollectionUtils.isNotEmpty(parsed)) {
            target.addAll(parsed);
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
        List<WebToolResult> collected = new ArrayList<>(webContext.getSearchResults());
        collected.addAll(webContext.getExtractResults());
        if (CollectionUtils.isNotEmpty(collected)) {
            Map<String, WebToolResult> referencesByUrl = new LinkedHashMap<>();
            for (WebToolResult reference : collected) {
                if (reference != null && StringUtils.isNotBlank(reference.url())) {
                    referencesByUrl.putIfAbsent(reference.url(), reference);
                }
            }
            String references = JSON.toJSONString(referencesByUrl.values());
            context.setReferences(references);
            emit(context, references, com.fons.cloud.ai.agent.constants.AgentMessageType.REFERENCE);
        }
    }

    @Override
    protected AgentRunContext createRunContext(AgentChatRequest request, String runId) {
        return new WebSearchAgentRunContext(request, runId);
    }


    /** WebSearchReactAgent 构建器；所有字段在 build 后作为共享只读配置使用。 */
    public static class Builder extends BaseAgentBuilder<Builder> {
        private final List<ToolCallback> tools;
        private final ToolRegistry toolsRegistry;
        private List<Interceptor> nativeInterceptors;

        private ReactAgentSystemPrompt systemPrompt;
        private int maxRounds = 5;

        public Builder(List<ToolCallback> tools, ChatModel chatModel, AgentTaskManager agentTaskManager, ToolRegistry toolsRegistry) {
            super(chatModel, agentTaskManager);
            this.tools = tools == null ? List.of() : List.copyOf(tools);
            this.toolsRegistry = java.util.Objects.requireNonNull(toolsRegistry,
                    "toolsRegistry cannot be null");

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

        /** 直接追加 Alibaba 原生 Interceptor，不再为高级能力创建 Fons4AI 镜像接口。 */
        public Builder nativeInterceptors(List<Interceptor> nativeInterceptors) {
            this.nativeInterceptors = nativeInterceptors == null
                    ? List.of() : List.copyOf(nativeInterceptors);
            return this;
        }

        /** 校验并创建可共享 Agent；请求态将在 start 时创建。 */
        public WebSearchReactAgent build() {
            WebSearchReactAgent reactAgent = new WebSearchReactAgent(tools, chatModel, agentTaskManager, toolsRegistry);
            reactAgent.systemPrompt = this.systemPrompt == null
                    ? ReactAgentSystemPrompt.defaultPrompt() : this.systemPrompt;
            reactAgent.maxRounds = this.maxRounds;
            if (this.nativeInterceptors == null) {
                this.nativeInterceptors = List.of();
            }
            reactAgent.nativeInterceptors = this.nativeInterceptors;
            applySharedConfig(reactAgent);
            return reactAgent;
        }
    }

}
