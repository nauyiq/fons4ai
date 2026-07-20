package com.fons.cloud.ai.agent.standard.react.websearch;

import com.fons.cloud.ai.agent.chat.AgentChatRequest;
import com.fons.cloud.ai.agent.chat.AgentExecutionContext;
import com.fons.cloud.ai.agent.constants.AgentType;
import com.fons.cloud.ai.agent.standard.react.ReactAgentRunContext;
import com.fons.cloud.ai.tool.model.WebExtractResult;
import com.fons.cloud.ai.tool.model.WebSearchResult;
import lombok.Getter;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Web Search ReAct 单次执行的引用聚合状态。
 * @author hongqy
 */
@Getter
public final class WebSearchAgentRunContext extends ReactAgentRunContext {

    /**
     * 网页搜索结果
     */
    private final List<WebSearchResult> searchResults = new CopyOnWriteArrayList<>();

    /**
     * 网页提取结果
     */
    private final List<WebExtractResult> extractResults = new CopyOnWriteArrayList<>();

    public WebSearchAgentRunContext(AgentChatRequest request, String runId) {
        super(AgentType.REACT, request, runId);
    }
}
