package com.fons.cloud.ai.agent.infrastructure.tools.tavily;

import com.fons.cloud.ai.agent.constants.ToolCategory;
import com.fons.cloud.ai.agent.infrastructure.tool.ToolProvider;
import com.fons.cloud.ai.agent.infrastructure.tool.ToolResultParser;
import com.fons.cloud.ai.agent.infrastructure.tools.tavily.parser.TavilyExtractResultParser;
import com.fons.cloud.ai.agent.infrastructure.tools.tavily.parser.TavilySearchResultParser;
import org.apache.commons.lang3.StringUtils;

/**
 * Tavily 搜索源实现。
 * Tavily MCP 服务器提供两个工具：
 * <ul>
 *   <li>tavily-search — 网页搜索</li>
 *   <li>tavily-extract — 网页内容提取</li>
 * </ul>
 * @author hongqy
 */
public class TavilySearchProvider implements ToolProvider {
    public static final String PROVIDER_NAME = "tavily";
    private static final TavilySearchResultParser TAVILY_SEARCH_RESULT_PARSER = new TavilySearchResultParser();
    private static final TavilyExtractResultParser TAVILY_EXTRACT_RESULT_PARSER = new TavilyExtractResultParser();

    @Override
    public String getProviderName() {
        return PROVIDER_NAME;
    }

    @Override
    public ToolCategory resolveCategory(String toolName) {
        if (StringUtils.isBlank(toolName)) {
            return ToolCategory.UNKNOWN;
        }
        if (toolName.contains("search")) {
            return ToolCategory.SEARCH;
        }
        if (toolName.contains("extract")) {
            return ToolCategory.EXTRACT;
        }
        if (toolName.contains("crawl")) {
            return ToolCategory.CRAWL;
        }
        return ToolCategory.UNKNOWN;
    }

    @Override
    public boolean supports(String toolName, String inputSchema) {
        return StringUtils.isNotBlank(toolName) && toolName.startsWith(PROVIDER_NAME);
    }

    @Override
    @SuppressWarnings("unchecked")
    public <T> ToolResultParser<T> getResultParser(ToolCategory category) {
        return switch (category) {
            case SEARCH ->  (ToolResultParser<T>) TAVILY_SEARCH_RESULT_PARSER;
            case EXTRACT -> (ToolResultParser<T>) TAVILY_EXTRACT_RESULT_PARSER;
            default -> null;
        };
    }
}
