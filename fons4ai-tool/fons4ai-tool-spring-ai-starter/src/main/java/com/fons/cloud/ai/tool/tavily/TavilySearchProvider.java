package com.fons.cloud.ai.tool.tavily;

import com.fons.cloud.ai.tool.constants.ToolCategory;
import com.fons.cloud.ai.tool.spi.ToolProvider;
import com.fons.cloud.ai.tool.spi.ToolResultParser;
import com.fons.cloud.ai.tool.tavily.parser.TavilyExtractResultParser;
import com.fons.cloud.ai.tool.tavily.parser.TavilySearchResultParser;
import org.apache.commons.lang3.StringUtils;

/**
 * Tavily 工具提供者。
 *
 * @author hongqy
 */
public class TavilySearchProvider implements ToolProvider {

    public static final String PROVIDER_NAME = "tavily";
    private static final TavilySearchResultParser SEARCH_RESULT_PARSER = new TavilySearchResultParser();
    private static final TavilyExtractResultParser EXTRACT_RESULT_PARSER = new TavilyExtractResultParser();

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
            case SEARCH -> (ToolResultParser<T>) SEARCH_RESULT_PARSER;
            case EXTRACT -> (ToolResultParser<T>) EXTRACT_RESULT_PARSER;
            default -> null;
        };
    }
}
