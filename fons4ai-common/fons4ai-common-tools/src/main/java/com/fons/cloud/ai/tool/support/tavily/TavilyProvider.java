package com.fons.cloud.ai.tool.support.tavily;

import com.fons.cloud.ai.tool.api.ToolProvider;
import com.fons.cloud.ai.tool.api.ToolResultParser;
import com.fons.cloud.ai.tool.common.constants.ToolCategory;
import com.fons.cloud.ai.tool.common.model.ToolInfo;
import com.fons.cloud.ai.tool.support.tavily.parser.TavilyExtractResultParser;
import com.fons.cloud.ai.tool.support.tavily.parser.TavilySearchResultParser;
import org.apache.commons.lang3.StringUtils;

/**
 * @author hongqy
 */
public class TavilyProvider implements ToolProvider {
    public static final String PROVIDER_NAME = "tavily";
    private final TavilySearchResultParser searchResultParser = new TavilySearchResultParser();
    private final TavilyExtractResultParser extractResultParser = new TavilyExtractResultParser();

    @Override
    public String providerName() {
        return PROVIDER_NAME;
    }
    @Override
    public boolean isSupport(String toolName, String inputSchema) {
        return StringUtils.isNotBlank(toolName) && toolName.startsWith(PROVIDER_NAME);
    }

    @Override
    public ToolCategory getToolCategory(String toolName) {
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
    @SuppressWarnings("unchecked")
    public <T> ToolResultParser<T> getResultParser(ToolInfo toolInfo) {
        if (toolInfo == null) {
            return null;
        }
        ToolCategory category = toolInfo.category();
        return switch (category) {
            case SEARCH ->  (ToolResultParser<T>) searchResultParser;
            case EXTRACT -> (ToolResultParser<T>) extractResultParser;
            default -> null;
        };
    }
}
