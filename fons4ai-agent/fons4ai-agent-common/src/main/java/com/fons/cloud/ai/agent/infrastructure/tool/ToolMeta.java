package com.fons.cloud.ai.agent.infrastructure.tool;

import com.fons.cloud.ai.agent.constants.ToolCategory;

/**
 * 工具元数据
 * @param toolName      原始工具名，如 "tavily-search"
 * @param providerName  搜索源标识，如 "tavily"
 * @param category      工具类型
 * @author hongqy
 */
public record ToolMeta(String toolName, String providerName, ToolCategory category) {

    /**
     * 未知的工具元数据
     */
    private static final ToolMeta UNKNOWN_META =
            new ToolMeta("", "unknown", ToolCategory.UNKNOWN);
    public static ToolMeta unknown() {
        return UNKNOWN_META;
    }

    /**
     * 是否是搜索类工具
     * @return
     */
    public boolean isSearch() {
        return category == ToolCategory.SEARCH;
    }

    /**
     * 是否是提取类工具
     * @return
     */
    public boolean isExtract() {
        return category == ToolCategory.EXTRACT;
    }

    /**
     * 是否是爬取类工具
     * @return
     */
    public boolean isCrawl() {
        return category == ToolCategory.CRAWL;
    }

    /**
     * 是否是未分类的工具
     * @return
     */
    public boolean isUnknown() {
        return category == ToolCategory.UNKNOWN;
    }


}
