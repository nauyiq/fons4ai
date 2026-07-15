package com.fons.cloud.ai.tool.model;

import com.fons.cloud.ai.tool.constants.ToolCategory;

/**
 * 工具元数据。
 *
 * @param toolName 原始工具名
 * @param providerName 工具提供者名称
 * @param category 工具分类
 * @author hongqy
 */
public record ToolMeta(String toolName, String providerName, ToolCategory category) {

    private static final ToolMeta UNKNOWN_META =
            new ToolMeta("", "unknown", ToolCategory.UNKNOWN);

    /**
     * 返回未知工具元数据。
     *
     * @return 未知工具元数据
     */
    public static ToolMeta unknown() {
        return UNKNOWN_META;
    }

    public boolean isWebTool() {
        return category == ToolCategory.SEARCH || category ==  ToolCategory.EXTRACT;
    }

    public boolean isSearch() {
        return category == ToolCategory.SEARCH;
    }

    public boolean isExtract() {
        return category == ToolCategory.EXTRACT;
    }

    public boolean isCrawl() {
        return category == ToolCategory.CRAWL;
    }

    public boolean isUnknown() {
        return category == ToolCategory.UNKNOWN;
    }
}
