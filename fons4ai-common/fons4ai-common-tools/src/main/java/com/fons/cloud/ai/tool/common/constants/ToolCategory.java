package com.fons.cloud.ai.tool.common.constants;

/**
 * 常用工具分类，用于工具注册时的分类标识。
 *
 * @author hongqy
 */
public enum ToolCategory {

    /** 搜索类：tavily-search、bing-search。 */
    SEARCH,

    /** 提取类：tavily-extract。 */
    EXTRACT,

    /** 爬取类：tavily-crawl。 */
    CRAWL,

    /** 未分类。 */
    UNKNOWN
}
