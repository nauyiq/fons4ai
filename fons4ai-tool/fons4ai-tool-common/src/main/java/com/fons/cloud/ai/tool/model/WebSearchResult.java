package com.fons.cloud.ai.tool.model;

/**
 * Web 搜索结果。
 *
 * @param url 结果 URL
 * @param title 页面标题
 * @param favicon 站点图标地址
 * @param content 搜索结果摘要
 * @author hongqy
 */
public record WebSearchResult(String url, String title, String favicon, String content) implements WebToolResult {
}
