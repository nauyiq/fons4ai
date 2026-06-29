package com.fons.cloud.ai.agent.response;

/**
 * 搜索结果
 * @author hongqy
 */
public record WebSearchResult(String url, String title, String favicon, String content) implements WebToolResult {

}
