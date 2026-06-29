package com.fons.cloud.ai.agent.response;

/**
 * 网页内容提取结果记录
 * @author hongqy
 * @param url        提取的网页 URL
 * @param rawContent 网页完整正文内容
 */
public record WebExtractResult(String url, String rawContent) implements WebToolResult {
}
