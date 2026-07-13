package com.fons.cloud.ai.tool.model;

/**
 * 网页内容提取结果。
 *
 * @param url 提取的网页 URL
 * @param rawContent 网页正文
 * @author hongqy
 */
public record WebExtractResult(String url, String rawContent) implements WebToolResult {
}
