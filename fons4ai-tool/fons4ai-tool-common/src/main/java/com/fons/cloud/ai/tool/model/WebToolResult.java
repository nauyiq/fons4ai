package com.fons.cloud.ai.tool.model;

/**
 * Web 工具的统一结果类型。
 *
 * @author hongqy
 */
public sealed interface WebToolResult permits WebSearchResult, WebExtractResult {

    /**
     * 返回结果对应的 URL。
     *
     * @return URL
     */
    String url();
}
