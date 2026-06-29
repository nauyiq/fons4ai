package com.fons.cloud.ai.agent.response;

/**
 * 涉及web相关的工具类， 比如网络搜索，网页内容提取等
 * @author hongqy
 */
public sealed interface WebToolResult
        permits WebSearchResult, WebExtractResult {

    /**
     * 请求的url
     * @return
     */
    String url();

}
