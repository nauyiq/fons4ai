package com.fons.cloud.ai.agent.infrastructure.tools.tavily.parser;

import com.fasterxml.jackson.databind.JsonNode;
import com.fons.cloud.ai.agent.response.WebSearchResult;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * Tavily 搜索结果解析器。
 * 解析 tavily-search 工具返回的 JSON，提取 url/title/content 字段。
 * <pre>
 * 典型返回格式（MCP 封装后）: [{ "text": { "results": [{ "url": "...", "title": "...", "content": "..." }] } }]
 * </pre>
 * @author hongqy
 */
@Slf4j
public class TavilySearchResultParser extends AbstractTavilyResultParser<WebSearchResult> {

    @Override
    protected List<WebSearchResult> parseResult(JsonNode resultsNode) {
        List<WebSearchResult> results = new ArrayList<>();
        for (JsonNode node : resultsNode) {
            String url = getSafe(node, "url");
            if (StringUtils.isNotBlank(url)) {
                results.add(new WebSearchResult(url,
                        getSafe(node, "title"),
                        getSafe(node, "favicon"),
                        getSafe(node, "content"))
                );
            }
        }
        return results;
    }


}
