package com.fons.cloud.ai.tool.tavily.parser;

import com.fasterxml.jackson.databind.JsonNode;
import com.fons.cloud.ai.tool.model.WebSearchResult;
import org.apache.commons.lang3.StringUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * Tavily 搜索结果解析器。
 *
 * @author hongqy
 */
public class TavilySearchResultParser extends AbstractTavilyResultParser<WebSearchResult> {

    @Override
    protected List<WebSearchResult> parseResult(JsonNode resultsNode) {
        List<WebSearchResult> results = new ArrayList<>();
        for (JsonNode node : resultsNode) {
            String url = getSafe(node, "url");
            if (StringUtils.isNotBlank(url)) {
                results.add(new WebSearchResult(
                        url,
                        getSafe(node, "title"),
                        getSafe(node, "favicon"),
                        getSafe(node, "content")));
            }
        }
        return results;
    }
}
