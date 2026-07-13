package com.fons.cloud.ai.tool.tavily.parser;

import com.fasterxml.jackson.databind.JsonNode;
import com.fons.cloud.ai.tool.model.WebExtractResult;
import org.apache.commons.lang3.StringUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * Tavily 网页提取结果解析器。
 *
 * @author hongqy
 */
public class TavilyExtractResultParser extends AbstractTavilyResultParser<WebExtractResult> {

    @Override
    protected List<WebExtractResult> parseResult(JsonNode resultsNode) {
        List<WebExtractResult> results = new ArrayList<>();
        for (JsonNode node : resultsNode) {
            String url = getSafe(node, "url");
            if (StringUtils.isNotBlank(url)) {
                results.add(new WebExtractResult(url, getSafe(node, "rawContent")));
            }
        }
        return results;
    }
}
