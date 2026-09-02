package com.fons.cloud.ai.tool.support.tavily.parser;

import com.fasterxml.jackson.databind.JsonNode;
import com.fons.cloud.ai.tool.common.model.web.WebExtractResult;
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
                results.add(WebExtractResult.builder()
                                .url(url)
                                .rawContent( getSafe(node, "rawContent"))
                        .build());
            }
        }
        return results;
    }
}
