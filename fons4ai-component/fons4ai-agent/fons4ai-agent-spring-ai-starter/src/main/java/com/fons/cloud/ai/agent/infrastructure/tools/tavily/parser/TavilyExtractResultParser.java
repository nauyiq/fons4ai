package com.fons.cloud.ai.agent.infrastructure.tools.tavily.parser;

import com.fasterxml.jackson.databind.JsonNode;
import com.fons.cloud.ai.agent.response.WebExtractResult;
import org.apache.commons.lang3.StringUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * Tavily 提取结果解析器。
 * <pre>
 *     典型返回格式（MCP 封装后）:
 *     [{ "text": { "results": [{ "url": "...", "raw_content": "..." }] } }]
 * </pre>
 * @author hongqy
 */
public class TavilyExtractResultParser extends AbstractTavilyResultParser<WebExtractResult> {

    @Override
    protected List<WebExtractResult> parseResult(JsonNode resultsNode) {
        List<WebExtractResult> results = new ArrayList<>();
        for (JsonNode node : resultsNode) {
            String url = getSafe(node, "url");
            if (StringUtils.isNotBlank(url)) {
                results.add(new WebExtractResult(url,
                        getSafe(node, "rawContent"))
                );
            }
        }
        return results;
    }
}
