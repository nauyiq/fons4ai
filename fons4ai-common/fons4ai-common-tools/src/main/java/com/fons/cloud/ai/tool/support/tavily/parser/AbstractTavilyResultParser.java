package com.fons.cloud.ai.tool.support.tavily.parser;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fons.cloud.ai.tool.api.ToolResultParser;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.List;

/**
 * Tavily 结果解析器公共基类。
 *
 * @param <T> 解析结果类型
 * @author hongqy
 */
@Slf4j
public abstract class AbstractTavilyResultParser<T> implements ToolResultParser<T> {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Override
    public List<T> parse(String result) {
        List<T> results = new ArrayList<>();
        try {
            JsonNode root = MAPPER.readTree(result);
            if (!root.isArray() || root.isEmpty()) {
                return results;
            }
            JsonNode textNode = root.get(0).get("text");
            if (textNode == null || textNode.isNull()) {
                return results;
            }
            JsonNode textJson = textNode.isTextual() ? MAPPER.readTree(textNode.asText()) : textNode;
            JsonNode resultsNode = textJson.get("results");
            if (resultsNode == null || !resultsNode.isArray()) {
                return results;
            }
            return parseResult(resultsNode);
        } catch (Exception e) {
            log.warn("解析 Tavily 结果失败: {}", e.getMessage(), e);
            return results;
        }
    }

    protected abstract List<T> parseResult(JsonNode resultsNode);

    protected String getSafe(JsonNode node, String field) {
        JsonNode value = node.get(field);
        return value == null || value.isNull() ? null : value.asText();
    }
}
