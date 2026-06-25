package com.fons.cloud.ai.agent.infrastructure.tools.tavily.parser;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fons.cloud.ai.agent.infrastructure.tool.ToolResultParser;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.List;

/**
 * 抽象的 Tavily 结果解析器。
 * <pre>
 *     提取公用的部分结果， 即results的内容提取出来  ["text:results:[{}, {}]"]
 * </pre>
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
            JsonNode first = root.get(0);
            JsonNode textNode = first.get("text");
            if (textNode == null || textNode.isNull()) {
                return results;
            }

            JsonNode textJson = textNode.isTextual() ? MAPPER.readTree(textNode.asText()) : textNode;
            // Tavily响应结果
            JsonNode resultsNode = textJson.get("results");
            if (resultsNode == null || !resultsNode.isArray()) {
                return results;
            }

            results = parseResult(resultsNode);

        } catch (Exception e) {
            log.warn("解析 Tavily 结果失败: {}", e.getMessage(), e);
        }

        return results;
    }

    /**
     * 由子类控制如何解析 Tavily 响应后的 "results"字段内容
     * @param resultsNode "results"字段内容
     * @return
     */
    protected abstract List<T> parseResult(JsonNode resultsNode);



    protected String getSafe(JsonNode node, String field) {
        JsonNode value = node.get(field);
        return value == null || value.isNull() ? null : value.asText();
    }
}
