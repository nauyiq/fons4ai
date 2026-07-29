package com.fons.cloud.ai.rag.langchain.document;

import com.fons.cloud.ai.rag.common.document.ParsedDocument;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * LangChain4j 文档适配器。
 * <p>
 * 只将 MinerU/common 中立 {@link ParsedDocument} 转换为 LangChain4j 原生 {@link Document}；
 * 不参与 native 路径，也不承担 LangChain4j 对象的往返序列化。
 * <p>
 * 元数据映射规则：仅保留 LangChain4j {@link Metadata} 支持的标量类型
 * （String、Integer、Long、Float、Double；UUID 由调用方自行处理），非标量值（含 Boolean）转为字符串，不用 JSON 序列化替代普通映射。
 *
 * @author hongqy
 */
public final class LangChain4jDocumentAdapter {

    /**
     * 将中立 {@link ParsedDocument} 转换为 LangChain4j 原生 {@link Document}。
     *
     * @param parsedDocument 中立解析结果，不可为 null
     * @return LangChain4j 原生 Document
     */
    public Document toDocument(ParsedDocument parsedDocument) {
        if (parsedDocument == null) {
            throw new IllegalArgumentException("parsedDocument 不可为空");
        }
        String content = parsedDocument.content();
        if (content == null || content.isBlank()) {
            throw new IllegalArgumentException("解析内容不可为空");
        }
        Metadata metadata = toMetadata(parsedDocument.metadata());
        return Document.from(content, metadata);
    }

    /**
     * 将中立元数据 Map 转换为 LangChain4j {@link Metadata}。
     * <p>
     * 标量值原样保留；非标量值转为字符串；null 值跳过。
     *
     * @param metadata 中立元数据，可为 null 或空
     * @return LangChain4j Metadata
     */
    private Metadata toMetadata(Map<String, Object> metadata) {
        if (metadata == null || metadata.isEmpty()) {
            return Metadata.from(Map.of());
        }
        Map<String, Object> filtered = new LinkedHashMap<>(metadata.size());
        for (Map.Entry<String, Object> entry : metadata.entrySet()) {
            Object value = entry.getValue();
            if (value == null) {
                continue;
            }
            if (isSupportedScalar(value)) {
                filtered.put(entry.getKey(), value);
            } else {
                filtered.put(entry.getKey(), String.valueOf(value));
            }
        }
        return Metadata.from(filtered);
    }

    /**
     * 判断值是否为 LangChain4j Metadata 支持的标量类型。
     * <p>
     * LangChain4j Metadata 支持 String、Integer、Long、Float、Double（及 UUID），
     * 不支持 Boolean，Boolean 等非标量值由调用方转为字符串。
     */
    private static boolean isSupportedScalar(Object value) {
        return value instanceof String
                || value instanceof Integer
                || value instanceof Long
                || value instanceof Float
                || value instanceof Double;
    }
}
