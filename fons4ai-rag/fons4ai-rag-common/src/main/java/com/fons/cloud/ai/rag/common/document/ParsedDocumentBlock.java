package com.fons.cloud.ai.rag.common.document;

import java.util.Map;

/**
 * 中立 provider 的可选分段内容。
 * <p>
 * 不承担 Spring AI/LangChain4j 原生对象的无损序列化职责。
 *
 * @param content  分段文本内容
 * @param metadata 分段元数据，不可变
 * @author hongqy
 */
public record ParsedDocumentBlock(
        String content,
        Map<String, Object> metadata
) {
}
