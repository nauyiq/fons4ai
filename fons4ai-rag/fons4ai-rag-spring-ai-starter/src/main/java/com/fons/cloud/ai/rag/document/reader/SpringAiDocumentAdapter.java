package com.fons.cloud.ai.rag.document.reader;

import com.fons.cloud.ai.rag.common.document.ParsedDocument;
import org.springframework.ai.document.Document;

import java.util.List;
import java.util.Map;

/**
 * 将 MinerU 中立 {@link ParsedDocument} 转为 Spring AI 原生 {@link Document} 列表。
 * <p>
 * 只处理 MinerU 路径，不参与 native 路径，也不承担 Spring AI 对象的往返序列化。
 * MinerU V1 blocks 为空时转为一个 Spring AI Document。
 * <p>
 * 保留 MinerU Markdown 的结构性空白，不执行压缩。
 *
 * @author hongqy
 */
public final class SpringAiDocumentAdapter {

    /**
     * 将中立 ParsedDocument 转为 Spring AI Document 列表。
     *
     * @param parsedDocument MinerU 中立解析结果
     * @return 包含一个或多个 Spring AI Document 的列表
     */
    public List<Document> toDocuments(ParsedDocument parsedDocument) {
        if (parsedDocument == null || parsedDocument.content() == null || parsedDocument.content().isBlank()) {
            return List.of();
        }

        Map<String, Object> metadata = parsedDocument.metadata() != null
                ? parsedDocument.metadata()
                : Map.of();

        Document doc = new Document(parsedDocument.content(), metadata);
        return List.of(doc);
    }
}
