package com.fons.cloud.ai.rag.infrastructure.chunking;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;

import java.util.List;

/**
 * 输出精确输入字符来源范围的 LangChain4j 文档分块扩展。
 *
 * <p>范围使用 Java {@link String} 的 UTF-16 半开下标；调用方应使用每个结果的
 * {@link SourceAwareTextSegment#sourceRanges()} 做精确来源映射。</p>
 *
 * @author hongqy
 */
public interface SourceAwareDocumentSplitter extends DocumentSplitter {

    /**
     * 分块并输出每个片段对输入文档的精确贡献范围。
     *
     * @param document 输入文档，不可为空
     * @return 带来源范围的不可变分块结果
     */
    List<SourceAwareTextSegment> splitWithSpans(Document document);
}
