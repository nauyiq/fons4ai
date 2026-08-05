package com.fons.cloud.ai.rag.langchain.document;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.segment.TextSegment;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link LangChain4jDocumentSplitter} 单元测试。
 *
 * @author hongqy
 */
class LangChain4jDocumentSplitterTest {

    /** AC-001: chunkSize=500/overlap=50 分块正确性 */
    @Test
    void shouldSplitDocumentWithChunkSizeAndOverlap() {
        LangChain4jDocumentSplitter splitter = new LangChain4jDocumentSplitter(500, 50);
        String text = "a".repeat(1200);
        Document doc = Document.from(text);

        List<TextSegment> segments = splitter.split(doc);

        assertFalse(segments.isEmpty());
        for (TextSegment seg : segments) {
            assertTrue(seg.text().length() <= 500,
                    "片段长度 " + seg.text().length() + " 超过 chunkSize 500");
        }
    }

    /** AC-002: 多段落文档在段落边界优先切分 */
    @Test
    void shouldSplitAtParagraphBoundaryFirst() {
        LangChain4jDocumentSplitter splitter = new LangChain4jDocumentSplitter(100, 0);
        String para1 = "第一段落内容。这是第一段的文字。";
        String para2 = "第二段落内容。这是第二段的文字。";
        String para3 = "第三段落内容。这是第三段的文字。";
        Document doc = Document.from(para1 + "\n\n" + para2 + "\n\n" + para3);

        List<TextSegment> segments = splitter.split(doc);

        assertFalse(segments.isEmpty());
        // 至少有一个片段以段落开头
        boolean hasParagraphStart = segments.stream()
                .anyMatch(seg -> seg.text().startsWith("第一段落") || seg.text().startsWith("第二段落"));
        assertTrue(hasParagraphStart, "应在段落边界切分");
    }

    /** AC-005: TextSegment 继承原始 Document 的 metadata */
    @Test
    void shouldInheritMetadataFromDocument() {
        LangChain4jDocumentSplitter splitter = new LangChain4jDocumentSplitter(1000, 0);
        Metadata metadata = Metadata.from("source", "test.pdf");
        metadata.put("format", "pdf");
        Document doc = Document.from("测试内容".repeat(100), metadata);

        List<TextSegment> segments = splitter.split(doc);

        assertFalse(segments.isEmpty());
        TextSegment first = segments.get(0);
        assertEquals("test.pdf", first.metadata().toMap().get("source"));
        assertEquals("pdf", first.metadata().toMap().get("format"));
    }

    /** null Document 返回空列表 */
    @Test
    void shouldReturnEmptyListForNullDocument() {
        LangChain4jDocumentSplitter splitter = new LangChain4jDocumentSplitter(500, 50);
        List<TextSegment> segments = splitter.split((Document) null);
        assertTrue(segments.isEmpty());
    }

    /** 空 Document 列表返回空列表 */
    @Test
    void shouldReturnEmptyListForEmptyDocumentList() {
        LangChain4jDocumentSplitter splitter = new LangChain4jDocumentSplitter(500, 50);
        List<TextSegment> segments = splitter.split(List.of());
        assertTrue(segments.isEmpty());
    }

    /** 多 Document 列表分块 */
    @Test
    void shouldSplitMultipleDocuments() {
        LangChain4jDocumentSplitter splitter = new LangChain4jDocumentSplitter(100, 0);
        Document doc1 = Document.from("文档一".repeat(50));
        Document doc2 = Document.from("文档二".repeat(50));

        List<TextSegment> segments = splitter.split(List.of(doc1, doc2));

        assertFalse(segments.isEmpty());
        assertTrue(segments.size() >= 2);
    }

    /** AC-006: chunkSize=0 构造器抛出异常 */
    @Test
    void shouldThrowWhenChunkSizeIsZero() {
        assertThrows(IllegalArgumentException.class,
                () -> new LangChain4jDocumentSplitter(0, 0));
    }

    /** AC-006: chunkSize 为负数构造器抛出异常 */
    @Test
    void shouldThrowWhenChunkSizeIsNegative() {
        assertThrows(IllegalArgumentException.class,
                () -> new LangChain4jDocumentSplitter(-1, 0));
    }

    /** AC-006: overlap 为负数构造器抛出异常 */
    @Test
    void shouldThrowWhenOverlapIsNegative() {
        assertThrows(IllegalArgumentException.class,
                () -> new LangChain4jDocumentSplitter(500, -1));
    }

    /** AC-006: overlap >= chunkSize 构造器抛出异常 */
    @Test
    void shouldThrowWhenOverlapGreaterOrEqualChunkSize() {
        assertThrows(IllegalArgumentException.class,
                () -> new LangChain4jDocumentSplitter(500, 500));
    }
}
