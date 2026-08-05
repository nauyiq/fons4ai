package com.fons.cloud.ai.rag.langchain.document;

import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.ai.rag.common.document.DocumentParseRequest;
import com.fons.cloud.ai.rag.common.document.DocumentParserRegistry;
import com.fons.cloud.ai.rag.common.document.DocumentParserSelector;
import com.fons.cloud.ai.rag.common.document.DocumentSource;
import com.fons.cloud.ai.rag.common.document.DocumentSources;
import com.fons.cloud.ai.rag.common.document.ParserSelection;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.segment.TextSegment;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link LangChain4jDocumentParserFacade#parseAndSplit} 单元测试。
 *
 * @author hongqy
 */
class LangChain4jDocumentParserFacadeSplitTest {

    private LangChain4jDocumentParserFacade createFacade(int chunkSize, int overlap) {
        LangChain4jNativeDocumentParser nativeParser = new LangChain4jNativeDocumentParser();
        DocumentParserRegistry<Document> registry = new DocumentParserRegistry<>();
        registry.register(nativeParser);
        DocumentParserSelector<Document> selector = new DocumentParserSelector<>(registry);
        LangChain4jDocumentSplitter splitter = new LangChain4jDocumentSplitter(chunkSize, overlap);
        return new LangChain4jDocumentParserFacade(selector, splitter);
    }

    /** AC-003: parseAndSplit 等价于先 parse 再 split */
    @Test
    void shouldParseAndSplitEquivalently() {
        LangChain4jDocumentParserFacade facade = createFacade(100, 0);
        String text = "段落一内容。".repeat(50);

        List<TextSegment> segments;
        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream(text.getBytes(StandardCharsets.UTF_8)),
                "test.txt", "text/plain", 4096);
        try {
            DocumentParseRequest request = new DocumentParseRequest(
                    source, DocumentType.TEXT, "txt",
                    ParserSelection.defaultNative(), Map.of(), Map.of());
            segments = facade.parseAndSplit(request);
        } finally {
            source.close();
        }

        assertFalse(segments.isEmpty());

        // 验证等价于两步
        DocumentSource source2 = DocumentSources.fromInputStream(
                new ByteArrayInputStream(text.getBytes(StandardCharsets.UTF_8)),
                "test.txt", "text/plain", 4096);
        try {
            DocumentParseRequest request2 = new DocumentParseRequest(
                    source2, DocumentType.TEXT, "txt",
                    ParserSelection.defaultNative(), Map.of(), Map.of());
            Document doc = facade.parse(request2);
            LangChain4jDocumentSplitter splitter = new LangChain4jDocumentSplitter(100, 0);
            List<TextSegment> twoStep = splitter.split(doc);
            assertEquals(twoStep.size(), segments.size());
        } finally {
            source2.close();
        }
    }

    /** AC-003: parseAndSplit 返回非空 TextSegment */
    @Test
    void shouldReturnNonEmptySegments() {
        LangChain4jDocumentParserFacade facade = createFacade(50, 10);
        String text = "测试内容".repeat(100);

        List<TextSegment> segments;
        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream(text.getBytes(StandardCharsets.UTF_8)),
                "test.txt", "text/plain", 4096);
        try {
            DocumentParseRequest request = new DocumentParseRequest(
                    source, DocumentType.TEXT, "txt",
                    ParserSelection.defaultNative(), Map.of(), Map.of());
            segments = facade.parseAndSplit(request);
        } finally {
            source.close();
        }

        assertFalse(segments.isEmpty());
        assertTrue(segments.size() > 1);
    }

    /** split(Document) 对已解析文档分块 */
    @Test
    void shouldSplitAlreadyParsedDocument() {
        LangChain4jDocumentParserFacade facade = createFacade(50, 10);
        String text = "测试内容".repeat(100);

        Document doc;
        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream(text.getBytes(StandardCharsets.UTF_8)),
                "test.txt", "text/plain", 4096);
        try {
            DocumentParseRequest request = new DocumentParseRequest(
                    source, DocumentType.TEXT, "txt",
                    ParserSelection.defaultNative(), Map.of(), Map.of());
            doc = facade.parse(request);
        } finally {
            source.close();
        }

        List<TextSegment> segments = facade.split(doc);

        assertFalse(segments.isEmpty());
        assertTrue(segments.size() > 1);
    }
}
