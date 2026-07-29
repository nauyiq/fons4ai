package com.fons.cloud.ai.rag.document.reader;

import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.ai.rag.common.document.DocumentParseResult;
import com.fons.cloud.ai.rag.common.document.DocumentParseRequest;
import com.fons.cloud.ai.rag.common.document.DocumentSource;
import com.fons.cloud.ai.rag.common.document.DocumentSources;
import com.fons.cloud.ai.rag.common.document.ParserSelection;
import com.fons.cloud.ai.rag.common.document.ParsedDocument;
import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;

import java.io.ByteArrayInputStream;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link SpringAiDocumentAdapter} 测试。
 * <p>
 * 验证 MinerU ParsedDocument 到 Spring AI Document 的转换，
 * 以及 Markdown 结构性空白保持。
 *
 * @author hongqy
 */
class SpringAiDocumentAdapterTest {

    @Test
    void shouldConvertParsedDocumentToSpringAiDocument() {
        String md = "# Title\n\nContent";
        ParsedDocument parsed = new ParsedDocument(md, "MARKDOWN", Map.of(), List.of(), List.of());

        SpringAiDocumentAdapter adapter = new SpringAiDocumentAdapter();
        List<Document> docs = adapter.toDocuments(parsed);

        assertEquals(1, docs.size());
        assertEquals(md, docs.get(0).getText());
    }

    @Test
    void shouldPreserveMarkdownWhitespaceStructure() {
        String md = "# Title\n\n\n\nText   with   spaces\n\t\tTabbed\n\n- List\n\n$$formula$$\n";
        ParsedDocument parsed = new ParsedDocument(md, "MARKDOWN", Map.of(), List.of(), List.of());

        SpringAiDocumentAdapter adapter = new SpringAiDocumentAdapter();
        List<Document> docs = adapter.toDocuments(parsed);

        // 字符级保持，不经过空白压缩
        assertEquals(md, docs.get(0).getText());
    }

    @Test
    void shouldReturnEmptyForNullContent() {
        SpringAiDocumentAdapter adapter = new SpringAiDocumentAdapter();
        List<Document> docs = adapter.toDocuments(null);
        assertTrue(docs.isEmpty());
    }

    @Test
    void shouldReturnEmptyForBlankContent() {
        ParsedDocument parsed = new ParsedDocument("  ", "MARKDOWN", Map.of(), List.of(), List.of());
        SpringAiDocumentAdapter adapter = new SpringAiDocumentAdapter();
        List<Document> docs = adapter.toDocuments(parsed);
        assertTrue(docs.isEmpty());
    }

    @Test
    void shouldPreserveMetadata() {
        Map<String, Object> metadata = Map.of("source", "mineru", "page", 1);
        ParsedDocument parsed = new ParsedDocument("content", "MARKDOWN", metadata, List.of(), List.of());

        SpringAiDocumentAdapter adapter = new SpringAiDocumentAdapter();
        List<Document> docs = adapter.toDocuments(parsed);

        assertEquals(1, docs.size());
        assertEquals("mineru", docs.get(0).getMetadata().get("source"));
    }
}
