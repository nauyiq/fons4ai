package com.fons.cloud.ai.rag.langchain.document;

import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.ai.rag.common.document.DocumentParseRequest;
import com.fons.cloud.ai.rag.common.document.DocumentParseResult;
import com.fons.cloud.ai.rag.common.document.DocumentParserCapability;
import com.fons.cloud.ai.rag.common.document.DocumentSource;
import com.fons.cloud.ai.rag.common.document.DocumentSources;
import com.fons.cloud.ai.rag.common.document.ParserSelection;
import dev.langchain4j.data.document.Document;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link LangChain4jNativeDocumentParser} 单元测试。
 * <p>
 * 验证 native 使用 Apache Tika 直通返回 LangChain4j 原生 {@link Document}，不经过中立模型转换。
 *
 * @author hongqy
 */
class LangChain4jNativeDocumentParserTest {

    private final LangChain4jNativeDocumentParser parser = new LangChain4jNativeDocumentParser();

    @Test
    void shouldReturnNativeCapability() {
        DocumentParserCapability capability = parser.capability();

        assertEquals(LangChain4jNativeDocumentParser.PROVIDER_ID, capability.provider());
        assertTrue(capability.available());
        assertTrue(capability.supportedDocumentTypes().contains(DocumentType.PDF));
        assertTrue(capability.supportedDocumentTypes().contains(DocumentType.TEXT));
        assertTrue(capability.supportedFileExtensions().contains("pdf"));
        assertTrue(capability.supportedFileExtensions().contains("txt"));
    }

    @Test
    void shouldParseTextAndReturnNativeDocument() {
        String text = "Hello LangChain4j native parser";
        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream(text.getBytes(StandardCharsets.UTF_8)),
                "test.txt", "text/plain", 4096);
        try {
            DocumentParseRequest request = new DocumentParseRequest(
                    source, DocumentType.TEXT, "txt",
                    ParserSelection.defaultNative(), Map.of(), Map.of());

            DocumentParseResult<Document> result = parser.parse(request);

            Document doc = result.payload();
            assertNotNull(doc);
            assertTrue(doc.text().contains("Hello LangChain4j native parser"),
                    "原生 Document 文本应包含原始内容");
        } finally {
            source.close();
        }
    }

    @Test
    void shouldReturnNativeTrace() {
        String text = "trace test";
        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream(text.getBytes(StandardCharsets.UTF_8)),
                "test.txt", "text/plain", 4096);
        try {
            DocumentParseRequest request = new DocumentParseRequest(
                    source, DocumentType.TEXT, "txt",
                    ParserSelection.defaultNative(), Map.of(), Map.of());

            DocumentParseResult<Document> result = parser.parse(request);

            assertEquals("native", result.parseTrace().provider());
            assertEquals("apache-tika", result.parseTrace().backend());
            assertEquals("TEXT", result.parseTrace().outputFormat());
            assertEquals("TEXT", result.parseTrace().sourceType());
        } finally {
            source.close();
        }
    }

    @Test
    void shouldReturnSameDocumentInstanceFromPayload() {
        String text = "reference test";
        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream(text.getBytes(StandardCharsets.UTF_8)),
                "test.txt", "text/plain", 4096);
        try {
            DocumentParseRequest request = new DocumentParseRequest(
                    source, DocumentType.TEXT, "txt",
                    ParserSelection.defaultNative(), Map.of(), Map.of());

            DocumentParseResult<Document> result = parser.parse(request);

            // payload 返回的就是原生 Document 实例，不经过重建
            assertSame(result.payload(), result.payload());
            assertNotNull(result.payload().text());
        } finally {
            source.close();
        }
    }
}
