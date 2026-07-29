package com.fons.cloud.ai.rag.langchain.document;

import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.ai.rag.common.document.DocumentParseRequest;
import com.fons.cloud.ai.rag.common.document.DocumentParseResult;
import com.fons.cloud.ai.rag.common.document.DocumentParserRegistry;
import com.fons.cloud.ai.rag.common.document.DocumentParserSelector;
import com.fons.cloud.ai.rag.common.document.DocumentSource;
import com.fons.cloud.ai.rag.common.document.DocumentSources;
import com.fons.cloud.ai.rag.common.document.ParserSelection;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link LangChain4jDocumentParserFacade} 和 {@link LangChain4jDocumentParserAdapterFactory} 单元测试。
 *
 * @author hongqy
 */
class LangChain4jDocumentParserFacadeTest {

    private final LangChain4jDocumentParserFacade facade;

    LangChain4jDocumentParserFacadeTest() {
        LangChain4jNativeDocumentParser nativeParser = new LangChain4jNativeDocumentParser();
        DocumentParserRegistry<Document> registry = new DocumentParserRegistry<>();
        registry.register(nativeParser);
        DocumentParserSelector<Document> selector = new DocumentParserSelector<>(registry);
        this.facade = new LangChain4jDocumentParserFacade(selector);
    }

    @Test
    void shouldParseWithDefaultNativeAndReturnDocument() {
        String text = "facade default native test";
        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream(text.getBytes(StandardCharsets.UTF_8)),
                "test.txt", "text/plain", 4096);
        try {
            DocumentParseRequest request = new DocumentParseRequest(
                    source, DocumentType.TEXT, "txt",
                    ParserSelection.defaultNative(), Map.of(), Map.of());

            Document doc = facade.parse(request);

            assertNotNull(doc);
            assertTrue(doc.text().contains("facade default native test"));
        } finally {
            source.close();
        }
    }

    @Test
    void shouldParseWithTrace() {
        String text = "trace facade test";
        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream(text.getBytes(StandardCharsets.UTF_8)),
                "test.txt", "text/plain", 4096);
        try {
            DocumentParseRequest request = new DocumentParseRequest(
                    source, DocumentType.TEXT, "txt",
                    ParserSelection.defaultNative(), Map.of(), Map.of());

            DocumentParseResult<Document> result = facade.parseWithTrace(request);

            assertNotNull(result.payload());
            assertTrue(result.payload().text().contains("trace facade test"));
            assertEquals("native", result.parseTrace().provider());
            assertEquals("DEFAULT 选型，使用 native provider",
                    result.parseTrace().selectionReason());
        } finally {
            source.close();
        }
    }

    @Test
    void shouldAdaptToStandardDocumentParser() {
        LangChain4jDocumentParserAdapterFactory factory =
                new LangChain4jDocumentParserAdapterFactory(facade);

        DocumentParser parser = factory.asDocumentParser(
                DocumentType.TEXT, "test.txt",
                ParserSelection.defaultNative(), null, null);

        String text = "standard parser adapter test";
        Document doc = parser.parse(
                new ByteArrayInputStream(text.getBytes(StandardCharsets.UTF_8)));

        assertNotNull(doc);
        assertTrue(doc.text().contains("standard parser adapter test"));
    }

    @Test
    void shouldInferExtensionFromFileName() {
        LangChain4jDocumentParserAdapterFactory factory =
                new LangChain4jDocumentParserAdapterFactory(facade);

        // 文件名无扩展名时默认 txt，仍能正常解析
        DocumentParser parser = factory.asDocumentParser(
                DocumentType.TEXT, "nofile",
                ParserSelection.defaultNative(), null, null);

        Document doc = parser.parse(
                new ByteArrayInputStream("no extension test".getBytes(StandardCharsets.UTF_8)));

        assertNotNull(doc);
        assertTrue(doc.text().contains("no extension test"));
    }
}
