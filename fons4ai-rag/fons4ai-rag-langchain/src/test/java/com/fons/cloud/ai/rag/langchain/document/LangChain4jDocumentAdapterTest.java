package com.fons.cloud.ai.rag.langchain.document;

import com.fons.cloud.ai.rag.common.document.ParsedDocument;
import dev.langchain4j.data.document.Document;
import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link LangChain4jDocumentAdapter} 单元测试。
 *
 * @author hongqy
 */
class LangChain4jDocumentAdapterTest {

    private final LangChain4jDocumentAdapter adapter = new LangChain4jDocumentAdapter();

    @Test
    void shouldConvertContentToDocument() {
        ParsedDocument parsed = new ParsedDocument("hello world", "MARKDOWN",
                Map.of(), List.of(), List.of());

        Document doc = adapter.toDocument(parsed);

        assertNotNull(doc);
        assertEquals("hello world", doc.text());
    }

    @Test
    void shouldFailWhenContentIsNull() {
        ParsedDocument parsed = new ParsedDocument(null, "MARKDOWN",
                Map.of(), List.of(), List.of());

        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                () -> adapter.toDocument(parsed));
        assertTrue(ex.getMessage().contains("解析内容不可为空"));
    }

    @Test
    void shouldFailWhenContentIsBlank() {
        ParsedDocument parsed = new ParsedDocument("   ", "MARKDOWN",
                Map.of(), List.of(), List.of());

        assertThrows(IllegalArgumentException.class, () -> adapter.toDocument(parsed));
    }

    @Test
    void shouldPreserveScalarMetadata() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("source", "mineru");
        metadata.put("page", 3);
        metadata.put("confidence", 0.95);
        ParsedDocument parsed = new ParsedDocument("c", "MARKDOWN",
                metadata, List.of(), List.of());

        Document doc = adapter.toDocument(parsed);

        Map<String, Object> result = doc.metadata().toMap();
        assertEquals("mineru", result.get("source"));
        assertEquals(3, result.get("page"));
        assertEquals(0.95, result.get("confidence"));
    }

    @Test
    void shouldConvertBooleanMetadataToString() {
        // LangChain4j Metadata 不支持 Boolean，应转为字符串
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("verified", true);
        ParsedDocument parsed = new ParsedDocument("c", "MARKDOWN",
                metadata, List.of(), List.of());

        Document doc = adapter.toDocument(parsed);

        Map<String, Object> result = doc.metadata().toMap();
        Object verified = result.get("verified");
        assertNotNull(verified);
        assertTrue(verified instanceof String, "Boolean 应转为字符串");
        assertEquals("true", verified);
    }

    @Test
    void shouldConvertNonScalarMetadataToString() {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("tags", List.of("a", "b"));
        ParsedDocument parsed = new ParsedDocument("c", "MARKDOWN",
                metadata, List.of(), List.of());

        Document doc = adapter.toDocument(parsed);

        Map<String, Object> result = doc.metadata().toMap();
        Object tagsValue = result.get("tags");
        assertNotNull(tagsValue);
        assertTrue(tagsValue instanceof String, "非标量值应转为字符串");
        assertEquals(String.valueOf(List.of("a", "b")), tagsValue);
    }

    @Test
    void shouldHandleEmptyMetadata() {
        ParsedDocument parsed = new ParsedDocument("c", "MARKDOWN",
                Map.of(), List.of(), List.of());

        Document doc = adapter.toDocument(parsed);

        assertNotNull(doc.metadata());
        assertTrue(doc.metadata().toMap().isEmpty());
    }
}
