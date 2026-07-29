package com.fons.cloud.ai.rag.common.document;

import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link DocumentParseRequest} Map 边界校验测试。
 *
 * @author hongqy
 */
class DocumentParseRequestTest {

    private static DocumentSource dummySource() {
        return DocumentSources.fromInputStream(
                new java.io.ByteArrayInputStream(new byte[]{1}), "test.txt", "text/plain", 1024);
    }

    @Test
    void shouldDefaultSelectionWhenNull() {
        DocumentParseRequest request = new DocumentParseRequest(
                dummySource(), com.fons.cloud.ai.rag.common.constants.DocumentType.TEXT,
                "txt", null, Map.of(), Map.of());
        assertEquals(ParserSelectionMode.DEFAULT, request.parserSelection().mode());
    }

    @Test
    void shouldNormalizeFileExtension() {
        DocumentParseRequest request = new DocumentParseRequest(
                dummySource(), com.fons.cloud.ai.rag.common.constants.DocumentType.TEXT,
                ".TXT", null, Map.of(), Map.of());
        assertEquals("txt", request.fileExtension());
    }

    @Test
    void shouldRejectNullSource() {
        assertThrows(NullPointerException.class, () -> new DocumentParseRequest(
                null, com.fons.cloud.ai.rag.common.constants.DocumentType.TEXT,
                "txt", null, Map.of(), Map.of()));
    }

    @Test
    void shouldRejectBlankExtension() {
        assertThrows(IllegalArgumentException.class, () -> new DocumentParseRequest(
                dummySource(), com.fons.cloud.ai.rag.common.constants.DocumentType.TEXT,
                "  ", null, Map.of(), Map.of()));
    }

    @Test
    void shouldRejectNullValueInMap() {
        Map<String, Object> options = new LinkedHashMap<>();
        options.put("key", null);
        assertThrows(IllegalArgumentException.class, () -> new DocumentParseRequest(
                dummySource(), com.fons.cloud.ai.rag.common.constants.DocumentType.TEXT,
                "txt", null, options, Map.of()));
    }

    @Test
    void shouldRejectDisallowedValueType() {
        Map<String, Object> options = Map.of("bad", new Object());
        assertThrows(IllegalArgumentException.class, () -> new DocumentParseRequest(
                dummySource(), com.fons.cloud.ai.rag.common.constants.DocumentType.TEXT,
                "txt", null, options, Map.of()));
    }

    @Test
    void shouldRejectMapOver64Entries() {
        Map<String, Object> options = new LinkedHashMap<>();
        for (int i = 0; i < 65; i++) {
            options.put("key" + i, i);
        }
        assertThrows(IllegalArgumentException.class, () -> new DocumentParseRequest(
                dummySource(), com.fons.cloud.ai.rag.common.constants.DocumentType.TEXT,
                "txt", null, options, Map.of()));
    }

    @Test
    void shouldAcceptAllowedValueTypes() {
        Map<String, Object> options = new LinkedHashMap<>();
        options.put("str", "value");
        options.put("num", 42);
        options.put("bool", true);
        options.put("list", List.of("a", "b"));
        assertDoesNotThrow(() -> new DocumentParseRequest(
                dummySource(), com.fons.cloud.ai.rag.common.constants.DocumentType.TEXT,
                "txt", null, options, Map.of()));
    }

    @Test
    void shouldReturnImmutableMap() {
        DocumentParseRequest request = new DocumentParseRequest(
                dummySource(), com.fons.cloud.ai.rag.common.constants.DocumentType.TEXT,
                "txt", null, Map.of("k", "v"), Map.of());
        assertThrows(UnsupportedOperationException.class, () -> request.options().put("new", "val"));
    }

    @Test
    void shouldRejectBlankKeyInMap() {
        Map<String, Object> options = new LinkedHashMap<>();
        options.put("  ", "value");
        assertThrows(IllegalArgumentException.class, () -> new DocumentParseRequest(
                dummySource(), com.fons.cloud.ai.rag.common.constants.DocumentType.TEXT,
                "txt", null, options, Map.of()));
    }

    @Test
    void shouldRejectListWithDisallowedItem() {
        Map<String, Object> options = Map.of("bad", List.of(new Object()));
        assertThrows(IllegalArgumentException.class, () -> new DocumentParseRequest(
                dummySource(), com.fons.cloud.ai.rag.common.constants.DocumentType.TEXT,
                "txt", null, options, Map.of()));
    }
}
