package com.fons.cloud.ai.rag.common.document;

import com.fons.cloud.ai.rag.common.constants.DocumentType;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link DocumentParserRegistry} 注册和查找测试。
 *
 * @author hongqy
 */
class DocumentParserRegistryTest {

    /**
     * 创建测试用 fake provider。
     */
    private static DocumentParseProvider<String> fakeProvider(String id, boolean available) {
        return new DocumentParseProvider<>() {
            @Override
            public DocumentParserCapability capability() {
                return new DocumentParserCapability(
                        id,
                        Set.of(DocumentType.PDF),
                        Set.of("pdf"),
                        Set.of(),
                        available,
                        0
                );
            }

            @Override
            public DocumentParseResult<String> parse(DocumentParseRequest request) {
                return new DocumentParseResult<>("result", null);
            }
        };
    }

    @Test
    void shouldRegisterAndFindProvider() {
        DocumentParserRegistry<String> registry = new DocumentParserRegistry<>();
        registry.register(fakeProvider("native", true));

        assertNotNull(registry.find("native"));
        assertNotNull(registry.find("NATIVE"));
        assertNotNull(registry.find(" Native "));
    }

    @Test
    void shouldRejectDuplicateProvider() {
        DocumentParserRegistry<String> registry = new DocumentParserRegistry<>();
        registry.register(fakeProvider("native", true));

        DocumentParseException ex = assertThrows(DocumentParseException.class,
                () -> registry.register(fakeProvider("native", true)));
        assertEquals(DocumentParseError.DUPLICATE_PROVIDER, ex.getError());
    }

    @Test
    void shouldReturnNullForUnknownProvider() {
        DocumentParserRegistry<String> registry = new DocumentParserRegistry<>();
        assertNull(registry.find("unknown"));
        assertNull(registry.find(null));
        assertNull(registry.find("  "));
    }

    @Test
    void allShouldReturnImmutableList() {
        DocumentParserRegistry<String> registry = new DocumentParserRegistry<>();
        registry.register(fakeProvider("native", true));
        registry.register(fakeProvider("mineru", true));

        assertEquals(2, registry.all().size());
        assertThrows(UnsupportedOperationException.class, () -> registry.all().add(fakeProvider("x", true)));
    }

    @Test
    void isEmptyShouldReflectRegistryState() {
        DocumentParserRegistry<String> registry = new DocumentParserRegistry<>();
        assertTrue(registry.isEmpty());
        registry.register(fakeProvider("native", true));
        assertTrue(!registry.isEmpty());
    }
}
