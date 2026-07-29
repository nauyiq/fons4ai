package com.fons.cloud.ai.rag.common.document;

import com.fons.cloud.ai.rag.common.constants.DocumentType;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link DocumentParserSelector} 选择、校验、执行和轨迹测试。
 * <p>
 * 覆盖 AC-002、AC-003、AC-004、AC-009。
 *
 * @author hongqy
 */
class DocumentParserSelectorTest {

    // ---- 测试辅助 ----

    /**
     * 创建测试用 fake provider，带解析调用计数器。
     */
    private static FakeProvider fakeProvider(String id, boolean available,
                                              Set<DocumentType> types, Set<String> exts,
                                              Set<ParserFeature> features) {
        return new FakeProvider(id, available, types, exts, features);
    }

    /**
     * 创建默认 PDF 解析请求。
     */
    private static DocumentParseRequest pdfRequest(ParserSelection selection) {
        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream(new byte[]{1}), "test.pdf", null, 1024);
        return new DocumentParseRequest(
                source, DocumentType.PDF, "pdf", selection, Map.of(), Map.of());
    }

    static class FakeProvider implements DocumentParseProvider<String> {
        final String id;
        final boolean available;
        final Set<DocumentType> types;
        final Set<String> exts;
        final Set<ParserFeature> features;
        final AtomicInteger parseCount = new AtomicInteger(0);
        String payload = "parsed";

        FakeProvider(String id, boolean available, Set<DocumentType> types, Set<String> exts, Set<ParserFeature> features) {
            this.id = id;
            this.available = available;
            this.types = types;
            this.exts = exts;
            this.features = features;
        }

        @Override
        public DocumentParserCapability capability() {
            return new DocumentParserCapability(id, types, exts, features, available, 0);
        }

        @Override
        public DocumentParseResult<String> parse(DocumentParseRequest request) {
            parseCount.incrementAndGet();
            return new DocumentParseResult<>(payload,
                    new ParseTrace(id, 0L, request.documentType().name(), "TEXT", "1.0", "test", "fake"));
        }
    }

    // ---- DEFAULT 选择测试 ----

    @Test
    void defaultShouldSelectNative() {
        FakeProvider nativeProvider = fakeProvider("native", true,
                Set.of(DocumentType.PDF), Set.of("pdf"), Set.of());
        DocumentParserRegistry<String> registry = new DocumentParserRegistry<>();
        registry.register(nativeProvider);
        DocumentParserSelector<String> selector = new DocumentParserSelector<>(registry);

        DocumentParseProvider<String> selected = selector.select(pdfRequest(ParserSelection.defaultNative()));
        assertSame(nativeProvider, selected);
    }

    @Test
    void defaultParseShouldReturnResultWithTrace() {
        FakeProvider nativeProvider = fakeProvider("native", true,
                Set.of(DocumentType.PDF), Set.of("pdf"), Set.of());
        DocumentParserRegistry<String> registry = new DocumentParserRegistry<>();
        registry.register(nativeProvider);
        DocumentParserSelector<String> selector = new DocumentParserSelector<>(registry);

        DocumentParseResult<String> result = selector.parse(pdfRequest(ParserSelection.defaultNative()));

        assertEquals("parsed", result.payload());
        assertNotNull(result.parseTrace());
        assertEquals("native", result.parseTrace().provider());
        assertEquals("PDF", result.parseTrace().sourceType());
        assertTrue(result.parseTrace().durationNanos() >= 0);
        assertTrue(result.parseTrace().selectionReason().contains("DEFAULT"));
    }

    // ---- EXPLICIT 选择测试 ----

    @Test
    void explicitShouldSelectSpecifiedProvider() {
        FakeProvider mineruProvider = fakeProvider("mineru", true,
                Set.of(DocumentType.PDF), Set.of("pdf"), Set.of());
        DocumentParserRegistry<String> registry = new DocumentParserRegistry<>();
        registry.register(mineruProvider);
        DocumentParserSelector<String> selector = new DocumentParserSelector<>(registry);

        DocumentParseProvider<String> selected = selector.select(
                pdfRequest(ParserSelection.explicit("mineru", Set.of())));
        assertSame(mineruProvider, selected);
    }

    @Test
    void explicitShouldBeCaseInsensitive() {
        FakeProvider mineruProvider = fakeProvider("mineru", true,
                Set.of(DocumentType.PDF), Set.of("pdf"), Set.of());
        DocumentParserRegistry<String> registry = new DocumentParserRegistry<>();
        registry.register(mineruProvider);
        DocumentParserSelector<String> selector = new DocumentParserSelector<>(registry);

        DocumentParseProvider<String> selected = selector.select(
                pdfRequest(ParserSelection.explicit("MINERU", Set.of())));
        assertSame(mineruProvider, selected);
    }

    // ---- 失败矩阵测试 ----

    @Test
    void shouldFailWhenProviderNotFound() {
        DocumentParserRegistry<String> registry = new DocumentParserRegistry<>();
        DocumentParserSelector<String> selector = new DocumentParserSelector<>(registry);

        DocumentParseException ex = assertThrows(DocumentParseException.class,
                () -> selector.select(pdfRequest(ParserSelection.explicit("mineru", Set.of()))));
        assertEquals(DocumentParseError.PROVIDER_NOT_FOUND, ex.getError());
        assertEquals("mineru", ex.getProvider());
    }

    @Test
    void shouldFailWhenProviderUnavailable() {
        FakeProvider mineruProvider = fakeProvider("mineru", false,
                Set.of(DocumentType.PDF), Set.of("pdf"), Set.of());
        DocumentParserRegistry<String> registry = new DocumentParserRegistry<>();
        registry.register(mineruProvider);
        DocumentParserSelector<String> selector = new DocumentParserSelector<>(registry);

        DocumentParseException ex = assertThrows(DocumentParseException.class,
                () -> selector.select(pdfRequest(ParserSelection.explicit("mineru", Set.of()))));
        assertEquals(DocumentParseError.PROVIDER_UNAVAILABLE, ex.getError());
    }

    @Test
    void shouldFailWhenDocumentTypeNotSupported() {
        FakeProvider mineruProvider = fakeProvider("mineru", true,
                Set.of(DocumentType.IMAGE), Set.of("png"), Set.of());
        DocumentParserRegistry<String> registry = new DocumentParserRegistry<>();
        registry.register(mineruProvider);
        DocumentParserSelector<String> selector = new DocumentParserSelector<>(registry);

        DocumentParseException ex = assertThrows(DocumentParseException.class,
                () -> selector.select(pdfRequest(ParserSelection.explicit("mineru", Set.of()))));
        assertEquals(DocumentParseError.UNSUPPORTED_DOCUMENT_TYPE, ex.getError());
    }

    @Test
    void shouldFailWhenExtensionNotSupported() {
        // provider 声明支持 PDF 类型但扩展名只有 png
        FakeProvider provider = fakeProvider("native", true,
                Set.of(DocumentType.PDF, DocumentType.IMAGE), Set.of("png"), Set.of());
        DocumentParserRegistry<String> registry = new DocumentParserRegistry<>();
        registry.register(provider);
        DocumentParserSelector<String> selector = new DocumentParserSelector<>(registry);

        DocumentParseException ex = assertThrows(DocumentParseException.class,
                () -> selector.select(pdfRequest(ParserSelection.defaultNative())));
        assertEquals(DocumentParseError.UNSUPPORTED_DOCUMENT_TYPE, ex.getError());
    }

    @Test
    void shouldFailWhenRequiredFeatureNotSupported() {
        FakeProvider nativeProvider = fakeProvider("native", true,
                Set.of(DocumentType.PDF), Set.of("pdf"), Set.of()); // 不支持任何特性
        DocumentParserRegistry<String> registry = new DocumentParserRegistry<>();
        registry.register(nativeProvider);
        DocumentParserSelector<String> selector = new DocumentParserSelector<>(registry);

        DocumentParseException ex = assertThrows(DocumentParseException.class,
                () -> selector.select(pdfRequest(
                        ParserSelection.explicit("native", Set.of(ParserFeature.OCR)))));
        assertEquals(DocumentParseError.REQUIRED_FEATURE_UNSUPPORTED, ex.getError());
    }

    // ---- 无 fallback 测试 ----

    @Test
    void nativeFailureShouldNotInvokeOtherProvider() {
        FakeProvider nativeProvider = fakeProvider("native", true,
                Set.of(DocumentType.PDF), Set.of("pdf"), Set.of());
        // 让 native 解析时抛异常
        FakeProvider failingNative = new FakeProvider("native", true,
                Set.of(DocumentType.PDF), Set.of("pdf"), Set.of()) {
            @Override
            public DocumentParseResult<String> parse(DocumentParseRequest request) {
                parseCount.incrementAndGet();
                throw new DocumentParseException(DocumentParseError.PROVIDER_FAILURE, "native",
                        "native 解析失败", null);
            }
        };
        FakeProvider mineruProvider = fakeProvider("mineru", true,
                Set.of(DocumentType.PDF), Set.of("pdf"), Set.of());

        DocumentParserRegistry<String> registry = new DocumentParserRegistry<>();
        registry.register(failingNative);
        registry.register(mineruProvider);
        DocumentParserSelector<String> selector = new DocumentParserSelector<>(registry);

        // DEFAULT 模式下 native 失败
        assertThrows(DocumentParseException.class,
                () -> selector.parse(pdfRequest(ParserSelection.defaultNative())));

        // native 被调用一次
        assertEquals(1, failingNative.parseCount.get());
        // mineru 零调用 -- 无 fallback
        assertEquals(0, mineruProvider.parseCount.get());
    }

    @Test
    void explicitMineruFailureShouldNotFallbackToNative() {
        FakeProvider nativeProvider = fakeProvider("native", true,
                Set.of(DocumentType.PDF), Set.of("pdf"), Set.of());
        FakeProvider failingMineru = new FakeProvider("mineru", true,
                Set.of(DocumentType.PDF), Set.of("pdf"), Set.of()) {
            @Override
            public DocumentParseResult<String> parse(DocumentParseRequest request) {
                parseCount.incrementAndGet();
                throw new DocumentParseException(DocumentParseError.PROVIDER_FAILURE, "mineru",
                        "mineru 解析失败", null);
            }
        };

        DocumentParserRegistry<String> registry = new DocumentParserRegistry<>();
        registry.register(nativeProvider);
        registry.register(failingMineru);
        DocumentParserSelector<String> selector = new DocumentParserSelector<>(registry);

        assertThrows(DocumentParseException.class,
                () -> selector.parse(pdfRequest(ParserSelection.explicit("mineru", Set.of()))));

        assertEquals(1, failingMineru.parseCount.get());
        assertEquals(0, nativeProvider.parseCount.get());
    }

    // ---- trace 合并与 map 测试 ----

    @Test
    void parseShouldMergeProviderTraceWithSelectionTrace() {
        FakeProvider nativeProvider = new FakeProvider("native", true,
                Set.of(DocumentType.PDF), Set.of("pdf"), Set.of()) {
            @Override
            public DocumentParseResult<String> parse(DocumentParseRequest request) {
                parseCount.incrementAndGet();
                // provider 写入自己的 version 和 backend
                return new DocumentParseResult<>("payload",
                        new ParseTrace("native", 100L, "PDF", "TEXT", "2.0", "pipeline", null));
            }
        };
        DocumentParserRegistry<String> registry = new DocumentParserRegistry<>();
        registry.register(nativeProvider);
        DocumentParserSelector<String> selector = new DocumentParserSelector<>(registry);

        DocumentParseResult<String> result = selector.parse(pdfRequest(ParserSelection.defaultNative()));

        // selector 覆盖 provider 和 durationNanos，但保留 provider 写入的 version 和 backend
        assertEquals("native", result.parseTrace().provider());
        assertEquals("2.0", result.parseTrace().providerVersion());
        assertEquals("pipeline", result.parseTrace().backend());
        assertEquals("PDF", result.parseTrace().sourceType());
        assertNotNull(result.parseTrace().selectionReason());
    }

    @Test
    void mapShouldPreserveTrace() {
        FakeProvider nativeProvider = fakeProvider("native", true,
                Set.of(DocumentType.PDF), Set.of("pdf"), Set.of());
        DocumentParserRegistry<String> registry = new DocumentParserRegistry<>();
        registry.register(nativeProvider);
        DocumentParserSelector<String> selector = new DocumentParserSelector<>(registry);

        DocumentParseResult<String> result = selector.parse(pdfRequest(ParserSelection.defaultNative()));
        ParseTrace originalTrace = result.parseTrace();

        DocumentParseResult<List<String>> mapped = result.map(s -> List.of(s, "extra"));

        assertEquals(List.of("parsed", "extra"), mapped.payload());
        // trace 引用保持不变
        assertSame(originalTrace, mapped.parseTrace());
    }

    // ---- 不同 R 类型隔离测试 ----

    @Test
    void registryShouldBeTypeIsolated() {
        // String 类型的 registry
        DocumentParserRegistry<String> strRegistry = new DocumentParserRegistry<>();
        strRegistry.register(fakeProvider("native", true,
                Set.of(DocumentType.PDF), Set.of("pdf"), Set.of()));

        // Integer 类型的 registry
        DocumentParserRegistry<Integer> intRegistry = new DocumentParserRegistry<>();
        intRegistry.register(new DocumentParseProvider<>() {
            @Override
            public DocumentParserCapability capability() {
                return new DocumentParserCapability("native",
                        Set.of(DocumentType.PDF), Set.of("pdf"), Set.of(), true, 0);
            }

            @Override
            public DocumentParseResult<Integer> parse(DocumentParseRequest request) {
                return new DocumentParseResult<>(42, null);
            }
        });

        DocumentParserSelector<String> strSelector = new DocumentParserSelector<>(strRegistry);
        DocumentParserSelector<Integer> intSelector = new DocumentParserSelector<>(intRegistry);

        DocumentParseResult<String> strResult = strSelector.parse(pdfRequest(ParserSelection.defaultNative()));
        DocumentParseResult<Integer> intResult = intSelector.parse(pdfRequest(ParserSelection.defaultNative()));

        assertEquals("parsed", strResult.payload());
        assertEquals(42, intResult.payload());
    }

    // ---- payload 引用保持测试 ----

    @Test
    void nativePayloadReferenceShouldBePreserved() {
        Object originalPayload = new Object();
        DocumentParseProvider<Object> provider = new DocumentParseProvider<>() {
            @Override
            public DocumentParserCapability capability() {
                return new DocumentParserCapability("native",
                        Set.of(DocumentType.PDF), Set.of("pdf"), Set.of(), true, 0);
            }

            @Override
            public DocumentParseResult<Object> parse(DocumentParseRequest request) {
                return new DocumentParseResult<>(originalPayload, null);
            }
        };

        DocumentParserRegistry<Object> registry = new DocumentParserRegistry<>();
        registry.register(provider);
        DocumentParserSelector<Object> selector = new DocumentParserSelector<>(registry);

        DocumentParseResult<Object> result = selector.parse(pdfRequest(ParserSelection.defaultNative()));
        assertSame(originalPayload, result.payload());
    }
}
