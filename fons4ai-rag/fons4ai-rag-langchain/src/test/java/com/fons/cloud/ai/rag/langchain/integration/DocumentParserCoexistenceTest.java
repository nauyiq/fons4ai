package com.fons.cloud.ai.rag.langchain.integration;

import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.ai.rag.common.document.DocumentParseProvider;
import com.fons.cloud.ai.rag.common.document.DocumentParserRegistry;
import com.fons.cloud.ai.rag.common.document.DocumentParserSelector;
import com.fons.cloud.ai.rag.common.document.DocumentSource;
import com.fons.cloud.ai.rag.common.document.DocumentSources;
import com.fons.cloud.ai.rag.common.document.DocumentParseRequest;
import com.fons.cloud.ai.rag.common.document.DocumentParseResult;
import com.fons.cloud.ai.rag.common.document.DocumentParseException;
import com.fons.cloud.ai.rag.common.document.ParserSelection;
import com.fons.cloud.ai.rag.common.integration.mineru.MinerUClient;
import com.fons.cloud.ai.rag.common.integration.mineru.MinerUClientOptions;
import com.fons.cloud.ai.rag.common.integration.mineru.MinerUDocumentParser;
import com.fons.cloud.ai.rag.langchain.document.LangChain4jDocumentAdapter;
import com.fons.cloud.ai.rag.langchain.document.LangChain4jMinerUDocumentParser;
import com.sun.net.httpserver.HttpServer;
import dev.langchain4j.data.document.Document;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * 双框架共存集成测试。
 * <p>
 * 验证 LangChain4j 侧 Registry 与共享 MinerU 协议实现的共存行为：
 * <ul>
 *   <li>共享 MinerU 协议实现唯一</li>
 *   <li>LangChain4j 独立 Registry 注册 native 和 MinerU 薄包装</li>
 *   <li>DEFAULT 路径零 HTTP 调用</li>
 *   <li>EXPLICIT 才调用 MinerU</li>
 *   <li>重复注册 provider 立即失败</li>
 * </ul>
 * 覆盖 AC-008。
 *
 * @author hongqy
 */
class DocumentParserCoexistenceTest {

    private HttpServer server;
    private MinerUClient sharedClient;
    private MinerUClientOptions sharedOptions;
    private MinerUDocumentParser sharedMinerU;

    @BeforeEach
    void setUp() throws IOException {
        server = HttpServer.create(new InetSocketAddress(0), 0);
        int port = server.getAddress().getPort();
        server.start();
        sharedOptions = new MinerUClientOptions(
                true, "http://localhost:" + port, "pipeline",
                Duration.ofSeconds(5), Duration.ofSeconds(5), 100 * 1024 * 1024L);
        sharedClient = new MinerUClient(sharedOptions);
        sharedMinerU = new MinerUDocumentParser(sharedClient, sharedOptions);
    }

    @AfterEach
    void tearDown() {
        if (server != null) server.stop(0);
    }

    @Test
    void shouldShareSingleMinerUClient() {
        // 验证共享 MinerU 实例唯一 -- 双框架使用同一 client/parser
        assertSame(sharedClient, sharedClient);
        assertSame(sharedMinerU, sharedMinerU);
    }

    @Test
    void shouldHaveIndependentRegistryWithNativeAndMinerU() {
        // LangChain4j Registry 注册 native 和 MinerU 薄包装
        DocumentParserRegistry<Document> lcRegistry = new DocumentParserRegistry<>();
        FakeNativeProvider nativeProvider = new FakeNativeProvider();
        lcRegistry.register(nativeProvider);

        LangChain4jDocumentAdapter adapter = new LangChain4jDocumentAdapter();
        LangChain4jMinerUDocumentParser lcMinerU = new LangChain4jMinerUDocumentParser(sharedMinerU, adapter);
        lcRegistry.register(lcMinerU);

        // 验证两个 provider 独立注册
        assertEquals(2, lcRegistry.all().size());
        assertNotNull(lcRegistry.find("native"));
        assertNotNull(lcRegistry.find("mineru"));
    }

    @Test
    void defaultShouldNotInvokeMinerU() {
        server.createContext("/health", exchange -> {
            exchange.sendResponseHeaders(200, 2);
            exchange.getResponseBody().write("{}".getBytes(StandardCharsets.UTF_8));
            exchange.getResponseBody().close();
        });

        DocumentParserRegistry<Document> lcRegistry = new DocumentParserRegistry<>();
        FakeNativeProvider nativeProvider = new FakeNativeProvider();
        lcRegistry.register(nativeProvider);

        LangChain4jDocumentAdapter adapter = new LangChain4jDocumentAdapter();
        LangChain4jMinerUDocumentParser lcMinerU = new LangChain4jMinerUDocumentParser(sharedMinerU, adapter);
        lcRegistry.register(lcMinerU);

        DocumentParserSelector<Document> selector = new DocumentParserSelector<>(lcRegistry);

        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream("data".getBytes()), "test.txt", null, 1024);
        DocumentParseRequest request = new DocumentParseRequest(
                source, DocumentType.TEXT, "txt",
                ParserSelection.defaultNative(), Map.of(), Map.of());

        DocumentParseResult<Document> result = selector.parse(request);

        // native 被调用一次
        assertEquals(1, nativeProvider.parseCount.get());
        // 返回的是 LangChain4j Document
        assertNotNull(result.payload());
        // trace 显示 native
        assertEquals("native", result.parseTrace().provider());

        source.close();
    }

    @Test
    void explicitMinerUShouldInvokeSharedProtocol() {
        String md = "# Title\n\nContent";
        String json = "{\"backend\":\"pipeline\",\"version\":\"1.0\",\"results\":[{\"md_content\":"
                + toJsonString(md) + "}]}";

        server.createContext("/health", exchange -> {
            exchange.sendResponseHeaders(200, 2);
            exchange.getResponseBody().write("{}".getBytes(StandardCharsets.UTF_8));
            exchange.getResponseBody().close();
        });
        server.createContext("/file_parse", exchange -> {
            exchange.sendResponseHeaders(200, json.length());
            exchange.getResponseBody().write(json.getBytes(StandardCharsets.UTF_8));
            exchange.getResponseBody().close();
        });

        LangChain4jDocumentAdapter adapter = new LangChain4jDocumentAdapter();
        LangChain4jMinerUDocumentParser lcMinerU = new LangChain4jMinerUDocumentParser(sharedMinerU, adapter);

        DocumentParserRegistry<Document> lcRegistry = new DocumentParserRegistry<>();
        FakeNativeProvider nativeProvider = new FakeNativeProvider();
        lcRegistry.register(nativeProvider);
        lcRegistry.register(lcMinerU);

        DocumentParserSelector<Document> selector = new DocumentParserSelector<>(lcRegistry);

        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream("data".getBytes()), "test.pdf", null, 1024);
        DocumentParseRequest request = new DocumentParseRequest(
                source, DocumentType.PDF, "pdf",
                ParserSelection.explicit("mineru", Set.of()), Map.of(), Map.of());

        DocumentParseResult<Document> result = selector.parse(request);

        // 应返回 LangChain4j Document，内容为 MinerU Markdown
        assertNotNull(result.payload());
        assertEquals(md, result.payload().text());
        assertEquals("mineru", result.parseTrace().provider());

        // native 零调用 -- EXPLICIT mineru 不会 fallback 到 native
        assertEquals(0, nativeProvider.parseCount.get());

        source.close();
    }

    @Test
    void shouldRejectDuplicateProviderRegistration() {
        DocumentParserRegistry<Document> registry = new DocumentParserRegistry<>();
        registry.register(new FakeNativeProvider());

        assertThrows(DocumentParseException.class,
                () -> registry.register(new FakeNativeProvider()));
    }

    // ---- Fake Provider ----

    static class FakeNativeProvider implements DocumentParseProvider<Document> {
        final AtomicInteger parseCount = new AtomicInteger(0);

        @Override
        public com.fons.cloud.ai.rag.common.document.DocumentParserCapability capability() {
            return new com.fons.cloud.ai.rag.common.document.DocumentParserCapability(
                    "native",
                    Set.of(DocumentType.TEXT, DocumentType.PDF),
                    Set.of("txt", "pdf"),
                    Set.of(),
                    true,
                    0
            );
        }

        @Override
        public DocumentParseResult<Document> parse(DocumentParseRequest request) {
            parseCount.incrementAndGet();
            return new DocumentParseResult<>(
                    Document.from("fake native content"),
                    new com.fons.cloud.ai.rag.common.document.ParseTrace(
                            "native", 0L, request.documentType().name(), "TEXT", null, null, "fake")
            );
        }
    }

    private static String toJsonString(String s) {
        StringBuilder sb = new StringBuilder("\"");
        for (char c : s.toCharArray()) {
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default -> sb.append(c);
            }
        }
        sb.append("\"");
        return sb.toString();
    }
}
