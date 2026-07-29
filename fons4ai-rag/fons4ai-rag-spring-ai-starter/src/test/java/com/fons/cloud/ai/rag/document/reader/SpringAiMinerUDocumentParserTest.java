package com.fons.cloud.ai.rag.document.reader;

import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.ai.rag.common.document.DocumentParseRequest;
import com.fons.cloud.ai.rag.common.document.DocumentParseResult;
import com.fons.cloud.ai.rag.common.document.DocumentSource;
import com.fons.cloud.ai.rag.common.document.DocumentSources;
import com.fons.cloud.ai.rag.common.document.ParserSelection;
import com.fons.cloud.ai.rag.common.document.ParsedDocument;
import com.fons.cloud.ai.rag.common.document.ParseTrace;
import com.fons.cloud.ai.rag.common.integration.mineru.MinerUDocumentParser;
import com.fons.cloud.ai.rag.common.integration.mineru.MinerUClient;
import com.fons.cloud.ai.rag.common.integration.mineru.MinerUClientOptions;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link SpringAiMinerUDocumentParser} 测试。
 * <p>
 * 验证委托共享 MinerU provider 并通过 map 完成唯一一次类型转换。
 *
 * @author hongqy
 */
class SpringAiMinerUDocumentParserTest {

    private HttpServer server;
    private MinerUClient client;
    private MinerUClientOptions options;

    @BeforeEach
    void setUp() throws IOException {
        server = HttpServer.create(new InetSocketAddress(0), 0);
        int port = server.getAddress().getPort();
        server.start();
        options = new MinerUClientOptions(
                true, "http://localhost:" + port, "pipeline",
                Duration.ofSeconds(5), Duration.ofSeconds(5), 100 * 1024 * 1024L);
        client = new MinerUClient(options);
    }

    @AfterEach
    void tearDown() {
        if (server != null) server.stop(0);
    }

    @Test
    void shouldDelegateToMinerUAndAdaptResult() {
        String md = "# Title\n\n- item1\n\n| A | B |\n|---|---|\n| 1 | 2 |\n";
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

        MinerUDocumentParser delegate = new MinerUDocumentParser(client, options);
        SpringAiDocumentAdapter adapter = new SpringAiDocumentAdapter();
        SpringAiMinerUDocumentParser parser = new SpringAiMinerUDocumentParser(delegate, adapter);

        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream("data".getBytes()), "test.pdf", null, 1024);
        DocumentParseRequest request = new DocumentParseRequest(
                source, DocumentType.PDF, "pdf",
                ParserSelection.explicit("mineru", Set.of()), Map.of(), Map.of());

        DocumentParseResult<List<Document>> result = parser.parse(request);

        // 应返回 Spring AI Document 列表
        assertNotNull(result.payload());
        assertEquals(1, result.payload().size());
        // Markdown 内容字符级保持
        assertEquals(md, result.payload().get(0).getText());
        // trace 保留
        assertNotNull(result.parseTrace());
        assertEquals("mineru", result.parseTrace().provider());

        source.close();
    }

    @Test
    void capabilityShouldDelegateToMinerU() {
        server.createContext("/health", exchange -> {
            exchange.sendResponseHeaders(200, 2);
            exchange.getResponseBody().write("{}".getBytes(StandardCharsets.UTF_8));
            exchange.getResponseBody().close();
        });

        MinerUDocumentParser delegate = new MinerUDocumentParser(client, options);
        SpringAiDocumentAdapter adapter = new SpringAiDocumentAdapter();
        SpringAiMinerUDocumentParser parser = new SpringAiMinerUDocumentParser(delegate, adapter);

        assertEquals("mineru", parser.capability().provider());
        assertTrue(parser.capability().available());
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
