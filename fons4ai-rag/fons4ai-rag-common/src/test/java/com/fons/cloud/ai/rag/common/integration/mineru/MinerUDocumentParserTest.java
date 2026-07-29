package com.fons.cloud.ai.rag.common.integration.mineru;

import com.fons.cloud.ai.rag.common.constants.DocumentType;
import com.fons.cloud.ai.rag.common.document.DocumentParseError;
import com.fons.cloud.ai.rag.common.document.DocumentParseException;
import com.fons.cloud.ai.rag.common.document.DocumentParseRequest;
import com.fons.cloud.ai.rag.common.document.DocumentParseResult;
import com.fons.cloud.ai.rag.common.document.DocumentParserCapability;
import com.fons.cloud.ai.rag.common.document.DocumentSource;
import com.fons.cloud.ai.rag.common.document.DocumentSources;
import com.fons.cloud.ai.rag.common.document.ParserFeature;
import com.fons.cloud.ai.rag.common.document.ParserSelection;
import com.fons.cloud.ai.rag.common.document.ParsedDocument;
import com.sun.net.httpserver.HttpServer;
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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link MinerUDocumentParser} provider 测试。
 * <p>
 * 覆盖 capability 声明、旧 Office 格式拒绝、解析成功 trace、开关/健康检查失败和 Markdown 保持。
 *
 * @author hongqy
 */
class MinerUDocumentParserTest {

    private HttpServer server;
    private MinerUClientOptions options;
    private MinerUClient client;

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
        if (server != null) {
            server.stop(0);
        }
    }

    // ---- Capability ----

    @Test
    void capabilityShouldDeclareOfficialFormatsOnly() {
        // 健康检查需要 Mock
        server.createContext("/health", exchange -> {
            exchange.sendResponseHeaders(200, 2);
            exchange.getResponseBody().write("{}".getBytes(StandardCharsets.UTF_8));
            exchange.getResponseBody().close();
        });

        MinerUDocumentParser parser = new MinerUDocumentParser(client, options);
        DocumentParserCapability cap = parser.capability();

        assertEquals("mineru", cap.provider());
        assertTrue(cap.available());

        // 官方支持的扩展名
        Set<String> exts = cap.supportedFileExtensions();
        assertTrue(exts.contains("pdf"));
        assertTrue(exts.contains("png"));
        assertTrue(exts.contains("jpg"));
        assertTrue(exts.contains("jpeg"));
        assertTrue(exts.contains("docx"));
        assertTrue(exts.contains("pptx"));
        assertTrue(exts.contains("xlsx"));

        // 旧 Office 格式不在 MinerU capability 中
        assertFalse(exts.contains("doc"));
        assertFalse(exts.contains("ppt"));
        assertFalse(exts.contains("xls"));
    }

    @Test
    void capabilityShouldBeUnavailableWhenDisabled() {
        MinerUClientOptions disabled = MinerUClientOptions.disabled();
        MinerUClient disabledClient = new MinerUClient(disabled);
        MinerUDocumentParser parser = new MinerUDocumentParser(disabledClient, disabled);

        DocumentParserCapability cap = parser.capability();
        assertFalse(cap.available());
    }

    @Test
    void capabilityShouldBeAvailableWhenEnabledRegardlessOfHealth() {
        // capability 只检查开关，不触发 HTTP 健康检查
        // 即使服务不可达，capability.available 仍为 true（开关开启时）
        server.stop(0);
        server = null;

        MinerUDocumentParser parser = new MinerUDocumentParser(client, options);
        DocumentParserCapability cap = parser.capability();
        assertTrue(cap.available());
    }

    @Test
    void capabilityShouldSupportAllFeatures() {
        server.createContext("/health", exchange -> {
            exchange.sendResponseHeaders(200, 2);
            exchange.getResponseBody().write("{}".getBytes(StandardCharsets.UTF_8));
            exchange.getResponseBody().close();
        });

        MinerUDocumentParser parser = new MinerUDocumentParser(client, options);
        DocumentParserCapability cap = parser.capability();

        Set<ParserFeature> features = cap.features();
        assertTrue(features.contains(ParserFeature.OCR));
        assertTrue(features.contains(ParserFeature.TABLE));
        assertTrue(features.contains(ParserFeature.FORMULA));
        assertTrue(features.contains(ParserFeature.LAYOUT));
    }

    // ---- 解析成功 ----

    @Test
    void parseShouldReturnParsedDocumentWithMarkdown() {
        String md = "# Title\n\nContent with  multiple  spaces\n\n| A | B |\n|---|---|\n| 1 | 2 |\n";
        String json = "{\"backend\":\"pipeline\",\"version\":\"1.2.0\",\"results\":[{\"md_content\":"
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

        MinerUDocumentParser parser = new MinerUDocumentParser(client, options);
        DocumentParseRequest request = buildRequest(DocumentType.PDF, "pdf");

        DocumentParseResult<ParsedDocument> result = parser.parse(request);

        ParsedDocument doc = result.payload();
        assertEquals(md, doc.content());
        assertEquals("MARKDOWN", doc.contentFormat());
        assertTrue(doc.blocks().isEmpty());
        assertTrue(doc.assets().isEmpty());

        // trace 应包含 version 和 backend
        assertNotNull(result.parseTrace());
        assertEquals("mineru", result.parseTrace().provider());
        assertEquals("1.2.0", result.parseTrace().providerVersion());
        assertEquals("pipeline", result.parseTrace().backend());
        assertEquals("MARKDOWN", result.parseTrace().outputFormat());
    }

    @Test
    void parseShouldPreserveMarkdownWhitespace() {
        // 包含连续空格、制表符和多个换行 -- 不应被压缩
        String md = "# Title\n\n\n\nText   with   spaces\n\t\tTabbed\n\n- List\n\n$$formula$$\n";
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

        MinerUDocumentParser parser = new MinerUDocumentParser(client, options);
        DocumentParseRequest request = buildRequest(DocumentType.PDF, "pdf");

        DocumentParseResult<ParsedDocument> result = parser.parse(request);

        // 字符级保持，不经过空白压缩
        assertEquals(md, result.payload().content());
    }

    // ---- 解析失败 ----

    @Test
    void shouldFailWhenDisabled() {
        MinerUClientOptions disabled = MinerUClientOptions.disabled();
        MinerUClient disabledClient = new MinerUClient(disabled);
        MinerUDocumentParser parser = new MinerUDocumentParser(disabledClient, disabled);

        DocumentParseException ex = assertThrows(DocumentParseException.class,
                () -> parser.parse(buildRequest(DocumentType.PDF, "pdf")));
        assertEquals(DocumentParseError.PROVIDER_UNAVAILABLE, ex.getError());
    }

    @Test
    void shouldFailWhenHealthCheckFails() {
        server.stop(0);
        server = null;

        MinerUDocumentParser parser = new MinerUDocumentParser(client, options);

        DocumentParseException ex = assertThrows(DocumentParseException.class,
                () -> parser.parse(buildRequest(DocumentType.PDF, "pdf")));
        assertEquals(DocumentParseError.PROVIDER_UNAVAILABLE, ex.getError());
    }

    @Test
    void shouldFailWhenFileTooLarge() {
        server.createContext("/health", exchange -> {
            exchange.sendResponseHeaders(200, 2);
            exchange.getResponseBody().write("{}".getBytes(StandardCharsets.UTF_8));
            exchange.getResponseBody().close();
        });

        MinerUClientOptions smallLimit = new MinerUClientOptions(
                true, "http://localhost:" + server.getAddress().getPort(), "pipeline",
                Duration.ofSeconds(5), Duration.ofSeconds(5), 3L);
        MinerUClient smallClient = new MinerUClient(smallLimit);
        MinerUDocumentParser parser = new MinerUDocumentParser(smallClient, smallLimit);

        DocumentParseException ex = assertThrows(DocumentParseException.class,
                () -> parser.parse(buildRequest(DocumentType.PDF, "pdf")));
        assertEquals(DocumentParseError.FILE_TOO_LARGE, ex.getError());
    }

    @Test
    void shouldFailWithHttpError() {
        server.createContext("/health", exchange -> {
            exchange.sendResponseHeaders(200, 2);
            exchange.getResponseBody().write("{}".getBytes(StandardCharsets.UTF_8));
            exchange.getResponseBody().close();
        });
        server.createContext("/file_parse", exchange -> {
            exchange.sendResponseHeaders(500, 0);
            exchange.getResponseBody().close();
        });

        MinerUDocumentParser parser = new MinerUDocumentParser(client, options);

        DocumentParseException ex = assertThrows(DocumentParseException.class,
                () -> parser.parse(buildRequest(DocumentType.PDF, "pdf")));
        assertEquals(DocumentParseError.HTTP_ERROR, ex.getError());
    }

    // ---- 辅助方法 ----

    private DocumentParseRequest buildRequest(DocumentType type, String ext) {
        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream("data".getBytes()), "test." + ext, null, 1024);
        return new DocumentParseRequest(
                source, type, ext, ParserSelection.explicit("mineru", Set.of()),
                Map.of(), Map.of());
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
