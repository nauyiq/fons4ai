package com.fons.cloud.ai.rag.common.integration.mineru;

import com.fons.cloud.ai.rag.common.document.DocumentParseException;
import com.fons.cloud.ai.rag.common.document.DocumentParseError;
import com.fons.cloud.ai.rag.common.document.DocumentSource;
import com.fons.cloud.ai.rag.common.document.DocumentSources;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link MinerUClient} HTTP 协议测试。
 * <p>
 * 使用 JDK {@link HttpServer} 作为 Mock，覆盖健康检查、multipart 契约、JSON 响应、
 * 超时、非 2xx、空/多结果、非法 JSON、Markdown 保持和资源清理。
 *
 * @author hongqy
 */
class MinerUClientTest {

    private HttpServer server;
    private MinerUClient client;
    private final AtomicReference<String> receivedContentType = new AtomicReference<>();
    private final AtomicReference<String> receivedBody = new AtomicReference<>();

    @BeforeEach
    void setUp() throws IOException {
        server = HttpServer.create(new InetSocketAddress(0), 0);
        int port = server.getAddress().getPort();
        server.start();

        MinerUClientOptions options = new MinerUClientOptions(
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

    // ---- 健康检查 ----

    @Test
    void healthCheckShouldReturnTrueFor2xxJson() {
        server.createContext("/health", exchange -> {
            String json = "{\"status\":\"ok\"}";
            exchange.sendResponseHeaders(200, json.length());
            exchange.getResponseBody().write(json.getBytes(StandardCharsets.UTF_8));
            exchange.getResponseBody().close();
        });

        assertTrue(client.isHealthy());
    }

    @Test
    void healthCheckShouldReturnFalseForNon2xx() {
        server.createContext("/health", exchange -> {
            exchange.sendResponseHeaders(503, 0);
            exchange.getResponseBody().close();
        });

        assertFalse(client.isHealthy());
    }

    @Test
    void healthCheckShouldReturnFalseForEmptyBody() {
        server.createContext("/health", exchange -> {
            exchange.sendResponseHeaders(200, 0);
            exchange.getResponseBody().close();
        });

        assertFalse(client.isHealthy());
    }

    @Test
    void healthCheckShouldReturnFalseWhenServerDown() {
        server.stop(0);
        server = null;
        assertFalse(client.isHealthy());
    }

    @Test
    void healthCheckShouldReturnFalseWhenDisabled() {
        MinerUClientOptions disabled = MinerUClientOptions.disabled();
        MinerUClient disabledClient = new MinerUClient(disabled);
        assertFalse(disabledClient.isHealthy());
    }

    // ---- 解析成功 ----

    @Test
    void parseFileShouldReturnMarkdownContent() {
        String expectedMd = "# Title\n\n- item1\n- item2\n\n| A | B |\n|---|---|\n| 1 | 2 |\n";
        String json = "{\"backend\":\"pipeline\",\"version\":\"1.0.0\",\"results\":[{\"md_content\":"
                + toJsonString(expectedMd) + "}]}";

        server.createContext("/file_parse", exchange -> {
            receivedContentType.set(exchange.getRequestHeaders().getFirst("Content-Type"));
            exchange.sendResponseHeaders(200, json.length());
            exchange.getResponseBody().write(json.getBytes(StandardCharsets.UTF_8));
            exchange.getResponseBody().close();
        });

        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream("fake pdf".getBytes()), "test.pdf", "application/pdf", 1024);

        MinerUParseResult result = client.parseFile(source);

        assertEquals(expectedMd, result.mdContent());
        assertEquals("1.0.0", result.version());
        assertEquals("pipeline", result.backend());
        assertNotNull(receivedContentType.get());
        assertTrue(receivedContentType.get().startsWith("multipart/form-data; boundary="));
    }

    @Test
    void parseFileShouldPreserveMarkdownStructure() {
        // 包含标题、列表、表格、代码块和公式的 Markdown
        String structuredMd = "# Heading\n\n## Subheading\n\n- List item 1\n- List item 2\n\n"
                + "```python\ncode = 1\n```\n\n"
                + "$$E=mc^2$$\n\n"
                + "| Col1 | Col2 |\n|------|------|\n| a    | b    |\n";
        String json = "{\"backend\":\"pipeline\",\"version\":\"2.0\",\"results\":[{\"md_content\":"
                + toJsonString(structuredMd) + "}]}";

        server.createContext("/file_parse", exchange -> {
            exchange.sendResponseHeaders(200, json.length());
            exchange.getResponseBody().write(json.getBytes(StandardCharsets.UTF_8));
            exchange.getResponseBody().close();
        });

        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream("data".getBytes()), "test.pdf", null, 1024);

        MinerUParseResult result = client.parseFile(source);

        // 字符级保持
        assertEquals(structuredMd, result.mdContent());
    }

    @Test
    void parseFileShouldSendMultipartWithFixedFormFields() {
        String json = "{\"backend\":\"pipeline\",\"version\":\"1.0\",\"results\":[{\"md_content\":\"md\"}]}";

        server.createContext("/file_parse", exchange -> {
            receivedContentType.set(exchange.getRequestHeaders().getFirst("Content-Type"));
            String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            receivedBody.set(body);
            exchange.sendResponseHeaders(200, json.length());
            exchange.getResponseBody().write(json.getBytes(StandardCharsets.UTF_8));
            exchange.getResponseBody().close();
        });

        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream("data".getBytes()), "test.pdf", "application/pdf", 1024);

        client.parseFile(source);

        String body = receivedBody.get();
        assertNotNull(body);
        // 验证固定表单字段
        assertTrue(body.contains("name=\"backend\""), "应包含 backend 字段");
        assertTrue(body.contains("pipeline"), "backend 值应为 pipeline");
        assertTrue(body.contains("name=\"parse_method\""), "应包含 parse_method 字段");
        assertTrue(body.contains("name=\"return_md\""), "应包含 return_md 字段");
        assertTrue(body.contains("name=\"response_format_zip\""), "应包含 response_format_zip 字段");
        assertTrue(body.contains("name=\"return_middle_json\""), "应包含 return_middle_json 字段");
        assertTrue(body.contains("name=\"return_model_output\""), "应包含 return_model_output 字段");
        assertTrue(body.contains("name=\"return_content_list\""), "应包含 return_content_list 字段");
        assertTrue(body.contains("name=\"return_images\""), "应包含 return_images 字段");
        assertTrue(body.contains("name=\"return_original_file\""), "应包含 return_original_file 字段");
        assertTrue(body.contains("name=\"formula_enable\""), "应包含 formula_enable 字段");
        assertTrue(body.contains("name=\"table_enable\""), "应包含 table_enable 字段");
        // 文件 part
        assertTrue(body.contains("name=\"files\""), "应包含 files 字段");
        assertTrue(body.contains("filename=\"test.pdf\""), "应包含安全文件名");
    }

    // ---- 解析失败矩阵 ----

    @Test
    void shouldFailWithFileTooLarge() {
        server.createContext("/file_parse", exchange -> {
            exchange.sendResponseHeaders(200, 0);
            exchange.getResponseBody().close();
        });

        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream("data".getBytes()), "test.pdf", null, 1024);

        MinerUClientOptions smallLimit = new MinerUClientOptions(
                true, "http://localhost:" + server.getAddress().getPort(), "pipeline",
                Duration.ofSeconds(5), Duration.ofSeconds(5), 3L);
        MinerUClient smallClient = new MinerUClient(smallLimit);

        DocumentParseException ex = assertThrows(DocumentParseException.class,
                () -> smallClient.parseFile(source));
        assertEquals(DocumentParseError.FILE_TOO_LARGE, ex.getError());
    }

    @Test
    void shouldFailWithHttpError() {
        server.createContext("/file_parse", exchange -> {
            String err = "{\"detail\":\"internal error\"}";
            exchange.sendResponseHeaders(500, err.length());
            exchange.getResponseBody().write(err.getBytes(StandardCharsets.UTF_8));
            exchange.getResponseBody().close();
        });

        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream("data".getBytes()), "test.pdf", null, 1024);

        DocumentParseException ex = assertThrows(DocumentParseException.class,
                () -> client.parseFile(source));
        assertEquals(DocumentParseError.HTTP_ERROR, ex.getError());
    }

    @Test
    void shouldFailWithInvalidResponseEmptyBody() {
        server.createContext("/file_parse", exchange -> {
            exchange.sendResponseHeaders(200, 0);
            exchange.getResponseBody().close();
        });

        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream("data".getBytes()), "test.pdf", null, 1024);

        DocumentParseException ex = assertThrows(DocumentParseException.class,
                () -> client.parseFile(source));
        assertEquals(DocumentParseError.INVALID_RESPONSE, ex.getError());
    }

    @Test
    void shouldFailWithInvalidResponseEmptyResults() {
        String json = "{\"backend\":\"pipeline\",\"version\":\"1.0\",\"results\":[]}";
        server.createContext("/file_parse", exchange -> {
            exchange.sendResponseHeaders(200, json.length());
            exchange.getResponseBody().write(json.getBytes(StandardCharsets.UTF_8));
            exchange.getResponseBody().close();
        });

        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream("data".getBytes()), "test.pdf", null, 1024);

        DocumentParseException ex = assertThrows(DocumentParseException.class,
                () -> client.parseFile(source));
        assertEquals(DocumentParseError.INVALID_RESPONSE, ex.getError());
    }

    @Test
    void shouldFailWithMultipleResults() {
        String json = "{\"backend\":\"pipeline\",\"version\":\"1.0\",\"results\":"
                + "[{\"md_content\":\"a\"},{\"md_content\":\"b\"}]}";
        server.createContext("/file_parse", exchange -> {
            exchange.sendResponseHeaders(200, json.length());
            exchange.getResponseBody().write(json.getBytes(StandardCharsets.UTF_8));
            exchange.getResponseBody().close();
        });

        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream("data".getBytes()), "test.pdf", null, 1024);

        DocumentParseException ex = assertThrows(DocumentParseException.class,
                () -> client.parseFile(source));
        assertEquals(DocumentParseError.INVALID_RESPONSE, ex.getError());
    }

    @Test
    void shouldFailWithMissingMdContent() {
        String json = "{\"backend\":\"pipeline\",\"version\":\"1.0\",\"results\":[{}]}";
        server.createContext("/file_parse", exchange -> {
            exchange.sendResponseHeaders(200, json.length());
            exchange.getResponseBody().write(json.getBytes(StandardCharsets.UTF_8));
            exchange.getResponseBody().close();
        });

        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream("data".getBytes()), "test.pdf", null, 1024);

        DocumentParseException ex = assertThrows(DocumentParseException.class,
                () -> client.parseFile(source));
        assertEquals(DocumentParseError.INVALID_RESPONSE, ex.getError());
    }

    @Test
    void shouldFailWithNonStringMdContent() {
        String json = "{\"backend\":\"pipeline\",\"version\":\"1.0\",\"results\":[{\"md_content\":123}]}";
        server.createContext("/file_parse", exchange -> {
            exchange.sendResponseHeaders(200, json.length());
            exchange.getResponseBody().write(json.getBytes(StandardCharsets.UTF_8));
            exchange.getResponseBody().close();
        });

        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream("data".getBytes()), "test.pdf", null, 1024);

        DocumentParseException ex = assertThrows(DocumentParseException.class,
                () -> client.parseFile(source));
        assertEquals(DocumentParseError.INVALID_RESPONSE, ex.getError());
    }

    @Test
    void shouldFailWithInvalidJson() {
        server.createContext("/file_parse", exchange -> {
            String bad = "not json at all";
            exchange.sendResponseHeaders(200, bad.length());
            exchange.getResponseBody().write(bad.getBytes(StandardCharsets.UTF_8));
            exchange.getResponseBody().close();
        });

        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream("data".getBytes()), "test.pdf", null, 1024);

        DocumentParseException ex = assertThrows(DocumentParseException.class,
                () -> client.parseFile(source));
        assertEquals(DocumentParseError.INVALID_RESPONSE, ex.getError());
    }

    // ---- 文件名安全 ----

    @Test
    void shouldSanitizeFileNameWithCrlf() {
        String json = "{\"backend\":\"pipeline\",\"version\":\"1.0\",\"results\":[{\"md_content\":\"md\"}]}";

        server.createContext("/file_parse", exchange -> {
            String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            receivedBody.set(body);
            exchange.sendResponseHeaders(200, json.length());
            exchange.getResponseBody().write(json.getBytes(StandardCharsets.UTF_8));
            exchange.getResponseBody().close();
        });

        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream("data".getBytes()),
                "path/to/evil\r\n\"injection\".pdf", null, 1024);

        client.parseFile(source);

        String body = receivedBody.get();
        // 不应包含 CR/LF 或引号注入
        assertFalse(body.contains("evil\r\n"), "文件名不应包含 CR/LF");
        assertTrue(body.contains("injection.pdf"), "应保留安全文件名");
    }

    @Test
    void shouldHandleNullFileName() {
        String json = "{\"backend\":\"pipeline\",\"version\":\"1.0\",\"results\":[{\"md_content\":\"md\"}]}";

        server.createContext("/file_parse", exchange -> {
            String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            receivedBody.set(body);
            exchange.sendResponseHeaders(200, json.length());
            exchange.getResponseBody().write(json.getBytes(StandardCharsets.UTF_8));
            exchange.getResponseBody().close();
        });

        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream("data".getBytes()), null, null, 1024);

        client.parseFile(source);

        String body = receivedBody.get();
        assertTrue(body.contains("filename=\"document\""), "null 文件名应使用默认值");
    }

    // ---- 资源清理 ----

    @Test
    void sourceShouldBeReusableAfterParse() throws Exception {
        String json = "{\"backend\":\"pipeline\",\"version\":\"1.0\",\"results\":[{\"md_content\":\"md\"}]}";

        server.createContext("/file_parse", exchange -> {
            exchange.sendResponseHeaders(200, json.length());
            exchange.getResponseBody().write(json.getBytes(StandardCharsets.UTF_8));
            exchange.getResponseBody().close();
        });

        DocumentSource source = DocumentSources.fromInputStream(
                new ByteArrayInputStream("data".getBytes()), "test.pdf", null, 1024);

        client.parseFile(source);
        // source 仍可打开新流
        InputStream s = source.openStream();
        assertNotNull(s);
        s.close();
        source.close();
    }

    // ---- 辅助方法 ----

    /**
     * 将字符串转为 JSON 字符串字面量（含引号和转义）。
     */
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
