package com.fons.cloud.ai.capability.ocr.official;

import com.fons.cloud.ai.capability.ocr.PaddleOcrDocumentParser;
import com.fons.cloud.ai.capability.ocr.PaddleOcrDocumentParsers;
import com.fons.cloud.ai.capability.ocr.PaddleOcrDocumentRequest;
import com.fons.cloud.ai.capability.ocr.PaddleOcrDocumentResult;
import com.fons.cloud.ai.capability.ocr.PaddleOcrDocumentStreamRequest;
import com.fons.cloud.ai.capability.ocr.PaddleOcrProvider;
import com.fons.cloud.ai.capability.constants.AiCapabilityResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.ByteArrayInputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * PaddleOCR 官方异步协议的契约测试。
 */
class PaddleOcrOfficialDocumentParserTest {

    private HttpServer server;

    @AfterEach
    void stopServer() {
        if (server != null) {
            server.stop(0);
        }
    }

    @Test
    void shouldSubmitPollAndDownloadMarkdownWithoutExposingToken() throws Exception {
        AtomicReference<String> requestBody = new AtomicReference<>();
        AtomicReference<String> authorization = new AtomicReference<>();
        AtomicInteger sourceOpenCount = new AtomicInteger();
        server = startServer();
        server.createContext("/api/v2/ocr/jobs", exchange -> {
            authorization.set(exchange.getRequestHeaders().getFirst("Authorization"));
            requestBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            reply(exchange, 200, "{\"code\":0,\"data\":{\"jobId\":\"job-1\"}}");
        });
        server.createContext("/api/v2/ocr/jobs/job-1", exchange -> reply(exchange, 200,
                "{\"code\":0,\"data\":{\"state\":\"done\",\"resultUrl\":{\"jsonUrl\":\"" + baseUri() + "/result.jsonl\"}}}"));
        server.createContext("/result.jsonl", exchange -> reply(exchange, 200,
                "{\"result\":{\"layoutParsingResults\":[{\"markdown\":{\"text\":\"# official\",\"images\":{\"images/chart.png\":\"https://image.example/chart.png\"}},\"outputImages\":{\"layout\":\"https://image.example/layout.jpg\"}}]}}\n"));

        PaddleOcrDocumentResult result = officialParser().parse(new PaddleOcrDocumentStreamRequest(
                "report.pdf", () -> {
                    sourceOpenCount.incrementAndGet();
                    return new ByteArrayInputStream(new byte[]{1, 2});
                }));

        assertEquals(PaddleOcrProvider.PADDLEOCR_OFFICIAL, result.provider());
        assertEquals("# official", result.markdown());
        assertEquals(Map.of("images/chart.png", "https://image.example/chart.png"), result.pages().getFirst().markdownImages());
        assertEquals(Map.of("layout", "https://image.example/layout.jpg"), result.pages().getFirst().outputImages());
        assertEquals("Bearer test-token", authorization.get());
        assertTrue(requestBody.get().contains("PaddleOCR-VL-1.6"));
        assertTrue(requestBody.get().contains("name=\"file\""));
        assertEquals(1, sourceOpenCount.get());
        assertFalse(result.toString().contains("test-token"));
    }

    @Test
    void shouldMapFailedJobToSafeServiceException() throws Exception {
        server = startServer();
        server.createContext("/api/v2/ocr/jobs", exchange ->
                reply(exchange, 200, "{\"code\":0,\"data\":{\"jobId\":\"job-2\"}}"));
        server.createContext("/api/v2/ocr/jobs/job-2", exchange ->
                reply(exchange, 200, "{\"code\":0,\"data\":{\"state\":\"failed\",\"errorMsg\":\"invalid file\"}}"));

        BusinessRuntimeException exception = assertThrows(BusinessRuntimeException.class,
                () -> officialParser().parse(new PaddleOcrDocumentRequest("report.pdf", new byte[]{1})));

        assertEquals(AiCapabilityResultCode.PADDLEOCR_DOCUMENT_PARSE_FAILED.getCode(), exception.getCode());
    }

    private PaddleOcrDocumentParser officialParser() {
        return PaddleOcrDocumentParsers.create(PaddleOcrProvider.PADDLEOCR_OFFICIAL,
                new PaddleOcrOfficialOptions(baseUri(), "test-token", Duration.ofSeconds(2), Duration.ofSeconds(2), Duration.ZERO));
    }

    private HttpServer startServer() throws IOException {
        HttpServer httpServer = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        httpServer.start();
        return httpServer;
    }

    private URI baseUri() {
        return URI.create("http://127.0.0.1:" + server.getAddress().getPort());
    }

    private static void reply(HttpExchange exchange, int status, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.sendResponseHeaders(status, bytes.length);
        exchange.getResponseBody().write(bytes);
        exchange.close();
    }
}
