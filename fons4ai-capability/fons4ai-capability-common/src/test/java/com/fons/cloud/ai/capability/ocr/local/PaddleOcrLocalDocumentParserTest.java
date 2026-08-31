package com.fons.cloud.ai.capability.ocr.local;

import com.fons.cloud.ai.capability.ocr.PaddleOcrDocumentParser;
import com.fons.cloud.ai.capability.ocr.PaddleOcrDocumentParsers;
import com.fons.cloud.ai.capability.ocr.PaddleOcrDocumentRequest;
import com.fons.cloud.ai.capability.ocr.PaddleOcrDocumentResult;
import com.fons.cloud.ai.capability.ocr.PaddleOcrProvider;
import com.fons.cloud.ai.capability.constants.AiCapabilityResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * paddleocr-local 协议契约测试。
 */
class PaddleOcrLocalDocumentParserTest {

    private HttpServer server;

    @AfterEach
    void stopServer() {
        if (server != null) {
            server.stop(0);
        }
    }

    @Test
    void shouldSendFixedLayoutParsingOptionsAndReadMarkdown() throws Exception {
        AtomicReference<String> requestBody = new AtomicReference<>();
        server = startServer();
        server.createContext("/layout-parsing", exchange -> {
            requestBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            reply(exchange, 200, "{\"errorCode\":0,\"result\":{\"layoutParsingResults\":[{\"markdown\":{\"text\":\"# local\"}}]}}");
        });

        PaddleOcrDocumentResult result = localParser().parse(new PaddleOcrDocumentRequest("scan.jpg", new byte[]{3, 4}));

        assertEquals(PaddleOcrProvider.PADDLEOCR_LOCAL, result.provider());
        assertEquals("# local", result.markdown());
        assertTrue(requestBody.get().contains("\"fileType\":1"));
        assertTrue(requestBody.get().contains("\"returnMarkdownImages\":false"));
        assertTrue(requestBody.get().contains("\"visualize\":false"));
        assertTrue(requestBody.get().contains("\"restructurePages\":true"));
        assertTrue(requestBody.get().contains("\"concatenatePages\":true"));
    }

    @Test
    void shouldRejectNonZeroServiceErrorCode() throws Exception {
        server = startServer();
        server.createContext("/layout-parsing", exchange ->
                reply(exchange, 200, "{\"errorCode\":422,\"errorMsg\":\"unsupported\"}"));

        BusinessRuntimeException exception = assertThrows(BusinessRuntimeException.class,
                () -> localParser().parse(new PaddleOcrDocumentRequest("scan.jpg", new byte[]{3})));

        assertEquals(AiCapabilityResultCode.PADDLEOCR_DOCUMENT_PARSE_FAILED.getCode(), exception.getCode());
    }

    private PaddleOcrDocumentParser localParser() {
        return PaddleOcrDocumentParsers.create(PaddleOcrProvider.PADDLEOCR_LOCAL,
                new PaddleOcrLocalOptions(baseUri(), Duration.ofSeconds(2)));
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
