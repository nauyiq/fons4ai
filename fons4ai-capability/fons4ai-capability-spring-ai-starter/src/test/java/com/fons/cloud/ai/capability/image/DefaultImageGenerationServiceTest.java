package com.fons.cloud.ai.capability.image;

import com.fons.cloud.ai.capability.config.ImageGenerationProperties;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DefaultImageGenerationServiceTest {

    @Test
    void shouldPreserveQwenRequestAndResponseMapping() throws IOException {
        AtomicReference<String> authorization = new AtomicReference<>();
        AtomicReference<String> requestBody = new AtomicReference<>();
        HttpServer server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext("/generate", exchange -> {
            authorization.set(exchange.getRequestHeaders().getFirst("Authorization"));
            requestBody.set(new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8));
            byte[] response = """
                    {"output":{"choices":[{"message":{"content":[{"image":"https://example.com/image.png"}]}}]}}
                    """.getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, response.length);
            exchange.getResponseBody().write(response);
            exchange.close();
        });
        server.start();

        try {
            ImageGenerationProperties properties = new ImageGenerationProperties();
            properties.setProvider(ImageGenProvider.QWEN);
            properties.setApiKey("test-key");
            properties.setModel("image-model");
            properties.setBaseUrl("http://127.0.0.1:" + server.getAddress().getPort() + "/generate");

            String imageUrl = new DefaultImageGenerationService(properties).generateImage("draw a lake");

            assertEquals("https://example.com/image.png", imageUrl);
            assertEquals("Bearer test-key", authorization.get());
            assertTrue(requestBody.get().contains("\"model\":\"image-model\""));
            assertTrue(requestBody.get().contains("\"text\":\"draw a lake\""));
            assertTrue(requestBody.get().contains("\"size\":\"1664*928\""));
        } finally {
            server.stop(0);
        }
    }
}
