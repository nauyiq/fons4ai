package com.fons.cloud.ai.capability.ocr.local;

import com.fons.cloud.ai.capability.ocr.PaddleOcrDocumentParser;
import com.fons.cloud.ai.capability.ocr.PaddleOcrDocumentRequest;
import com.fons.cloud.ai.capability.ocr.PaddleOcrDocumentResult;
import com.fons.cloud.ai.capability.ocr.PaddleOcrJsonSupport;
import com.fons.cloud.ai.capability.ocr.PaddleOcrProvider;
import com.fons.cloud.ai.capability.constants.AiCapabilityResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpTimeoutException;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 调用方自部署 PaddleOCR-VL-1.6 layout-parsing 服务的协议适配器。
 * <p>
 * 所有 V1 参数固定，避免调用方在无版本约束下改变 Markdown 输出语义。
 *
 * @author hongqy
 */
public final class PaddleOcrLocalDocumentParser implements PaddleOcrDocumentParser {

    private static final String LAYOUT_PARSING_PATH = "/layout-parsing";

    private final PaddleOcrLocalOptions options;
    private final HttpClient httpClient;

    /**
     * @param options 已校验的本地服务调用选项
     */
    public PaddleOcrLocalDocumentParser(PaddleOcrLocalOptions options) {
        this.options = options;
        this.httpClient = HttpClient.newBuilder().connectTimeout(options.requestTimeout()).build();
    }

    @Override
    public PaddleOcrProvider provider() {
        return PaddleOcrProvider.PADDLEOCR_LOCAL;
    }

    @Override
    public PaddleOcrDocumentResult parse(PaddleOcrDocumentRequest request) {
        if (request == null) {
            throw BusinessRuntimeException.of(AiCapabilityResultCode.PADDLEOCR_DOCUMENT_REQUEST_INVALID);
        }
        Instant startedAt = Instant.now();
        HttpRequest httpRequest = HttpRequest.newBuilder()
                .uri(pathUri())
                .timeout(options.requestTimeout())
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(requestPayload(request), StandardCharsets.UTF_8))
                .build();
        HttpResponse<String> response = send(httpRequest);
        String markdown = parseMarkdown(response);
        return new PaddleOcrDocumentResult(markdown, provider(), Duration.between(startedAt, Instant.now()));
    }

    private String requestPayload(PaddleOcrDocumentRequest request) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("file", Base64.getEncoder().encodeToString(request.content()));
        payload.put("fileType", request.isPdf() ? 0 : 1);
        payload.put("returnMarkdownImages", false);
        payload.put("visualize", false);
        payload.put("restructurePages", true);
        payload.put("concatenatePages", true);
        return PaddleOcrJsonSupport.toJson(payload);
    }

    private String parseMarkdown(HttpResponse<String> response) {
        if (response.statusCode() < 200 || response.statusCode() >= 300) {
            throw BusinessRuntimeException.of(AiCapabilityResultCode.PADDLEOCR_DOCUMENT_PARSE_FAILED);
        }
        try {
            Map<String, Object> root = PaddleOcrJsonSupport.parseObject(response.body());
            if (PaddleOcrJsonSupport.requiredInt(root, "errorCode") != 0) {
                throw BusinessRuntimeException.of(AiCapabilityResultCode.PADDLEOCR_DOCUMENT_PARSE_FAILED);
            }
            Map<String, Object> result = PaddleOcrJsonSupport.requiredObject(root, "result");
            List<Object> pages = PaddleOcrJsonSupport.requiredArray(result, "layoutParsingResults");
            if (pages.size() != 1) {
                throw new IllegalArgumentException("本地结果必须包含唯一 layoutParsingResults 页面");
            }
            Map<String, Object> page = PaddleOcrJsonSupport.asObject(pages.getFirst(), "layoutParsingResults");
            return PaddleOcrJsonSupport.requiredString(PaddleOcrJsonSupport.requiredObject(page, "markdown"), "text");
        } catch (BusinessRuntimeException exception) {
            throw exception;
        } catch (IllegalArgumentException exception) {
            throw BusinessRuntimeException.of(AiCapabilityResultCode.PADDLEOCR_DOCUMENT_RESPONSE_INVALID.getCode(), exception);
        }
    }

    private HttpResponse<String> send(HttpRequest request) {
        try {
            return httpClient.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        } catch (HttpTimeoutException exception) {
            throw BusinessRuntimeException.of(AiCapabilityResultCode.PADDLEOCR_DOCUMENT_PARSE_TIMEOUT.getCode(), exception);
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw BusinessRuntimeException.of(AiCapabilityResultCode.PADDLEOCR_DOCUMENT_PARSE_FAILED.getCode(), exception);
        } catch (IOException exception) {
            throw BusinessRuntimeException.of(AiCapabilityResultCode.PADDLEOCR_DOCUMENT_PARSE_FAILED.getCode(), exception);
        }
    }

    private URI pathUri() {
        String base = options.baseUri().toString();
        return URI.create((base.endsWith("/") ? base.substring(0, base.length() - 1) : base) + LAYOUT_PARSING_PATH);
    }

}
