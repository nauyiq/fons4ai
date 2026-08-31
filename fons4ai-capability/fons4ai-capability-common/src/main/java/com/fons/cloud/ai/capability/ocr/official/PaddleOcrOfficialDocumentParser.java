package com.fons.cloud.ai.capability.ocr.official;

import com.fons.cloud.ai.capability.ocr.PaddleOcrDocumentParser;
import com.fons.cloud.ai.capability.ocr.PaddleOcrDocumentPageResult;
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
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

/**
 * PaddleOCR 官方异步文档解析协议适配器。
 * <p>
 * 协议为提交任务、轮询状态、下载 JSONL 结果三步；不持久化 jobId，不在失败时改派 local Provider。
 *
 * @author hongqy
 */
public final class PaddleOcrOfficialDocumentParser implements PaddleOcrDocumentParser {

    private static final String JOBS_PATH = "/api/v2/ocr/jobs";
    private static final String MODEL = "PaddleOCR-VL-1.6";

    private final PaddleOcrOfficialOptions options;
    private final HttpClient httpClient;

    /**
     * @param options 已校验的官方调用选项
     */
    public PaddleOcrOfficialDocumentParser(PaddleOcrOfficialOptions options) {
        this.options = options;
        this.httpClient = HttpClient.newBuilder().connectTimeout(options.requestTimeout()).build();
    }

    @Override
    public PaddleOcrProvider provider() {
        return PaddleOcrProvider.PADDLEOCR_OFFICIAL;
    }

    @Override
    public PaddleOcrDocumentResult parse(PaddleOcrDocumentRequest request) {
        if (request == null) {
            throw BusinessRuntimeException.of(AiCapabilityResultCode.PADDLEOCR_DOCUMENT_REQUEST_INVALID);
        }
        Instant startedAt = Instant.now();
        String jobId = submit(request);
        String resultUrl = pollForResultUrl(jobId);
        List<PaddleOcrDocumentPageResult> pages = parsePages(downloadResult(resultUrl));
        String markdown = pages.stream().map(PaddleOcrDocumentPageResult::markdown).collect(Collectors.joining("\n\n"));
        return new PaddleOcrDocumentResult(markdown, pages, provider(), Duration.between(startedAt, Instant.now()));
    }

    private String submit(PaddleOcrDocumentRequest request) {
        String boundary = "----fons4aiPaddleOcr" + UUID.randomUUID().toString().replace("-", "");
        HttpRequest httpRequest = HttpRequest.newBuilder()
                .uri(pathUri(JOBS_PATH))
                .timeout(options.requestTimeout())
                .header("Authorization", "Bearer " + options.accessToken())
                .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                .POST(HttpRequest.BodyPublishers.ofByteArray(multipartBody(boundary, request)))
                .build();
        Map<String, Object> data = unwrap(send(httpRequest, "提交官方解析任务"));
        try {
            return PaddleOcrJsonSupport.requiredString(data, "jobId");
        } catch (IllegalArgumentException exception) {
            throw invalidResponse(exception);
        }
    }

    private String pollForResultUrl(String jobId) {
        Instant deadline = Instant.now().plus(options.pollTimeout());
        while (Instant.now().isBefore(deadline)) {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(pathUri(JOBS_PATH + "/" + jobId))
                    .timeout(options.requestTimeout())
                    .header("Authorization", "Bearer " + options.accessToken())
                    .GET()
                    .build();
            Map<String, Object> data = unwrap(send(request, "查询官方解析任务"));
            String state;
            try {
                state = PaddleOcrJsonSupport.requiredString(data, "state");
            } catch (IllegalArgumentException exception) {
                throw invalidResponse(exception);
            }
            if ("done".equals(state)) {
                try {
                    return PaddleOcrJsonSupport.requiredString(
                            PaddleOcrJsonSupport.requiredObject(data, "resultUrl"), "jsonUrl");
                } catch (IllegalArgumentException exception) {
                    throw invalidResponse(exception);
                }
            }
            if ("failed".equals(state)) {
                throw BusinessRuntimeException.of(AiCapabilityResultCode.PADDLEOCR_DOCUMENT_PARSE_FAILED);
            }
            if (!("pending".equals(state) || "running".equals(state))) {
                throw BusinessRuntimeException.of(AiCapabilityResultCode.PADDLEOCR_DOCUMENT_RESPONSE_INVALID);
            }
            sleepBeforeNextPoll(deadline);
        }
        throw BusinessRuntimeException.of(AiCapabilityResultCode.PADDLEOCR_DOCUMENT_PARSE_TIMEOUT);
    }

    private String downloadResult(String resultUrl) {
        URI uri;
        try {
            uri = URI.create(resultUrl);
        } catch (IllegalArgumentException exception) {
            throw invalidResponse(exception);
        }
        HttpRequest request = HttpRequest.newBuilder().uri(uri).timeout(options.requestTimeout()).GET().build();
        return send(request, "下载官方解析结果").body();
    }

    private List<PaddleOcrDocumentPageResult> parsePages(String jsonl) {
        List<PaddleOcrDocumentPageResult> pages = new ArrayList<>();
        for (String line : jsonl.split("\\R")) {
            if (line.isBlank()) {
                continue;
            }
            try {
                Map<String, Object> result = PaddleOcrJsonSupport.requiredObject(
                        PaddleOcrJsonSupport.parseObject(line), "result");
                for (Object page : PaddleOcrJsonSupport.requiredArray(result, "layoutParsingResults")) {
                    Map<String, Object> pageObject = PaddleOcrJsonSupport.asObject(page, "layoutParsingResults");
                    Map<String, Object> markdownObject = PaddleOcrJsonSupport.requiredObject(pageObject, "markdown");
                    String markdown = PaddleOcrJsonSupport.requiredString(markdownObject, "text");
                    pages.add(new PaddleOcrDocumentPageResult(markdown,
                            optionalImageUrls(markdownObject, "images"), optionalImageUrls(pageObject, "outputImages")));
                }
            } catch (IllegalArgumentException exception) {
                throw invalidResponse(exception);
            }
        }
        if (pages.isEmpty()) {
            throw BusinessRuntimeException.of(AiCapabilityResultCode.PADDLEOCR_DOCUMENT_RESPONSE_INVALID);
        }
        return pages;
    }

    private Map<String, String> optionalImageUrls(Map<String, Object> source, String field) {
        Object value = source.get(field);
        if (value == null) {
            return Map.of();
        }
        Map<String, Object> images = PaddleOcrJsonSupport.asObject(value, field);
        Map<String, String> result = new LinkedHashMap<>();
        for (Map.Entry<String, Object> entry : images.entrySet()) {
            if (!(entry.getValue() instanceof String imageUrl) || imageUrl.isBlank()) {
                throw new IllegalArgumentException("JSON 图片地址 " + entry.getKey() + " 必须是非空字符串");
            }
            result.put(entry.getKey(), imageUrl);
        }
        return result;
    }

    private Map<String, Object> unwrap(HttpResponse<String> response) {
        if (response.statusCode() < 200 || response.statusCode() >= 300) {
            throw BusinessRuntimeException.of(AiCapabilityResultCode.PADDLEOCR_DOCUMENT_PARSE_FAILED);
        }
        try {
            Map<String, Object> root = PaddleOcrJsonSupport.parseObject(response.body());
            Object code = root.get("code");
            if (code instanceof Number number && number.intValue() != 0) {
                throw BusinessRuntimeException.of(AiCapabilityResultCode.PADDLEOCR_DOCUMENT_PARSE_FAILED);
            }
            return PaddleOcrJsonSupport.requiredObject(root, "data");
        } catch (BusinessRuntimeException exception) {
            throw exception;
        } catch (IllegalArgumentException exception) {
            throw invalidResponse(exception);
        }
    }

    private HttpResponse<String> send(HttpRequest request, String operation) {
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

    private void sleepBeforeNextPoll(Instant deadline) {
        Duration remaining = Duration.between(Instant.now(), deadline);
        Duration sleep = options.pollInterval().compareTo(remaining) < 0 ? options.pollInterval() : remaining;
        if (sleep.isZero() || sleep.isNegative()) {
            return;
        }
        try {
            Thread.sleep(sleep);
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw BusinessRuntimeException.of(AiCapabilityResultCode.PADDLEOCR_DOCUMENT_PARSE_FAILED.getCode(), exception);
        }
    }

    private URI pathUri(String path) {
        String base = options.baseUri().toString();
        return URI.create((base.endsWith("/") ? base.substring(0, base.length() - 1) : base) + path);
    }

    private byte[] multipartBody(String boundary, PaddleOcrDocumentRequest request) {
        String prefix = "--" + boundary + "\r\n"
                + "Content-Disposition: form-data; name=\"model\"\r\n\r\n" + MODEL + "\r\n"
                + "--" + boundary + "\r\n"
                + "Content-Disposition: form-data; name=\"file\"; filename=\"" + safeFileName(request.fileName()) + "\"\r\n"
                + "Content-Type: application/octet-stream\r\n\r\n";
        byte[] prefixBytes = prefix.getBytes(StandardCharsets.UTF_8);
        byte[] content = request.content();
        byte[] suffix = ("\r\n--" + boundary + "--\r\n").getBytes(StandardCharsets.UTF_8);
        byte[] result = new byte[prefixBytes.length + content.length + suffix.length];
        System.arraycopy(prefixBytes, 0, result, 0, prefixBytes.length);
        System.arraycopy(content, 0, result, prefixBytes.length, content.length);
        System.arraycopy(suffix, 0, result, prefixBytes.length + content.length, suffix.length);
        return result;
    }

    private String safeFileName(String fileName) {
        String name = fileName.replace('\\', '/');
        int separator = name.lastIndexOf('/');
        String baseName = separator >= 0 ? name.substring(separator + 1) : name;
        String cleaned = baseName.replaceAll("[\\r\\n\"]", "");
        return cleaned.isBlank() ? "document" : cleaned;
    }

    private BusinessRuntimeException invalidResponse(IllegalArgumentException cause) {
        return BusinessRuntimeException.of(AiCapabilityResultCode.PADDLEOCR_DOCUMENT_RESPONSE_INVALID.getCode(), cause);
    }
}
