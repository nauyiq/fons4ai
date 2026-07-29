package com.fons.cloud.ai.rag.common.integration.mineru;

import com.fons.cloud.ai.rag.common.document.DocumentParseException;
import com.fons.cloud.ai.rag.common.document.DocumentParseError;
import com.fons.cloud.ai.rag.common.document.DocumentSource;

import java.io.IOException;
import java.io.InputStream;
import java.io.SequenceInputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpConnectTimeoutException;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpTimeoutException;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * MinerU HTTP 协议客户端。
 * <p>
 * 使用 JDK {@link HttpClient} 实现，不引入额外 HTTP 框架。
 * 健康检查 {@code GET /health}，同步解析 {@code POST /file_parse}（multipart/form-data）。
 * 协议依据以 MinerU 官方 {@code /health}、{@code /file_parse} 文档及官方 {@code fast_api.py} 响应实现为准。
 * <p>
 * 安全约束：不记录文档正文、认证信息或原始响应全文；错误摘要限长脱敏。
 *
 * @author hongqy
 */
public final class MinerUClient {

    /** 限制错误摘要最大长度 */
    private static final int ERROR_SUMMARY_MAX_LEN = 200;

    private final MinerUClientOptions options;
    private final HttpClient httpClient;

    /**
     * @param options MinerU 配置选项，不可为 null
     */
    public MinerUClient(MinerUClientOptions options) {
        this.options = options;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(options.connectTimeout())
                .build();
    }

    /**
     * 健康检查。
     * <p>
     * V1 在每次显式 MinerU 解析前检查，不缓存健康结果。
     *
     * @return 2xx 且返回可解析 JSON 时返回 true；否则 false
     */
    public boolean isHealthy() {
        if (!options.enabled()) {
            return false;
        }
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(options.baseUrl() + "/health"))
                    .timeout(options.readTimeout())
                    .GET()
                    .build();
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            if (response.statusCode() < 200 || response.statusCode() >= 300) {
                return false;
            }
            // 响应应为可解析 JSON
            String body = response.body();
            if (body == null || body.isBlank()) {
                return false;
            }
            // 简单校验是否为 JSON 格式（不以简单非 JSON 字符开头）
            String trimmed = body.stripLeading();
            return trimmed.startsWith("{") || trimmed.startsWith("[");
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * 同步解析文件。
     * <p>
     * V1 每次只发送一个 {@code files} part。
     *
     * @param source 可重复文档源
     * @return MinerU 解析结果
     * @throws DocumentParseException 超时、HTTP 错误、响应非法或解析失败时抛出
     */
    public MinerUParseResult parseFile(DocumentSource source) {
        // 先校验真实文件大小
        if (source.size() > options.maxFileSize()) {
            throw new DocumentParseException(DocumentParseError.FILE_TOO_LARGE, "mineru",
                    "文件大小超过上限: " + options.maxFileSize() + " 字节", null);
        }

        String boundary = generateBoundary();
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(options.baseUrl() + "/file_parse"))
                .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                .timeout(options.readTimeout())
                .POST(HttpRequest.BodyPublishers.ofInputStream(
                        () -> buildMultipartBody(boundary, source)))
                .build();

        HttpResponse<String> response;
        try {
            response = httpClient.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        } catch (HttpConnectTimeoutException e) {
            throw new DocumentParseException(DocumentParseError.CONNECTION_TIMEOUT, "mineru",
                    "连接 MinerU 超时", e);
        } catch (HttpTimeoutException e) {
            throw new DocumentParseException(DocumentParseError.READ_TIMEOUT, "mineru",
                    "读取 MinerU 响应超时", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new DocumentParseException(DocumentParseError.IO_ERROR, "mineru",
                    "请求被中断", e);
        } catch (IOException e) {
            throw new DocumentParseException(DocumentParseError.IO_ERROR, "mineru",
                    "请求 MinerU 失败: " + e.getMessage(), e);
        }

        if (response.statusCode() < 200 || response.statusCode() >= 300) {
            throw new DocumentParseException(DocumentParseError.HTTP_ERROR, "mineru",
                    "MinerU HTTP " + response.statusCode() + ": " + sanitizeSummary(response.body()), null);
        }

        return parseResponse(response.body());
    }

    /**
     * 解析 MinerU JSON 响应。
     * <p>
     * 顶层读取 backend、version、results；
     * V1 要求 results 为唯一文件结果，并从其 md_content 读取 Markdown。
     * 结果为空、多结果、缺失 md_content 或类型不符都是 INVALID_RESPONSE。
     */
    MinerUParseResult parseResponse(String jsonBody) {
        if (jsonBody == null || jsonBody.isBlank()) {
            throw new DocumentParseException(DocumentParseError.INVALID_RESPONSE, "mineru",
                    "MinerU 响应为空", null);
        }

        Map<String, Object> root;
        try {
            root = parseJson(jsonBody);
        } catch (Exception e) {
            throw new DocumentParseException(DocumentParseError.INVALID_RESPONSE, "mineru",
                    "MinerU 响应 JSON 解析失败: " + e.getMessage(), e);
        }

        Object resultsObj = root.get("results");
        if (resultsObj == null) {
            throw new DocumentParseException(DocumentParseError.INVALID_RESPONSE, "mineru",
                    "MinerU 响应缺少 results 字段", null);
        }

        // results 应为列表且只有一个元素
        List<?> results;
        if (resultsObj instanceof List<?> list) {
            results = list;
        } else {
            throw new DocumentParseException(DocumentParseError.INVALID_RESPONSE, "mineru",
                    "MinerU results 不是数组", null);
        }

        if (results.isEmpty()) {
            throw new DocumentParseException(DocumentParseError.INVALID_RESPONSE, "mineru",
                    "MinerU results 为空", null);
        }
        if (results.size() > 1) {
            throw new DocumentParseException(DocumentParseError.INVALID_RESPONSE, "mineru",
                    "MinerU results 包含多个结果，V1 只支持单文件", null);
        }

        Object firstResult = results.get(0);
        if (!(firstResult instanceof Map<?, ?> resultMap)) {
            throw new DocumentParseException(DocumentParseError.INVALID_RESPONSE, "mineru",
                    "MinerU results[0] 不是对象", null);
        }

        Object mdContent = resultMap.get("md_content");
        if (mdContent == null) {
            throw new DocumentParseException(DocumentParseError.INVALID_RESPONSE, "mineru",
                    "MinerU 响应缺少 md_content 字段", null);
        }
        if (!(mdContent instanceof String md)) {
            throw new DocumentParseException(DocumentParseError.INVALID_RESPONSE, "mineru",
                    "MinerU md_content 不是字符串", null);
        }

        String backend = root.get("backend") instanceof String b ? b : null;
        String version = root.get("version") instanceof String v ? v : null;

        return new MinerUParseResult(md, version, backend);
    }

    // ---- multipart 构建 ----

    /**
     * 构建 multipart/form-data 请求体。
     * <p>
     * 使用 SequenceInputStream 连接多个流，避免将文件全部读入内存。
     * 文件名只使用 basename 并过滤 CR/LF 和引号，防止 multipart header 注入。
     */
    InputStream buildMultipartBody(String boundary, DocumentSource source) {
        List<InputStream> streams = new ArrayList<>();

        // 固定表单字段
        Map<String, String> formFields = new LinkedHashMap<>();
        formFields.put("backend", options.backend());
        formFields.put("parse_method", "auto");
        formFields.put("return_md", "true");
        formFields.put("response_format_zip", "false");
        formFields.put("return_middle_json", "false");
        formFields.put("return_model_output", "false");
        formFields.put("return_content_list", "false");
        formFields.put("return_images", "false");
        formFields.put("return_original_file", "false");
        formFields.put("formula_enable", "true");
        formFields.put("table_enable", "true");

        for (var entry : formFields.entrySet()) {
            streams.add(toStream("--" + boundary + "\r\n"));
            streams.add(toStream("Content-Disposition: form-data; name=\"" + entry.getKey() + "\"\r\n\r\n"));
            streams.add(toStream(entry.getValue() + "\r\n"));
        }

        // 文件 part
        String safeFileName = sanitizeFileName(source.fileName());
        String contentType = source.contentType() != null ? source.contentType() : "application/octet-stream";
        streams.add(toStream("--" + boundary + "\r\n"));
        streams.add(toStream("Content-Disposition: form-data; name=\"files\"; filename=\"" + safeFileName + "\"\r\n"));
        streams.add(toStream("Content-Type: " + contentType + "\r\n\r\n"));
        streams.add(source.openStream());
        streams.add(toStream("\r\n"));

        // 结束标记
        streams.add(toStream("--" + boundary + "--\r\n"));

        return new SequenceInputStream(Collections.enumeration(streams));
    }

    /**
     * 生成随机 boundary。
     */
    private static String generateBoundary() {
        Random random = new Random();
        StringBuilder sb = new StringBuilder("----fons4aiMinerU");
        for (int i = 0; i < 16; i++) {
            sb.append(Integer.toHexString(random.nextInt(16)));
        }
        return sb.toString();
    }

    /**
     * 文件名安全处理：只取 basename，过滤 CR/LF 和引号，防止 multipart header 注入。
     */
    static String sanitizeFileName(String fileName) {
        if (fileName == null || fileName.isBlank()) {
            return "document";
        }
        // 取 basename
        String name = fileName;
        int lastSep = Math.max(fileName.lastIndexOf('/'), fileName.lastIndexOf('\\'));
        if (lastSep >= 0) {
            name = fileName.substring(lastSep + 1);
        }
        // 过滤 CR/LF 和引号
        name = name.replaceAll("[\\r\\n\"]", "");
        if (name.isBlank()) {
            return "document";
        }
        return name;
    }

    /**
     * 限制错误摘要长度并脱敏，不记录完整响应。
     */
    private static String sanitizeSummary(String text) {
        if (text == null) {
            return "";
        }
        String summary = text.strip();
        if (summary.length() > ERROR_SUMMARY_MAX_LEN) {
            summary = summary.substring(0, ERROR_SUMMARY_MAX_LEN) + "...";
        }
        return summary;
    }

    private static InputStream toStream(String s) {
        return new java.io.ByteArrayInputStream(s.getBytes(StandardCharsets.UTF_8));
    }

    // ---- 简易 JSON 解析（不引入外部依赖） ----

    /**
     * 简易 JSON 解析，仅支持 MinerU 响应所需的对象、数组、字符串、数字、布尔和 null。
     */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> parseJson(String json) {
        JsonParser parser = new JsonParser(json);
        Object result = parser.parseValue();
        parser.skipWhitespace();
        if (parser.pos < parser.json.length()) {
            throw new IllegalArgumentException("JSON 尾部有多余字符");
        }
        if (result instanceof Map) {
            return (Map<String, Object>) result;
        }
        throw new IllegalArgumentException("JSON 顶层不是对象");
    }

    /**
     * 极简 JSON 解析器，仅用于 MinerU 响应。
     */
    private static final class JsonParser {
        final String json;
        int pos;

        JsonParser(String json) {
            this.json = json;
            this.pos = 0;
        }

        void skipWhitespace() {
            while (pos < json.length() && Character.isWhitespace(json.charAt(pos))) {
                pos++;
            }
        }

        Object parseValue() {
            skipWhitespace();
            if (pos >= json.length()) {
                throw new IllegalArgumentException("JSON 意外结束");
            }
            char c = json.charAt(pos);
            if (c == '{') return parseObject();
            if (c == '[') return parseArray();
            if (c == '"') return parseString();
            if (c == 't' || c == 'f') return parseBoolean();
            if (c == 'n') return parseNull();
            return parseNumber();
        }

        Map<String, Object> parseObject() {
            Map<String, Object> map = new LinkedHashMap<>();
            pos++; // skip {
            skipWhitespace();
            if (pos < json.length() && json.charAt(pos) == '}') {
                pos++;
                return map;
            }
            while (true) {
                skipWhitespace();
                String key = parseString();
                skipWhitespace();
                if (pos >= json.length() || json.charAt(pos) != ':') {
                    throw new IllegalArgumentException("JSON 对象缺少冒号");
                }
                pos++; // skip :
                Object value = parseValue();
                map.put(key, value);
                skipWhitespace();
                if (pos >= json.length()) {
                    throw new IllegalArgumentException("JSON 对象意外结束");
                }
                char next = json.charAt(pos);
                if (next == ',') {
                    pos++;
                    continue;
                }
                if (next == '}') {
                    pos++;
                    break;
                }
                throw new IllegalArgumentException("JSON 对象缺少逗号或右括号");
            }
            return map;
        }

        List<Object> parseArray() {
            List<Object> list = new ArrayList<>();
            pos++; // skip [
            skipWhitespace();
            if (pos < json.length() && json.charAt(pos) == ']') {
                pos++;
                return list;
            }
            while (true) {
                Object value = parseValue();
                list.add(value);
                skipWhitespace();
                if (pos >= json.length()) {
                    throw new IllegalArgumentException("JSON 数组意外结束");
                }
                char next = json.charAt(pos);
                if (next == ',') {
                    pos++;
                    continue;
                }
                if (next == ']') {
                    pos++;
                    break;
                }
                throw new IllegalArgumentException("JSON 数组缺少逗号或右括号");
            }
            return list;
        }

        String parseString() {
            if (pos >= json.length() || json.charAt(pos) != '"') {
                throw new IllegalArgumentException("JSON 字符串缺少引号");
            }
            pos++; // skip "
            StringBuilder sb = new StringBuilder();
            while (pos < json.length()) {
                char c = json.charAt(pos++);
                if (c == '"') {
                    return sb.toString();
                }
                if (c == '\\') {
                    if (pos >= json.length()) {
                        throw new IllegalArgumentException("JSON 转义意外结束");
                    }
                    char esc = json.charAt(pos++);
                    switch (esc) {
                        case '"' -> sb.append('"');
                        case '\\' -> sb.append('\\');
                        case '/' -> sb.append('/');
                        case 'n' -> sb.append('\n');
                        case 't' -> sb.append('\t');
                        case 'r' -> sb.append('\r');
                        case 'b' -> sb.append('\b');
                        case 'f' -> sb.append('\f');
                        case 'u' -> {
                            if (pos + 4 > json.length()) {
                                throw new IllegalArgumentException("JSON unicode 转义不完整");
                            }
                            String hex = json.substring(pos, pos + 4);
                            sb.append((char) Integer.parseInt(hex, 16));
                            pos += 4;
                        }
                        default -> throw new IllegalArgumentException("JSON 无效转义: \\" + esc);
                    }
                } else {
                    sb.append(c);
                }
            }
            throw new IllegalArgumentException("JSON 字符串未闭合");
        }

        Object parseBoolean() {
            if (json.startsWith("true", pos)) {
                pos += 4;
                return Boolean.TRUE;
            }
            if (json.startsWith("false", pos)) {
                pos += 5;
                return Boolean.FALSE;
            }
            throw new IllegalArgumentException("JSON 无效布尔值");
        }

        Object parseNull() {
            if (json.startsWith("null", pos)) {
                pos += 4;
                return null;
            }
            throw new IllegalArgumentException("JSON 无效 null 值");
        }

        Object parseNumber() {
            int start = pos;
            while (pos < json.length()) {
                char c = json.charAt(pos);
                if (Character.isDigit(c) || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') {
                    pos++;
                } else {
                    break;
                }
            }
            if (start == pos) {
                throw new IllegalArgumentException("JSON 无效数字");
            }
            String numStr = json.substring(start, pos);
            if (numStr.contains(".") || numStr.contains("e") || numStr.contains("E")) {
                return Double.parseDouble(numStr);
            }
            return Long.parseLong(numStr);
        }
    }
}
