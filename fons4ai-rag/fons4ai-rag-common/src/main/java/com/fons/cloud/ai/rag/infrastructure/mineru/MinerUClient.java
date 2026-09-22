package com.fons.cloud.ai.rag.infrastructure.mineru;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;
import com.alibaba.fastjson2.annotation.JSONField;
import com.fons.cloud.ai.rag.api.DocumentSource;
import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;
import com.fons.cloud.common.result.R;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.SequenceInputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpConnectTimeoutException;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpTimeoutException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Flow;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * MinerU 同步 HTTP 客户端。
 *
 * <p>客户端只负责协议传输和响应解码，不理解 RAG 领域模型。文件内容采用流式
 * multipart 发送，响应 JSON 在 MinerU 基础设施内部使用 Fastjson2 解析。</p>
 *
 * @author hongqy
 */
public final class MinerUClient {

    /** 本客户端共享的连接参数、启用状态和传输字节上限。 */
    private final MinerUOptions options;
    /** 首次请求时延迟创建的可复用 HTTP 客户端，通过 volatile 安全发布。 */
    private volatile HttpClient httpClient;

    public MinerUClient(MinerUOptions options) {
        if (options == null) {
            throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
        }
        this.options = options;
    }

    /** 显式探测服务状态；文档解析链路不会为每个文件自动调用该方法。 */
    public boolean isHealthy() {
        if (!options.isEnabled()) {
            return false;
        }
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(options.getBaseUrl() + "/health"))
                    .timeout(options.getReadTimeout())
                    .GET()
                    .build();
            HttpResponse<String> response = httpClient().send(request,
                    limitedStringBody(options.getMaxResponseSize(), new AtomicBoolean()));
            if (response.statusCode() < 200 || response.statusCode() >= 300
                    || response.body() == null || response.body().isBlank()) {
                return false;
            }
            JSON.parseObject(response.body());
            return true;
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            return false;
        } catch (Exception ignored) {
            return false;
        }
    }

    /**
     * 将单个文件提交给 MinerU，并使用 RAG 错误码表达传输或协议失败。
     */
    public R<MinerUParsePayload> parseFile(DocumentSource source) {
        if (!options.isEnabled()) {
            return R.failed(RagResultCode.DOCUMENT_PARSER_UNAVAILABLE);
        }
        if (source == null) {
            return R.failed(RagResultCode.DOCUMENT_SOURCE_INVALID);
        }
        long sourceSize;
        try {
            sourceSize = source.size();
        } catch (RuntimeException exception) {
            return R.failed(RagResultCode.DOCUMENT_SOURCE_INVALID);
        }
        if (sourceSize >= 0 && sourceSize > options.getMaxFileSize()) {
            return R.failed(RagResultCode.DOCUMENT_FILE_TOO_LARGE);
        }

        String boundary = "----fons4aiMinerU" + UUID.randomUUID().toString().replace("-", "");
        UploadState uploadState = new UploadState();
        AtomicBoolean responseTooLarge = new AtomicBoolean();
        HttpRequest request;
        try {
            request = HttpRequest.newBuilder()
                    .uri(URI.create(options.getBaseUrl() + "/file_parse"))
                    .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                    .timeout(options.getReadTimeout())
                    .POST(HttpRequest.BodyPublishers.ofInputStream(
                            () -> buildMultipartBody(boundary, source, uploadState)))
                    .build();
        } catch (RuntimeException exception) {
            return R.failed(RagResultCode.DOCUMENT_PARSER_FAILED);
        }

        HttpResponse<String> response;
        try {
            response = httpClient().send(
                    request, limitedStringBody(options.getMaxResponseSize(), responseTooLarge));
        } catch (HttpConnectTimeoutException exception) {
            return R.failed(RagResultCode.DOCUMENT_PARSER_CONNECTION_TIMEOUT);
        } catch (HttpTimeoutException exception) {
            return R.failed(RagResultCode.DOCUMENT_PARSER_READ_TIMEOUT);
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            return R.failed(RagResultCode.DOCUMENT_PARSER_IO_ERROR);
        } catch (IOException exception) {
            if (uploadState.tooLarge.get()) {
                return R.failed(RagResultCode.DOCUMENT_FILE_TOO_LARGE);
            }
            if (uploadState.sourceInvalid.get()) {
                return R.failed(RagResultCode.DOCUMENT_SOURCE_INVALID);
            }
            if (uploadState.sourceFailed.get()) {
                return R.failed(RagResultCode.DOCUMENT_SOURCE_READ_FAILED);
            }
            if (responseTooLarge.get()) {
                return R.failed(RagResultCode.DOCUMENT_PARSER_RESPONSE_TOO_LARGE);
            }
            return R.failed(RagResultCode.DOCUMENT_PARSER_IO_ERROR);
        } catch (RuntimeException exception) {
            if (uploadState.tooLarge.get()) {
                return R.failed(RagResultCode.DOCUMENT_FILE_TOO_LARGE);
            }
            if (uploadState.sourceInvalid.get()) {
                return R.failed(RagResultCode.DOCUMENT_SOURCE_INVALID);
            }
            if (uploadState.sourceFailed.get()) {
                return R.failed(RagResultCode.DOCUMENT_SOURCE_READ_FAILED);
            }
            if (responseTooLarge.get()) {
                return R.failed(RagResultCode.DOCUMENT_PARSER_RESPONSE_TOO_LARGE);
            }
            return R.failed(RagResultCode.DOCUMENT_PARSER_FAILED);
        }

        if (response.statusCode() < 200 || response.statusCode() >= 300) {
            return R.failed(RagResultCode.DOCUMENT_PARSER_HTTP_ERROR);
        }
        return parseResponse(response.body());
    }

    R<MinerUParsePayload> parseResponse(String body) {
        if (body == null || body.isBlank()) {
            return R.failed(RagResultCode.DOCUMENT_PARSER_RESPONSE_INVALID);
        }
        try {
            MinerUResponse response = JSON.parseObject(body, MinerUResponse.class);
            if (response == null) {
                return R.failed(RagResultCode.DOCUMENT_PARSER_RESPONSE_INVALID);
            }
            MinerUFileResult result = singleResult(response.getResults())
                    .toJavaObject(MinerUFileResult.class);
            String markdown = textOrNull(result.getMarkdown());
            List<Map<String, Object>> contentItems = contentItems(result.getContentList());
            if ((markdown == null || markdown.isBlank()) && contentItems.isEmpty()) {
                return R.failed(RagResultCode.DOCUMENT_PARSER_RESPONSE_INVALID);
            }
            return R.success(new MinerUParsePayload(
                    markdown, textOrNull(response.getBackend()), contentItems));
        } catch (BusinessRuntimeException exception) {
            return R.failed(RagResultCode.DOCUMENT_PARSER_RESPONSE_INVALID);
        } catch (RuntimeException exception) {
            return R.failed(RagResultCode.DOCUMENT_PARSER_RESPONSE_INVALID);
        }
    }

    InputStream buildMultipartBody(String boundary, DocumentSource source) {
        return buildMultipartBody(boundary, source, new UploadState());
    }

    private InputStream buildMultipartBody(
            String boundary, DocumentSource source, UploadState uploadState) {
        List<InputStream> streams = new ArrayList<>();
        Map<String, String> formFields = new LinkedHashMap<>();
        formFields.put("backend", options.getBackend());
        formFields.put("parse_method", "auto");
        formFields.put("return_md", "true");
        formFields.put("response_format_zip", "false");
        formFields.put("return_middle_json", "false");
        formFields.put("return_model_output", "false");
        formFields.put("return_content_list", "true");
        formFields.put("return_images", "false");
        formFields.put("return_original_file", "false");
        formFields.put("formula_enable", "true");
        formFields.put("table_enable", "true");

        for (Map.Entry<String, String> entry : formFields.entrySet()) {
            streams.add(asStream("--" + boundary + "\r\n"));
            streams.add(asStream("Content-Disposition: form-data; name=\""
                    + entry.getKey() + "\"\r\n\r\n"));
            streams.add(asStream(entry.getValue() + "\r\n"));
        }

        String fileName;
        String mediaType;
        try {
            fileName = source.fileName();
            mediaType = source.mediaType();
        } catch (RuntimeException exception) {
            uploadState.sourceInvalid.set(true);
            throw exception;
        }
        streams.add(asStream("--" + boundary + "\r\n"));
        streams.add(asStream("Content-Disposition: form-data; name=\"files\"; filename=\""
                + sanitizeFileName(fileName) + "\"\r\n"));
        streams.add(asStream("Content-Type: " + sanitizeMediaType(mediaType)
                + "\r\n\r\n"));
        InputStream sourceStream;
        try {
            sourceStream = source.openStream();
            if (sourceStream == null) {
                uploadState.sourceFailed.set(true);
                throw new IllegalStateException("DocumentSource returned null stream");
            }
        } catch (RuntimeException exception) {
            uploadState.sourceFailed.set(true);
            throw exception;
        }
        streams.add(new LimitedSourceStream(
                sourceStream, options.getMaxFileSize(), uploadState));
        streams.add(asStream("\r\n--" + boundary + "--\r\n"));
        return new SequenceInputStream(Collections.enumeration(streams));
    }

    private static JSONObject singleResult(Object results) {
        if (results == null) {
            throw invalidResponse();
        }
        if (results instanceof JSONObject resultObject) {
            if (resultObject.size() != 1) {
                throw invalidResponse();
            }
            return asJsonObject(resultObject.values().iterator().next());
        }
        if (results instanceof JSONArray resultArray) {
            if (resultArray.size() != 1) {
                throw invalidResponse();
            }
            return asJsonObject(resultArray.get(0));
        }
        throw invalidResponse();
    }

    private static JSONObject asJsonObject(Object value) {
        if (value instanceof JSONObject jsonObject) {
            return jsonObject;
        }
        throw invalidResponse();
    }

    private static List<Map<String, Object>> contentItems(Object contentValue) {
        if (contentValue == null) {
            return List.of();
        }
        List<?> values;
        if (contentValue instanceof String text) {
            values = JSON.parseArray(text);
        } else if (contentValue instanceof JSONArray array) {
            values = array;
        } else if (contentValue instanceof List<?> list) {
            values = list;
        } else {
            throw invalidResponse();
        }
        if (values == null) {
            throw invalidResponse();
        }
        List<Map<String, Object>> items = new ArrayList<>(values.size());
        for (Object value : values) {
            if (!(value instanceof Map<?, ?> map)) {
                throw invalidResponse();
            }
            items.add(toStringKeyMap(map));
        }
        return items;
    }

    private static Map<String, Object> toStringKeyMap(Map<?, ?> source) {
        Map<String, Object> target = new LinkedHashMap<>();
        for (Map.Entry<?, ?> entry : source.entrySet()) {
            if (!(entry.getKey() instanceof String key)) {
                throw invalidResponse();
            }
            target.put(key, toJavaValue(entry.getValue()));
        }
        return target;
    }

    private static Object toJavaValue(Object value) {
        if (value instanceof Map<?, ?> map) {
            return toStringKeyMap(map);
        }
        if (value instanceof List<?> list) {
            List<Object> values = new ArrayList<>(list.size());
            for (Object element : list) {
                values.add(toJavaValue(element));
            }
            return values;
        }
        return value;
    }

    private static String textOrNull(Object value) {
        return value instanceof String text && !text.isBlank()
                ? text
                : null;
    }

    static String sanitizeFileName(String fileName) {
        if (fileName == null || fileName.isBlank()) {
            return "document";
        }
        int separator = Math.max(fileName.lastIndexOf('/'), fileName.lastIndexOf('\\'));
        String name = separator >= 0 ? fileName.substring(separator + 1) : fileName;
        name = name.replaceAll("[\\r\\n\"]", "");
        return name.isBlank() ? "document" : name;
    }

    private static String sanitizeMediaType(String mediaType) {
        if (mediaType == null || !mediaType.matches("[A-Za-z0-9.+-]+/[A-Za-z0-9.+-]+")) {
            return "application/octet-stream";
        }
        return mediaType;
    }

    private static InputStream asStream(String value) {
        return new ByteArrayInputStream(value.getBytes(StandardCharsets.UTF_8));
    }

    static HttpResponse.BodyHandler<String> limitedStringBody(
            long maxBytes, AtomicBoolean tooLarge) {
        return ignored -> new LimitedStringSubscriber(maxBytes, tooLarge);
    }

    /** 请求体可能被 HTTP 客户端重试，因此大小和读取状态按一次调用汇总。 */
    private static final class UploadState {
        /** 本次调用的任一上传流是否超过文件字节上限，跨请求体重开保留。 */
        private final AtomicBoolean tooLarge = new AtomicBoolean();
        /** 本次调用是否遇到无法正常打开的来源，用于归类来源无效响应。 */
        private final AtomicBoolean sourceInvalid = new AtomicBoolean();
        /** 本次调用是否发生来源读取失败，用于与超限及 HTTP 失败区分。 */
        private final AtomicBoolean sourceFailed = new AtomicBoolean();
    }

    /** 按实际读取字节限制单个上传流，并把失败状态汇总到本次调用。 */
    private static final class LimitedSourceStream extends FilterInputStream {
        /** 当前文件流允许读取的最大字节数，不包含 multipart 协议边界。 */
        private final long maxBytes;
        /** 同次调用共享的上传状态，不随请求体流重新打开而重置。 */
        private final UploadState state;
        /** 当前流已读取的文件字节数，每次创建新流从 0 开始。 */
        private long count;

        private LimitedSourceStream(InputStream source, long maxBytes, UploadState state) {
            super(source);
            this.maxBytes = maxBytes;
            this.state = state;
        }

        @Override
        public int read() throws IOException {
            try {
                int value = in.read();
                if (value >= 0 && ++count > maxBytes) {
                    state.tooLarge.set(true);
                    throw new IOException("DocumentSource exceeds MinerU upload limit");
                }
                return value;
            } catch (IOException exception) {
                if (!state.tooLarge.get()) {
                    state.sourceFailed.set(true);
                }
                throw exception;
            }
        }

        @Override
        public int read(byte[] bytes, int offset, int length) throws IOException {
            if (length == 0) {
                return 0;
            }
            try {
                long remaining = maxBytes - count;
                int allowed = remaining > 0
                        ? (int) Math.min(length, remaining) : 1;
                int read = in.read(bytes, offset, allowed);
                if (read > 0 && (count += read) > maxBytes) {
                    state.tooLarge.set(true);
                    throw new IOException("DocumentSource exceeds MinerU upload limit");
                }
                return read;
            } catch (IOException exception) {
                if (!state.tooLarge.get()) {
                    state.sourceFailed.set(true);
                }
                throw exception;
            }
        }
    }

    /** 先限制响应字节缓冲，再在接收完成后统一按 UTF-8 解码。 */
    private static final class LimitedStringSubscriber
            implements HttpResponse.BodySubscriber<String> {
        /** 响应体缓冲字节上限，不是字符串字符数上限。 */
        private final long maxBytes;
        /** 向请求调用方报告响应体超限，便于映射稳定的框架响应码。 */
        private final AtomicBoolean tooLarge;
        /** 已接收且未超过上限的响应字节，完成前不进行分段字符串解码。 */
        private final ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        /** 最终解码结果；超限或传输失败时异常完成，由客户端转成 R 失败。 */
        private final CompletableFuture<String> body = new CompletableFuture<>();
        /** HTTP 响应订阅，用于逐批拉取和超限时取消；订阅前为 null。 */
        private Flow.Subscription subscription;

        private LimitedStringSubscriber(long maxBytes, AtomicBoolean tooLarge) {
            this.maxBytes = maxBytes;
            this.tooLarge = tooLarge;
        }

        @Override
        public CompletableFuture<String> getBody() {
            return body;
        }

        @Override
        public void onSubscribe(Flow.Subscription subscription) {
            this.subscription = subscription;
            subscription.request(1);
        }

        @Override
        public void onNext(List<ByteBuffer> buffers) {
            for (ByteBuffer buffer : buffers) {
                if (buffer.remaining() > maxBytes - bytes.size()) {
                    tooLarge.set(true);
                    subscription.cancel();
                    body.completeExceptionally(new IOException("MinerU response exceeds size limit"));
                    return;
                }
                byte[] chunk = new byte[buffer.remaining()];
                buffer.get(chunk);
                bytes.writeBytes(chunk);
            }
            subscription.request(1);
        }

        @Override
        public void onError(Throwable error) {
            body.completeExceptionally(error);
        }

        @Override
        public void onComplete() {
            body.complete(bytes.toString(StandardCharsets.UTF_8));
        }
    }

    private HttpClient httpClient() {
        HttpClient current = httpClient;
        if (current != null) {
            return current;
        }
        synchronized (this) {
            if (httpClient == null) {
                // 延迟创建避免仅做配置装配时启动网络选择器。
                httpClient = HttpClient.newBuilder()
                        .connectTimeout(options.getConnectTimeout())
                        .build();
            }
            return httpClient;
        }
    }

    private static BusinessRuntimeException invalidResponse() {
        return BusinessRuntimeException.of(
                RagResultCode.DOCUMENT_PARSER_RESPONSE_INVALID);
    }

    /** MinerU 顶层响应，只保留协议解码阶段需要的字段。 */
    private static final class MinerUResponse {

        /** 供应商响应中的后端标识，仅用于协议载荷，不进入文档元数据。 */
        private String backend;
        /** 供应商文件结果容器，具体形状由专用解码步骤校验，不向统一契约暴露。 */
        private Object results;

        public String getBackend() {
            return backend;
        }

        public void setBackend(String backend) {
            this.backend = backend;
        }

        public Object getResults() {
            return results;
        }

        public void setResults(Object results) {
            this.results = results;
        }
    }

    /** 单个文件的供应商结果，异构内容列表留到专用解码步骤处理。 */
    private static final class MinerUFileResult {

        /** md_content 对应的可选 Markdown 正文。 */
        @JSONField(name = "md_content")
        private String markdown;

        /** content_list 原始值，可为 JSON 文本或内容数组，解码后再交给映射器。 */
        @JSONField(name = "content_list")
        private Object contentList;

        public String getMarkdown() {
            return markdown;
        }

        public void setMarkdown(String markdown) {
            this.markdown = markdown;
        }

        public Object getContentList() {
            return contentList;
        }

        public void setContentList(Object contentList) {
            this.contentList = contentList;
        }
    }

}
