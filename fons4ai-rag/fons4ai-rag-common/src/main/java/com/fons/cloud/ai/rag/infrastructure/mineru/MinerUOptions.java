package com.fons.cloud.ai.rag.infrastructure.mineru;

import com.fons.cloud.ai.rag.common.constants.RagResultCode;
import com.fons.cloud.common.base.exception.BusinessRuntimeException;

import java.net.URI;
import java.time.Duration;

/**
 * MinerU 解析服务的连接与资源边界。
 *
 * <p>该配置属于基础设施，不参与文档解析契约，也不会进入解析结果。</p>
 *
 * @author hongqy
 */
public final class MinerUOptions {

    /** 默认上传文件上限，100 MiB，以字节计量。 */
    public static final long DEFAULT_MAX_FILE_SIZE = 100L * 1024 * 1024;
    /** 默认 HTTP 响应体上限，64 MiB，以解码前的字节计量。 */
    public static final long DEFAULT_MAX_RESPONSE_SIZE = 64L * 1024 * 1024;
    /** 默认建立 HTTP 连接的等待上限，10 秒。 */
    public static final Duration DEFAULT_CONNECT_TIMEOUT = Duration.ofSeconds(10);
    /** 默认单次 HTTP 请求超时，5 分钟，不是逐次读操作的空闲超时。 */
    public static final Duration DEFAULT_READ_TIMEOUT = Duration.ofMinutes(5);
    /** 默认提交给 MinerU 的解析后端参数，不进入业务契约。 */
    public static final String DEFAULT_BACKEND = "pipeline";

    /** 是否允许使用 MinerU；该配置开关不等于实时健康检查结果。 */
    private final boolean enabled;
    /** 绝对 HTTP(S) 服务地址，去除末尾斜杠；禁用且未配置时为 null。 */
    private final String baseUrl;
    /** MinerU 协议中的后端选择参数，空白时使用默认值。 */
    private final String backend;
    /** HTTP 客户端连接超时；未提供或非正时使用默认值。 */
    private final Duration connectTimeout;
    /** 单次 HTTP 请求超时；未提供或非正时使用默认值，不表示模型输入预算。 */
    private final Duration readTimeout;
    /** 文件上传字节上限，同时检查已知大小和实际读取量；非正配置使用默认值。 */
    private final long maxFileSize;
    /** HTTP 响应体字节上限，在 JSON 解码前约束缓冲量；非正配置使用默认值。 */
    private final long maxResponseSize;

    public MinerUOptions(
            boolean enabled,
            String baseUrl,
            String backend,
            Duration connectTimeout,
            Duration readTimeout,
            long maxFileSize) {
        this(enabled, baseUrl, backend, connectTimeout, readTimeout,
                maxFileSize, DEFAULT_MAX_RESPONSE_SIZE);
    }

    public MinerUOptions(
            boolean enabled,
            String baseUrl,
            String backend,
            Duration connectTimeout,
            Duration readTimeout,
            long maxFileSize,
            long maxResponseSize) {
        this.enabled = enabled;
        this.baseUrl = normalizeBaseUrl(enabled, baseUrl);
        this.backend = normalizeBackend(backend);
        this.connectTimeout = positiveOrDefault(connectTimeout, DEFAULT_CONNECT_TIMEOUT);
        this.readTimeout = positiveOrDefault(readTimeout, DEFAULT_READ_TIMEOUT);
        this.maxFileSize = maxFileSize > 0 ? maxFileSize : DEFAULT_MAX_FILE_SIZE;
        this.maxResponseSize = maxResponseSize > 0
                ? maxResponseSize : DEFAULT_MAX_RESPONSE_SIZE;
    }

    /** 创建默认禁用配置，供按条件装配的上层模块使用。 */
    public static MinerUOptions disabled() {
        return new MinerUOptions(false, null, DEFAULT_BACKEND,
                DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT, DEFAULT_MAX_FILE_SIZE);
    }

    public boolean isEnabled() {
        return enabled;
    }

    public String getBaseUrl() {
        return baseUrl;
    }

    public String getBackend() {
        return backend;
    }

    public Duration getConnectTimeout() {
        return connectTimeout;
    }

    public Duration getReadTimeout() {
        return readTimeout;
    }

    public long getMaxFileSize() {
        return maxFileSize;
    }

    public long getMaxResponseSize() {
        return maxResponseSize;
    }

    private static String normalizeBaseUrl(boolean enabled, String value) {
        if (!enabled && (value == null || value.isBlank())) {
            return null;
        }
        if (value == null || value.isBlank()) {
            throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
        }
        URI uri;
        try {
            uri = URI.create(value.strip());
        } catch (IllegalArgumentException exception) {
            throw BusinessRuntimeException.of(
                    RagResultCode.INVALID_ARGUMENT.getCode(),
                    RagResultCode.INVALID_ARGUMENT.getMessage(),
                    exception);
        }
        if (!uri.isAbsolute()
                || (!("http".equalsIgnoreCase(uri.getScheme()))
                && !("https".equalsIgnoreCase(uri.getScheme())))) {
            throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
        }
        String normalized = value.strip();
        while (normalized.endsWith("/")) {
            normalized = normalized.substring(0, normalized.length() - 1);
        }
        return normalized;
    }

    private static String normalizeBackend(String value) {
        String normalized = value == null || value.isBlank() ? DEFAULT_BACKEND : value.strip();
        if (!normalized.matches("[A-Za-z0-9_-]+")) {
            throw BusinessRuntimeException.of(RagResultCode.INVALID_ARGUMENT);
        }
        return normalized;
    }

    private static Duration positiveOrDefault(Duration value, Duration defaultValue) {
        return value == null || value.isZero() || value.isNegative() ? defaultValue : value;
    }
}
