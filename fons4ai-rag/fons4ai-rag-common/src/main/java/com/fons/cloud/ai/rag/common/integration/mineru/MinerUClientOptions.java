package com.fons.cloud.ai.rag.common.integration.mineru;

import java.time.Duration;
import java.util.Objects;

/**
 * MinerU 客户端配置选项。
 * <p>
 * 无 Spring 注解，由两个框架模块各自的配置绑定类转换为此对象。
 * common 不依赖 Spring 类型。
 *
 * @param enabled         是否启用 MinerU，默认 false
 * @param baseUrl         MinerU API 基础地址，必须为绝对 HTTP/HTTPS URI
 * @param backend         MinerU 后端，默认 pipeline
 * @param connectTimeout  连接超时，默认 10s
 * @param readTimeout     读取超时，默认 5m
 * @param maxFileSize     文件大小上限（字节），默认 100MB
 * @author hongqy
 */
public record MinerUClientOptions(
        boolean enabled,
        String baseUrl,
        String backend,
        Duration connectTimeout,
        Duration readTimeout,
        long maxFileSize
) {

    /** 默认文件大小上限：100 MB */
    public static final long DEFAULT_MAX_FILE_SIZE = 100L * 1024 * 1024;

    /** 默认连接超时 */
    public static final Duration DEFAULT_CONNECT_TIMEOUT = Duration.ofSeconds(10);

    /** 默认读取超时 */
    public static final Duration DEFAULT_READ_TIMEOUT = Duration.ofMinutes(5);

    /** 默认后端 */
    public static final String DEFAULT_BACKEND = "pipeline";

    /**
     * 构造器校验配置合法性。
     */
    public MinerUClientOptions {
        if (enabled) {
            Objects.requireNonNull(baseUrl, "base-url 不可为空（enabled=true 时）");
            if (baseUrl.isBlank()) {
                throw new IllegalArgumentException("base-url 不可为空白");
            }
            if (!baseUrl.startsWith("http://") && !baseUrl.startsWith("https://")) {
                throw new IllegalArgumentException("base-url 必须为 HTTP/HTTPS URI: " + baseUrl);
            }
        }
        backend = (backend == null || backend.isBlank()) ? DEFAULT_BACKEND : backend;
        connectTimeout = (connectTimeout == null || connectTimeout.isZero() || connectTimeout.isNegative())
                ? DEFAULT_CONNECT_TIMEOUT : connectTimeout;
        readTimeout = (readTimeout == null || readTimeout.isZero() || readTimeout.isNegative())
                ? DEFAULT_READ_TIMEOUT : readTimeout;
        if (maxFileSize <= 0) {
            maxFileSize = DEFAULT_MAX_FILE_SIZE;
        }
    }

    /**
     * 创建默认禁用的配置。
     */
    public static MinerUClientOptions disabled() {
        return new MinerUClientOptions(false, null, DEFAULT_BACKEND,
                DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT, DEFAULT_MAX_FILE_SIZE);
    }
}
