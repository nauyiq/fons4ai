package com.fons.cloud.ai.capability.ocr.official;

import com.fons.cloud.ai.capability.ocr.PaddleOcrProviderOptions;

import java.net.URI;
import java.time.Duration;
import java.util.Objects;

/**
 * PaddleOCR 官方异步文档解析调用选项。
 * <p>
 * accessToken 仅用于构造 HTTP Authorization 请求头，不会写入结果或异常消息。
 *
 * @param baseUri 官方 API 基础地址，必须为 HTTPS；测试可使用回环 HTTP 地址
 * @param accessToken AI Studio Access Token，不可为空白
 * @param requestTimeout 一次提交、查询或结果下载的超时
 * @param pollTimeout 从提交到结果可用的总轮询超时
 * @param pollInterval 两次状态查询之间的固定间隔
 * @author hongqy
 */
public record PaddleOcrOfficialOptions(
        URI baseUri,
        String accessToken,
        Duration requestTimeout,
        Duration pollTimeout,
        Duration pollInterval
) implements PaddleOcrProviderOptions {

    /** 官方服务默认基础地址。 */
    public static final URI DEFAULT_BASE_URI = URI.create("https://paddleocr.aistudio-app.com");

    /**
     * 校验地址、凭据和所有超时边界。
     */
    public PaddleOcrOfficialOptions {
        validateBaseUri(baseUri, true);
        if (accessToken == null || accessToken.isBlank()) {
            throw new IllegalArgumentException("官方 Access Token 不可为空");
        }
        requestTimeout = positiveDuration(requestTimeout, "单次请求超时");
        pollTimeout = positiveDuration(pollTimeout, "轮询总超时");
        if (pollInterval == null || pollInterval.isNegative()) {
            throw new IllegalArgumentException("轮询间隔不可为空或为负数");
        }
    }

    /**
     * 使用官方地址和安全默认超时创建选项；调用方仍必须显式提供 Token。
     *
     * @param accessToken AI Studio Access Token
     * @return 官方调用选项
     */
    public static PaddleOcrOfficialOptions defaults(String accessToken) {
        return new PaddleOcrOfficialOptions(DEFAULT_BASE_URI, accessToken,
                Duration.ofSeconds(30), Duration.ofMinutes(10), Duration.ofSeconds(3));
    }

    @Override
    public String toString() {
        return "PaddleOcrOfficialOptions[baseUri=" + baseUri + ", accessToken=***, requestTimeout="
                + requestTimeout + ", pollTimeout=" + pollTimeout + ", pollInterval=" + pollInterval + "]";
    }

    private static Duration positiveDuration(Duration value, String name) {
        Objects.requireNonNull(value, name + "不可为空");
        if (value.isZero() || value.isNegative()) {
            throw new IllegalArgumentException(name + "必须大于零");
        }
        return value;
    }

    private static void validateBaseUri(URI value, boolean official) {
        Objects.requireNonNull(value, "基础地址不可为空");
        String scheme = value.getScheme();
        if (scheme == null || !("https".equalsIgnoreCase(scheme)
                || (!official && "http".equalsIgnoreCase(scheme))
                || (official && "http".equalsIgnoreCase(scheme) && isLoopback(value)))) {
            throw new IllegalArgumentException("基础地址必须使用 HTTPS，测试仅允许回环 HTTP 地址");
        }
        if (value.getHost() == null || value.getHost().isBlank()) {
            throw new IllegalArgumentException("基础地址必须包含主机名");
        }
    }

    private static boolean isLoopback(URI value) {
        return "localhost".equalsIgnoreCase(value.getHost()) || "127.0.0.1".equals(value.getHost())
                || "::1".equals(value.getHost());
    }
}
