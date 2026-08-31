package com.fons.cloud.ai.capability.ocr.local;

import com.fons.cloud.ai.capability.ocr.PaddleOcrProviderOptions;

import java.net.URI;
import java.time.Duration;
import java.util.Objects;

/**
 * 调用方自部署 PaddleOCR layout-parsing 服务的选项。
 *
 * @param baseUri 自部署服务基础地址，HTTP/HTTPS 均可，由调用方部署网络策略负责保护
 * @param requestTimeout 一次 layout-parsing 请求的超时
 * @author hongqy
 */
public record PaddleOcrLocalOptions(URI baseUri, Duration requestTimeout) implements PaddleOcrProviderOptions {

    /**
     * 校验自部署服务地址和请求超时。
     */
    public PaddleOcrLocalOptions {
        Objects.requireNonNull(baseUri, "基础地址不可为空");
        String scheme = baseUri.getScheme();
        if (scheme == null || !("http".equalsIgnoreCase(scheme) || "https".equalsIgnoreCase(scheme))) {
            throw new IllegalArgumentException("本地服务基础地址必须使用 HTTP 或 HTTPS");
        }
        if (baseUri.getHost() == null || baseUri.getHost().isBlank()) {
            throw new IllegalArgumentException("基础地址必须包含主机名");
        }
        Objects.requireNonNull(requestTimeout, "单次请求超时不可为空");
        if (requestTimeout.isZero() || requestTimeout.isNegative()) {
            throw new IllegalArgumentException("单次请求超时必须大于零");
        }
    }
}
