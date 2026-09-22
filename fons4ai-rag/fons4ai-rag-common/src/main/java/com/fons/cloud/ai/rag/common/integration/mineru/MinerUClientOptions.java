package com.fons.cloud.ai.rag.common.integration.mineru;

import com.fons.cloud.ai.rag.infrastructure.mineru.MinerUOptions;

import java.time.Duration;

/**
 * 旧框架装配代码使用的 MinerU 配置兼容入口。
 *
 * <p>新代码应使用 {@link MinerUOptions}。该类会在两个技术框架完成统一契约适配后删除。</p>
 *
 * @author hongqy
 * @deprecated 迁移到 {@link MinerUOptions}
 */
@Deprecated(forRemoval = true)
public final class MinerUClientOptions {

    public static final long DEFAULT_MAX_FILE_SIZE = MinerUOptions.DEFAULT_MAX_FILE_SIZE;
    public static final Duration DEFAULT_CONNECT_TIMEOUT = MinerUOptions.DEFAULT_CONNECT_TIMEOUT;
    public static final Duration DEFAULT_READ_TIMEOUT = MinerUOptions.DEFAULT_READ_TIMEOUT;
    public static final String DEFAULT_BACKEND = MinerUOptions.DEFAULT_BACKEND;

    private final MinerUOptions delegate;

    public MinerUClientOptions(
            boolean enabled,
            String baseUrl,
            String backend,
            Duration connectTimeout,
            Duration readTimeout,
            long maxFileSize) {
        this.delegate = new MinerUOptions(
                enabled, baseUrl, backend, connectTimeout, readTimeout, maxFileSize);
    }

    public static MinerUClientOptions disabled() {
        return new MinerUClientOptions(false, null, DEFAULT_BACKEND,
                DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT, DEFAULT_MAX_FILE_SIZE);
    }

    public boolean enabled() {
        return delegate.isEnabled();
    }

    public String baseUrl() {
        return delegate.getBaseUrl();
    }

    public String backend() {
        return delegate.getBackend();
    }

    public Duration connectTimeout() {
        return delegate.getConnectTimeout();
    }

    public Duration readTimeout() {
        return delegate.getReadTimeout();
    }

    public long maxFileSize() {
        return delegate.getMaxFileSize();
    }

    public MinerUOptions toOptions() {
        return delegate;
    }
}
