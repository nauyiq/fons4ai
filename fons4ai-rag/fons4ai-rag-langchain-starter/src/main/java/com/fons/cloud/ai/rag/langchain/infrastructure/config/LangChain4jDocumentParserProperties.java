package com.fons.cloud.ai.rag.langchain.infrastructure.config;

import com.fons.cloud.ai.rag.common.integration.mineru.MinerUClientOptions;
import jakarta.annotation.PostConstruct;
import org.springframework.boot.context.properties.ConfigurationProperties;

import java.time.Duration;
import java.util.Objects;

/**
 * LangChain4j 文档解析器配置属性。
 * <p>
 * 绑定 {@code sys.rag.document-parser} 前缀配置，转成 common {@link MinerUClientOptions}。
 * V1 {@code default-provider} 只允许 {@code native}，其他值启动校验失败。
 *
 * @author hongqy
 */
@ConfigurationProperties(prefix = "sys.rag.document-parser")
public class LangChain4jDocumentParserProperties {

    /** 默认 provider 标识 */
    private static final String DEFAULT_PROVIDER = "native";

    /** 默认 provider，V1 只允许 native */
    private String defaultProvider = DEFAULT_PROVIDER;

    /** MinerU 配置 */
    private MinerU mineru = new MinerU();

    /**
     * @return 默认 provider 标识
     */
    public String getDefaultProvider() {
        return defaultProvider;
    }

    /**
     * @param defaultProvider 默认 provider 标识
     */
    public void setDefaultProvider(String defaultProvider) {
        this.defaultProvider = defaultProvider;
    }

    /**
     * @return MinerU 配置
     */
    public MinerU getMineru() {
        return mineru;
    }

    /**
     * @param mineru MinerU 配置
     */
    public void setMineru(MinerU mineru) {
        this.mineru = mineru;
    }

    /**
     * 校验 default-provider 只允许 native。
     */
    @PostConstruct
    void validate() {
        if (!DEFAULT_PROVIDER.equalsIgnoreCase(defaultProvider)) {
            throw new IllegalArgumentException(
                    "sys.rag.document-parser.default-provider 只允许 native，当前值: " + defaultProvider);
        }
    }

    /**
     * 转换为 common {@link MinerUClientOptions}。
     *
     * @return MinerU 客户端配置选项
     */
    public MinerUClientOptions toMinerUOptions() {
        Objects.requireNonNull(mineru, "mineru 配置不可为空");
        return new MinerUClientOptions(
                mineru.isEnabled(),
                mineru.getBaseUrl(),
                mineru.getBackend(),
                mineru.getConnectTimeout(),
                mineru.getReadTimeout(),
                mineru.getMaxFileSize().toBytes()
        );
    }

    /**
     * MinerU 配置。
     */
    public static class MinerU {

        /** 是否启用 MinerU，默认 false */
        private boolean enabled = false;

        /** MinerU API 基础地址 */
        private String baseUrl = "http://localhost:8000";

        /** MinerU 后端，默认 pipeline */
        private String backend = "pipeline";

        /** 连接超时，默认 10s */
        private Duration connectTimeout = Duration.ofSeconds(10);

        /** 读取超时，默认 5m */
        private Duration readTimeout = Duration.ofMinutes(5);

        /** 文件大小上限，默认 100MB */
        private org.springframework.util.unit.DataSize maxFileSize =
                org.springframework.util.unit.DataSize.ofMegabytes(100);

        /**
         * @return 是否启用 MinerU
         */
        public boolean isEnabled() {
            return enabled;
        }

        /**
         * @param enabled 是否启用 MinerU
         */
        public void setEnabled(boolean enabled) {
            this.enabled = enabled;
        }

        /**
         * @return MinerU API 基础地址
         */
        public String getBaseUrl() {
            return baseUrl;
        }

        /**
         * @param baseUrl MinerU API 基础地址
         */
        public void setBaseUrl(String baseUrl) {
            this.baseUrl = baseUrl;
        }

        /**
         * @return MinerU 后端
         */
        public String getBackend() {
            return backend;
        }

        /**
         * @param backend MinerU 后端
         */
        public void setBackend(String backend) {
            this.backend = backend;
        }

        /**
         * @return 连接超时
         */
        public Duration getConnectTimeout() {
            return connectTimeout;
        }

        /**
         * @param connectTimeout 连接超时
         */
        public void setConnectTimeout(Duration connectTimeout) {
            this.connectTimeout = connectTimeout;
        }

        /**
         * @return 读取超时
         */
        public Duration getReadTimeout() {
            return readTimeout;
        }

        /**
         * @param readTimeout 读取超时
         */
        public void setReadTimeout(Duration readTimeout) {
            this.readTimeout = readTimeout;
        }

        /**
         * @return 文件大小上限
         */
        public org.springframework.util.unit.DataSize getMaxFileSize() {
            return maxFileSize;
        }

        /**
         * @param maxFileSize 文件大小上限
         */
        public void setMaxFileSize(org.springframework.util.unit.DataSize maxFileSize) {
            this.maxFileSize = maxFileSize;
        }
    }
}
