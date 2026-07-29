package com.fons.cloud.ai.rag.infrastructure.config;

import com.fons.cloud.ai.capability.multimodal.ImageRecognitionService;
import com.fons.cloud.ai.rag.common.integration.mineru.MinerUClient;
import com.fons.cloud.ai.rag.common.integration.mineru.MinerUClientOptions;
import com.fons.cloud.ai.rag.common.integration.mineru.MinerUDocumentParser;
import com.fons.cloud.ai.rag.common.document.DocumentParserRegistry;
import com.fons.cloud.ai.rag.common.document.DocumentParserSelector;
import com.fons.cloud.ai.rag.document.reader.DocumentReaderFacade;
import com.fons.cloud.ai.rag.document.reader.DocumentReaderStrategy;
import com.fons.cloud.ai.rag.document.reader.SpringAiDocumentAdapter;
import com.fons.cloud.ai.rag.document.reader.SpringAiMinerUDocumentParser;
import com.fons.cloud.ai.rag.document.reader.SpringAiNativeDocumentParser;
import com.fons.cloud.ai.rag.document.reader.support.*;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.ai.document.Document;

import java.time.Duration;
import java.util.List;

/**
 * Spring AI 文档读取自动配置。
 * <p>
 * 注册 native strategy、MinerU 共享组件、泛型 Registry/Selector 和 Facade。
 * MinerU 默认关闭，启用后通过健康检查和明确选择才参与解析。
 *
 * @author hongqy
 */
@Configuration
@EnableConfigurationProperties(DocumentReaderAutoConfiguration.DocumentParserProperties.class)
public class DocumentReaderAutoConfiguration {

    /**
     * 创建 JSON 读取策略。
     */
    @Bean
    public DocumentReaderStrategy jsonReaderStrategy() {
        return new JsonReaderStrategy();
    }

    /**
     * 创建 Markdown 读取策略。
     */
    @Bean
    public DocumentReaderStrategy markdownReaderStrategy() {
        return new MarkdownReaderStrategy();
    }

    /**
     * 创建 PDF 读取策略。
     */
    @Bean
    public DocumentReaderStrategy pdfReaderStrategy() {
        return new PdfReaderStrategy();
    }

    /**
     * 创建纯文本读取策略。
     */
    @Bean
    public DocumentReaderStrategy textReaderStrategy() {
        return new TextReaderStrategy();
    }

    /**
     * 创建 Word 文档读取策略。
     */
    @Bean
    public DocumentReaderStrategy documentReaderStrategy() {
        return new DocReaderStrategy();
    }

    /**
     * 创建图片读取策略，需要多模态识别服务。
     */
    @Bean
    @ConditionalOnBean(ImageRecognitionService.class)
    public DocumentReaderStrategy imageReaderStrategy(ImageRecognitionService imageRecognitionService) {
        return new ImageReadStrategy(imageRecognitionService);
    }

    // ---- MinerU 共享组件 ----

    /**
     * 创建 MinerU 客户端配置选项。
     * <p>
     * 使用 @ConditionalOnMissingBean 保证双框架同时加载时只保留一份协议实现。
     */
    @Bean
    @ConditionalOnMissingBean
    public MinerUClientOptions minerUClientOptions(DocumentParserProperties properties) {
        DocumentParserProperties.Mineru mineru = properties.getMineru();
        return new MinerUClientOptions(
                mineru.isEnabled(),
                mineru.getBaseUrl(),
                mineru.getBackend(),
                parseDuration(mineru.getConnectTimeout()),
                parseDuration(mineru.getReadTimeout()),
                parseFileSize(mineru.getMaxFileSize())
        );
    }

    /**
     * 创建共享 MinerU HTTP 客户端。
     */
    @Bean
    @ConditionalOnMissingBean
    public MinerUClient minerUClient(MinerUClientOptions options) {
        return new MinerUClient(options);
    }

    /**
     * 创建共享 MinerU 文档解析 provider。
     */
    @Bean
    @ConditionalOnMissingBean
    public MinerUDocumentParser minerUDocumentParser(MinerUClient client, MinerUClientOptions options) {
        return new MinerUDocumentParser(client, options);
    }

    // ---- Spring AI 适配 ----

    /**
     * 创建 Spring AI Document 适配器。
     */
    @Bean
    public SpringAiDocumentAdapter springAiDocumentAdapter() {
        return new SpringAiDocumentAdapter();
    }

    /**
     * 创建 Spring AI native provider，聚合现有 strategy。
     */
    @Bean
    public SpringAiNativeDocumentParser springAiNativeDocumentParser(List<DocumentReaderStrategy> strategies) {
        return new SpringAiNativeDocumentParser(strategies);
    }

    /**
     * 创建 Spring AI MinerU 薄包装 provider。
     */
    @Bean
    public SpringAiMinerUDocumentParser springAiMinerUDocumentParser(
            MinerUDocumentParser minerUDocumentParser,
            SpringAiDocumentAdapter adapter) {
        return new SpringAiMinerUDocumentParser(minerUDocumentParser, adapter);
    }

    /**
     * 创建 Spring AI 泛型 Registry 并注册 native 和 MinerU provider。
     */
    @Bean("springAiDocumentParserRegistry")
    public DocumentParserRegistry<List<Document>> springAiDocumentParserRegistry(
            SpringAiNativeDocumentParser nativeParser,
            SpringAiMinerUDocumentParser mineruParser) {
        DocumentParserRegistry<List<Document>> registry = new DocumentParserRegistry<>();
        registry.register(nativeParser);
        registry.register(mineruParser);
        return registry;
    }

    /**
     * 创建 Spring AI 泛型 Selector。
     */
    @Bean
    public DocumentParserSelector<List<Document>> springAiDocumentParserSelector(
            @org.springframework.beans.factory.annotation.Qualifier("springAiDocumentParserRegistry")
            DocumentParserRegistry<List<Document>> registry) {
        return new DocumentParserSelector<>(registry);
    }

    /**
     * 创建文档读取门面，注入 selector 实现统一选型。
     */
    @Bean
    DocumentReaderFacade documentReaderFacade(
            List<DocumentReaderStrategy> strategies,
            DocumentParserSelector<List<Document>> selector) {
        return new DocumentReaderFacade(strategies, selector);
    }

    // ---- 配置属性 ----

    /**
     * 文档解析配置属性。
     */
    @ConfigurationProperties(prefix = "sys.rag.document-parser")
    public static class DocumentParserProperties {

        /** 默认 provider，V1 只允许 native */
        private String defaultProvider = "native";

        /** MinerU 配置 */
        private Mineru mineru = new Mineru();

        public String getDefaultProvider() { return defaultProvider; }
        public void setDefaultProvider(String defaultProvider) { this.defaultProvider = defaultProvider; }
        public Mineru getMineru() { return mineru; }
        public void setMineru(Mineru mineru) { this.mineru = mineru; }

        /**
         * 启动时校验 default-provider 配置。
         * <p>
         * V1 只允许 native，其他值启动校验失败，防止配置绕过 DEFAULT 语义。
         *
         * @throws IllegalArgumentException default-provider 非 native 时抛出
         */
        @jakarta.annotation.PostConstruct
        public void validate() {
            if (defaultProvider == null || !"native".equalsIgnoreCase(defaultProvider.trim())) {
                throw new IllegalArgumentException(
                        "sys.rag.document-parser.default-provider V1 只允许 native，当前值: " + defaultProvider);
            }
        }

        /**
         * MinerU 配置项。
         */
        public static class Mineru {
            /** 是否启用 */
            private boolean enabled = false;
            /** API 基础地址 */
            private String baseUrl = "http://localhost:8000";
            /** 后端 */
            private String backend = "pipeline";
            /** 连接超时 */
            private String connectTimeout = "10s";
            /** 读取超时 */
            private String readTimeout = "5m";
            /** 文件大小上限 */
            private String maxFileSize = "100MB";

            public boolean isEnabled() { return enabled; }
            public void setEnabled(boolean enabled) { this.enabled = enabled; }
            public String getBaseUrl() { return baseUrl; }
            public void setBaseUrl(String baseUrl) { this.baseUrl = baseUrl; }
            public String getBackend() { return backend; }
            public void setBackend(String backend) { this.backend = backend; }
            public String getConnectTimeout() { return connectTimeout; }
            public void setConnectTimeout(String connectTimeout) { this.connectTimeout = connectTimeout; }
            public String getReadTimeout() { return readTimeout; }
            public void setReadTimeout(String readTimeout) { this.readTimeout = readTimeout; }
            public String getMaxFileSize() { return maxFileSize; }
            public void setMaxFileSize(String maxFileSize) { this.maxFileSize = maxFileSize; }
        }
    }

    /**
     * 解析时长字符串（如 "10s"、"5m"）为 Duration。
     */
    private static Duration parseDuration(String value) {
        if (value == null || value.isBlank()) {
            return null;
        }
        return Duration.parse("PT" + value.toUpperCase());
    }

    /**
     * 解析文件大小字符串（如 "100MB"）为字节数。
     */
    private static long parseFileSize(String value) {
        if (value == null || value.isBlank()) {
            return MinerUClientOptions.DEFAULT_MAX_FILE_SIZE;
        }
        String upper = value.trim().toUpperCase();
        if (upper.endsWith("MB")) {
            return Long.parseLong(upper.substring(0, upper.length() - 2).trim()) * 1024 * 1024;
        }
        if (upper.endsWith("KB")) {
            return Long.parseLong(upper.substring(0, upper.length() - 2).trim()) * 1024;
        }
        if (upper.endsWith("GB")) {
            return Long.parseLong(upper.substring(0, upper.length() - 2).trim()) * 1024 * 1024 * 1024;
        }
        return Long.parseLong(upper);
    }
}
