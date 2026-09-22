package com.fons.cloud.ai.rag.infrastructure.config;

import com.fons.cloud.ai.rag.api.ChunkingStrategy;
import com.fons.cloud.ai.rag.api.DocumentChunkingService;
import com.fons.cloud.ai.rag.api.DocumentFormatDetector;
import com.fons.cloud.ai.rag.api.DocumentParser;
import com.fons.cloud.ai.rag.api.DocumentParsingService;
import com.fons.cloud.ai.rag.core.DefaultDocumentChunkingService;
import com.fons.cloud.ai.rag.core.DefaultDocumentParsingService;
import com.fons.cloud.ai.rag.infrastructure.detection.DefaultDocumentFormatDetector;
import com.fons.cloud.ai.rag.infrastructure.mineru.MinerUClient;
import com.fons.cloud.ai.rag.infrastructure.mineru.MinerUDocumentParser;
import com.fons.cloud.ai.rag.infrastructure.mineru.MinerUOptions;
import com.fons.cloud.ai.rag.infrastructure.chunking.LangChain4jRecursiveChunkingStrategy;
import com.fons.cloud.ai.rag.infrastructure.parsing.LangChain4jTikaTextDocumentParser;
import com.fons.cloud.ai.rag.langchain.infrastructure.config.LangChain4jDocumentParserAutoConfiguration;
import com.fons.cloud.ai.rag.langchain.infrastructure.config.LangChain4jDocumentParserProperties;
import dev.langchain4j.data.document.Document;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

import java.util.List;

/**
 * 将 LangChain4j 解析、分块实现装配到 rag-common 的中立编排服务。
 *
 * <p>旧解析 Facade 保留给尚未迁移的业务调用方；新服务只接收统一契约实现，
 * 不把 LangChain4j 的 Document 或 Splitter 暴露到 common。</p>
 */
@AutoConfiguration(after = LangChain4jDocumentParserAutoConfiguration.class)
@ConditionalOnClass(Document.class)
@EnableConfigurationProperties(LangChain4jDocumentParserProperties.class)
public class LangChain4jRagCommonAutoConfiguration {

    /** 格式识别属于 common 的基础设施实现，业务可自行替换。 */
    @Bean
    @ConditionalOnMissingBean(DocumentFormatDetector.class)
    public DocumentFormatDetector ragDocumentFormatDetector() {
        return new DefaultDocumentFormatDetector();
    }

    /** LangChain4j 只解析能够证明为纯文本的文档。 */
    @Bean
    @ConditionalOnMissingBean(LangChain4jTikaTextDocumentParser.class)
    public LangChain4jTikaTextDocumentParser langChain4jTikaTextDocumentParser() {
        return new LangChain4jTikaTextDocumentParser();
    }

    /** 新旧调用链共用一份配置值，但互不共用旧版解析结果类型。 */
    @Bean
    @ConditionalOnMissingBean(MinerUOptions.class)
    public MinerUOptions ragCommonMinerUOptions(
            LangChain4jDocumentParserProperties properties) {
        LangChain4jDocumentParserProperties.MinerU mineru = properties.getMineru();
        return new MinerUOptions(
                mineru.isEnabled(),
                mineru.getBaseUrl(),
                mineru.getBackend(),
                mineru.getConnectTimeout(),
                mineru.getReadTimeout(),
                mineru.getMaxFileSize().toBytes());
    }

    @Bean
    @ConditionalOnMissingBean(MinerUClient.class)
    public MinerUClient ragCommonMinerUClient(MinerUOptions options) {
        return new MinerUClient(options);
    }

    /** MinerU 不依赖 LangChain4j；未启用时由 common 编排在选择前排除。 */
    @Bean
    @ConditionalOnMissingBean(MinerUDocumentParser.class)
    public MinerUDocumentParser ragCommonMinerUDocumentParser(
            MinerUClient client, MinerUOptions options) {
        return new MinerUDocumentParser(client, options);
    }

    /** 具体递归切分算法留在 LangChain4j 适配层。 */
    @Bean
    @ConditionalOnMissingBean(LangChain4jRecursiveChunkingStrategy.class)
    public LangChain4jRecursiveChunkingStrategy langChain4jRecursiveChunkingStrategy() {
        return new LangChain4jRecursiveChunkingStrategy();
    }

    /** 第一步汇聚所有统一 Parser，再交给 common 按格式和选择策略编排。 */
    @Bean
    @ConditionalOnMissingBean(DocumentParsingService.class)
    public DocumentParsingService ragDocumentParsingService(
            DocumentFormatDetector detector, List<DocumentParser> parsers) {
        return new DefaultDocumentParsingService(detector, parsers);
    }

    /** 分块服务只选择并验收策略，真正计算边界由被选中的技术实现负责。 */
    @Bean
    @ConditionalOnMissingBean(DocumentChunkingService.class)
    public DocumentChunkingService ragDocumentChunkingService(
            List<ChunkingStrategy> strategies) {
        return new DefaultDocumentChunkingService(strategies);
    }
}
