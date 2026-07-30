package com.fons.cloud.ai.rag.langchain.infrastructure.config;

import com.fons.cloud.ai.rag.common.document.DocumentParserRegistry;
import com.fons.cloud.ai.rag.common.document.DocumentParserSelector;
import com.fons.cloud.ai.rag.common.integration.mineru.MinerUClient;
import com.fons.cloud.ai.rag.common.integration.mineru.MinerUDocumentParser;
import com.fons.cloud.ai.rag.langchain.document.LangChain4jDocumentAdapter;
import com.fons.cloud.ai.rag.langchain.document.LangChain4jDocumentParserAdapterFactory;
import com.fons.cloud.ai.rag.langchain.document.LangChain4jDocumentParserFacade;
import com.fons.cloud.ai.rag.langchain.document.LangChain4jDocumentSplitter;
import com.fons.cloud.ai.rag.langchain.document.LangChain4jMinerUDocumentParser;
import com.fons.cloud.ai.rag.langchain.document.LangChain4jNativeDocumentParser;
import dev.langchain4j.data.document.Document;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * LangChain4j 文档解析器自动配置。
 * <p>
 * 注册 MinerU 配置绑定、共享 client、共享中立 parser、native provider、MinerU 薄包装 provider、
 * 泛型 Registry、Selector、Facade 和适配器工厂。
 * <p>
 * 共享 {@link MinerUClient} 和中立 {@link MinerUDocumentParser} 使用 {@link ConditionalOnMissingBean} 发布，
 * 与 Spring AI 模块同时加载时只保留一份协议实现。
 *
 * @author hongqy
 */
@Configuration
@ConditionalOnClass(Document.class)
@EnableConfigurationProperties({LangChain4jDocumentParserProperties.class, LangChain4jDocumentSplitterProperties.class})
public class LangChain4jDocumentParserAutoConfiguration {

    /**
     * 注册共享 MinerU HTTP 客户端，双框架共存时只保留一份。
     *
     * @param properties 配置属性
     * @return MinerU 客户端
     */
    @Bean
    @ConditionalOnMissingBean
    public MinerUClient minerUClient(LangChain4jDocumentParserProperties properties) {
        return new MinerUClient(properties.toMinerUOptions());
    }

    /**
     * 注册共享中立 MinerU 文档解析 provider，双框架共存时只保留一份。
     *
     * @param client     MinerU 客户端
     * @param properties 配置属性
     * @return 共享 MinerU 文档解析 provider
     */
    @Bean
    @ConditionalOnMissingBean
    public MinerUDocumentParser minerUDocumentParser(MinerUClient client,
                                                     LangChain4jDocumentParserProperties properties) {
        return new MinerUDocumentParser(client, properties.toMinerUOptions());
    }

    /**
     * 注册 LangChain4j 文档适配器。
     *
     * @return LangChain4j 文档适配器
     */
    @Bean
    public LangChain4jDocumentAdapter langChain4jDocumentAdapter() {
        return new LangChain4jDocumentAdapter();
    }

    /**
     * 注册 LangChain4j native 文档解析 provider。
     *
     * @return LangChain4j native provider
     */
    @Bean
    public LangChain4jNativeDocumentParser langChain4jNativeDocumentParser() {
        return new LangChain4jNativeDocumentParser();
    }

    /**
     * 注册 LangChain4j MinerU 薄包装 provider。
     *
     * @param minerUDocumentParser 共享 MinerU provider
     * @param adapter              LangChain4j 文档适配器
     * @return LangChain4j MinerU 薄包装 provider
     */
    @Bean
    public LangChain4jMinerUDocumentParser langChain4jMinerUDocumentParser(
            MinerUDocumentParser minerUDocumentParser, LangChain4jDocumentAdapter adapter) {
        return new LangChain4jMinerUDocumentParser(minerUDocumentParser, adapter);
    }

    /**
     * 注册 LangChain4j 独立泛型 Registry，只注册本框架 native 和 MinerU 薄包装 provider。
     *
     * @param nativeParser LangChain4j native provider
     * @param minerUParser LangChain4j MinerU 薄包装 provider
     * @return 泛型 Registry
     */
    @Bean
    public DocumentParserRegistry<Document> langChain4jDocumentParserRegistry(
            LangChain4jNativeDocumentParser nativeParser, LangChain4jMinerUDocumentParser minerUParser) {
        DocumentParserRegistry<Document> registry = new DocumentParserRegistry<>();
        registry.register(nativeParser);
        registry.register(minerUParser);
        return registry;
    }

    /**
     * 注册 LangChain4j 泛型选择器。
     *
     * @param registry 泛型 Registry
     * @return 泛型选择器
     */
    @Bean
    public DocumentParserSelector<Document> langChain4jDocumentParserSelector(
            DocumentParserRegistry<Document> registry) {
        return new DocumentParserSelector<>(registry);
    }

    /**
     * 注册 LangChain4j 文档分块器。
     *
     * @param properties 分块配置属性
     * @return 文档分块器
     */
    @Bean
    @ConditionalOnMissingBean
    public LangChain4jDocumentSplitter langChain4jDocumentSplitter(LangChain4jDocumentSplitterProperties properties) {
        return new LangChain4jDocumentSplitter(
                properties.getStrategy(),
                properties.getChunkSize(),
                properties.getOverlap(),
                properties.getTitleLevel());
    }

    /**
     * 注册 LangChain4j 文档解析 Facade。
     *
     * @param selector 泛型选择器
     * @param splitter 文档分块器
     * @return LangChain4j 文档解析 Facade
     */
    @Bean
    public LangChain4jDocumentParserFacade langChain4jDocumentParserFacade(
            DocumentParserSelector<Document> selector,
            LangChain4jDocumentSplitter splitter) {
        return new LangChain4jDocumentParserFacade(selector, splitter);
    }

    /**
     * 注册 LangChain4j 标准 DocumentParser 适配器工厂。
     *
     * @param facade LangChain4j 文档解析 Facade
     * @return 适配器工厂
     */
    @Bean
    public LangChain4jDocumentParserAdapterFactory langChain4jDocumentParserAdapterFactory(
            LangChain4jDocumentParserFacade facade) {
        return new LangChain4jDocumentParserAdapterFactory(facade);
    }
}
