package com.fons.cloud.ai.rag.langchain.infrastructure.config;

import com.fons.cloud.ai.rag.common.document.DocumentParserRegistry;
import com.fons.cloud.ai.rag.common.document.DocumentParserSelector;
import com.fons.cloud.ai.rag.common.integration.mineru.MinerUClient;
import com.fons.cloud.ai.rag.common.integration.mineru.MinerUDocumentParser;
import com.fons.cloud.ai.rag.langchain.document.LangChain4jDocumentAdapter;
import com.fons.cloud.ai.rag.langchain.document.LangChain4jDocumentParserAdapterFactory;
import com.fons.cloud.ai.rag.langchain.document.LangChain4jDocumentParserFacade;
import com.fons.cloud.ai.rag.langchain.document.LangChain4jMinerUDocumentParser;
import com.fons.cloud.ai.rag.langchain.document.LangChain4jNativeDocumentParser;
import dev.langchain4j.data.document.Document;
import org.junit.jupiter.api.Test;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * {@link LangChain4jDocumentParserAutoConfiguration} 自动配置测试。
 *
 * @author hongqy
 */
class LangChain4jDocumentParserAutoConfigurationTest {

    private final ApplicationContextRunner contextRunner = new ApplicationContextRunner()
            .withConfiguration(AutoConfigurations.of(LangChain4jDocumentParserAutoConfiguration.class));

    @Test
    void shouldRegisterAllBeans() {
        contextRunner.run(context -> {
            assertThat(context).hasSingleBean(MinerUClient.class);
            assertThat(context).hasSingleBean(MinerUDocumentParser.class);
            assertThat(context).hasSingleBean(LangChain4jDocumentAdapter.class);
            assertThat(context).hasSingleBean(LangChain4jNativeDocumentParser.class);
            assertThat(context).hasSingleBean(LangChain4jMinerUDocumentParser.class);
            assertThat(context).hasSingleBean(DocumentParserRegistry.class);
            assertThat(context).hasSingleBean(DocumentParserSelector.class);
            assertThat(context).hasSingleBean(LangChain4jDocumentParserFacade.class);
            assertThat(context).hasSingleBean(LangChain4jDocumentParserAdapterFactory.class);
        });
    }

    @Test
    void shouldRegisterNativeAndMinerUProvidersInRegistry() {
        contextRunner.run(context -> {
            DocumentParserRegistry<Document> registry = context.getBean(DocumentParserRegistry.class);
            // 只注册 native 和 mineru 两个 provider
            assertThat(registry.all()).hasSize(2);
            assertThat(registry.find("native")).isNotNull();
            assertThat(registry.find("mineru")).isNotNull();
        });
    }

    @Test
    void shouldFailWhenDefaultProviderNotNative() {
        contextRunner
                .withPropertyValues("sys.rag.document-parser.default-provider=mineru")
                .run(context -> {
                    assertThat(context).hasFailed();
                    assertThat(context.getStartupFailure())
                            .rootCause()
                            .isInstanceOf(IllegalArgumentException.class);
                });
    }

    @Test
    void shouldBindMinerUProperties() {
        contextRunner
                .withPropertyValues(
                        "sys.rag.document-parser.mineru.enabled=true",
                        "sys.rag.document-parser.mineru.base-url=http://mineru:8080",
                        "sys.rag.document-parser.mineru.backend=pipeline",
                        "sys.rag.document-parser.mineru.connect-timeout=20s",
                        "sys.rag.document-parser.mineru.read-timeout=10m",
                        "sys.rag.document-parser.mineru.max-file-size=50MB"
                )
                .run(context -> {
                    LangChain4jDocumentParserProperties props = context
                            .getBean(LangChain4jDocumentParserProperties.class);
                    assertThat(props.getMineru().isEnabled()).isTrue();
                    assertThat(props.getMineru().getBaseUrl()).isEqualTo("http://mineru:8080");
                    assertThat(props.getMineru().getBackend()).isEqualTo("pipeline");
                    assertThat(props.toMinerUOptions().enabled()).isTrue();
                    assertThat(props.toMinerUOptions().maxFileSize()).isEqualTo(50L * 1024 * 1024);
                });
    }

    @Test
    void shouldDefaultMinerUDisabled() {
        contextRunner.run(context -> {
            LangChain4jDocumentParserProperties props = context
                    .getBean(LangChain4jDocumentParserProperties.class);
            assertThat(props.getMineru().isEnabled()).isFalse();
            assertThat(props.toMinerUOptions().enabled()).isFalse();
        });
    }
}
