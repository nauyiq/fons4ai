package com.fons.cloud.ai.tool.config;

import com.fons.cloud.ai.tool.registry.ToolsRegistry;
import com.fons.cloud.ai.tool.tavily.TavilyConfigProperties;
import com.fons.cloud.ai.tool.tavily.TavilySearchProvider;
import com.fons.cloud.ai.tool.tavily.TavilyWebSearchTools;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import static org.assertj.core.api.Assertions.assertThat;
class ToolsAutoConfigurationTest {

    private final ApplicationContextRunner contextRunner = new ApplicationContextRunner()
            .withUserConfiguration(ToolsAutoConfiguration.class);

    @Test
    void shouldBindNewTavilyPrefixAndLoadToolBeans() {
        contextRunner
                .withBean(TavilyWebSearchTools.class, NoOpTavilyWebSearchTools::new)
                .withPropertyValues(
                        "sys.tool.tavily.api-key=test-key",
                        "sys.tool.tavily.mcp-url=https://example.com",
                        "sys.tool.tavily.request-timeout-seconds=30")
                .run(context -> {
                    assertThat(context).hasNotFailed();
                    assertThat(context).hasSingleBean(ToolsRegistry.class);
                    assertThat(context).hasSingleBean(TavilySearchProvider.class);
                    TavilyConfigProperties properties = context.getBean(TavilyConfigProperties.class);
                    assertThat(properties.getApiKey()).isEqualTo("test-key");
                    assertThat(properties.getRequestTimeoutSeconds()).isEqualTo(30);
                });
    }

    @Test
    void shouldNotBindOldTavilyPrefix() {
        contextRunner
                .withPropertyValues(
                        "sys.tavily.api-key=old-key",
                        "sys.tavily.mcp-url=https://example.com")
                .run(context -> {
                    assertThat(context).hasNotFailed();
                    assertThat(context).doesNotHaveBean(TavilyWebSearchTools.class);
                    TavilyConfigProperties properties = context.getBean(TavilyConfigProperties.class);
                    assertThat(properties.getApiKey()).isNull();
                    assertThat(properties.getMcpUrl()).isNull();
                });
    }

    private static final class NoOpTavilyWebSearchTools extends TavilyWebSearchTools {

        private NoOpTavilyWebSearchTools() {
            super(null, null);
        }

        @Override
        public void afterPropertiesSet() {
            // 测试自动配置时禁止建立外部 MCP 连接。
        }
    }
}
