package com.fons.cloud.ai.tool.config;

import com.fons.cloud.ai.tool.registry.ToolsRegistry;
import com.fons.cloud.ai.tool.spi.ToolProvider;
import com.fons.cloud.ai.tool.tavily.TavilyConfigProperties;
import com.fons.cloud.ai.tool.tavily.TavilySearchProvider;
import com.fons.cloud.ai.tool.tavily.TavilyWebSearchTools;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.List;

/**
 * 工具管理自动配置。
 *
 * @author hongqy
 */
@Configuration
@EnableConfigurationProperties(TavilyConfigProperties.class)
public class ToolsAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public ToolsRegistry toolsRegistry(List<ToolProvider> toolProviders) {
        return new ToolsRegistry(toolProviders);
    }

    @Bean
    @ConditionalOnMissingBean
    @ConditionalOnProperty(
            prefix = "sys.tavily",
            name = {"api-key", "mcp-url"})
    public TavilyWebSearchTools tavilyWebSearchTools(
            TavilyConfigProperties properties,
            ToolsRegistry toolsRegistry) {
        return new TavilyWebSearchTools(properties, toolsRegistry);
    }

    @Bean
    @ConditionalOnMissingBean
    public TavilySearchProvider tavilySearchProvider() {
        return new TavilySearchProvider();
    }
}
