package com.fons.cloud.ai.agent.infrastructure.config;

import com.fons.cloud.ai.agent.infrastructure.tools.ToolsRegistry;
import com.fons.cloud.ai.agent.infrastructure.tools.tavily.TavilyConfigProperties;
import com.fons.cloud.ai.agent.infrastructure.tools.tavily.TavilyWebSearchTools;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * @author hongqy
 */
@Configuration
@EnableConfigurationProperties({TavilyConfigProperties.class})
public class ToolsAutoConfigurations {

    @Bean
    @ConditionalOnMissingBean
    public TavilyWebSearchTools tavilyWebSearchTools(TavilyConfigProperties properties) {
        return new TavilyWebSearchTools(properties);
    }

    @Bean
    @ConditionalOnMissingBean
    public ToolsRegistry toolsRegistry() {
        return new ToolsRegistry();
    }

}
