package com.fons.cloud.ai.agent.infrastructure.config;

import com.fons.cloud.ai.agent.infrastructure.tools.websearch.TavilyConfigProperties;
import com.fons.cloud.ai.agent.infrastructure.tools.websearch.TavilyWebSearchTools;
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
    public TavilyWebSearchTools tavilyWebSearchTools(TavilyConfigProperties properties) {
        return new TavilyWebSearchTools(properties);
    }

}
