package com.fons.cloud.ai.agent.infrastructure.config;

import com.fons.cloud.ai.agent.infrastructure.service.ImageGenerationService;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * @author hongqy
 */
@Configuration
@EnableConfigurationProperties(ImageGenerationProperties.class)
public class ImageGenerationConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public ImageGenerationService imageGenerationService(ImageGenerationProperties properties) {
        return new ImageGenerationService(properties);
    }

}
