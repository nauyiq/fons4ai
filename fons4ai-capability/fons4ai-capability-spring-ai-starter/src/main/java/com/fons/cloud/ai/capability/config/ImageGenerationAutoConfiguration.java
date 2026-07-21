package com.fons.cloud.ai.capability.config;

import com.fons.cloud.ai.capability.image.ImageGenerationService;
import com.fons.cloud.ai.capability.image.DefaultImageGenerationService;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * 图像生成自动配置。
 *
 * @author hongqy
 */
@Configuration
@EnableConfigurationProperties(ImageGenerationProperties.class)
public class ImageGenerationAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public ImageGenerationService imageGenerationService(ImageGenerationProperties properties) {
        return new DefaultImageGenerationService(properties);
    }
}
