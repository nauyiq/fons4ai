package com.fons.cloud.ai.capability.config;

import com.fons.cloud.ai.capability.multimodal.ImageRecognitionService;
import com.fons.cloud.ai.capability.multimodal.SpringAiImageRecognitionService;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * 多模态图片识别自动配置。
 *
 * @author hongqy
 */
@Configuration
@EnableConfigurationProperties(MultimodalProperties.class)
@ConditionalOnProperty(name = "sys.multimodal.enabled", havingValue = "true")
public class MultimodalAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public ImageRecognitionService imageRecognitionService(MultimodalProperties properties) {
        return new SpringAiImageRecognitionService(properties);
    }
}
