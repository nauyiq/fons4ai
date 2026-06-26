package com.fons.cloud.ai.rag.infrastructure.config;

import com.fons.cloud.ai.rag.config.MultipleModalConfigProperties;
import com.fons.cloud.ai.rag.infrastructure.multiplemodal.MultipleModalChatModel;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * @author hongqy
 */
@Configuration
@EnableConfigurationProperties(MultipleModalConfigProperties.class)
@ConditionalOnProperty(name = "sys.multiple-modal.enabled", havingValue = "true")
public class MultipleModalAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public MultipleModalChatModel multipleModalChatModel(MultipleModalConfigProperties properties) {
        return new MultipleModalChatModel(properties);
    }

}
