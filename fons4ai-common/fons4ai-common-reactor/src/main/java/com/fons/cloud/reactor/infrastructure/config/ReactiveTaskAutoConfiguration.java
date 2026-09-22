package com.fons.cloud.reactor.infrastructure.config;

import com.fons.cloud.reactor.api.ReactiveTaskRunFactory;
import com.fons.cloud.reactor.core.DefaultReactiveTaskRunFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * @author hongqy
 */
@Configuration
public class ReactiveTaskAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public ReactiveTaskRunFactory reactiveTaskRunFactory() {
        return new DefaultReactiveTaskRunFactory();
    }

}
