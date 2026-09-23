package com.fons.cloud.reactor.infrastructure.config;

import com.fons.cloud.reactor.api.ReactiveTaskRunFactory;
import com.fons.cloud.reactor.core.DefaultReactiveTaskRunFactory;
import com.fons.cloud.reactor.core.ReactiveRunManager;
import org.redisson.api.RedissonClient;
import org.springframework.boot.autoconfigure.AutoConfigureAfter;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * @author hongqy
 */
@Configuration
@AutoConfigureAfter(name = "org.redisson.spring.starter.RedissonAutoConfigurationV2")
public class ReactiveTaskAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public ReactiveTaskRunFactory reactiveTaskRunFactory() {
        return new DefaultReactiveTaskRunFactory();
    }

    /**
     * 仅在应用已经提供 RedissonClient 时启用分布式运行管理器。
     * 不创建 Redis 连接，也不改变没有 Redis 的响应式任务运行方式。
     */
    @Bean
    @ConditionalOnBean(RedissonClient.class)
    @ConditionalOnMissingBean
    public ReactiveRunManager reactiveRunManager(RedissonClient redissonClient) {
        return new ReactiveRunManager(redissonClient);
    }

}
