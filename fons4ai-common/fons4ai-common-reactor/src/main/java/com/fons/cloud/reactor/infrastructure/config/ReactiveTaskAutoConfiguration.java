package com.fons.cloud.reactor.infrastructure.config;

import com.fons.cloud.reactor.api.ReactiveTaskInterceptor;
import com.fons.cloud.reactor.api.ReactiveTaskRunFactory;
import com.fons.cloud.reactor.core.DefaultReactiveTaskRunFactory;
import com.fons.cloud.reactor.core.ReactiveRunManager;
import com.fons.cloud.reactor.infrastructure.config.properties.ReactiveRunProperties;
import org.redisson.api.RedissonClient;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.AutoConfigureAfter;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * @author hongqy
 */
@Configuration
@EnableConfigurationProperties(ReactiveRunProperties.class)
@AutoConfigureAfter(name = "org.redisson.spring.starter.RedissonAutoConfigurationV2")
public class ReactiveTaskAutoConfiguration {

    /**
     * 默认运行工厂：注入容器中全部运行扩展，未注册任何扩展时与无扩展工厂行为一致。
     *
     * @param interceptorProvider 运行扩展提供器
     * @return 响应式任务运行句柄工厂
     */
    @Bean
    @ConditionalOnMissingBean
    public ReactiveTaskRunFactory reactiveTaskRunFactory(
            ObjectProvider<ReactiveTaskInterceptor> interceptorProvider) {
        return new DefaultReactiveTaskRunFactory(interceptorProvider.orderedStream().toList());
    }

    /**
     * 仅在应用已经提供 RedissonClient 时启用分布式运行管理器。
     * 不创建 Redis 连接，也不改变没有 Redis 的响应式任务运行方式。
     */
    @Bean
    @ConditionalOnBean(RedissonClient.class)
    @ConditionalOnMissingBean
    public ReactiveRunManager reactiveRunManager(RedissonClient redissonClient, ReactiveRunProperties properties) {
        return new ReactiveRunManager(redissonClient, properties);
    }

}
