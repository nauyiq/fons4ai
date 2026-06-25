package com.fons.cloud.ai.agent.infrastructure.config;

import com.fons.cloud.ai.agent.core.AgentTaskManager;
import org.redisson.api.RedissonClient;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * @author hongqy
 */
@Configuration
public class AgentAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    @ConditionalOnBean(RedissonClient.class)
    public AgentTaskManager agentTaskManager(RedissonClient redissonClient) {
        return new AgentTaskManager(redissonClient);
    }

}
