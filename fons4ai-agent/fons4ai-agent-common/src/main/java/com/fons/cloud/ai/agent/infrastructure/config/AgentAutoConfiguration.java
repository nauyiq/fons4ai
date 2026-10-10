package com.fons.cloud.ai.agent.infrastructure.config;

import com.fons.cloud.ai.agent.api.AgentRegistry;
import com.fons.cloud.ai.agent.core.AgentTaskManager;
import com.fons.cloud.ai.agent.core.DefaultAgentRegistry;
import com.fons.cloud.reactor.core.ReactiveRunManager;
import com.fons.cloud.reactor.infrastructure.config.ReactiveTaskAutoConfiguration;
import org.springframework.boot.autoconfigure.AutoConfigureAfter;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * @author hongqy
 */
@Configuration
@AutoConfigureAfter(ReactiveTaskAutoConfiguration.class)
public class AgentAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public AgentTaskManager agentTaskManager(ReactiveRunManager reactiveRunManager) {
        return new AgentTaskManager(reactiveRunManager);
    }

    @Bean
    @ConditionalOnMissingBean
    public AgentRegistry agentRegistry() {
        return new DefaultAgentRegistry();
    }

}
