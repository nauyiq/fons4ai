package com.fons.cloud.ai.agent.infrastructure.config;

import com.fons.cloud.ai.agent.infrastructure.middleware.FonsAgentTraceMiddleware;
import com.fons.cloud.ai.agent.observability.api.AgentTraceRecorder;
import com.fons.cloud.ai.agent.observability.autoconfigure.AgentObservabilityProperties;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * @author hongqy
 */
@Configuration
@ConditionalOnProperty(
        prefix = AgentObservabilityProperties.PREFIX,
        name = "enabled",
        havingValue = "true")
public class AgentScopeObservabilityAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public FonsAgentTraceMiddleware fonsAgentTraceMiddleware(
            AgentTraceRecorder recorder) {
        return new FonsAgentTraceMiddleware(recorder);
    }


}
