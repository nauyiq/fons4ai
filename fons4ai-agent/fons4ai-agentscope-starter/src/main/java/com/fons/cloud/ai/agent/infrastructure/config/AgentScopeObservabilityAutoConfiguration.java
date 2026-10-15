package com.fons.cloud.ai.agent.infrastructure.config;

import com.fons.cloud.ai.agent.infrastructure.middleware.FonsAgentTraceMiddleware;
import com.fons.cloud.ai.agent.infrastructure.observability.AgentScopeTraceSupport;
import com.fons.cloud.ai.agent.observability.api.AgentTraceRecorder;
import com.fons.cloud.ai.agent.observability.autoconfigure.AgentObservabilityProperties;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.beans.factory.ObjectProvider;

/**
 * @author hongqy
 */
@Configuration
public class AgentScopeObservabilityAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    @ConditionalOnProperty(prefix = AgentObservabilityProperties.PREFIX, name = "enabled", havingValue = "true")
    public FonsAgentTraceMiddleware fonsAgentTraceMiddleware(
            AgentTraceRecorder recorder) {
        return new FonsAgentTraceMiddleware(recorder);
    }

    /** 关闭观测时仍提供轻量no-op入口，业务Agent不依赖某个Exporter是否启用。 */
    @Bean
    @ConditionalOnMissingBean
    public AgentScopeTraceSupport agentScopeTraceSupport(ObjectProvider<AgentTraceRecorder> recorder) {
        return new AgentScopeTraceSupport(recorder.getIfAvailable(AgentTraceRecorder::noop));
    }

}
