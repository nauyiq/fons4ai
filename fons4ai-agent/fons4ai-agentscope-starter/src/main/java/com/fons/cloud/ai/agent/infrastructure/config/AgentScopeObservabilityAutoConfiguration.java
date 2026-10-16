package com.fons.cloud.ai.agent.infrastructure.config;

import com.fons.cloud.ai.agent.infrastructure.middleware.FonsAgentTraceMiddleware;
import com.fons.cloud.ai.agent.infrastructure.observability.AgentScopeTracer;
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
public class AgentScopeObservabilityAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    @ConditionalOnProperty(prefix = AgentObservabilityProperties.PREFIX, name = "enabled", havingValue = "true")
    public FonsAgentTraceMiddleware fonsAgentTraceMiddleware(
            AgentTraceRecorder recorder) {
        return new FonsAgentTraceMiddleware(recorder);
    }

    /**
     * SDK 原生 Tracer 桥接配置。
     *
     * <p>独立开关默认关闭；启用后采集注解作用域内的普通 Agent 与模型调用。
     * Harness 分支通过所有权标记交给原生中间件采集，避免重复 Generation。</p>
     */
    @Configuration(proxyBeanMethods = false)
    @ConditionalOnProperty(prefix = AgentObservabilityProperties.PREFIX, name = "enabled", havingValue = "true")
    static class AgentScopeTracerConfiguration {

        /**
         * 安装 SDK Tracer 桥接；Bean 销毁时恢复安装前状态。
         *
         * @param recorder 共享的 Trace Recorder
         * @return 已安装的 Tracer 实例
         */
        @Bean(destroyMethod = "shutdown")
        @ConditionalOnMissingBean
        @ConditionalOnProperty(prefix = AgentObservabilityProperties.PREFIX + ".agentscope-tracer",
                name = "enabled", havingValue = "true")
        public AgentScopeTracer agentScopeTracer(AgentTraceRecorder recorder) {
            AgentScopeTracer tracer = new AgentScopeTracer(recorder);
            tracer.install();
            return tracer;
        }
    }
}
