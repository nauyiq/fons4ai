package com.fons.cloud.ai.agent.observability.autoconfigure;

import com.fons.cloud.ai.agent.observability.api.AgentTraceRecorder;
import com.fons.cloud.ai.agent.observability.api.TraceSink;
import com.fons.cloud.ai.agent.observability.core.DefaultAgentTraceRecorder;
import com.fons.cloud.ai.agent.observability.core.JsonlTraceSink;
import com.fons.cloud.ai.agent.observability.core.LangfuseTraceSink;
import io.opentelemetry.api.OpenTelemetry;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.List;

/**
 * Agent 可观测性 Spring Boot 自动配置入口。
 *
 * <p>该配置负责组装 TraceSink 和 AgentTraceRecorder。具体 Agent 框架只需要注入
 * {@link OpenTelemetry} 与 {@link AgentTraceRecorder}，不需要自行创建 SDK 或 Exporter。</p>
 *
 * @author hongqy
 */
@AutoConfiguration
@EnableConfigurationProperties(AgentObservabilityProperties.class)
public class AgentObservabilityAutoConfiguration {

    /**
     * Agent 可观测性总开关打开后的 Bean 配置。
     *
     * @author hongqy
     */
    @Configuration(proxyBeanMethods = false)
    @ConditionalOnProperty(
            prefix = AgentObservabilityProperties.PREFIX,
            name = "enabled",
            havingValue = "true")
    static class EnabledConfiguration {

        /**
         * 使用全部 TraceSink 创建通用 Recorder；没有 Sink 时使用无操作实现。
         *
         * @param sinkProvider Spring 容器中的 TraceSink 提供器
         * @return Agent Trace Recorder
         */
        @Bean
        @ConditionalOnMissingBean(AgentTraceRecorder.class)
        AgentTraceRecorder agentTraceRecorder(ObjectProvider<TraceSink> sinkProvider) {
            List<TraceSink> sinks = sinkProvider.orderedStream().toList();
            return sinks.isEmpty()
                    ? AgentTraceRecorder.noop()
                    : new DefaultAgentTraceRecorder(sinks);
        }

        /**
         * Langfuse OpenTelemetry 运行环境配置。
         *
         * @author hongqy
         */
        @Configuration(proxyBeanMethods = false)
        @ConditionalOnProperty(
                prefix = AgentObservabilityProperties.PREFIX + ".langfuse",
                name = "enabled",
                havingValue = "true")
        static class LangfuseConfiguration {
            /**
             * 创建复用当前 OpenTelemetry 上下文的 Langfuse TraceSink。
             *
             * @param openTelemetry OpenTelemetry 实例
             * @return Langfuse TraceSink
             */
            @Bean
            @ConditionalOnMissingBean(LangfuseTraceSink.class)
            LangfuseTraceSink langfuseTraceSink(OpenTelemetry openTelemetry) {
                return new LangfuseTraceSink(openTelemetry);
            }
        }

        /**
         * 本地 JSONL 输出配置。
         *
         * @author hongqy
         */
        @Configuration(proxyBeanMethods = false)
        @ConditionalOnProperty(
                prefix = AgentObservabilityProperties.PREFIX + ".jsonl",
                name = "enabled",
                havingValue = "true")
        static class JsonlConfiguration {

            /**
             * 创建本地 JSONL TraceSink。
             *
             * @param properties Agent 可观测性配置
             * @return JSONL TraceSink
             */
            @Bean
            @ConditionalOnMissingBean(JsonlTraceSink.class)
            JsonlTraceSink jsonlTraceSink(AgentObservabilityProperties properties) {
                return new JsonlTraceSink(properties.getJsonl().getDirectory());
            }
        }
    }
}
