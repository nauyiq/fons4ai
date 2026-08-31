package com.fons.cloud.ai.agent.organization;

import cn.hutool.core.codec.Base64;
import cn.hutool.core.util.StrUtil;
import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.exporter.otlp.http.trace.OtlpHttpSpanExporter;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.resources.Resource;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.export.BatchSpanProcessor;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.nio.charset.StandardCharsets;

/**
 * @author hongqy
 */
@Configuration
@EnableConfigurationProperties(OrganizationConfigProperties.class)
@ConditionalOnProperty(prefix = "sys.agent.organization", name = "enabled", havingValue = "true")
public class OrganizationAutoConfiguration {

    @ConditionalOnMissingBean
    @Bean(destroyMethod = "close")
    public OpenTelemetrySdk openTelemetrySdk(OrganizationConfigProperties properties) {
        String auth = Base64.encode(properties.getAccessId() + StrUtil.COLON + properties.getAccessSecret(), StandardCharsets.UTF_8);

        OtlpHttpSpanExporter exporter =  OtlpHttpSpanExporter.builder()
                .setEndpoint(properties.getEndpoint())
                .addHeader("Authorization", "Basic " + auth)
                .build();

        Resource resource =
                Resource.getDefault()
                        .merge(Resource.create(
                                        Attributes.builder()
                                                .put("service.name", properties.getProjectName())
                                                .build()));

        SdkTracerProvider tracerProvider =
                SdkTracerProvider.builder()
                        .setResource(resource)
                        .addSpanProcessor(BatchSpanProcessor.builder(exporter).build()).build();

        GlobalOpenTelemetry.resetForTest();
        return OpenTelemetrySdk.builder()
                .setTracerProvider(tracerProvider)
                .buildAndRegisterGlobal();
    }
}
