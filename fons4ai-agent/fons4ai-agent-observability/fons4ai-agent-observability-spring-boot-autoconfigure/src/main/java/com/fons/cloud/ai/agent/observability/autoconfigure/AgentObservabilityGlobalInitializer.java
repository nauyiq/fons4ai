package com.fons.cloud.ai.agent.observability.autoconfigure;

import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.OpenTelemetry;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.exporter.otlp.http.trace.OtlpHttpSpanExporter;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.resources.Resource;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.export.BatchSpanProcessor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.config.BeanDefinition;
import org.springframework.beans.factory.support.BeanDefinitionRegistry;
import org.springframework.beans.factory.support.RootBeanDefinition;
import org.springframework.boot.context.properties.bind.Bindable;
import org.springframework.boot.context.properties.bind.Binder;
import org.springframework.context.ApplicationContextInitializer;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.core.Ordered;

import java.nio.charset.StandardCharsets;
import java.util.Base64;

/**
 * 在 Spring Bean 创建前初始化全局 OpenTelemetry 的启动器。
 *
 * <p>AgentScope 原生 OtelTracingMiddleware 通过 GlobalOpenTelemetry 获取 Tracer，
 * 因此必须在 DataSource 等组件首次访问全局实例前完成初始化。</p>
 *
 * @author hongqy
 */
@Slf4j
public final class AgentObservabilityGlobalInitializer
        implements ApplicationContextInitializer<ConfigurableApplicationContext>, Ordered {

    /** 注册到 Spring 容器中的 OpenTelemetry Bean 名称。 */
    private static final String OPEN_TELEMETRY_BEAN_NAME = "fonsAgentObservabilityOpenTelemetry";

    /** Langfuse v4 原生 OpenTelemetry 数据接入版本。 */
    private static final String LANGFUSE_INGESTION_VERSION = "4";


    @Override
    public int getOrder() {
        return Ordered.HIGHEST_PRECEDENCE;
    }

    @Override
    public void initialize(ConfigurableApplicationContext applicationContext) {
        AgentObservabilityProperties properties = Binder.get(applicationContext.getEnvironment())
                .bind(
                        AgentObservabilityProperties.PREFIX,
                        Bindable.of(AgentObservabilityProperties.class))
                .orElseGet(AgentObservabilityProperties::new);

        if (!properties.isEnabled() || !properties.getLangfuse().isEnabled()) {
            return;
        }

        synchronized (AgentObservabilityGlobalInitializer.class) {
            initializeGlobalOpenTelemetry(applicationContext, properties);
        }
    }

    /**
     * 根据全局实例状态选择复用或覆盖，并注册对应的 Spring Bean。
     *
     * @param applicationContext 当前 Spring 应用上下文
     * @param properties Agent 可观测性配置
     */
    private static void initializeGlobalOpenTelemetry(
            ConfigurableApplicationContext applicationContext,
            AgentObservabilityProperties properties) {
        if (GlobalOpenTelemetry.isSet() && !properties.isOverrideGlobal()) {
            registerOpenTelemetryBean(applicationContext, GlobalOpenTelemetry.get(), false);
            return;
        }

        OpenTelemetrySdk sdk = createOpenTelemetrySdk(properties);
        if (GlobalOpenTelemetry.isSet()) {
            GlobalOpenTelemetry.resetForTest();
        }

        try {
            GlobalOpenTelemetry.set(sdk);
            registerOpenTelemetryBean(applicationContext, sdk, true);
        } catch (IllegalStateException exception) {
            sdk.close();
            OpenTelemetry existing = GlobalOpenTelemetry.getOrNoop();
            registerOpenTelemetryBean(applicationContext, existing, false);
            log.warn(
                    "Global OpenTelemetry was occupied concurrently; reused the current instance", exception);
        }
    }

    /**
     * 根据 Langfuse 配置创建 OpenTelemetry SDK。
     *
     * @param properties Agent 可观测性配置
     * @return 尚未注册为全局实例的 OpenTelemetry SDK
     */
    private static OpenTelemetrySdk createOpenTelemetrySdk(AgentObservabilityProperties properties) {
        AgentObservabilityProperties.LangfuseProperties langfuse = properties.getLangfuse();
        String endpoint = requireText(langfuse.getEndpoint(), "sys.agent.observability.langfuse.endpoint");
        String accessId = requireText(langfuse.getAccessId(), "sys.agent.observability.langfuse.access-id");
        String accessSecret = requireText(
                langfuse.getAccessSecret(),
                "sys.agent.observability.langfuse.access-secret");
        String serviceName = requireText(
                langfuse.getServiceName(),
                "sys.agent.observability.langfuse.service-name");

        String credentials = accessId + ":" + accessSecret;
        String authorization = Base64.getEncoder().encodeToString(
                credentials.getBytes(StandardCharsets.UTF_8));

        OtlpHttpSpanExporter exporter = OtlpHttpSpanExporter.builder()
                .setEndpoint(endpoint)
                .addHeader("Authorization", "Basic " + authorization)
                .addHeader("x-langfuse-ingestion-version", LANGFUSE_INGESTION_VERSION)
                .build();

        Resource resource = Resource.getDefault().merge(Resource.create(
                Attributes.builder()
                        .put("service.name", serviceName)
                        .build()));

        SdkTracerProvider tracerProvider = SdkTracerProvider.builder()
                .setResource(resource)
                .addSpanProcessor(BatchSpanProcessor.builder(exporter).build())
                .build();

        return OpenTelemetrySdk.builder()
                .setTracerProvider(tracerProvider)
                .build();
    }

    /**
     * 把最终使用的 OpenTelemetry 注册为 Spring 主 Bean。
     *
     * @param applicationContext 当前 Spring 应用上下文
     * @param openTelemetry 最终使用的 OpenTelemetry 实例
     * @param ownedByApplication 当前应用是否负责关闭该实例
     */
    private static void registerOpenTelemetryBean(
            ConfigurableApplicationContext applicationContext,
            OpenTelemetry openTelemetry,
            boolean ownedByApplication) {
        BeanDefinitionRegistry registry = (BeanDefinitionRegistry) applicationContext.getBeanFactory();
        if (registry.containsBeanDefinition(OPEN_TELEMETRY_BEAN_NAME)) {
            return;
        }

        RootBeanDefinition beanDefinition = new RootBeanDefinition();
        beanDefinition.setBeanClass(OpenTelemetry.class);
        beanDefinition.setInstanceSupplier(() -> openTelemetry);
        beanDefinition.setPrimary(true);
        beanDefinition.setRole(BeanDefinition.ROLE_INFRASTRUCTURE);
        if (ownedByApplication) {
            beanDefinition.setDestroyMethodName("close");
        }
        registry.registerBeanDefinition(OPEN_TELEMETRY_BEAN_NAME, beanDefinition);
    }

    /**
     * 读取必填文本配置，并在缺失时给出明确的配置键提示。
     *
     * @param value 配置值
     * @param propertyName 配置键
     * @return 去除首尾空白后的配置值
     */
    private static String requireText(String value, String propertyName) {
        if (value == null || value.isBlank()) {
            throw new IllegalStateException("Required property is missing: " + propertyName);
        }
        return value.trim();
    }
}
