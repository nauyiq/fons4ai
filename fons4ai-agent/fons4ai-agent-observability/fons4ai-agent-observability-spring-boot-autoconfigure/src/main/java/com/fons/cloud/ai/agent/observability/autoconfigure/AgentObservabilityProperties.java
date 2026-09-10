package com.fons.cloud.ai.agent.observability.autoconfigure;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;

import java.nio.file.Path;

/**
 * Agent 可观测性自动配置属性。
 *
 * <p>全部框架级配置统一使用 {@code sys.agent.observability} 前缀。</p>
 *
 * @author hongqy
 */
@Getter
@ConfigurationProperties(prefix = AgentObservabilityProperties.PREFIX)
public class AgentObservabilityProperties {

    /** Agent 可观测性配置前的统一前缀。 */
    public static final String PREFIX = "sys.agent.observability";

    /** 是否启用 Agent 可观测性自动配置。 */
    @Setter
    private boolean enabled;

    /** 全局 OpenTelemetry 已存在时，是否覆盖为当前模块创建的实例。 */
    @Setter
    private boolean overrideGlobal;

    /** Langfuse 输出配置。 */
    private final LangfuseProperties langfuse = new LangfuseProperties();

    /** 本地 JSONL 输出配置。 */
    private final JsonlProperties jsonl = new JsonlProperties();

    /**
     * Langfuse OTLP 输出配置。
     *
     * @author hongqy
     */
    @Setter
    @Getter
    public static class LangfuseProperties {

        /** 是否启用 Langfuse TraceSink 和 OpenTelemetry SDK 自动配置。 */
        private boolean enabled;

        /** Langfuse OTLP/HTTP Trace 完整接收地址。 */
        private String endpoint;

        /** 写入 OpenTelemetry Resource 的服务名称。 */
        private String serviceName = "fons4ai-agent";

        /** Langfuse 项目公钥或访问标识。 */
        private String accessId;

        /** Langfuse 项目密钥。 */
        private String accessSecret;

    }

    /**
     * 本地 JSONL 输出配置。
     *
     * @author hongqy
     */
    @Setter
    @Getter
    public static class JsonlProperties {

        /** 是否启用本地 JSONL TraceSink。 */
        private boolean enabled;

        /** JSONL 轨迹文件保存目录。 */
        private Path directory = Path.of("logs", "agent-traces");

    }
}
