package com.fons.cloud.ai.agent.organization;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * @author hongqy
 */
@Getter
@Setter
@ConfigurationProperties(prefix = "sys.agent.organization")
public class OrganizationConfigProperties {

    /**
     * 能否使用
     */
    private Boolean enabled;

    /**
     * 可观察服务暴露的端点
     */
    private String endpoint;

    /**
     * 项目名称
     */
    private String projectName;

    /**
     * 访问ID
     */
    private String accessId;

    /**
     * 访问密钥
     */
    private String accessSecret;
}
