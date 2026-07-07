package com.fons.cloud.ai.agent.infrastructure.config;

import com.fons.cloud.ai.agent.constants.ImageGenProvider;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * @author hongqy
 */
@Getter
@Setter
@ToString
@ConfigurationProperties("sys.image-generation")
public class ImageGenerationProperties {

    /**
     * 图像生成服务提供者
     */
    private ImageGenProvider provider;

    /**
     * 请求路径
     */
    private String baseUrl;

    /**
     * API密钥
     */
    private String apiKey;

    /**
     * 模型名称
     */
    private String model;
}
