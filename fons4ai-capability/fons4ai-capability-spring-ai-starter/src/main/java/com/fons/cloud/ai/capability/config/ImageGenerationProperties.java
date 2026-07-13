package com.fons.cloud.ai.capability.config;

import com.fons.cloud.ai.capability.image.ImageGenProvider;
import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * 图像生成配置。
 *
 * @author hongqy
 */
@Getter
@Setter
@ConfigurationProperties("sys.ai.image-generation")
public class ImageGenerationProperties {

    /** 图像生成服务提供者。 */
    private ImageGenProvider provider;

    /** 图像生成服务请求地址。 */
    private String baseUrl;

    /** 图像生成服务 API Key。 */
    private String apiKey;

    /** 图像生成模型名称。 */
    private String model;
}
