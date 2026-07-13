package com.fons.cloud.ai.capability.config;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * 多模态图片识别配置。
 *
 * @author hongqy
 */
@Getter
@Setter
@ConfigurationProperties("sys.ai.multimodal")
public class MultimodalProperties {

    /** 是否启用多模态能力。 */
    private Boolean enabled;

    /** 模型服务 API Key。 */
    private String apiKey;

    /** OpenAI 兼容服务地址。 */
    private String baseUrl;

    /** 模型名称。 */
    private String model;

    /** 模型温度。 */
    private Double temperature = 0.2D;

    /** 图片识别提示词；为空时使用内置默认值。 */
    private String recognizeUserMessage;
}
