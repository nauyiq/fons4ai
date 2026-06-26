package com.fons.cloud.ai.rag.config;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * @author hongqy
 */
@Getter
@Setter
@ConfigurationProperties("sys.multiple-modal")
public class MultipleModalConfigProperties {

    private Boolean enabled;
    private String apiKey;
    private String baseUrl;
    private String model;
    private Double temperature = 0.2D;
    private String recognizeUserMessage;


}
