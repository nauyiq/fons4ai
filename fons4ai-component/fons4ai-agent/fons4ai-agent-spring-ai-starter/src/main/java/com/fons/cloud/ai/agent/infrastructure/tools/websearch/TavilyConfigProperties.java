package com.fons.cloud.ai.agent.infrastructure.tools.websearch;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * @author hongqy
 */
@Getter
@Setter
@ConfigurationProperties("fons4ai.tavily")
public class TavilyConfigProperties {

    /**
     * Tavily 搜索引擎 API Key
     */
    private String apiKey;

    /**
     * Tavily 搜索引擎 MCP URL
     */
    private String mcpUrl;

    /**
     * Tavily 搜索引擎请求超时时间（秒）
     */
    private Integer requestTimeoutSeconds = 300;



}
