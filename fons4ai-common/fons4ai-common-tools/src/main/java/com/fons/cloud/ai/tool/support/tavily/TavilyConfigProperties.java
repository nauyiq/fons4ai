package com.fons.cloud.ai.tool.support.tavily;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Tavily 工具配置。
 * @author hongqy
 */
@Getter
@Setter
@ConfigurationProperties("sys.tavily")
public class TavilyConfigProperties {

    /** Tavily API Key。 */
    private String apiKey;

    /** Tavily MCP URL。 */
    private String mcpUrl;

    /** 请求超时时间，单位为秒。 */
    private Integer requestTimeoutSeconds = 300;

}
