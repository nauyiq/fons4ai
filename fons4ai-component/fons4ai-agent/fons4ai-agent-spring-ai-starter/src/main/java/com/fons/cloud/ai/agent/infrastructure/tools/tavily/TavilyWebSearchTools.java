package com.fons.cloud.ai.agent.infrastructure.tools.tavily;

import io.modelcontextprotocol.client.McpClient;
import io.modelcontextprotocol.client.McpSyncClient;
import io.modelcontextprotocol.client.transport.HttpClientStreamableHttpTransport;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.mcp.SyncMcpToolCallbackProvider;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.http.HttpHeaders;

import java.net.http.HttpRequest;
import java.time.Duration;

/**
 * Tavily 网络搜索工具
 * @author hongqy
 */
@Slf4j
@RequiredArgsConstructor
public class TavilyWebSearchTools implements InitializingBean {

    /**
     * 执行工具回调
     */
    @Getter
    private ToolCallback[] toolCallbacks;

    /**
     * Tavily 配置属性
     */
    private final TavilyConfigProperties properties;


    @Override
    public void afterPropertiesSet() throws Exception {
        log.info("初始化Tavily网页搜索回调");

        // tavily 搜索引擎
        String authorizationHeader = "Bearer " + properties.getApiKey();

        HttpRequest.Builder requestBuilder = HttpRequest.newBuilder()
                .header(HttpHeaders.AUTHORIZATION, authorizationHeader);

        HttpClientStreamableHttpTransport transport = HttpClientStreamableHttpTransport
                .builder(properties.getMcpUrl())
                .requestBuilder(requestBuilder).build();

        McpSyncClient mcpSyncClient = McpClient.sync(transport).requestTimeout(Duration.ofSeconds(properties.getRequestTimeoutSeconds())).build();
        SyncMcpToolCallbackProvider provider = SyncMcpToolCallbackProvider.builder().mcpClients(mcpSyncClient).build();
        this.toolCallbacks = provider.getToolCallbacks();

        log.info("初始化Tavily网页搜索回调完成，工具数量: {}", this.toolCallbacks.length);
    }
}
