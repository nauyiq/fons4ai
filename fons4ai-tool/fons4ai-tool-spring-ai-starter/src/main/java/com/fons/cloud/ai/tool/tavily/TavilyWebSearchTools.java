package com.fons.cloud.ai.tool.tavily;

import com.fons.cloud.ai.tool.registry.ToolsRegistry;
import io.modelcontextprotocol.client.McpClient;
import io.modelcontextprotocol.client.McpSyncClient;
import io.modelcontextprotocol.client.transport.HttpClientStreamableHttpTransport;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.mcp.SyncMcpToolCallbackProvider;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.beans.factory.InitializingBean;

import java.net.http.HttpRequest;
import java.time.Duration;

/**
 * Tavily Web 搜索工具接入。
 *
 * @author hongqy
 */
@Slf4j
@RequiredArgsConstructor
public class TavilyWebSearchTools implements InitializingBean {

    private static final String AUTHORIZATION_HEADER = "Authorization";

    @Getter
    private ToolCallback[] toolCallbacks;

    private final TavilyConfigProperties properties;
    private final ToolsRegistry toolsRegistry;

    @Override
    public void afterPropertiesSet() {
        log.info("初始化 Tavily Web 搜索回调");
        String authorizationHeader = "Bearer " + properties.getApiKey();
        HttpRequest.Builder requestBuilder = HttpRequest.newBuilder()
                .header(AUTHORIZATION_HEADER, authorizationHeader);
        HttpClientStreamableHttpTransport transport = HttpClientStreamableHttpTransport
                .builder(properties.getMcpUrl())
                .endpoint("/mcp")
                .requestBuilder(requestBuilder)
                .build();
        McpSyncClient mcpSyncClient = McpClient.sync(transport)
                .requestTimeout(Duration.ofSeconds(properties.getRequestTimeoutSeconds()))
                .build();
        SyncMcpToolCallbackProvider provider = SyncMcpToolCallbackProvider.builder()
                .mcpClients(mcpSyncClient)
                .build();
        toolCallbacks = provider.getToolCallbacks();
        toolsRegistry.register(toolCallbacks, new TavilySearchProvider());
        log.info("Tavily Web 搜索回调初始化完成，工具数量: {}", toolCallbacks.length);
    }
}
