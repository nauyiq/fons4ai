package com.fons.cloud.ai.tool.registry;

import com.fons.cloud.ai.tool.constants.ToolCategory;
import com.fons.cloud.ai.tool.model.ToolMeta;
import com.fons.cloud.ai.tool.spi.ToolProvider;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.tool.ToolCallback;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

/**
 * Spring AI 工具注册中心。
 *
 * @author hongqy
 */
public class ToolsRegistry implements ToolRegistry {

    private final Map<String, ToolProvider> providers = new HashMap<>();
    private final Map<String, ToolMeta> toolMetaMap = new HashMap<>();

    public ToolsRegistry(Collection<ToolProvider> toolProviders) {
        if (toolProviders != null) {
            for (ToolProvider toolProvider : toolProviders) {
                providers.put(toolProvider.getProviderName(), toolProvider);
            }
        }
    }

    /**
     * 自动识别提供者并注册工具。
     *
     * @param toolCallbacks Spring AI 工具回调
     */
    public void register(ToolCallback[] toolCallbacks) {
        register(toolCallbacks, "");
    }

    /**
     * 使用指定提供者注册工具。
     *
     * @param toolCallbacks Spring AI 工具回调
     * @param toolProvider 工具提供者
     */
    public void register(ToolCallback[] toolCallbacks, ToolProvider toolProvider) {
        if (toolCallbacks == null || toolCallbacks.length == 0) {
            return;
        }
        for (ToolCallback tool : toolCallbacks) {
            registerTool(
                    tool.getToolDefinition().name(),
                    toolProvider);
        }
    }

    /**
     * 使用提供者名称注册工具。
     *
     * @param toolCallbacks Spring AI 工具回调
     * @param providerName 工具提供者名称；为空时自动识别
     */
    public void register(ToolCallback[] toolCallbacks, String providerName) {
        if (toolCallbacks == null || toolCallbacks.length == 0) {
            return;
        }
        if (StringUtils.isNotBlank(providerName)) {
            ToolProvider toolProvider = providers.get(providerName);
            for (ToolCallback tool : toolCallbacks) {
                registerTool(
                        tool.getToolDefinition().name(),
                        toolProvider);
            }
            return;
        }

        for (ToolCallback tool : toolCallbacks) {
            String name = tool.getToolDefinition().name();
            String inputSchema = tool.getToolDefinition().inputSchema();
            ToolProvider matchedProvider = providers.values().stream()
                    .filter(provider -> provider.supports(name, inputSchema))
                    .findFirst()
                    .orElse(null);
            registerTool(name, matchedProvider);
        }
    }

    @Override
    public ToolMeta getToolMeta(String toolName) {
        return toolMetaMap.getOrDefault(toolName, ToolMeta.unknown());
    }

    @Override
    public ToolProvider getToolProvider(String toolName) {
        ToolMeta toolMeta = getToolMeta(toolName);
        return providers.get(toolMeta.providerName());
    }

    private void registerTool(String toolName, ToolProvider provider) {
        if (provider == null) {
            toolMetaMap.put(toolName, new ToolMeta(toolName, "unknown", ToolCategory.UNKNOWN));
            return;
        }
        ToolCategory toolCategory = provider.resolveCategory(toolName);
        toolMetaMap.put(toolName, new ToolMeta(toolName, provider.getProviderName(), toolCategory));
    }

}
