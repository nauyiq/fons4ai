package com.fons.cloud.ai.agent.infrastructure.tools;

import cn.hutool.extra.spring.SpringUtil;
import com.fons.cloud.ai.agent.constants.ToolCategory;
import com.fons.cloud.ai.agent.infrastructure.tool.ToolMeta;
import com.fons.cloud.ai.agent.infrastructure.tool.ToolProvider;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.beans.factory.SmartInitializingSingleton;

import java.util.HashMap;
import java.util.Map;

/**
 * 工具注册中心
 * 在工具注册阶段（从 MCP 获取 ToolCallback 后）对工具进行分类打标， 生成 {@link ToolMeta} 元数据，供 Agent 在运行时查询
 * @author hongqy
 */
@Slf4j
public class ToolsRegistry implements SmartInitializingSingleton {

    /**
     * 程序自动装备加载的工具源
     */
    private final Map<String, ToolProvider> providers = new HashMap<>();

    private final Map<String, ToolMeta> toolMetaMap = new HashMap<>();

    /**
     * 自动注册工具
     * @param toolCallbacks
     */
    public void register(ToolCallback[] toolCallbacks) {
        this.register(toolCallbacks, null);
    }

    /**
     * 指定工具源注册工具
     * @param toolCallbacks
     * @param providerName
     */
    public void register(ToolCallback[] toolCallbacks, String providerName) {
        if (toolCallbacks == null || toolCallbacks.length == 0) {
            return;
        }
        if (StringUtils.isNotBlank(providerName)) {
            ToolProvider toolProvider = providers.get(providerName);
            // 存在时直接注册工具
            for (ToolCallback tool : toolCallbacks) {
                String name = tool.getToolDefinition().name();
                String inputSchema = tool.getToolDefinition().inputSchema();
                registerTool(name, inputSchema, toolProvider);
            }
        } else {
            for (ToolCallback tool : toolCallbacks) {
                String name = tool.getToolDefinition().name();
                String inputSchema = tool.getToolDefinition().inputSchema();
                ToolProvider existingProvider = null;
                for (ToolProvider provider : this.providers.values()) {
                    if (provider.supports(name, inputSchema)) {
                        existingProvider = provider;
                        break;
                    }
                }
                registerTool(name, inputSchema, existingProvider);
            }
        }

    }

    /**
     * 根据工具名查找元数据
     * @param toolName
     * @return
     */
    public ToolMeta getToolMeta(String toolName) {
        return toolMetaMap.getOrDefault(toolName, ToolMeta.unknown());
    }

    /**
     * 获取工具提供者
     * @param toolName
     * @return
     */
    public ToolProvider getToolProvider(String toolName) {
        return providers.get(toolName);
    }

    private void registerTool(String toolName, String inputSchema, ToolProvider provider) {
        if (provider == null) {
            // Provider不存在, 注册为未知
            toolMetaMap.put(toolName, new ToolMeta(toolName, inputSchema, ToolCategory.UNKNOWN));
        } else {
            ToolCategory toolCategory = provider.resolveCategory(toolName);
            toolMetaMap.put(toolName, new ToolMeta(toolName, inputSchema, toolCategory));
        }
    }



    @Override
    public void afterSingletonsInstantiated() {
        // 读取所有的ToolProvider
        Map<String, ToolProvider> beans = SpringUtil.getBeansOfType(ToolProvider.class);
        if (MapUtils.isNotEmpty(beans)) {
            for (ToolProvider bean : beans.values()) {
                providers.put(bean.getProviderName(), bean);
            }
        }
    }
}
