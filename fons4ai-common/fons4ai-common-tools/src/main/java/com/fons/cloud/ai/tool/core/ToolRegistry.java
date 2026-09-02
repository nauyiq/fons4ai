package com.fons.cloud.ai.tool.core;

import cn.hutool.core.lang.Assert;
import com.fons.cloud.ai.tool.api.ToolProvider;
import com.fons.cloud.ai.tool.api.ToolResultParser;
import com.fons.cloud.ai.tool.common.model.ToolInfo;
import com.fons.cloud.common.base.exception.SystemIntervalException;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 工具注册中心
 * @author hongqy
 */
@Slf4j
public class ToolRegistry  {
    private final Map<String, ToolProvider> PROVIDERS = new ConcurrentHashMap<>();
    private final Map<String, ToolInfo> ALL_TOOLS = new ConcurrentHashMap<>();

    /**
     * 注册工具提供者
     * @param toolProvider 工具提供者
     */
    public void register(ToolProvider toolProvider, List<ToolInfo> supportToolInfos) {
        Assert.notNull(toolProvider, () -> SystemIntervalException.of("Tool provider cannot be null"));
        Assert.notEmpty(toolProvider.providerName(), () -> SystemIntervalException.of("Tool provider name cannot be empty"));

        // 校验工具提供者名称是否重复， 如果重复注册则抛出异常
        if (PROVIDERS.containsKey(toolProvider.providerName())) {
            throw SystemIntervalException.of("Tool provider already exists");
        }
        // 注册提供者
        PROVIDERS.put(toolProvider.providerName(), toolProvider);
        // 注册工具， 对于大多数业务情况 都是通过工具名直接过去工具信息 然后获取对应的解析器解析工具结果
        for (ToolInfo toolInfo : supportToolInfos) {
            if (ALL_TOOLS.containsKey(toolInfo.toolName())) {
                log.warn("Tool already exists, toolName: {}", toolInfo.toolName());
            } else {
                ALL_TOOLS.put(toolInfo.toolName(), toolInfo);
            }
        }
    }

    /**
     * 获取工具信息
     * @param toolName
     * @return
     */
    public ToolInfo getToolInfo(String toolName) {
        return ALL_TOOLS.get(toolName);
    }

    /**
     * 获取工具提供者
     * @param toolName
     * @return
     */
    public ToolProvider getProvider(String toolName) {
        if (StringUtils.isBlank(toolName)) {
            return null;
        }
        ToolInfo toolInfo = getToolInfo(toolName);
        if (toolInfo == null) {
            return null;
        }
        return PROVIDERS.get(toolInfo.providerName());
    }

    /**
     * 获取工具结果解析器
     * @param toolName 工具名称
     * @return
     */
    public <T>ToolResultParser<T> getToolResultParser(String toolName) {
        ToolInfo toolInfo = getToolInfo(toolName);
        if (toolInfo == null) {
            return null;
        }
        ToolProvider provider = PROVIDERS.get(toolInfo.providerName());
        if (provider == null) {
            return null;
        }
        return provider.getResultParser(toolInfo);
    }

}
