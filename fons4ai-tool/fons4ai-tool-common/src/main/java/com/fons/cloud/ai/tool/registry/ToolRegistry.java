package com.fons.cloud.ai.tool.registry;

import com.fons.cloud.ai.tool.model.ToolMeta;
import com.fons.cloud.ai.tool.spi.ToolProvider;

/**
 * 工具注册信息的只读查询契约。
 *
 * @author hongqy
 */
public interface ToolRegistry {

    /**
     * 根据工具名查询元数据。
     *
     * @param toolName 工具名
     * @return 工具元数据，不存在时返回未知元数据
     */
    ToolMeta getToolMeta(String toolName);

    /**
     * 根据工具名查询提供者。
     *
     * @param toolName 工具名
     * @return 工具提供者，不存在时返回 null
     */
    ToolProvider getToolProvider(String toolName);
}
