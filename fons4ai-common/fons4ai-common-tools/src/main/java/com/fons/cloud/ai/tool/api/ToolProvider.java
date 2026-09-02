package com.fons.cloud.ai.tool.api;

import com.fons.cloud.ai.tool.common.constants.ToolCategory;
import com.fons.cloud.ai.tool.common.model.ToolInfo;

/**
 * 工具提供者
 * @author hongqy
 */
public interface ToolProvider {

    /**
     * 工具提供者名称
     */
    String providerName();

    /**
     * 判断工具是否属于当前提供者
     * @param toolName    工具名
     * @param inputSchema 工具输入参数结构
     * @return            是否支持
     */
    boolean isSupport(String toolName, String inputSchema);

    /**
     * 根据工具名称获取工具类别
     * @param toolName
     * @return
     */
    ToolCategory getToolCategory(String toolName);

    /**
     * 获取工具结果解析器
     * @param toolInfo 工具信息
     */
    <T> ToolResultParser<T> getResultParser(ToolInfo toolInfo);

}
