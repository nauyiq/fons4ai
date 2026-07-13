package com.fons.cloud.ai.tool.spi;

import com.fons.cloud.ai.tool.constants.ToolCategory;

/**
 * 工具提供者。
 *
 * @author hongqy
 */
public interface ToolProvider {

    /**
     * 返回系统内唯一的工具提供者名称。
     *
     * @return 工具提供者名称
     */
    String getProviderName();

    /**
     * 根据工具名解析工具分类。
     *
     * @param toolName 工具名
     * @return 工具分类
     */
    ToolCategory resolveCategory(String toolName);

    /**
     * 判断工具是否属于当前提供者。
     *
     * @param toolName 工具名
     * @param inputSchema 工具输入参数模式
     * @return 是否支持
     */
    boolean supports(String toolName, String inputSchema);

    /**
     * 返回指定分类的结果解析器。
     *
     * @param category 工具分类
     * @param <T> 解析结果类型
     * @return 结果解析器
     */
    <T> ToolResultParser<T> getResultParser(ToolCategory category);
}
