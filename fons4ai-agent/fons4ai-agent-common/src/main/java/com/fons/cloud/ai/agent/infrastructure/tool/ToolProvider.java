package com.fons.cloud.ai.agent.infrastructure.tool;

import com.fons.cloud.ai.agent.constants.ToolCategory;

/**
 * 工具提供者
 * @author hongqy
 */
public interface ToolProvider {

    /**
     * 工具提供者名称, 如 "tavily"、"bing"。
     * 要求整个系统中是唯一的，
     * @return
     */
    String getProviderName();

    /**
     * 根据工具名解析工具类别
     * @param toolName
     * @return
     */
    ToolCategory resolveCategory(String toolName);

    /**
     * 判断某个 MCP 工具是否属于当前提供者
     * <p>
     * 实现方式可以是工具名前缀匹配、输入参数 schema 检测等。
     *
     * @param toolName    工具名
     * @param inputSchema 调用工具所用参数的模式。
     * @return true 表示输入的参数属于该提供者
     */
    boolean supports(String toolName, String inputSchema);

    /**
     * 获取工具返回结果解析器
     * <pre>
     *     支持同一个提供商下的不同工具列表返回的结果类型不同，  因此到了真正业务落地时需要自行关注结果解析器的类型
     * </pre>
     * @param category
     * @return
     * @param <T>
     */
    <T> ToolResultParser<T> getResultParser(ToolCategory category);
}
