package com.fons.cloud.ai.agent.infrastructure.tool;

import java.util.List;

/**
 * 工具返回结果解析策略
 * @author hongqy
 */
public interface ToolResultParser<T> {

    /**
     * 解析工具返回的原始 JSON，提取统一的结果模型。
     * @param result 工具返回的原始字符串
     * @return 解析后的结果列表
     */
    List<T> parse(String result);

}
