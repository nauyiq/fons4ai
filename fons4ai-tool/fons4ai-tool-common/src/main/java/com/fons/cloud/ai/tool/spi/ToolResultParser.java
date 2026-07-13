package com.fons.cloud.ai.tool.spi;

import java.util.List;

/**
 * 工具返回结果解析策略。
 *
 * @param <T> 解析后的结果类型
 * @author hongqy
 */
public interface ToolResultParser<T> {

    /**
     * 解析工具返回的原始内容。
     *
     * @param result 工具返回的原始字符串
     * @return 解析后的结果列表
     */
    List<T> parse(String result);
}
