package com.fons.cloud.ai.tool.common.model;

import com.fons.cloud.ai.tool.common.constants.ToolCategory;

/**
 * 工具信息
 * @param toolName      工具名称
 * @param providerName  工具提供者名称
 * @param category      工具类别
 * @author hongqy
 */
public record ToolInfo(String toolName, String providerName, ToolCategory category){





}
