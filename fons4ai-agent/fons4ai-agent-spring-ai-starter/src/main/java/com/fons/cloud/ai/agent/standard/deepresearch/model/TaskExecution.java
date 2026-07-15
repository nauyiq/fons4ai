package com.fons.cloud.ai.agent.standard.deepresearch.model;

import com.fons.cloud.ai.tool.model.WebToolResult;

import java.util.List;

/**
 * @author hongqy
 */
public record TaskExecution(TaskResult taskResult, List<WebToolResult> toolResults) {


    public static TaskExecution failed(String id, String message) {
        return new TaskExecution(new TaskResult(id, false, null, message), List.of());
    }
}
