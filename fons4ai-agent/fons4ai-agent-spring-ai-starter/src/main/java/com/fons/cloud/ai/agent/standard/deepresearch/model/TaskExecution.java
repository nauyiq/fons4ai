package com.fons.cloud.ai.agent.standard.deepresearch.model;

import com.fons.cloud.ai.tool.common.model.web.WebBaseResult;

import java.util.List;

/**
 * @author hongqy
 */
public record TaskExecution(TaskResult taskResult, List<WebBaseResult> toolResults) {


    public static TaskExecution failed(String id, String message) {
        return new TaskExecution(new TaskResult(id, false, null, message), List.of());
    }
}
