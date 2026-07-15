package com.fons.cloud.ai.agent.standard.deepresearch.model;

/**
 * 任务结果
 * @author hongqy
 * @param taskId  任务ID
 * @param success 是否成功
 * @param output  输出结果
 * @param error   错误信息
 */
public record TaskResult(String taskId, boolean success, String output, String error) {
}
