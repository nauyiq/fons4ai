package com.fons.cloud.ai.agent.standard.deepresearch.model;

/**
 * 计划任务
 * @author hongqy
 * @param id          id
 * @param toolName    计划中必须实际调用的工具名称
 * @param instruction 工具调用指令
 * @param order       排序字段
 */
public record PlanTask(String id, String toolName, String instruction, int order) {

    /**
     * 兼容旧的三字段计划任务；旧调用方会在执行前因缺少 toolName 被拒绝。
     */
    public PlanTask(String id, String instruction, int order) {
        this(id, null, instruction, order);
    }
}
