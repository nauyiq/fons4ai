package com.fons.cloud.ai.agent.standard.deepresearch.model;

/**
 * 计划任务
 * @author hongqy
 * @param id          id
 * @param instruction 简介
 * @param order       排序字段
 */
public record PlanTask(String id, String instruction, int order) {
}
