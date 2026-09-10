package com.fons.cloud.ai.agent.observability.model;

/**
 * Run 或节点的轨迹状态，不与任一 Agent 框架的状态类型耦合。
 *
 * @author hongqy
 */
public enum TraceStatus {
    /**
     * 权柄已创建但尚未开始执行。
     */
    CREATED,
    /**
     * 正在执行。
     */
    RUNNING,
    /**
     * 执行成功。
     */
    SUCCEEDED,
    /**
     * 执行失败。
     */
    FAILED,
    /**
     * 被调用方主动取消。
     */
    CANCELLED,
    /**
     * 因 HITL 等原因暂停当前执行分段。
     */
    SUSPENDED,
    /**
     * 在进入主要执行链路前被拒绝。
     */
    REJECTED,
    /**
     * 执行超时。
     */
    TIMED_OUT
}
