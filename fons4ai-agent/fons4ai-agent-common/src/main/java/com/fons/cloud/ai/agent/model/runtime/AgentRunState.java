package com.fons.cloud.ai.agent.model.runtime;

/**
 * 智能体执行状态。WAITING_APPROVAL 表示顶层Run当前执行分段已结束，并且被一个或
 * 多个人工审批阻塞；审批目标可以是顶层Agent，也可以是其内部执行单元。其余终态不可逆。
 *
 * @author hongqy
 */
public enum AgentRunState {

    /**
     * 已创建但尚未首次订阅执行。
     */
    CREATED,

    /**
     * 正在执行模型、工具或 Graph。
     */
    RUNNING,

    /**
     * 顶层Run当前执行分段已暂停，等待一个或多个人工审批完成。
     */
    WAITING_APPROVAL,

    /**
     * 正常完成并形成最终回答。
     */
    COMPLETED,

    /**
     * 执行异常失败。
     */
    FAILED,

    /**
     * 收到用户或下游取消。
     */
    CANCELLED,

    /**
     * 启动前被任务占用等框架条件拒绝。
     */
    REJECTED,

    /**
     * 最终决定为拒绝且点位配置为终止。
     */
    APPROVAL_REJECTED,

    /**
     * 审批等待超时。
     */
    TIMED_OUT;

    /**
     * @return 当前状态是否已经结束
     */
    public boolean isTerminal() {
        return this == COMPLETED || this == FAILED || this == CANCELLED || this == REJECTED
                || this == APPROVAL_REJECTED || this == TIMED_OUT;
    }

    /**
     * @return 当前状态是否已不允许任何执行活动。WAITING_APPROVAL 表示当前流分段已结束、
     * 分段的订阅/租约/事件流均已收口，框架不得再绑定或发射任何资源，但其并非不可逆终态，
     * 仍可被取消、超时或审批拒绝终结。
     */
    public boolean isSegmentEnd() {
        return isTerminal() || this == WAITING_APPROVAL;
    }
}
