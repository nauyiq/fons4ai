package com.fons.cloud.ai.agent.approval;

/** 审批拒绝后的框架处理方式。 */
public enum ApprovalRejectionMode {
    /** 安全默认值：拒绝后终止原 Run。 */
    TERMINATE,
    /** 将拒绝意见作为不可信反馈恢复原 Run，由 Agent 继续响应或重新规划。 */
    RESUME_WITH_FEEDBACK
}
