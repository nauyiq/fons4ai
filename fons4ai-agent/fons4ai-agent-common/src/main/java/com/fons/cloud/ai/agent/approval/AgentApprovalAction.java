package com.fons.cloud.ai.agent.approval;

/**
 * 下游完成鉴权和业务审批编排后，可以对一次 HITL 中断提交的动作。
 *
 * <p>动作只决定当前已授权步骤如何继续，不能为 Agent 新增工具、技能或资源权限。</p>
 * @author hongqy
 */
public enum AgentApprovalAction {

    /** 按中断发生时冻结的原动作继续执行。 */
    APPROVE,

    /** 使用目标 Adapter 校验后的受控参数替换原动作；不支持编辑的 Adapter 必须拒绝。 */
    EDIT,

    /** 不执行原动作；根据审批点配置终止 Run，或把意见作为不可信反馈交还 Agent。 */
    REJECT
}
