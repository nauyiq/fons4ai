# Agent 运行时审批 Spec Review

> 功能标识：`agent-runtime-approval`  
> 当前方案：`CR-001` 轻量可插拔 HITL  
> 评审角色：独立 Spec Reviewer  
> 评审日期：2026-07-18  
> 评审结论：**PASS**

## 1. 结论

当前实现符合 CR-001 的收缩目标：框架只提供审批点、Policy、中断契约、同进程 Runtime、Run 状态及恢复回调，不再实现 Redisson 审批平台、跨进程恢复编排、审计平台或各 Agent 专属恢复协议。

独立复审未发现 Critical 或 Important。2026-07-20 负责人回复“通过”，人工可读性 Gate 已完成。

## 2. 需求与范围核对

| 核对项 | 结果 | 证据摘要 |
| --- | --- | --- |
| 默认关闭与显式启用 | 通过 | `AgentRunOptions.approvalProfileId` 为空时不进入 Policy；自动配置默认关闭。 |
| 统一轻量契约 | 通过 | Common 只保留 Policy、Interrupt、Decision、Runtime、Requirement 等公共类型。 |
| 流式与非流式 | 通过 | 流式输出审批事件；非流式返回 `WAITING_APPROVAL`；二者共享同一 Run。 |
| 决定语义 | 通过 | 支持批准、受控编辑、拒绝终止、拒绝携带意见恢复；多工具批次不开放 EDIT。 |
| 同进程一致性 | 通过 | 首决定、幂等重试、冲突、超时、取消和 discard 均有确定状态。 |
| 共享 Agent 隔离 | 通过 | 请求态位于 RunContext；React、Plan、Skills 共享实例测试覆盖隔离。 |
| Agent 审批点 | 通过 | React/Web 工具前；Skills 原生工具中断；Plan 三个阶段点；Base 不设置全局审批点。 |
| 安全边界 | 通过 | 事件只含安全摘要；完整参数留在 continuation/原生 metadata；审批不能提升技能或工具权限。 |
| 延期能力边界 | 通过 | 跨进程恢复、审批 UI、鉴权、多人聚合、通知、审计和保留治理明确归属未来扩展或下游。 |

## 3. 复审中关闭的问题

- 已解决回执不再长期持有 continuation，只保留短期幂等结果。
- Pending 转 Resolved 时先发布回执、再移除 Pending，消除查询不可见窗口。
- 多工具批次允许动作收窄为 `APPROVE/REJECT`，不再暴露无法安全映射的 EDIT。
- Graph Saver 注释已明确：持久化 checkpoint 不等于公共中断可跨重启恢复。
- 重型错误码、事件、Persistence、Recovery、Audit/Query 类型及失效测试已清理。

## 4. Evidence

- clean 全量回归：Common 18、Tool Common 1、Starter 79，共 98 项；0 failure、0 error、1 Windows 符号链接权限跳过。
- Reactor：6 个模块全部 `SUCCESS`。
- 重型审批类型源码/测试引用扫描：通过。
- Common 轻量审批契约供应商边界扫描：通过。
- `git diff --check`：通过，仅有行尾转换提示。

## 5. Gate

- Spec Review：通过。
- Critical/Important：无。
- 下游真实业务、多实例宿主和审批 UI 联调：按用户决策等待首个接入方，不作为当前框架仓库阻塞项。
- 人工 Gate：2026-07-20 负责人回复“通过”，已确认能够理解触发、中断、恢复以及应用重启后的行为。
