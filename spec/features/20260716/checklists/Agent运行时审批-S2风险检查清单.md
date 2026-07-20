# Agent 运行时审批 S2 风险检查清单

> 功能标识：`agent-runtime-approval`  
> 当前口径：CR-001 轻量、可插拔、默认同进程 HITL  
> 检查日期：2026-07-18

## 1. 公共契约与兼容

- [x] 审批公共契约位于 Agent Common，不依赖 Spring AI、Alibaba 或 Redisson 类型。
- [x] 保留既有 `Agent.start/stream/call` 入口；新增审批通过 `AgentRunOptions` 显式启用。
- [x] 审批默认关闭；未启用的 Run 不创建中断。
- [x] 流式事件和非流式 `WAITING_APPROVAL` 使用同一个 Run 生命周期。
- [x] 公共决定仅保留恢复所需字段，不保存 actor、规则快照或持久化版本。
- [ ] 下游使用真实发布制品后执行二进制兼容检查。

## 2. 状态与副作用边界

- [x] `WAITING_APPROVAL` 为非终态；拒绝终止、超时、取消均为明确终态。
- [x] 首个合法决定只执行一次 continuation；相同幂等键不重复执行，冲突决定拒绝。
- [x] Base 不设置全局审批点，也不保存 checkpoint。
- [x] React/Web 仅在 `react.before-tool` 暂停，审批前工具副作用为 0。
- [x] Skills 仅在业务工具前通过 Alibaba 原生 HITL 暂停；`read_skill` 和白名单资源读取不属于人工审批。
- [x] Plan 仅保留 `after-plan`、`before-task`、`before-report` 三个无副作用阶段点。
- [x] 拒绝默认终止；显式 `RESUME_WITH_FEEDBACK` 时不执行被拒绝动作，意见按不可信输入返回 Agent。

## 3. 共享、隔离与恢复

- [x] Agent 实例只共享构建配置，请求态集中在 `AgentRunContext`。
- [x] Runtime 可共享，Pending 中断按 interruptId/runId 隔离。
- [x] 两个并发 Run 的消息、工具、技能激活、Graph delegate、中断和终态测试互不串用。
- [x] Skills/Plan 复用同一 Run 的 Alibaba thread、Saver 和 checkpoint 继续执行。
- [x] 应用重启后，默认进程内 Runtime 的未决中断明确失效，不伪装成可跨进程恢复。
- [ ] 跨进程公共恢复、恢复所有权竞争和多实例故障转移等待未来独立能力设计，不属于当前实现。

## 4. 安全与下游责任

- [x] 中断事件只暴露技术标识、允许动作和脱敏摘要，不发送完整工具参数。
- [x] EDIT 只允许修改 Agent 明确支持的受控参数；不能新增工具或技能权限。
- [x] Skills 路径、正文和资源大小限制及符号链接逃逸防护保留。
- [x] 自动配置只创建进程内 Runtime，不创建 Redis、加密、审计或清理组件。
- [x] 身份鉴权、业务授权、多审批人聚合、UI、通知、审计及保留期明确归属下游。
- [ ] 真实模型、真实工具和宿主安全策略在首个下游接入时联调。

## 5. 删除与静态检查

- [x] 删除 Redisson 审批 Persistence、PayloadProtector、Retention、Audit/Query、RecoveryRegistry。
- [x] 删除 Base/React/Web/Plan/Skills 专属 ResumeHandler/RecoveryAdapter。
- [x] 删除审批专用 Skills Gate/Permission/Pending 类型及对应失效测试。
- [x] 重型类型引用扫描无残留。
- [x] `git diff --check` 通过，仅有行尾转换提示。

## 6. 自动验证与 Gate

- [x] Agent Common 18 项：0 failure、0 error。
- [x] Agent Starter 79 项：0 failure、0 error、1 项 Windows 符号链接权限跳过。
- [x] Tool Common 1 项通过；Reactor 6 项目 `BUILD SUCCESS`。
- [x] 独立 Spec Review 通过，无 Critical/Important。
- [x] 独立 Code Review 通过，无 Critical/Important。
- [x] 负责人可读性 Gate：2026-07-20 负责人回复“通过”，确认当前代码和接入方式可理解。

## 7. 当前回滚步骤

1. 下游停止给新 Run 设置 `approvalEnabled=true`，或关闭 `fons4ai.agent.approval.enabled`。
2. 对当前进程中仍 Pending 的中断作批准、拒绝、超时或取消处理；不要静默丢弃业务决定。
3. 回滚 HITL 调用入口后继续使用既有非审批 `start/stream/call`。
4. 不需要迁移 Redis 审批数据或审计数据，因为轻量 Runtime 不创建这些持久化结构。

> 本清单不把尚无下游的真实联调包装成已完成证据；自动验证覆盖的是框架内确定性行为。
