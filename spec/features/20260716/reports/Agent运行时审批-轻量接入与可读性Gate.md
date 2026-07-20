# Agent 运行时审批：轻量接入与可读性 Gate

## 1. 负责人只需要理解的四个对象

1. `AgentApprovalPolicy`：下游判断某个工具动作是否需要审批。
2. `HumanInterrupt`：框架返回的安全中断摘要，不包含完整工具参数。
3. `AgentApprovalDecision`：下游完成鉴权后提交 `APPROVE`、`EDIT` 或 `REJECT`。
4. `HumanInTheLoopRuntime`：统一的中断查询和恢复入口；下游不实现 ResumeHandler。

## 2. ReactAgent 实际流程

```text
start/call
  -> BaseAgent 为请求创建独立 AgentRunContext 并注册任务
  -> ReactAgent 调用模型
  -> 无工具：输出最终回答并结束
  -> 有工具：进入 react.before-tool
       -> Policy 直通：执行工具
       -> Policy 要求审批：注册一次性 continuation，返回 WAITING_APPROVAL
  -> 下游调用 HumanInTheLoopRuntime.resume(...)
       -> APPROVE：原参数执行一次
       -> EDIT：只允许校验并替换单个工具的 arguments，然后执行一次
       -> REJECT + TERMINATE：不执行工具，Run 进入 APPROVAL_REJECTED
       -> REJECT + RESUME_WITH_FEEDBACK：不执行工具，把意见作为不可信 Observation 交回模型
  -> 工具结果进入下一轮模型
  -> BaseAgent 统一完成、失败、取消、超时、清理 TaskManager 和 onFinish
```

BaseAgent 不提供 `before-run`、`before-finalize` 等全局审批点，也不保存 Redis、checkpoint
或 Agent 专属恢复路由。工具副作用只从 ReactAgent 的统一获准执行入口发生。

## 3. 最小接入示例

下游提供策略并注入 Runtime：

```java
@Bean
AgentApprovalPolicy approvalPolicy() {
    return (context, options) -> context.point().equals(ReactAgent.BEFORE_TOOL)
            ? ApprovalRequirement.required(
                    "高风险工具", "执行前确认",
                    Set.of(AgentApprovalAction.APPROVE,
                            AgentApprovalAction.EDIT,
                            AgentApprovalAction.REJECT),
                    Duration.ofMinutes(5))
            : ApprovalRequirement.notRequired();
}

ReactAgent agent = ReactAgent.builder(tools, chatModel, taskManager)
        .humanInTheLoopRuntime(humanInTheLoopRuntime)
        .build();
```

非流式调用在中断时立即得到 WAITING：

```java
AgentRunResult waiting = agent.call(request,
        new AgentRunOptions("risk-profile", Map.of()));

if (waiting.getState() == AgentRunState.WAITING_APPROVAL) {
    AgentApprovalDecision decision = AgentApprovalDecision.forInterrupt(
            waiting.getPendingApprovalId(), waiting.getRunId(),
            AgentApprovalAction.APPROVE, "approved", Map.of(), "decision-1");
    Mono<HumanDecisionResult> resumed = humanInTheLoopRuntime.resume(
            waiting.getPendingApprovalId(), decision);
}
```

流式调用使用同一个 Run；`events()` 收到 `approval_required` / `run_paused`，
`completion()` 返回同一个 `WAITING_APPROVAL + interruptId`，恢复入口不变：

```java
AgentRun run = agent.start(request, options);
Flux<String> events = run.events();
Mono<AgentRunResult> firstResult = run.completion();
```

审批 UI、审批人鉴权、多审批人聚合、通知和业务审计属于下游。下游只提交最终决定，
不得把 `actorId` 字符串当作框架授权证明。

## 4. 生命周期和限制

- Runtime 可以被共享 Agent 共用，但每个中断按 `interruptId + runId` 隔离。
- 首个合法决定生效；相同决定幂等复用结果；冲突决定返回 `CONFLICT`。
- 超时进入 `TIMED_OUT`，不会执行原工具；取消进入 `CANCELLED`。
- 自研 Base/React/Web/Plan 的 V1 continuation 只保存在当前进程。应用重启后返回
  `NOT_FOUND`，不会伪造恢复成功。
- Spring AI Alibaba Skills 后续在 T013 使用其原生 thread/Saver/HITL 恢复，不复制本地 continuation。

## 5. 本阶段暂存代码和后续删除清单

为了遵守 T012 后的人工停止门禁，尚未开始迁移的 Skills/Plan 临时继承
`LegacyApprovalAgent`；旧 Coordinator/checkpoint、Redisson、审计、Retention、
RecoveryAdapter/ResumeHandler 及审批专用 Skills 类型仍在仓库中，但 Base/React 新链路
不再引用。它们将在 Gate 通过后的 T013～T016 按依赖顺序删除，不能作为新接入 API。

## 6. 请负责人确认

- 是否能从 `BaseAgent` 看懂 start、WAITING、恢复和终态清理？
- 是否能从 `ReactAgent` 看懂工具审批前无副作用，以及四种决定分支？
- 是否能仅使用 Policy、HumanInterrupt、Decision、Runtime 完成下游接入？
- 是否接受“自研 Agent 仅同进程恢复；Alibaba 使用原生 Saver”的 V1 边界？

四项均确认后，T012 Gate 通过，才可继续 T013～T016。
