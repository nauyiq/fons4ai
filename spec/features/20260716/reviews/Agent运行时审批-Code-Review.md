# Agent 运行时审批 Code Review

> 功能标识：`agent-runtime-approval`  
> 当前方案：`CR-001` 轻量可插拔 HITL  
> 评审角色：独立 Code Reviewer  
> 评审日期：2026-07-18  
> 评审结论：**PASS**

## 1. 结论

最终工作区未发现 Critical 或 Important。审批状态、取消竞态、请求快照、Skills 原生订阅代次和异常收口问题均已修复，并由 clean 编译和自动化测试验证。

## 2. 重点审查结果

| 审查项 | 结果 | 说明 |
| --- | --- | --- |
| 首决定与幂等 | 通过 | 回执校验 interruptId、runId、允许动作、idempotencyKey 和决定摘要；相同键同摘要幂等，其余冲突。 |
| discard/decide 竞争 | 通过 | 两者在同一 `PendingInterrupt` 对象锁上线性化；先取得锁的一方成为确定结果。 |
| 取消/恢复竞争 | 通过 | Base 在恢复前、取得恢复权后、注册后、绑定后均重新消费取消意图。 |
| 终态数据 | 通过 | 真实终态清空 `pendingApprovalId`，WAITING 快照才携带审批 ID。 |
| Policy 属性快照 | 通过 | Map/List/Set 递归冻结；数值只允许明确不可变类型，拒绝 `AtomicInteger` 等可变 Number。 |
| Skills 旧订阅 | 通过 | generation 条件绑定；旧订阅只能释放自己，不能覆盖恢复订阅。 |
| Skills 异常 | 通过 | 中断使旧 generation 失效后，Policy、Runtime 和直通恢复异常在当前调用栈显式结束 Run。 |
| React 清理 | 通过 | 无用待审批轮次字段、方法及 `capturePendingRound` 遗留调用均已删除。 |
| 公共契约边界 | 通过 | 轻量审批 API 不依赖 Spring AI、Alibaba 或 Redisson 类型。 |

## 3. 评审发现及关闭记录

- 编译阻断：`ReactAgent` 遗留 `capturePendingRound` 调用；已删除。
- 竞态：旧版 discard 先删除再标记；已改为对象锁内标记并条件删除。
- 可变快照：任意 Number 可穿透；已改为不可变数值白名单并补负向测试。
- Skills 异常吞没：旧 generation 过滤了真实异常；已增加当前调用栈显式收口。
- Skills 同步订阅覆盖：外层订阅可能覆盖嵌套恢复订阅；已增加 generation 条件绑定，初始 `streamExecute` 不再由 Base 二次绑定。
- clean 编译额外发现缺失 import 和内部 `result()` 名称遮蔽；均已修复，最终复核确认不改变运行语义。

## 4. 验证证据

- clean Maven 全量回归：98 项执行，0 failure、0 error、1 skip；6 个 Reactor 模块 `BUILD SUCCESS`。
- 新增覆盖：可变 Number 拒绝、错误回执关联、Policy 异常收口、旧 generation 条件绑定、取消等待后不可恢复、多工具不允许 EDIT。
- `git diff --check`：通过。
- 重型类型与 `capturePendingRound` 引用扫描：无源码/测试残留。

## 5. 非阻塞事项

- Mockito 提示未来 JDK 将禁止动态自附加 Java Agent；当前 JDK 21 测试通过，建议后续统一构建配置中显式配置 Mockito agent。
- 1 项文件系统符号链接测试在 Windows 权限不足时按既有条件跳过，不影响路径规范化和普通越权用例。
- 人工可读性 Gate 已由负责人于 2026-07-20 回复“通过”完成；Code Reviewer 的技术结论与人工 Gate 证据分别保留。
