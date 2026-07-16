# 智能体实例共享 Code Review

> 评审日期：2026-07-16  
> 评审角色：独立 Code Reviewer  
> 评审范围：公共 `Agent/AgentRun` 契约、`BaseAgent`、`AgentTaskManager`、ReAct、WebSearch、Plan-Execute、Skills、相关自动化测试  
> 评审边界：只审查实现是否满足既定需求和技术设计，不重新裁判需求；真实 Redis 多实例联调不作为本次框架任务门禁。

## 1. 门禁结论

**结论：阻塞。**

当前实现已经建立“共享 Agent + 每请求 RunContext”的主体结构，公共冷流入口、结构化同步结果、精确任务句柄、Redis compare-and-delete、各 Agent 的请求态迁移方向基本正确；现有 Surefire 报告显示 38 个测试均无失败。

但评审确认存在 1 个 Critical 和 6 个 Important。Critical 是 `RUNNING` 到任务注册之间的取消丢失竞态，会使 `AgentRun.cancel()` 对仍在执行的目标 Run 返回失败并继续执行。Important 还涉及 ReAct 工具取消、Plan checkpoint 释放顺序、Skills 资源快照授权一致性、符号链接枚举边界、技能目录快照的无界预加载，以及关键验收项缺少确定性测试。上述问题关闭前不建议通过 Code Review Gate。

## 2. 评审证据

- 已逐项静态审查生产代码和测试代码，并对照需求 `REQ-001` 至 `REQ-008`、验收标准 `AC-001` 至 `AC-010` 和技术设计中的运行、取消、安全、测试约束。
- 工作区已有 Surefire XML 报告共 38 个测试，`failures=0`、`errors=0`、`skipped=0`。
- 本评审尝试重新执行 `mvn.cmd -pl fons4ai-agent/fons4ai-agent-spring-ai-starter -am test`，但当前沙箱无法访问 Maven Central，且本地缺少父 POM `com.fons.cloud:fons4cloud:1.0.0`，因此未形成新的独立构建结果。这是评审环境依赖问题，不判定为代码失败。
- 未要求真实 Redis 多实例环境；Redis 结论来自代码契约和替身测试审查。

## 3. Critical

### C-01：Run 进入 RUNNING、任务尚未注册时取消会丢失

- 状态：**已验证（代码路径确定）**
- 定位：
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/BaseAgent.java:88`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/BaseAgent.java:95`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/BaseAgent.java:142`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/BaseAgent.java:152`
- 现象：`beginRun()` 先通过 `tryStart()` 把状态改成 `RUNNING`，随后才准备记忆并注册任务。若另一个线程恰在两者之间调用 `AgentRun.cancel()`，`cancelRun()` 会进入 RUNNING 分支并直接调用 `agentTaskManager.cancelTask(handle)`；由于任务尚未进入 `taskMap`，该调用返回 `false`，取消意图没有写入 RunContext，也不会在后续注册后重试，执行继续启动模型或 Graph。
- 影响：违反 `REQ-003`、`BR-005`、`AC-005`。用户明确取消目标 Run 时可能仍产生模型调用、工具副作用和费用；取消返回值也与实际运行状态不一致。
- 建议：在 RunContext 中增加原子的 cancellation-request 状态，把取消意图与 TaskManager 是否已经注册解耦。`beginRun()` 在记忆准备后、注册后、绑定 Disposable 前均检查取消意图；注册成功后若已请求取消，必须走精确 handle 的主动停止收口。不要只用“`cancelTask()` 返回 false 后再直接完成”修补，否则仍可能在注册竞态中留下已登记任务或启动底层执行。
- 必要测试：使用门闩卡住 `prepare/register` 窗口，确定性并发调用 `cancel()`，验证最终状态只能为 `CANCELLED`、模型/Graph/工具没有启动、任务租约没有残留、Hook 与 sink 只完成一次。

## 4. Important

### I-01：ReAct 工具任务未纳入 Run 的取消资源树

- 状态：**已验证（代码路径确定）**
- 定位：
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/react/ReactAgent.java:269`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/react/ReactAgent.java:276`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/react/ReactAgent.java:297`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/react/ReactAgent.java:317`
- 现象：每个工具调用通过 `Schedulers.boundedElastic().schedule(...)` 独立调度，但返回的调度 `Disposable` 没有保存到 `ReactAgentRunContext`，`ReactAgent` 也没有覆盖 `onRunCancelled()` 设置 `finalResultSent`。取消只会中断当前模型订阅，已经排队或运行的工具仍可能产生副作用、记录工具结果并触发迟到的 `onComplete`/下一轮调度。
- 影响：用户停止后工具仍可能访问外部系统；共享 Agent 虽不会直接串写另一个 RunContext，但目标 Run 的资源和副作用没有按生命周期释放，违反 `REQ-003`、`AC-006` 和资源释放设计。
- 建议：为每个 ReAct Run 建立组合 Disposable/工具任务集合；取消时先设置本 Run 的终止标志，再取消未开始/可中断的工具调度，并禁止完成回调继续调度下一轮。对不可中断的同步 `ToolCallback` 明确“最佳努力取消”契约，并在工具返回后丢弃迟到结果。

### I-02：Plan-Execute 在 Graph 订阅真正取消前释放 checkpoint

- 状态：**已验证（调用顺序确定），实际 saver 并发后果为推断**
- 定位：
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/deepresearch/PlanExecuteAgent.java:369`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/deepresearch/PlanExecuteAgent.java:376`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/runtime/AgentRunContext.java:227`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/runtime/AgentRunContext.java:233`
- 现象：`RunCancellation.dispose()` 明确先执行上层 cancellation handler、后 dispose 原生订阅；而 Plan 的 cancellation handler 会立即调用 `checkpointSaver.release()`。因此 Graph 仍可能处于节点执行或 checkpoint 写入期间，checkpoint 已被释放。
- 影响：具体 saver 若不支持“运行中 release”，可能发生写回已释放状态、恢复点残留或释放/保存竞态。
- 建议：取消 handler 只标记 `finished`；将 checkpoint 释放放到 Graph `doFinally` 或等价的原生终止回调中，并增加请求级 release-once 标志，保证先停止 Graph、再释放本 Run 的 checkpoint，正常、异常、取消竞争时仍只执行一次。

### I-03：Skills 的资源访问未绑定到本 Run 固定的 CatalogSnapshot

- 状态：**已验证（对象关系确定），目录热变更后的越权结果为推断**
- 定位：
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/skill/SkillsReactAgent.java:79`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/skill/SkillsReactAgent.java:82`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/skill/SkillsReactAgent.java:137`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/skill/SkillsReactAgent.java:152`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/skill/FileSystemSkillResourceResolver.java:35`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/skill/FileSystemSkillResourceResolver.java:127`
- 现象：技能正文和激活依据来自每 Run 的 `SkillCatalogSnapshot`，但资源工具调用共享的 `SkillResourceResolver`。默认文件系统 Resolver 自己持有另一个 Registry，并在每次读取时重新查询当前 `skillPath`。当 `autoReload` 或外部 Registry 更新发生时，旧 Run 已激活的同名技能可能读取到新目录资源，而不是其启动时固定目录。
- 影响：破坏“运行固定快照”和授权依据稳定性；同名技能被重新绑定目录时，旧 Run 的既有授权可能扩展到新资源集合。
- 建议：让 Resolver 的访问显式接收本 Run 的资源目录快照/版本令牌，或按 Run 创建绑定 `SkillCatalogSnapshot` 的安全 Resolver 视图。资源授权校验必须同时覆盖 `runId + skillName + catalogVersion/rootIdentity`，不能只检查技能名。
- 必要测试：Run A 激活技能后更新源 Registry 并启动 Run B，验证 A 仍只能读取旧快照资源，B 才能读取新资源。

### I-04：文件资源列表可通过符号链接读取授权目录外目标的元数据

- 状态：**代码语义推断，当前测试未覆盖**
- 定位：
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/skill/FileSystemSkillResourceResolver.java:64`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/skill/FileSystemSkillResourceResolver.java:68`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/skill/FileSystemSkillResourceResolver.java:188`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/skill/FileSystemSkillResourceResolver.java:193`
- 现象：直接读取路径会通过 `toRealPath()` 检查逃逸，这是正确的；但 `list()` 对遍历项只执行词法 `root.relativize(path)` 的 `isAllowed()`。`toDescriptor()` 的 `Files.isDirectory/size/probeContentType` 默认跟随符号链接，因此位于 `references/` 下、指向技能根外文件的链接仍可能返回外部目标的类型和大小。
- 影响：虽然正文读取会被拒绝，但仍违反“阻止符号链接逃逸”的资源枚举边界，并泄漏根目录外文件元数据。
- 建议：资源列表明确跳过 `Files.isSymbolicLink(path)`，或对每个条目执行真实路径校验后再描述；所有 `Files.*` 属性读取应统一使用不跟随链接策略。增加文件链接和目录链接的 list/read/describe 覆盖测试。

### I-05：SkillCatalogSnapshot 在安全上限校验前无界读取所有技能正文

- 状态：**已验证（代码路径确定）**
- 定位：
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/skill/SkillCatalogSnapshot.java:37`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/skill/SkillCatalogSnapshot.java:45`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/skill/SkillCatalogSnapshot.java:51`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/skill/GuardedSkillRegistry.java:97`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/skill/GuardedSkillRegistry.java:103`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/skill/GuardedSkillRegistry.java:152`
- 现象：`capture()` 先遍历 Registry 并读取每个技能的完整正文，之后才构造 `GuardedSkillRegistry` 检查 50 技能上限和正文字符上限。超量或超大正文已经进入内存；`autoReload=true` 时每个新 Run 都重复全量读取。限制使用 `String.length()`，也不是设计中的字节上限。
- 影响：外部 Registry 可在被拒绝前造成显著内存和 I/O 放大；同时每请求全量加载违背“轻量 Run”和渐进加载的性能目标。
- 建议：先读取并验证受限元数据列表，再按需延迟读取正文；若为了运行快照必须固定正文，应在读取流阶段实施单技能字节上限和总目录字节上限，超过立即中止，不能先完整加载。`autoReload` 应原子发布一次已验证目录版本，而不是由每个并发 Run 重复预热全文。

### I-06：AC-004/AC-005/AC-006/AC-009 的确定性自动化证据不足

- 状态：**已验证（测试代码审查）**
- 定位：
  - `fons4ai-agent/fons4ai-agent-common/src/test/java/com/fons/cloud/ai/agent/core/AgentTaskManagerTest.java:29`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/test/java/com/fons/cloud/ai/agent/standard/BaseAgentSharedInstanceTest.java:29`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/test/java/com/fons/cloud/ai/agent/standard/react/websearch/WebSearchReactAgentContractTest.java:27`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/test/java/com/fons/cloud/ai/agent/standard/deepresearch/PlanExecuteAgentGraphTest.java:99`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/test/java/com/fons/cloud/ai/agent/standard/skill/SkillsReactAgentContractTest.java:119`
- 缺口：
  1. 没有正常完成、异常、主动取消三方并发竞争的门闩测试，也没有 C-01 注册窗口取消测试。
  2. TaskManager 只验证远程停止“发送”，未通过捕获 Topic listener 验证目标实例只取消精确 runId；缺少旧 handle 迟到 cancel、TTL 归属变化和 CAS 失败保留新租约测试。
  3. WebSearch、Plan-Execute、Skills 的“共享实例并发”多数只检查 RunContext/集合对象不同，没有让同一真实 Agent 执行两个受控并发链路并验证输出、Hook、工具、引用和技能权限不串用。
  4. 没有覆盖 text/thinking/reference/recommend/error/stop 全部现有 JSON 协议的契约快照；当前只覆盖部分文本和停止内容。
- 影响：38 个测试通过不能完整证明 `AC-004`、`AC-005`、`AC-006`、`AC-009`，多个核心竞态会在现有测试全部通过的情况下存在。
- 建议：按技术设计 §6 的矩阵补充确定性替身测试；不要求真实 Redis，多实例行为可用两个 `AgentTaskManager`、共享原子 Bucket 和捕获的 Topic listener 模拟。

## 5. Suggestion

### S-01：`call()` 不消费事件流，unicast sink 会无界缓存全部流片段

- 状态：**已验证**
- 定位：`fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/runtime/AgentRunContext.java:33`、`fons4ai-agent/fons4ai-agent-common/src/main/java/com/fons/cloud/ai/agent/api/Agent.java:41`
- 建议：为非流式执行提供不保留客户端事件的 Run sink 策略，或为事件缓冲设置明确上限；最终上下文仍由同一执行链路聚合，不能另起模型调用。

### S-02：`stream(request)` 的下游订阅取消不会自动取消 AgentRun

- 状态：**已验证**
- 定位：`fons4ai-agent/fons4ai-agent-common/src/main/java/com/fons/cloud/ai/agent/api/Agent.java:27`、`fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/runtime/DefaultAgentRun.java:42`
- 建议：明确兼容入口的取消契约。若客户端断开应停止模型/Graph，可在 defer 内保存 Run 并将 `doOnCancel` 绑定到 `run.cancel()`；若只允许显式 `AgentRun.cancel()`，应在 API 文档中清楚声明，避免资源泄漏预期不一致。

### S-03：共享扩展对象的线程安全约束尚未在 Builder API 中显式表达

- 状态：**已验证**
- 定位：`ReactAgent` 的共享 `ToolCallback/Advisor`、`SkillsReactAgent` 的共享 `commonTools/nativeHooks/resourceResolver`、`BaseAgent` 的共享 `AgentChatHook`。
- 建议：在公共 Builder 和 JavaDoc 明确这些实例必须无请求状态且线程安全；对需要请求态的扩展提供 per-run factory。否则框架完成自身字段迁移后，业务扩展仍可能把请求态带回共享实例。

### S-04：Skills Builder 对 `toolExecutionTimeout` 未校验正值

- 状态：**已验证**
- 定位：`fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/skill/SkillsReactAgent.java:572`
- 建议：与 Plan 的 `taskTimeout` 一致，拒绝 zero/negative Duration，在构建期快速失败。

## 6. 已确认的正确实现

- `Agent.stream(request)` 使用 `Flux.defer`，每次订阅创建新 Run；`call(request)` 只通过同一 Run 的 `completion()` 等待结果，没有第二条模型执行链。
- `AgentChatRequest.snapshot()` 对 params 和 history 元素进行了防御性复制。
- `AgentRunContext.tryFinalize()` 使用 CAS 保证终态不可逆；任务完成使用精确 `AgentTaskHandle`。
- `AgentTaskManager.completeTask()` 不再发送“用户停止”消息；Redis 删除使用租约完整值的 compare-and-delete，迟到 complete 不会删除后续 Run。
- ReAct、WebSearch、Plan-Execute 的主要请求字段已迁移到各自 RunContext；Plan 的 Graph threadId 包含 runId，自建线程池只在 `close()` 关闭。
- Skills 每 Run 创建独立 Guarded Registry、Alibaba delegate 和激活集合；技能专属工具在执行入口再次检查本 Run 激活状态。
- ChatMemory 的当前问题由 BaseAgent 统一写入，现有 ReAct/Plan/Skills 输入路径没有再次显式追加当前问题。
- 日志抽查未发现完整请求、提示词、工具参数、技能正文或模型答案被直接序列化记录。

## 7. 复审条件

1. 修复 C-01，并补充确定性注册窗口取消测试。
2. 修复或明确关闭 I-01 至 I-05 的代码问题。
3. 补齐 I-06 中不依赖真实下游系统的框架级替身测试。
4. 重新执行 Agent Reactor 全量测试并提供通过证据。
5. 完成后进行独立 Code Re-Review；真实 Redis 多实例和下游业务联调不列为本次复审前置。

## 8. 2026-07-16 Code Re-Review

### 8.1 复审结论

**最终门禁结论：仍然阻塞。**

本轮修改已经关闭原评审中的大部分问题：ReAct 工具任务已进入 Run 取消资源树，Plan checkpoint 已移动到 Graph `doFinally` 并增加 release-once，默认文件资源解析器已绑定 Run 快照并过滤符号链接，技能目录先限制数量并按 UTF-8 字节限制正文，公共流取消、非流式事件排空、超时参数校验和多数验收测试也已补齐。

但是 **C-01 只关闭了“取消意图丢失”，没有关闭“注册过程中取消导致任务租约残留”**。在真实 `AgentTaskManager.registerTask()` 的 Redis 租约写入与本地 `taskMap.putIfAbsent()` 之间仍存在一个更窄的竞态窗口；当前新增测试使用 mock TaskManager，未模拟该有状态窗口。因此 Code Review Gate 仍被 1 个 Critical 阻塞。真实 Redis 多实例和下游业务联调继续不作为当前门禁。

### 8.2 新鲜验证证据

- Agent Reactor 全量构建：`BUILD SUCCESS`。
- Common：11 tests；Tool：1 test；Starter：40 tests；合计 52 个测试用例，失败 0、错误 0。
- `FileSystemSkillResourceResolverTest` 的 Windows 符号链接创建用例因当前权限条件跳过 1 项；生产代码静态检查已确认 list 过滤 `Files.isSymbolicLink(path)`。
- Common 供应商类型扫描通过。
- 共享 Agent 请求态字段扫描通过。
- 本轮复审直接检查了最终生产代码、测试代码和 Surefire XML；未要求真实 Redis 多实例环境。

### 8.3 原问题逐项状态

| 编号 | 复审状态 | 复审结论与证据 |
| --- | --- | --- |
| C-01 | **未关闭，仍为 Critical** | `AgentRunContext` 已增加 `cancellationRequested`，`BaseAgent` 也在执行边界重复检查，取消不会再启动模型；但取消若发生在真实 `registerTask()` 已写 Redis、尚未写本地 map 的窗口，取消 handler 的 `completeTask()` 查不到任务，注册返回后再次 dispose 已终止的 cancellation 也不会重新清理，最终留下本地任务和 Redis lease。详见 §8.4。 |
| I-01 | **已关闭** | `ReactAgent.onRunCancelled()` 先封闭轮次推进；每个 boundedElastic 工具任务通过 `trackDisposable()` 纳入 Run 资源树；工具返回后和聚合回调前均校验 Run 仍为 RUNNING，迟到结果被丢弃。`ReactAgentSharedInstanceTest.cancellingRunMustDiscardLateToolResultAndPreventNextRound` 覆盖该行为。 |
| I-02 | **已关闭** | Plan cancellation handler 只标记执行上下文停止，不再 release；checkpoint 由 Graph `doFinally` 释放，`checkpointReleased` CAS 保证一次。同步订阅创建失败单独释放。新增测试验证 cancellation handler 不提前释放。 |
| I-03 | **已关闭** | `SkillResourceResolver` 增加 `forRun(snapshot)` 契约；默认 `FileSystemSkillResourceResolver` 返回绑定 `SkillCatalogSnapshot` 的新实例，旧 Run 不再读取 reload 后的新路径。新增 run-bound resolver 测试覆盖目录切换。自定义可变 Resolver 必须按接口 JavaDoc 覆盖该方法。 |
| I-04 | **已关闭（自动化证据受 Windows 权限限制）** | `FileSystemSkillResourceResolver.list()` 明确过滤符号链接，read/describe 仍使用真实路径边界校验。符号链接测试已编写，但本机无创建权限时通过 assumption 跳过；静态代码契约可确认修复存在。 |
| I-05 | **主体关闭，剩余限制降级为 Suggestion** | Snapshot 现在先检查技能数量和重复名称，再读取正文；正文和 Guarded Registry 均按 UTF-8 字节上限拒绝，测试覆盖“超量目录不读取任何正文”和多字节字符。由于 Alibaba `SkillRegistry` 只提供返回完整 `String` 的接口，单个恶意实现仍可在返回前分配超大字符串；当前 Registry 属于应用配置级可信依赖，此限制不继续作为当前门禁，后续可扩展 bounded/streaming registry SPI。 |
| I-06 | **大部分关闭，剩余测试缺口降级为 Suggestion；C-01 对应缺口仍随 Critical 阻塞** | 已新增终态竞争、远程 listener 精确 runId、完整响应协议、Plan 实际并发 Graph、Skills 实际并发授权、ReAct 取消和 ChatMemory 隔离测试。尚缺真实状态 TaskManager 的“register 内部窗口取消”测试；WebSearch 仍以 RunContext 集合隔离和单次执行为主，没有独立的双请求实际并发引用测试；TTL/CAS 的少数失败分支也未全部形成门闩测试。 |
| S-01 | **已关闭** | `Agent.call()` 订阅并排空同一 Run 的事件流，再等待 `completion()`，避免无人订阅的 unicast sink 缓存全部片段，同时没有创建第二条执行链。 |
| S-02 | **已关闭** | `Agent.stream()` 在每次 defer 创建 Run 后用 `doOnCancel(run::cancel)` 绑定客户端断开；新增公共契约测试验证取消只作用于所属 Run。 |
| S-03 | **部分关闭，保留 Suggestion** | `Agent` 顶层 JavaDoc 已明确共享实现只能保存稳定配置和线程安全依赖，`SkillResourceResolver.forRun` 也声明可变实现责任；具体 Builder 的 `ToolCallback`、Advisor、Hook 参数仍可进一步标注线程安全或提供 per-run factory。 |
| S-04 | **已关闭** | Skills Builder 已拒绝 zero/negative `toolExecutionTimeout`。 |

### 8.4 仍阻塞的 Critical

#### C-01R：取消发生在 TaskManager 注册内部窗口时会残留任务与租约

- 状态：**已验证（并发时序由代码确定）**
- 定位：
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/BaseAgent.java:103`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/BaseAgent.java:112`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/BaseAgent.java:163`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/BaseAgent.java:186`
  - `fons4ai-agent/fons4ai-agent-common/src/main/java/com/fons/cloud/ai/agent/core/AgentTaskManager.java:127`
  - `fons4ai-agent/fons4ai-agent-common/src/main/java/com/fons/cloud/ai/agent/core/AgentTaskManager.java:138`
  - `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/test/java/com/fons/cloud/ai/agent/standard/BaseAgentSharedInstanceTest.java:112`
- 精确时序：
  1. Run 进入 `RUNNING`，调用真实 `registerTask()`。
  2. `bucket.setIfAbsent(leaseValue, ttl)` 成功，但执行尚未到 `taskMap.putIfAbsent()`。
  3. 另一线程调用 `cancel()`；`cancellationRequested` 成功置位，但 `cancelTask(handle)` 因本地 map 尚无任务返回 false。
  4. cancellation handler 先把 Run 收口为 `CANCELLED` 并调用 `completeTask(handle)`；此时仍查不到本地任务，Redis lease 未删除。
  5. 原注册线程继续把 TaskInfo 放入本地 map并返回成功。
  6. `BaseAgent` 的注册后检查发现取消，但 cancellation disposable 已在步骤 4 终止，重复 `dispose()` 不会再次触发 handler 或 `completeTask()`；方法返回且不会启动模型，但本地 TaskInfo 和 Redis lease 保留到 TTL/销毁。
- 当前测试为何没有捕获：`cancellationBetweenRunningAndRegistrationMustNotBeLost` 用 mock `AgentTaskManager` 阻塞 `registerTask()`，取消时没有真实写入状态；mock 返回成功后也不会形成需要二次清理的本地 TaskInfo/lease，因此只证明“模型不会启动和结果为 CANCELLED”，没有证明“注册产物被释放”。
- 修复建议：注册成功后的取消分支必须无条件对精确 handle 再执行一次清理，而不能只重复 dispose 已终止的 RunCancellation。可以让 `stopBeforeExecutionWhenCancelled(context, registered=true)` 显式调用 `agentTaskManager.completeTask(handle)`，或把“注册后取消补偿”做成 TaskManager 的原子协议；无论采用哪种方式，都必须保持 compare-and-delete 和精确 runId。
- 必要测试：使用有状态 TaskManager 替身或可控 Bucket，把执行卡在 `setIfAbsent` 成功与 `taskMap.putIfAbsent` 之间；取消后放行注册，断言 Run 为 CANCELLED、模型未启动、本地 `hasRunningTask(conversationId)` 为 false、lease 为空、Hook/sink/清理各一次。该测试不需要真实 Redis 多实例。

### 8.5 最终复审建议

1. 修复 C-01R 的注册后补偿清理并增加有状态确定性测试。
2. 重新运行 52 项 Agent Reactor 测试；新增竞态测试后测试总数应相应增加。
3. 对 C-01R 做一次独立定点 Re-Review。若无新增 Critical/Important，可将 Code Review Gate 调整为“通过”。
4. I-05、I-06、S-03 的剩余内容作为后续增强建议，不要求真实下游接入，也不阻塞本框架变更。

## 9. 2026-07-16 第二次 Code Re-Review

### 9.1 定点复审范围

本次仅复核 §8.4 的 C-01R，不重新扩展审查范围：

- `registerTask()` 返回后的取消补偿是否会再次精确清理任务；
- 注册内部窗口的有状态测试是否能够证明注册产物被清理；
- 最新全量回归是否仍然通过。

### 9.2 C-01R 关闭验证

**状态：已关闭。**

生产代码已将注册后的普通取消检查替换为 `stopAfterRegistrationWhenCancelled(context)`：

- `BaseAgent.java:103-115`：`registerTask()` 成功返回后立即进入专用取消补偿分支；
- `BaseAgent.java:201-207`：若取消意图已登记，先调用 `agentTaskManager.completeTask(handle)`，再 dispose 请求级取消句柄；
- `BaseAgent.java:210-215`：取消 handler 之前已执行过的首次 `completeTask()` 仍保持幂等，注册后第二次精确清理负责覆盖“首次清理时任务尚未进入本地 map”的窗口。

该顺序关闭了上轮报告中的完整竞态：即使取消 handler 已经把 Run 收口为 `CANCELLED`，注册线程在真正产生本地 TaskInfo/Redis lease 后仍会按同一 `conversationId + runId` 再执行一次 compare-and-delete 清理，不会启动模型、Graph 或工具。

测试也从无状态 mock 行为升级为有状态注册产物模拟：

- `BaseAgentSharedInstanceTest.java:112-140` 在 `registerTask()` 返回前使用门闩制造取消窗口；
- 注册放行时把 `registered` 设置为 true，模拟已经产生本地任务/租约；
- `completeTask()` 通过 CAS 把该状态从 true 清为 false；
- 最终断言 Run 为 `CANCELLED`、执行次数为 0、注册状态为 false，且 `completeTask()` 被调用两次，分别对应取消 handler 的首次尝试和注册返回后的补偿清理。

复审同时检查了相邻窗口：取消发生在注册后检查与 `setDisposable()` 之间时，`cancelTask()` 会先移除精确任务；随后 `setDisposable()` 失败并 dispose 请求级取消句柄，仍由统一 handler 收口为 `CANCELLED`，不会转成 FAILED 或继续执行。

### 9.3 回归证据

- Maven 全量回归：52 个测试条目；51 通过，1 个 Windows 符号链接权限条件跳过；0 failures，0 errors。
- 跳过项与 C-01R 无关，且其生产代码边界已在 §8.3 的 I-04 中完成静态确认。
- 本次没有要求真实 Redis 多实例或下游业务接入；C-01R 通过有状态确定性替身覆盖。

### 9.4 最终门禁结论

**Code Review Gate：通过。**

- 原报告的 Critical/Important 已全部关闭，或已明确降级为不阻塞当前框架交付的后续 Suggestion。
- C-01R 的任务注册窗口、取消终态和精确补偿清理已经形成生产代码闭环及确定性回归证据。
- 真实 Redis 多实例联调和下游业务联调不属于当前框架技术项目的任务门禁，可在未来实际接入阶段执行。

当前剩余非阻塞建议仍为：为可信 SkillRegistry 之外的实现补充 bounded/streaming SPI、继续扩展 WebSearch 双请求实际并发测试和少数 TTL/CAS 失败分支测试、在具体 Builder 上进一步标注共享扩展线程安全或提供 per-run factory。这些事项不影响本次 Code Review 通过。
