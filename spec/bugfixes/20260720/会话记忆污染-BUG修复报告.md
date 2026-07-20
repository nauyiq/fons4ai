# 会话记忆污染 BUG 修复报告

> Bug: `会话记忆污染`
> Status: Verified
> Completed: 2026-07-20

## 基本信息

- 模块/功能：`fons4ai-agent-spring-ai-starter` / `BaseAgent` 会话记忆生命周期
- 严重级别：高
- 严重级别依据：任务拒绝或执行失败会污染后续对话上下文，成功回答又未被统一记录，直接破坏同一会话后续推理的输入真实性。
- 影响范围：所有启用 `useChatMemory` 且通过 `BaseAgent` 执行的 Agent 实现，包括审批暂停/恢复链路。
- 报告人：Codex

## 问题描述

- 期望结果：只有成功完成的 Run 才将本轮用户消息与最终 Assistant 回答成对写入长期 ChatMemory；拒绝、失败或取消的 Run 不写入；框架已有长期记忆时不重复灌入调用方完整历史；历史消息不得跨会话写入。
- 实际结果：任务注册前即写入历史与用户问题，导致注册拒绝和执行失败仍污染记忆；调用方每轮携带完整历史时重复写入；成功后未统一写入最终 Assistant 回答；历史消息使用自身 `conversationId` 落库且未校验当前会话。
- 首次发现时间：2026-07-20
- 触发频率：必现

## 复现步骤

1. 构造启用 ChatMemory 的 `ReactAgent`，令同一会话的第一次任务注册返回 `CONVERSATION_BUSY`，第二次正常执行。
2. 检查第二次发送给模型的 Prompt，可观察到第一次被拒绝的用户问题。
3. 连续两次携带相同完整历史执行成功请求，可观察到历史重复出现，且第一轮 Assistant 最终回答缺失。
4. 将历史消息的 `conversationId` 改为其他会话，原实现仍会执行模型并把历史写入错误会话。

## 复现环境

- 环境/版本：Windows 工作区，JDK 21，Maven 3.9.8，当前工作树源码
- 账号/角色/权限：不涉及
- 配置/依赖/外部条件：测试使用 Mockito `ChatModel`、内存 ChatMemory 和模拟 `AgentTaskManager`
- 日志/截图/报错信息：RED 阶段 `ReactAgentSharedInstanceTest` 7 个用例中 3 个失败；审批恢复补充用例缺少原始用户消息 `execute`。

## 根因分析

- 关键线索：`BaseAgent.beginRun` 在 `registerTask` 前调用 `prepareChatMemory`；`prepareChatMemory` 直接修改共享 ChatMemory；`finishRun` 没有成功态记忆提交入口。
- 排查路径：沿 `start/call -> beginRun -> prepareChatMemory -> streamExecute -> finishRun` 检查同步、异步、任务拒绝、执行失败及审批恢复分支，并用稳定失败测试验证每个现象。
- 根因说明：会话记忆被当作“执行前输入准备”的副作用，而不是 Run 成功后的提交结果；缺少 Run 级暂存快照和统一提交边界。同时，历史导入没有明确当前会话的所有权校验，也没有定义调用方历史与框架长期记忆的优先级。
- 是否属于需求变更：否，属于现有会话隔离和执行生命周期契约的缺陷修复。

## 修复方案

- 修复策略：任务注册及 Disposable 绑定成功后再准备记忆；在 `AgentRunContext` 暂存本次模型输入与待提交增量；仅 `COMPLETED` 时追加待提交消息及最终 Assistant 回答。
- 最小改动说明：不修改公共请求、响应和 Agent SPI，仅在 `BaseAgent` 与 Run 上下文内部增加暂存/提交逻辑，并补充回归测试。
- 影响评估：拒绝、失败、取消和未完成审批不再写入长期记忆；成功 Run 写入用户消息与 Assistant 回答；已有框架记忆时忽略调用方重复历史；首次成功请求仍可导入调用方历史。
- 风险点：ChatMemory 的 `add` 接口不提供跨多条消息的底层原子事务；当前依赖同会话任务互斥保证提交期间没有并发 Run。记忆提交异常采用告警并保持任务终态收口，避免完成链路被增强能力阻断。
- 回滚方案：回退 `BaseAgent`、`AgentRunContext` 及本报告列出的两个测试文件变更，恢复执行前直接写入 ChatMemory 的旧逻辑。

## 变更文件

- `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/BaseAgent.java`：调整任务注册与记忆准备顺序，增加 Run 成功提交、历史去重和会话校验。
- `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/runtime/AgentRunContext.java`：增加模型输入快照和成功后待提交消息增量。
- `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/test/java/com/fons/cloud/ai/agent/standard/react/ReactAgentSharedInstanceTest.java`：覆盖拒绝、失败、成功提交、历史去重和会话冲突。
- `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/test/java/com/fons/cloud/ai/agent/standard/react/ReactAgentNativeResumeTest.java`：覆盖审批恢复成功后的用户/Assistant 成对记忆。

## 自动化测试

- RED 证据：首次运行新增共享实例测试时 `Tests run: 7, Failures: 3`，分别稳定暴露拒绝污染、历史重复/Assistant 缺失和跨会话历史未拦截；审批恢复补充测试在修复前因下一轮缺少 `execute` 失败。
- 新增/更新测试：新增 4 个共享实例 ChatMemory 用例，并扩展 1 个原生 checkpoint 审批恢复用例。
- 测试命令：`mvn -pl fons4ai-agent/fons4ai-agent-spring-ai-starter -am -Dtest=BaseAgentSharedInstanceTest,ReactAgentSharedInstanceTest,ReactAgentNativeResumeTest,SkillsReactAgentExecutionTest,PlanExecuteAgentGraphTest -Dsurefire.failIfNoSpecifiedTests=false -Dmaven.compiler.forceJavacCompilerUse=true -Dmaven.compiler.useIncrementalCompilation=false test`
- 测试结果：隔离副本、JDK 21 下 `Tests run: 28, Failures: 0, Errors: 0, Skipped: 0`，`BUILD SUCCESS`。
- 若无法自动化测试，原因：不适用。

## 手动验证

1. 在启用 ChatMemory 的真实 Agent 上，以同一 `conversationId` 先触发任务繁忙拒绝或模型失败，再发起成功请求。
2. 检查第二次请求的模型输入与 ChatMemory，确认不存在被拒绝或失败的问题。
3. 连续完成两轮对话，确认第一轮用户消息和 Assistant 最终回答各出现一次；再执行一次审批暂停/批准恢复并做后续追问，确认审批轮消息成对出现。

预期结果：拒绝、失败、取消、等待审批均不污染长期记忆；普通成功与审批恢复成功均在完成后成对提交用户消息和最终回答；跨会话历史在模型执行前失败。当前未连接真实模型或外部持久化 ChatMemory 做人工验证，行为已由自动化测试覆盖。

## 回归验证

- 回归范围：`BaseAgent` 共享实例与终态竞争、React 普通/工具/审批恢复、Skills React、PlanExecute 图执行。
- 验证命令或步骤：执行上述 5 个测试类的 Maven 定向回归。
- 验证结果：28 个用例全部通过；工作区直接编译曾受外部进程删除 `target/classes` 中 class 文件干扰，因此最终证据来自不含 `.git`、`.idea`、`target` 的隔离副本，同一源码和依赖环境下验证成功。

## 证据清单

| 结论 | 证据来源 | 证据等级 | 状态 |
| --- | --- | --- | --- |
| 复现信号 | RED 自动化测试：共享实例 3 个失败，审批恢复缺少原始用户消息 | L3 | 已验证 |
| 根因判断 | `beginRun`、`prepareChatMemory`、`finishRun` 代码路径与失败用例逐项对应 | L3 | 已验证 |
| 修复已生效 | 隔离环境最终定向回归 28/28 通过，Maven `BUILD SUCCESS` | L3 | 已验证 |

## 知识库同步

- Knowledge Sync Needed: yes
- 影响的真理源：`.specify/memory/项目运行架构文档.md` 中仍描述先初始化会话记忆、再登记任务的旧顺序，需要改为任务注册成功后准备 Run 级记忆、成功完成后提交。
- SQL DDL files: no
- DDL grouping: 不适用
- Suggested follow-up: `fons4ai-knowledge-summary`

## 后续事项

- 按需执行 `fons4ai-knowledge-summary`，同步运行架构文档中的 ChatMemory 生命周期口径。
- 若后续替换为外部持久化 ChatMemory，评估提供批量/事务提交 API，以获得多条消息的存储层原子性。
