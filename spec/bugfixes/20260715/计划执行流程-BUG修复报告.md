# 计划执行流程-BUG修复报告

> Bug: `PlanExecuteAgent` 生命周期与状态图失效
> Status: Verified
> Completed: 2026-07-15

## 基本信息

- 模块/功能：`fons4ai-agent-spring-ai-starter` / Deep Research Plan-Execute Agent
- 严重级别：阻断
- 影响范围：计划生成、工具执行、反思重规划、流式输出和引用采集
- 报告人：用户代码审查请求

## 问题描述

- 期望结果：Agent 能从澄清、主题、计划、按波次执行、批判与总结组成的图中正常流转；关闭或取消后才停止。
- 实际结果：新建上下文即被判断为关闭，错误又被当作用户停止而静默结束；重规划图边引用节点描述而非节点名；执行依赖与运行时装配不完整。
- 触发频率：必现。

## 复现步骤

1. 创建 `DeepResearchExecuteContext`，其 `finished` 初始值为 `false`。
2. 调用任一图节点；修复前 `isClose()` 返回 `!finished`，节点立即抛出任务已关闭异常。
3. 构建图；修复前 `compress -> plan` 使用 `COMPRESS.getDesc()`，与注册的 `compress` 节点名不一致。

## 复现环境

- 项目版本：当前工作区
- JDK：Java 21
- 构建工具：Apache Maven 3.9.8
- 外部条件：无需真实模型或工具服务，生命周期和图边错误可由单元测试及图编译复现。

## 根因分析

- 生命周期布尔语义被反向实现，且单任务完成后的关闭判断同样反向。
- 重规划边误用了展示文案字段。
- Agent 缺失可配置构建入口，工具、提示词、线程池和注册表可能为空。
- 每轮波次仅覆盖上一波依赖；模型生成的计划未经结构性校验。
- 可恢复消息上下文未写回图状态；总结输入未包含引用来源。
- 同波次并发调用工具时，父类中的已用工具集合不是线程安全的。

## 修复方案

- 纠正关闭/停止语义和执行后关闭判断，修正 `compress -> plan` 图边。
- 增加 `PlanExecuteAgent.Builder`，集中校验并装配工具、提示词、执行线程池、注册表、检查点和会话记忆。
- 校验任务 ID、指令、波次及重复 ID；无工具时拒绝执行型计划。
- 累积所有成功波次的依赖结果，并将任务结果、批判反馈和压缩快照同步写回图状态。
- 在首次规划前检查上下文长度，压缩后验证结果长度；将去重后的引用传递给最终总结。
- 为并发工具调用保护已用工具集合，并隔离引用解析失败，避免已成功工具调用被引用解析故障中断。
- 回滚方案：回退本报告“变更文件”列出的代码和测试改动；不涉及数据迁移或外部状态变更。

## 变更文件

- `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/deepresearch/PlanExecuteAgent.java`
- `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/deepresearch/PlanExecuteGraph.java`
- `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/deepresearch/model/DeepResearchExecuteContext.java`
- `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/main/java/com/fons/cloud/ai/agent/standard/BaseAgent.java`
- `fons4ai-agent/fons4ai-agent-spring-ai-starter/src/test/java/com/fons/cloud/ai/agent/standard/deepresearch/PlanExecuteAgentGraphTest.java`

## 自动化测试

- RED 证据：修复前的生命周期表达式在 `finished=false` 时直接得出“已关闭”；图边源名称与注册节点名不一致，均为最小失败信号。
- 新增测试：覆盖生命周期、图编译、计划校验和跨波次依赖保留。
- 测试命令：

```powershell
$env:JAVA_HOME='C:\hongqy\C\Java\jdk21'
$env:Path="$env:JAVA_HOME\bin;$env:Path"
& 'C:\hongqy\tool\apache-maven-3.9.8\bin\mvn.cmd' -pl fons4ai-agent/fons4ai-agent-spring-ai-starter -am test -DskipTests=false
```

- 测试结果：通过。`PlanExecuteAgentGraphTest` 4 项、受影响模块回归共 5 项均通过。

## 手动验证

1. 使用 Builder 创建 Agent，并配置至少一个工具和对应 `ToolRegistry`。
2. 发起一个含两个串行波次的研究请求，令第二波引用第一波任务 ID。
3. 确认第二波输入可获得第一波结果，最终响应同时包含正文和引用消息；取消时应停止后续节点。

预期结果：正常请求完整流转；取消仅在取消后生效；异常计划不会覆盖已有任务结果或调度非法波次。

## 回归验证

- 回归范围：Agent 公共基类并发工具记录、工具注册表依赖、Deep Research 图构建。
- 验证结果：Maven Reactor 构建和测试通过。

## 证据清单

| 结论 | 证据来源 | 证据等级 | 状态 |
| --- | --- | --- | --- |
| 复现信号 | 修复前生命周期表达式和图边字段 | L2 | 已验证 |
| 根因判断 | 受影响源码与图状态键定义 | L2 | 已验证 |
| 修复已生效 | Java 21 下 Maven Reactor 测试通过 | L3 | 已验证 |

## 知识库同步

- Knowledge Sync Needed: yes
- 影响的真理源：Agent 编排的任务生命周期、状态图与工具并发约束。
- SQL DDL files: no
- Suggested follow-up: `fons4ai-knowledge-summary`

## 后续事项

- 未接入真实模型和真实工具服务做端到端验证；需按手动验证步骤在目标运行环境补测。
- 检查点恢复的跨请求入口由调用方和图框架配置决定，本次仅保证运行中的消息快照写回图状态，不将其表述为已验证的跨请求恢复能力。
