# Agent 运行时审批实施报告

> 功能标识：`agent-runtime-approval`  
> 任务范围：`T014-T016`  
> 实现确认依据：用户已授权执行全部剩余任务，并于 2026-07-20 回复“通过”完成人工可读性 Gate  
> SDD 等级：`S2`  
> 完成日期：2026-07-20

## 1. 实施摘要

- 已完成任务：T014、T015、T016，包含实现、自动验证、Spec Review、Code Review 和人工可读性 Gate。
- 未完成任务：无。
- 阻塞任务：无。
- 实施结果：全部完成，框架仓库交付已收口。
- 是否可交付完成：是，当前 SDD 范围内的框架代码、测试、评审和人工 Gate 均已完成。
- 是否发布就绪：否，框架无下游接入，本轮不伪造真实业务联调结论。
- UI 设计确认状态：不适用，本功能不提供审批页面。
- 用户跳过设计确认：否。

## 2. 最终实现与变更范围

本轮交付的是轻量可插拔 HITL 边界，不是审批平台：

```text
下游启用 Profile 并提供 Policy
  -> Agent 到达自身副作用前审批点
  -> Runtime 生成 HumanInterrupt，Run 进入 WAITING_APPROVAL
  -> 下游完成展示、鉴权和业务审批后调用 Runtime.resume(...)
  -> APPROVE/EDIT 恢复原动作
     REJECT+TERMINATE 结束 Run
     REJECT+RESUME_WITH_FEEDBACK 不执行原动作，将不可信意见交回 Agent
```

主要变更文件范围：

- `fons4ai-agent-common/.../api/`、`approval/`、`constants/`：Run、审批公共契约和协议事件。
- `fons4ai-agent-spring-ai-starter/.../standard/BaseAgent.java`、`runtime/`、`approval/`：共享生命周期与进程内 Runtime。
- `standard/react/`、`react/websearch/`、`deepresearch/`、`skill/`：四条 Agent 审批流程及逐 Run 隔离。
- `autoconfigure/`、`resources/`：轻量自动配置及单一开关。
- 对应 Common/Starter 自动化测试、Review、报告和风险清单。

## 3. TDD / 等价验证记录

| 任务 | RED | GREEN | REFACTOR |
| --- | --- | --- | --- |
| T014 | Plan/Web 聚焦用例覆盖阶段中断、恢复和统一工具点位 | 24 项聚焦回归通过 | 删除专属恢复协议，收敛到公共入口 |
| T015 | 注释清单和既有源码可读性问题 | 公共 API、字段、入口和主流程注释补齐 | 删除过时/重复注释，保留关键不变量说明 |
| T016 | 独立 Review 与 clean 编译发现并发、快照、代次、缺失 import 和名称遮蔽问题 | 逐项修复并新增负向测试 | 删除重型实现、失效测试和死字段，最终复审通过 |

## 4. 验证结果

验证在隔离临时副本执行，避免 Trae Java Language Server 锁定或覆盖主工作区 `target`；副本排除了 `.git` 和所有旧 `target`，源码来自最终工作区。

```text
mvn -pl fons4ai-agent/fons4ai-agent-spring-ai-starter -am clean
    -Dmaven.compiler.forceJavacCompilerUse=true
    -Dmaven.compiler.useIncrementalCompilation=false test
```

- Common：18 项，0 failure，0 error。
- Tool Common：1 项，0 failure，0 error。
- Starter：79 项，0 failure，0 error，1 项 Windows 符号链接权限跳过。
- 合计：98 项执行，0 failure，0 error，1 skip。
- Reactor：6 个模块全部 `SUCCESS`，`BUILD SUCCESS`。
- `git diff --check`：通过，仅有 LF/CRLF 转换提示。
- 重型审批类型源码/测试扫描：无残留。
- Common 轻量审批契约供应商边界扫描：通过。
- 验证证据等级：L3。

## 5. Evidence Bundle

| 项目 | 内容 |
| --- | --- |
| 任务来源 | `spec/features/20260716/Agent运行时审批-任务规划.md`，T014-T016；用户明确授权继续执行全部剩余任务 |
| 变更范围 | Agent Common、Starter、四类 Agent 适配、测试和 SDD 实施产物 |
| 测试说明 | 覆盖默认关闭、流式/非流式 WAITING、批准、受控编辑、两种拒绝、超时、取消、幂等、共享隔离、Plan Graph、Skills 原生恢复与资源安全 |
| AC 覆盖 | T014-T016 对应 AC-001～AC-023 中 CR-001 保留范围均有源码、测试或 Review 证据 |
| Review 状态 | Spec Review：通过；Code Review：通过；无 Critical/Important |
| 人工 Gate | 2026-07-20 负责人回复“通过”，已完成 |
| 风险声明 | 公共 Runtime 默认只支持同进程恢复；跨进程、UI、鉴权、多人聚合、审计和真实宿主联调未交付 |

## 5.1 服务级 Evidence Matrix

不适用。当前交付为库/Starter，不新增独立可运行服务、HTTP/RPC/消息入口或部署单元。

## 6. Review 与人工 Gate

- Implementer 自检：已完成。
- Spec Review：通过；轻量范围、Agent 点位、同进程边界和延期能力符合 CR-001。
- Code Review：通过；并发、取消、幂等、快照和 Skills generation 问题均已修复并复审。
- Critical/Important 问题：已修复并复审，无未关闭项。
- 人工 Gate 适用性：适用，T016 明确要求负责人确认代码和接入方式可理解。
- 人工 Gate 状态：已通过；2026-07-20 负责人回复“通过”。
- 可交付完成判断：是；T016 已勾选，当前 SDD 范围已完成。

## 6.1 Harness 校验结果

- 校验来源：当前 `fons4ai-sdd-implement` 契约和独立 Reviewer。
- 校验命令：clean Maven 回归、静态残留扫描、供应商边界扫描、`git diff --check`。
- 校验结果：自动验证与独立 Review 通过。
- 未验证项：下游真实接入、多实例宿主、审批页面和业务鉴权；按用户决策暂不启动。
- 是否阻塞当前框架交付：否；T016 已通过人工 Gate 并完成。

## 7. 各 Agent 流程与 AC 覆盖

| Agent/范围 | 审批点与恢复方式 | 验证摘要 |
| --- | --- | --- |
| Base | 无全局审批点；只管理 Run、事件、等待、恢复和终态 | 流式/非流式、取消竞态、共享隔离 |
| React | `react.before-tool`；同进程 continuation | 审批前副作用为 0，批准/编辑/两种拒绝、批次动作限制 |
| Web | 复用 `react.before-tool` | 搜索、抓取统一走 React 工具入口 |
| Skills | `skills.before-tool`；Alibaba interruption/ToolFeedback | 原生恢复、delegate 代次、技能/工具权限不抬升 |
| Plan | `plan.after-plan`、`plan.before-task`、`plan.before-report` | checkpoint 后继恢复、目标副作用单次执行、并发隔离 |

## 8. 代码质量复盘

- 可读性、方法职责、命名、异常和日志检查：通过。
- 状态与不变量归属：Runtime 的 Pending/Resolved 内部对象维护首决定、幂等、超时和 discard 规则；Agent 只编排审批点。
- 共享边界：构建配置可共享，请求态全部进入 RunContext 或 Pending continuation。
- 基础设施边界：公共审批契约不依赖 Alibaba、Spring AI 或 Redisson；供应商能力只在 Starter 适配层。
- 新增依赖：否。
- 测试可读性：通过，新增用例集中描述竞态、快照和原生订阅代次。

## 9. DDL、DML 与数据验证状态

- DDL：不涉及。
- DML/Seed：不涉及。
- 字段映射或外部数据流：不涉及。
- 持久化结构：CR-001 已删除原 Redisson 审批平台方案；当前 Runtime 只保存进程内短期状态。

## 10. S2 门禁关闭情况

- Checklist：自动项、独立 Review 和人工 Gate 均已关闭。
- 回滚方案：已记录，可通过关闭审批开关或不传 Profile 回退到原流程。
- 兼容性：既有 `start/stream/call` 保留；真实发布制品二进制兼容由下游首次接入验证。
- 安全/权限：框架只传安全摘要；身份鉴权、业务授权和多人聚合归属下游。
- 事务/数据库：不适用。

## 11. 长期知识影响

- 是否产生长期知识影响：是。
- 影响类型：技术方案、公共接口契约和 Agent 编排边界。
- 影响说明：框架审批能力确定为轻量、可插拔、默认同进程；Alibaba 适配优先复用原生 HITL。
- 是否已修改知识库正文：否。
- 处理边界：仅在用户显式触发 `fons4ai-knowledge-summary` 后沉淀。

## 12. 风险与后续事项

- 当前 `InMemoryHumanInTheLoopRuntime` 不支持应用重启后恢复未决中断。
- 跨进程恢复、恢复所有权、多实例故障转移需要未来独立设计，不能由 Saver 配置隐式承诺。
- 审批 UI、通知、身份、业务权限、多审批人、审计和保留期由下游编排。
- Mockito 动态 agent 警告属于未来 JDK 构建兼容事项；当前 JDK 21 不阻塞。
- 当前 SDD 范围无剩余必做任务；下游真实接入、多实例宿主和审批 UI 等仍按未来接入需求独立规划。

## 13. CR-002 恢复逻辑与权限快照实施记录（2026-07-20）

> 当前任务范围：`T017-T019`  
> 实现确认依据：用户先确认“执行方案二”，随后明确回复“开始执行任务”  
> 当前结论：实现和聚焦测试已修改，自动化回归受依赖环境阻塞；T017-T019 暂不勾选。

### 13.1 已实施变更

- T017：新增 Starter 内部 `AlibabaResumeSupport`，React、Plan、Skills 统一复用 threadId 关联校验、checkpoint 查找和原生反馈配置；`AgentResumeRequest` 的公共字段和 `resume(...)` 签名保持不变，并在命令边界拒绝空决定。
- T018：Skills 在产生原生中断时，把目录指纹和已激活技能名作为最小权限快照写入原 checkpoint；恢复时先校验当前目录指纹，再恢复原激活集合。目录 reload、快照缺失或格式不一致会明确失败，审批决定本身不能授予新工具或资源权限。
- 未新增审批 Runtime、Store、配置开关、下游字段或持久化系统；Saver 和 `AgentResumeRequest` 接入方式不变。

### 13.2 TDD 与验证证据

| 任务 | 新增/扩展测试 | 当前状态 |
| --- | --- | --- |
| T017 | threadId 不匹配时不得访问 Saver、checkpoint 缺失异常、空决定边界 | 用例已写入，未获得 L3 执行证据 |
| T018 | 原激活技能恢复、reload 后目录不一致明确拒绝、新 Run 可见新目录 | 用例已写入，未获得 L3 执行证据 |

- `git diff --check`：通过。
- 公共契约静态复核：`AgentResumeRequest`、`ResumableAgent.resume`、APPROVE/EDIT/REJECT 枚举、Saver 注入入口未删除或改签名。
- Maven/离线验证阻塞原因与 `智能体实例共享-实施报告.md` §14.2 的记录一致：私有父 POM和当前 Alibaba/Spring BOM 不在可信本地缓存，安全策略禁止向 Maven Central 解析私有坐标。
- 证据等级：L2；未获得当前最终源码的 L3 编译和测试运行证据。

### 13.3 Review、Gate 与后续

- Spec Review：待执行。
- Code Review：待执行。
- 人工 Gate：待执行。
- T019：保持未完成；依赖恢复后需运行 React/Plan/Skills 原生恢复聚焦测试和完整 Starter 回归，再执行独立 Review 与用户 Gate。
- 长期知识：本轮未修改知识库正文。
