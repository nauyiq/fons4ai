# CR-xxx <变更标题>

> 功能标识：`<feature-slug>`
> 变更类型：微调 | 扩展 | 重构 | 数据结构变更 | 契约变更 | 纯文档修正
> SDD 等级：`S1|S2`
> 文档状态：正式 | 草案-待确认
> 创建日期：YYYY-MM-DD

## 1. 变更摘要

- 变更一句话说明：
- 本次变更结论：
- 是否建议新建 feature：否 | 是，原因

## 2. 变更原因

- 用户诉求：
- 业务或技术背景：
- 不变更的影响：

## 3. 当前状态检查

- 任务规划未完成项：无 | Txxx，处理方式
- 历史 CR 未完成项：无 | CR-xxx/Txxx，处理方式
- 文档与代码一致性：一致 | 存在差异，说明
- 技术设计与当前实现一致性：一致 | 存在差异，说明
- 前置阻塞：无 | 有，说明

## 4. 影响范围

- 需求影响：无 | 有，说明
- 技术设计影响：无 | 有，说明
- 代码影响：无 | 有，涉及文件
- 测试影响：无 | 有，涉及测试
- 接口/契约影响：无 | 有，说明
- 权限/安全影响：无 | 有，说明
- 兼容/回滚影响：无 | 有，说明

## 5. 需求与 AC 变化

- 新增 AC：无 | AC-xxx
- 变更 AC：无 | AC-xxx
- 删除 AC：无 | AC-xxx
- REQ/AC 映射调整：无 | 有，说明

## 6. 技术设计影响

- API/RPC/消息影响：不适用，原因 | 具体影响
- 领域对象/业务规则影响：不适用，原因 | 具体影响
- 状态流转影响：不适用，原因 | 具体影响
- 事务/一致性影响：不适用，原因 | 具体影响
- 工具包/依赖影响：不适用，原因 | 具体影响

## 7. 数据结构与 DDL 影响

- 是否涉及持久化结构变更：否 | 是
- SQL 当前结构快照：不适用 | `.specify/sql/<database_or_service>/<business_model>.sql`
- SQL DDL 动作：无 | 新增 | 更新 | 重命名
- DDL 分组：同一数据库/服务 + 强耦合业务模型可合并；不同数据库/服务必须拆分
- 存量表原始 DDL：无 | 已存在于 `.specify/sql/<database_or_service>/<business_model>.sql` | 待确认
- 执行型变更 DDL：不适用 | `<project-migration-path>.sql` | `spec/features/<yyyymmdd>/ddl-changes/CR-xxx-<database_or_service>-<business_model>.sql`
- DDL 执行方式：不适用 | 用户/DBA 手动执行 | 项目迁移流程执行
- DDL 执行确认：不适用 | 需要用户确认 | 需要只读数据库验证

## 8. 回归与回滚

- 回归风险：无 | 有，说明
- 回归验证范围：
- 回滚方案：不适用，原因 | 具体方案
- S2 风险门禁：不适用 | 需要，说明

## 9. 长期知识影响

- 是否产生长期知识影响：否 | 是
- 影响类型：无 | 业务规则 | 技术方案 | 数据结构 | 接口契约 | 治理规则 | 其他
- 影响说明：
- 处理边界：知识沉淀由 `fons4ai-knowledge-summary` 在用户显式触发后处理，本 CR 不生成知识同步任务或知识汇总交接任务。

## 10. 文档更新

- `需求说明书.md`：不更新 | 更新章节
- `<功能中文名>-技术设计说明书.md`：不更新 | 更新章节
- `<功能中文名>-任务规划.md`：追加任务 Txxx
- 变更记录：已追加 | 不适用，原因

## 11. 增量任务

可执行增量任务必须追加到 `<功能中文名>-任务规划.md`。本节只记录新增任务摘要和任务 ID，便于 CR 追踪。

| 任务 ID | 任务标题 | AC | 追加位置 |
| --- | --- | --- | --- |
| Txxx | 任务标题 | AC-xxx | `<功能中文名>-任务规划.md` |

### 11.1 任务规划追加片段

以下片段应追加到 `<功能中文名>-任务规划.md`，不得只保留在 CR 中。

- [ ] Txxx 任务标题
  - 通俗解释: 这个任务完成后，用户或系统会发生什么可感知变化。
  - AC: AC-xxx
  - 来源: CR-xxx
  - Files:
  - Depends: 无 | Txxx
  - Verification:
  - Quality: 确认可读性、DDD-lite/领域建模、方法长度、命名、重复代码、工具复用和依赖门禁
  - Done:

### 11.2 DDL 任务片段

仅当 `## 7. 数据结构与 DDL 影响` 声明涉及持久化结构变更时生成。

- [ ] Txxx 生成执行型 DDL 草案
  - 通俗解释: 完成后用户或 DBA 可以审核并手动执行数据库结构变更脚本。
  - AC: AC-xxx
  - 来源: CR-xxx
  - Files: spec/features/<yyyymmdd>/ddl-changes/CR-xxx-<database_or_service>-<business_model>.sql | <project-migration-path>.sql
  - Depends: Txxx
  - Verification: 对照原始 SQL DDL 与目标结构，确认 `ALTER TABLE` 或等价语句覆盖本次表结构变更，并标明执行前置条件和回滚策略。
  - Quality: 执行型变更 DDL 与 `.specify/sql/` 当前结构快照分离维护，不包含 MCP/Tool 信息、查询文本或来源路径。
  - Done: 用户确认实现后已生成执行型 DDL 草案，或已明确不适用原因。

- [ ] Txxx 确认 DDL 执行状态
  - 通俗解释: 完成后可以确认数据库结构变更是否已经由用户/DBA 或项目迁移流程执行。
  - AC: AC-xxx
  - 来源: CR-xxx
  - Files: spec/features/<yyyymmdd>/ddl-changes/CR-xxx-<database_or_service>-<business_model>.sql | <project-migration-path>.sql
  - Depends: Txxx
  - Verification: 用户确认 DDL 已执行，或通过只读数据库 MCP/查询确认字段、索引、约束和默认值已生效。
  - Quality: 不由 agent 直接执行生产 DDL；确认记录只描述执行状态和验证证据，不暴露敏感连接信息。
  - Done: DDL 执行状态已确认；若未执行，依赖该结构变更的实现任务不得标记为发布就绪。

- [ ] Txxx 同步 SQL 当前结构快照
  - 通俗解释: 完成后 `.specify/sql/` 中的数据库结构快照与已执行或已确认的真实结构保持一致。
  - AC: AC-xxx
  - 来源: CR-xxx
  - Files: .specify/sql/<database_or_service>/<business_model>.sql
  - Depends: Txxx
  - Verification: 对照已执行 DDL、只读数据库 MCP/查询结果或仓库正式迁移文件，确认 SQL 当前结构快照准确且不包含 MCP/Tool 信息。
  - Quality: SQL 文件按同库业务模型分组，跨库/跨服务不合并，不从实体、Mapper、ORM 注解或 Java 字段推断 DDL。
  - Done: SQL 当前结构快照已在 DDL 执行确认后更新；或已记录暂缓原因、责任方和确认依据。

## 12. 实现确认门禁

- 状态：等待用户确认
- 规划产物不等于实现授权。
- 生成本 CR 和增量任务后必须暂停，等待用户确认后才能进入业务代码实现。
- 用户确认执行且未指定任务 ID 时，默认执行全部未完成任务。
- 用户指定任务 ID 时，例如 `执行 T001,T002`，只执行指定任务。
- 确认执行后默认执行全部未完成任务；如需指定范围，请回复：执行 T001,T002。
