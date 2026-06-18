# CR-xxx <变更标题>

> 功能标识：`<feature-slug>`
> SDD 等级：`S1|S2`
> 文档状态：初稿 | 草案-待确认
> 创建日期：YYYY-MM-DD
> S1 填写规则：只记录受影响范围和必要增量任务；无接口、数据、回滚或风险影响时写 `不适用，原因`。

## 变更意图

说明要变更什么，以及为什么需要变更。

## 影响分析

### 需求影响

- 新增 AC：
- 变更 AC：
- 删除 AC：

### 设计影响

- API/UI/数据/模块影响：

### 代码影响

- 可能受影响的既有文件：
- 可能需要新增的文件：

### 测试影响

- 可能受影响的既有测试：
- 需要新增的测试：

### 知识同步清单

- 项目级索引：
- 领域业务架构：
- 领域技术架构：
- 领域数据架构：
- 知识卡片：
- 其他真理源：
- SQL DDL 当前结构快照：
- DDL 分组：同一数据库/服务 + 强耦合业务模型可合并；不同数据库/服务必须拆分
- SQL DDL 动作：无 | 新增 | 更新 | 重命名
- 存量表原始 DDL：无 | 已存在于 `.specify/sql/<database_or_service>/<business_model>.sql` | 待确认
- 执行型变更 DDL：不适用 | `<project-migration-path>.sql` | `spec/features/<yyyymmdd>/ddl-changes/CR-xxx-<database_or_service>-<business_model>.sql`
- 知识同步标记：Knowledge Sync Needed: no

## 回归与回滚

- 回归风险：
- 回滚方案：

## 实现确认门禁

- 状态：等待用户确认
- 规划产物不等于实现授权。
- 生成本 CR 后必须暂停，等待用户确认后才能进入业务代码实现。
- 用户确认执行且未指定任务 ID 时，默认执行全部未完成增量任务。
- 用户指定任务 ID 时，例如 `执行 T001,T002`，只执行指定任务。

## 文档更新

- `需求说明书.md`：
- `plan.md`：
- `tasks.md`：

## 增量任务

- [ ] Txxx 任务标题
  - AC: AC-xxx
  - Files:
  - Verification:
  - Quality: 确认可读性、DDD-lite/领域建模、方法长度、命名、重复代码、工具复用和依赖门禁
  - Done:

- [ ] Txxx 同步 DDL 知识文件
  - AC: AC-xxx
  - Files: .specify/sql/<database_or_service>/<business_model>.sql; .specify/memory/index.md; .specify/memory/domains/<domain-slug>/数据架构.md
  - Verification: SQL 文件与变更后的同库业务模型/表组一致，并在存在 index.md 和领域 数据架构.md 时完成索引
  - Quality: 确认 SQL 知识文件可读、分组正确，且没有重复写入未证实的结构事实
  - Done: DDL 知识更新完成，或已明确暂缓原因和负责人

- [ ] Txxx 生成存量表执行型变更 DDL
  - AC: AC-xxx
  - Files: spec/features/<yyyymmdd>/ddl-changes/CR-xxx-<database_or_service>-<business_model>.sql | <project-migration-path>.sql
  - Verification: 对照原始 SQL DDL 与目标结构确认 `ALTER TABLE` 或等价语句覆盖本次表结构变更，可供用户复制执行
  - Quality: 执行型变更 DDL 与 `.specify/sql/` 当前结构快照分离维护，并说明执行前置条件和回滚策略
  - Done: 用户确认实现后已生成执行型变更 DDL 文件，或已明确不适用原因
