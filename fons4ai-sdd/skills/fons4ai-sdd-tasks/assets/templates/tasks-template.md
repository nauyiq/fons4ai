# <功能名称> 任务拆解

> 功能标识：`<feature-slug>`
> SDD 等级：`S1|S2`
> 来源需求：`specs/features/<feature-slug>/spec.md`
> 来源方案：`specs/features/<feature-slug>/plan.md`
> 文档状态：初稿

## 执行策略

- MVP 范围：
- 任务依赖：
- 可并行分组：

## 实现确认门禁

- 状态：等待用户确认
- 规划产物不等于实现授权。
- 生成本 `tasks.md` 后必须暂停，等待用户确认后才能进入业务代码实现。
- 用户确认执行且未指定任务 ID 时，默认执行全部未完成任务。
- 用户指定任务 ID 时，例如 `执行 T001,T002`，只执行指定任务。

## 任务列表

- [ ] T001 用动宾短语描述任务
  - AC: AC-001
  - Files: path/to/source; path/to/test
  - Verification: 执行聚焦测试或手动检查
  - Quality: 确认可读性、DDD-lite/领域建模、方法长度、命名、重复代码、工具复用和依赖门禁
  - Done: 客观完成标准

## S2 质量门禁

仅 S2 使用本节。

- [ ] T999 回归或回滚门禁
  - AC: AC-001
  - Files: path/to/source; path/to/test
  - Verification: 执行聚焦测试、回归测试或手动检查
  - Quality: 确认风险控制代码保持可读，遵守 DDD-lite/领域边界，并只使用已批准工具或依赖
  - Done: 风险控制已验证

## 知识与 DDL 同步任务

当 `plan.md` 声明真理源或 `.specify/sql/` 影响时使用本节。涉及持久化数据模型新增或变更时，DDL 同步任务是必需项，除非 `plan.md` 已明确记录暂缓原因、负责人和用户确认。

- [ ] Txxx 同步 DDL 知识文件
  - AC: AC-xxx
  - Files: .specify/sql/<database_or_service>/<business_model>.sql; .specify/memory/data-architecture.md
  - Verification: 确认 SQL 文件与已实现的同库业务模型/表组一致，并在存在 data-architecture.md 时完成索引
  - Quality: 确认 SQL 知识文件可读、分组正确，且没有重复写入未证实的结构事实
  - Done: DDL 知识更新完成，或已明确暂缓原因和负责人

- [ ] Txxx 汇总知识影响
  - AC: AC-xxx
  - Files: <truth-source-path>
  - Verification: 确认长期架构或数据事实已通过合适的文档流程更新
  - Quality: 确认知识文本简洁、可追溯、不重复，且不写入未验证假设
  - Done: 知识更新完成，或已明确暂缓原因和负责人
