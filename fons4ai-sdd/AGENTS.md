# AGENTS.md

<!-- fons4ai-skill-routing: enabled -->

> 适用范围：本仓库及复制本文件的同类项目
> 规则状态：轻量入口 + Fons4AI 默认流程

# 项目简介

Fons4AI SDD 技能库，用于约束 AI agent 按“需求澄清、技术设计、任务拆解、确认执行、知识沉淀”的方式进行规范驱动开发。

## 快速导航

| 你想做什么 | 去哪里看 |
| --- | --- |
| 了解项目级业务、技术、数据事实 | `.specify/memory/index.md` |
| 查看业务领域细节 | `.specify/memory/domains/<domain-slug>/业务架构.md` |
| 查看领域技术落地 | `.specify/memory/domains/<domain-slug>/技术架构.md` |
| 查看领域数据设计 | `.specify/memory/domains/<domain-slug>/数据架构.md` |
| 查看最小事实单元 | `.specify/memory/domains/<domain-slug>/cards/` |
| 查看 agent 通用运行约束 | `.specify/rules/agent运行规则.md` |
| 查看代码编写约束 | `.specify/rules/代码编写规范.md` |
| 查看 SDD 功能产物 | `spec/features/<yyyymmdd>/` |
| 查看 BUG 修复报告 | `specs/bugfixes/<bug-slug>/bugfix-report.md` |

## 硬性规则

- 默认使用中文沟通、编写任务和交付说明。
- 修改代码或文档前，必须先读取相关上下文；不得凭空猜测业务逻辑、接口、字段、表结构或第三方 API。
- 必须优先遵循项目事实、知识库、规则文件和已有代码风格，而不是模型默认习惯。
- 必须优先复用已有代码、工具类、组件、规则和依赖；不得重复造轮子。
- 不得修改与当前需求无关的代码、文档、配置或格式。
- 不得引入新框架、新模块、新依赖或新技术路线，除非需求、方案或用户明确确认。
- 删除文件、删除核心逻辑、大范围重构、覆盖既有文档、修改数据库结构前，必须获得用户确认。
- 不得提交密钥、令牌、个人信息、生产数据或敏感日志。
- 信息不足时必须说明缺失信息并提出澄清问题；不得把待确认内容写成已确认事实。

## Fons4AI 技能路由

本文件顶部的 `<!-- fons4ai-skill-routing: enabled -->` 是 Fons4AI 全局技能自动触发标记。其他项目没有作用域内 `AGENTS.md` 声明该标记时，`fons4ai-*` 技能不得自动触发，除非用户显式指定技能或明确要求使用 Fons4AI/SDD 流程。

| 场景 | 默认技能 |
| --- | --- |
| 全新需求接入 | `fons4ai-sdd-requirements` -> `fons4ai-sdd-design` -> `fons4ai-sdd-tasks` |
| 用户确认执行 SDD 任务 | `fons4ai-sdd-implement` |
| 已有功能迭代 | `fons4ai-sdd-change` |
| BUG、异常、回归失败 | `fons4ai-bugfix-workflow` |
| 初始化项目知识库 | `fons4ai-project-knowledge-base-init` |
| 汇总已验证知识 | `fons4ai-knowledge-summary` |
| 生成项目规则 | `fons4ai-generate-project-rules` |

若无法判断属于哪类场景，先询问用户要执行的工作流，并给出推荐理由。

## SDD 规范

- SDD 只使用 `S1` 和 `S2`：`S1` 是默认级别，`S2` 用于跨核心模块、数据库迁移、公共 API、权限安全、兼容性、事务边界等高风险改动。
- 全新需求产物默认位于 `spec/features/<yyyymmdd>/`，需求说明书文件为 `<功能中文名>-需求说明书.md`，同级维护 `plan.md`、`tasks.md`。
- `需求说明书.md` 面向业务，使用业务术语描述背景、范围、角色场景、需求、规则、流程、数据口径、影响和验收标准；技术细节放入 `plan.md`。
- `plan.md` 面向实现，描述总体架构、核心规则落地、数据流、领域建模、关键代码片段、状态流转、接口契约、数据模型、事务一致性、迁移回滚和验证策略。
- `tasks.md` 面向执行，每个任务必须包含 `AC:`、`Files:`、`Verification:`、`Quality:`、`Done:`。
- 若关键需求含义、业务术语、数据语义、验收口径、兼容性、安全权限、迁移回滚或 SDD 等级存在歧义，必须先澄清，不得直接生成正式三件套。
- 只有用户明确要求“先按假设生成草案”时，才允许生成 `文档状态：草案-待确认` 的草案；草案不得进入设计、任务或实现阶段。
- 生成 `tasks.md` 或 CR 增量任务后必须暂停。用户回复 `执行`、`开始实现`、`继续执行` 时默认执行全部未完成任务；用户回复 `执行 T001,T002` 时只执行指定任务。
- `fons4ai-sdd-requirements`、`fons4ai-sdd-design`、`fons4ai-sdd-tasks`、`fons4ai-sdd-change` 只能生成或更新 SDD 产物，不得写业务代码。

## 知识库与规则

- 默认知识入口是 `.specify/memory/index.md`；不要全量读取 `.specify/memory/`，必须先定位再读取。
- 项目级知识库使用 `.specify/memory/业务架构.md`、`.specify/memory/技术架构.md`、`.specify/memory/数据架构.md` 保存总览。
- 领域级知识库使用 `.specify/memory/domains/<domain-slug>/业务架构.md`、`技术架构.md`、`数据架构.md` 保存领域细节。
- 知识卡片位于 `.specify/memory/domains/<domain-slug>/cards/`，用于保存可精准检索的业务规则、状态流转、接口契约、数据模型和治理事实。
- 项目规则默认位于 `.specify/rules/`，核心文件为 `agent运行规则.md` 和 `代码编写规范.md`。
- 已验证的长期业务、技术、数据、接口、治理事实必须通过 `fons4ai-knowledge-summary` 或项目指定流程汇总到知识库。
- 临时调试、未完成计划、废弃方案、未经验证的猜测不得写入长期知识库。

## 数据与 DDL

- 涉及持久化数据模型新增、删除、重命名、字段、索引、约束或关系变更时，必须明确 SQL/DDL 影响。
- `.specify/sql/**/*.sql` 是知识库快照，不替代项目自身迁移脚本。
- 知识库初始化技能不得生成、复制、导入、更新或占位创建 `.specify/sql/**/*.sql` 文件；数据库 MCP 或仓库 SQL 只作为数据架构事实来源。
- SDD 实现涉及存量表结构变更，且已有可确认原始 DDL 时，必须生成可复制执行的变更 DDL 脚本，并同步更新 SQL 知识快照。
- 多个数据库 MCP 或候选数据库无法唯一确认时，必须先询问用户选择，不得自行合并数据源。

## 代码修改

- 修改前按需读取 `AGENTS.md`、相关源码、测试、配置、SDD 产物、知识卡片、领域文档和规则文件。
- 工具包优先级：JDK 标准库能力 -> 项目已有工具类/基础组件 -> 项目已引入三方工具包 -> 新增依赖。
- 默认采用 DDD-lite：核心业务规则、状态流转、校验和不变量优先下沉到领域对象或领域方法；简单 CRUD 和纯查询不强行 DDD 化。
- 应用服务负责流程编排、事务、权限、外部协作和持久化协调；领域对象不得反向依赖 controller、repository、mapper、MQ、RPC、HTTP 等基础设施细节。
- 新增关键逻辑、复杂分支、领域字段含义时，添加简洁注释。

## 验证与交付

- 优先运行与变更最相关的自动化测试；无法测试时说明原因和手动验证步骤。
- 交付说明必须包含：变更内容、涉及文件、验证结果、未验证项、风险和是否需要知识汇总。
- 若变更影响知识库、规则、数据模型或 DDL，必须说明是否已同步，未同步时说明原因和后续动作。
