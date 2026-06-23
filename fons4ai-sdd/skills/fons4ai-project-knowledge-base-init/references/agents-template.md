# AGENTS.md

<!-- fons4ai-skill-routing: enabled -->

> 适用范围：本仓库
> 知识状态：已初始化 | 待初始化
> 更新日期：YYYY-MM-DD

# 项目简介

<一句话说明项目定位；事实不足时写：待确认>

## 快速导航

| 你想做什么 | 去哪里看 | 状态 |
| --- | --- | --- |
| 了解项目知识入口 | `.specify/memory/index.md` | 已存在/待初始化 |
| 查看业务领域知识 | `.specify/memory/domains/<domain-slug>/业务架构.md` | 已存在/按需创建 |
| 查看技术落地知识 | `.specify/memory/domains/<domain-slug>/技术架构.md` | 已存在/按需创建 |
| 查看数据架构知识 | `.specify/memory/domains/<domain-slug>/数据架构.md` | 已存在/按需创建 |
| 查看知识卡片 | `.specify/memory/domains/<domain-slug>/cards/` | 已存在/按需创建 |
| 查看 Agent 运行规则 | `.specify/rules/agent运行规则.md` | 已存在/可选生成 |
| 查看代码编写规范 | `.specify/rules/代码编写规范.md` | 已存在/可选生成 |
| 查看 SDD 团队协作规范 | `.specify/rules/sdd团队协作规范.md` | 已存在/可选生成 |
| 查看 SDD 功能产物 | `spec/features/<yyyymmdd>/` | 按需生成 |

## 硬性规则

- 默认使用中文沟通、编写文档和交付说明。
- 修改代码或文档前，必须先读取相关上下文。
- 不得凭空猜测业务逻辑、接口、字段、表结构或第三方 API。
- 优先遵循项目事实、知识库、规则文件和已有代码风格。
- 优先复用已有代码、工具类、组件、规则和依赖。
- 不得修改与当前需求无关的代码、文档、配置或格式。
- 不得引入新框架、新模块、新依赖或新技术路线，除非用户明确确认。
- 删除文件、删除核心逻辑、大范围重构、覆盖文档、修改数据库结构前，必须获得用户确认。
- 不得提交密钥、令牌、个人信息、生产数据或敏感日志。
- 信息不足时必须说明缺失信息并提出澄清问题。

## 上下文加载顺序

1. 先读本文件。
2. 再读 `.specify/memory/index.md`，只加载相关领域文档和知识卡片。
3. 按需读取 `.specify/rules/agent运行规则.md`、`.specify/rules/代码编写规范.md` 和 `.specify/rules/sdd团队协作规范.md`。
4. 再读取相关 SDD 产物、源码、测试、配置和文档。
5. 不得默认全量读取 `.specify/memory/`、`spec/`、`docs/` 或源码目录。

## Fons4AI 工作流

| 场景 | 推荐流程 |
| --- | --- |
| 全新需求 | `fons4ai-sdd-requirements` -> `fons4ai-sdd-design` -> `fons4ai-sdd-tasks` |
| 用户确认执行任务 | `fons4ai-sdd-implement` |
| 已有功能变更 | `fons4ai-sdd-change` |
| BUG 修复 | `fons4ai-bugfix-workflow` |
| 初始化知识库 | `fons4ai-project-knowledge-base-init` |
| 汇总已验证知识 | `fons4ai-knowledge-summary` |
| 生成项目规则 | `fons4ai-generate-project-rules` |

无法判断场景时，先询问用户要执行哪类工作流，并给出推荐理由。

## SDD 约束

- SDD 只使用 `S1` 和 `S2`。
- 新需求产物位于 `spec/features/<yyyymmdd>/`。
- 需求、设计、任务、变更阶段只生成文档，不写业务代码。
- 标准扩展场景必须优先复用已确认的专业工作流、领域知识卡片或团队规则；专业工作流结论必须先进入技术设计和任务规划，不能在实现阶段临场发挥。
- 生成任务规划或 CR 增量任务后必须暂停，等待用户确认执行。
- 用户回复 `执行` 时默认执行全部未完成任务；回复 `执行 T001,T002` 时只执行指定任务。

## 知识沉淀

- 已验证的长期业务、技术、数据、接口和治理事实，应通过 `fons4ai-knowledge-summary` 汇总到知识库。
- 临时调试、未完成计划、废弃方案、未经验证猜测不得写入长期知识库。
- `.specify/sql/**/*.sql` 不由知识库初始化技能生成；只能作为已有 SQL 或真实 DDL 参考。
