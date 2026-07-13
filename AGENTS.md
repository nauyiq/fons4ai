# Fons4AI Agent 规则

<!-- fons4ai-skill-routing: enabled -->

> 适用范围：本仓库
> 知识状态：基线已建立
> 更新日期：2026-07-13

# 项目简介

Fons4AI 是持续迭代的可插拔 AI Agent 开发框架，以 Spring Boot Starter 形式提供智能体执行引擎、工具编排、RAG 检索增强、多模态识别和图像生成等 AI 能力，供上层业务系统按需接入。

## 快速导航

| 你想做什么 | 去哪里看 | 状态 |
| --- | --- | --- |
| 查看知识库入口 | `.specify/memory/index.md` | 已建立 |
| 查看技术能力总览 | `.specify/memory/项目技术能力架构文档.md` | 已建立 |
| 查看运行架构 | `.specify/memory/项目运行架构文档.md` | 已建立 |
| 查看配置与资源 | `.specify/memory/项目配置与资源架构文档.md` | 已建立 |
| 深度建模能力域 | `.specify/memory/capabilities/<capability-slug>/` | 按需建立 |
| 查看项目规则 | `.specify/rules/` | 按需生成 |

## 硬性规则

- 默认使用中文沟通、编写文档和交付说明。
- 修改代码或文档前，必须先读取相关上下文。
- 不得凭空猜测业务逻辑、接口、字段、表结构或第三方 API。
- 优先遵循项目事实、知识库、规则文件和已有代码风格。
- 删除文件、覆盖文档、大范围重构、修改数据库结构前必须确认。
- 用户提供文档与源码不一致时，必须标记冲突并请求确认。

## 当前架构口径

- 项目按技术框架/基础设施项目建模，不抽象虚构业务领域。
- 项目能力域为：`agent-orchestration`、`tool-management`、`rag`、`ai-capability`。
- 当前 `fons4ai-agent` 和 `fons4ai-rag` 是实现模块，不等同于最终能力边界。
- Spring AI 是当前实现适配，不能被表述为项目长期唯一框架。
- 工具管理当前位于 Agent 模块，具备独立演进条件；图像生成和多模态能力后续归入 `ai-capability`。

## 知识库建设工作流

| 阶段 | 推荐技能 | 目的 |
| --- | --- | --- |
| 项目知识基线初始化 | `fons4ai-knowledge-bootstrap` | 建立项目级知识入口、技术能力域和项目级架构文档 |
| 技术能力域知识建模 | `fons4ai-domain-knowledge-modeling` | 深挖单个能力域的场景、适配、运行和资源模型 |
| 知识汇总治理 | `fons4ai-knowledge-summary` | 将已验证变更同步到长期知识库 |

## Fons4AI 技能路由

| 场景 | 推荐技能 |
| --- | --- |
| 正常新需求开发 | `fons4ai-sdd-feature-workflow` |
| 需求澄清/补需求说明书 | `fons4ai-sdd-requirements` |
| 技术设计补充 | `fons4ai-sdd-design` |
| 任务规划补充 | `fons4ai-sdd-tasks` |
| 用户确认执行 SDD 任务 | `fons4ai-sdd-implement` |
| 已有功能迭代 | `fons4ai-sdd-change` |
| 低风险小变更 | `fons4ai-sdd-quick-path` |
| BUG、异常、回归失败 | `fons4ai-bugfix-workflow` |
| 技术能力域深度建模 | `fons4ai-domain-knowledge-modeling` |
| 汇总已验证知识 | `fons4ai-knowledge-summary` |
| 生成项目规则 | `fons4ai-generate-project-rules` |

无法判断场景时，先询问用户要执行哪类工作流，并给出推荐理由。

## 知识库建设确认门禁

- 技术能力域中文名、slug、职责边界、核心能力和参考链路必须经过确认。
- 正式知识库生成前，必须先在对话中逐个提出阻塞性确认问题；没有剩余阻塞问题后才能写入正式基线。
- 单个实现方、单个配置或单个流程不得被写成最终标准，除非经过横向对比或用户确认。
- 源码中的预留能力、TODO 或占位实现必须明确标注，不得写成已交付能力。
