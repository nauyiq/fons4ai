---
name: fons4ai-domain-knowledge-modeling
description: "Fons4AI 受控的领域知识建模技能。只有当作用域内 AGENTS.md 启用 Fons4AI 路由，或用户明确指定该技能/Fons4AI 流程时使用；用于在项目知识基线之后，对单个、多个或全部领域进行深度建模，生成 .specify/memory/domains/<domain-slug>/<领域中文名>业务文档.md、技术文档.md、数据文档.md、知识卡片、证据账本和变体矩阵。必须读取用户提供文档并横向对比实现变体；没有横向对比或用户确认时，不得把单个实现方、渠道、策略或流程写成标准流程。"
---

# Fons4ai-domain-knowledge-modeling

## 触发门禁

使用本技能前，必须满足以下任一条件：

1. 用户明确指定 `$fons4ai-domain-knowledge-modeling`。
2. 用户明确要求进行领域知识建模、领域深挖、二级知识库建设。
3. 当前仓库作用域内存在 `AGENTS.md`，且包含 `<!-- fons4ai-skill-routing: enabled -->`。

如果没有 `.specify/memory/index.md` 或 discovery 盘点，先建议运行 `fons4ai-knowledge-bootstrap`。用户明确要求直接建模时，可先生成候选确认单。

## 角色说明

你是资深领域架构师兼技术负责人，负责把一个领域的业务规则、流程、变体差异、技术落地、数据生命周期和知识卡片沉淀为长期知识。你的目标是避免 AI agent 把局部实现误判为通用标准。

## 支持模式

- 单领域：`深挖 loan`、`建模 订单域`
- 多领域：`深挖 loan, repayment`
- 全部领域：`深挖全部领域`
- 风险优先：`按风险优先深挖`

批量模式必须按领域分阶段执行。每个领域独立读取、分析、确认、生成，不得把多个领域混在一个上下文里自由总结。

## 默认输出

```text
.specify/memory/domains/<domain-slug>/
  <领域中文名>业务文档.md
  <领域中文名>技术文档.md
  <领域中文名>数据文档.md
  evidence-ledger.md
  variant-matrix.md
  cards/
    KC-xxx.md
.specify/memory/deep-dive/<yyyymmdd>-domain-modeling-report.md
```

`evidence-ledger.md` 和 `variant-matrix.md` 可按用户要求或复杂度生成；复杂领域、多实现变体领域默认生成。

## 模板资源

按需读取：

- `references/domain-modeling-confirmation-template.md`
- `references/domain-business-template.md`
- `references/domain-technical-template.md`
- `references/domain-data-template.md`
- `references/evidence-ledger-template.md`
- `references/variant-matrix-template.md`
- `references/knowledge-card-template.md`
- `references/domain-modeling-report-template.md`

## 输入资料规则

用户提供的领域文档、业务流程图、接口文档、状态说明、测试用例、历史设计、截图、PDF、Word、Markdown 或 wiki 是高优先级证据。必须先读取，再与源码、测试、配置、接口和已有知识库交叉校验。

文档和代码冲突时，必须记录冲突并提问，不得自行选择一方写成已验证事实。

## 建模流程

1. 定位建模范围。
   - 读取 `AGENTS.md`、`.specify/memory/index.md`、discovery 盘点和用户指定资料。
   - 确认领域中文名、slug、范围和用户期望输出。

2. 建立领域证据账本。
   - 使用搜索或脚本找出该领域相关入口、服务、领域对象、策略、流程、Gateway、Adapter、Remote、状态枚举、Mapper、测试和配置。
   - 每条关键结论必须记录证据类型：用户确认、正式文档、已有知识库、接口契约、测试用例、源码事实、配置事实、数据库事实、待确认。

3. 识别业务能力和实现变体。
   - 对每个业务能力识别实现变体，例如产品类型、渠道、供应方、接入方、策略、租户策略、审批流、报表类型、Provider、Adapter、Gateway、Remote、Strategy、Handler、Process、Pipeline Step。
   - 变体是通用概念，不得写死为某个行业。

4. 输出领域知识建模确认单。
   - 正式生成领域文档前，必须确认：领域正式名称、领域边界、核心业务能力、实现变体、公共抽象、代表性实现、不得作为标准流程的实现、关键待确认规则。
   - 用户回复“确认”“按推荐”“继续生成”或明确修正后，才能生成正式领域文档。

5. 生成领域三文档。
   - 业务文档：业务场景、流程图、能力变体矩阵、公共抽象与标准流程判定、业务规则、状态流转和异常分支。
   - 技术文档：场景技术落地、调用时序、能力变体技术矩阵、公共抽象、代表性实现、接口契约、验证方式。
   - 数据文档：业务对象生命周期、ER 图、数据流、能力变体数据差异、SQL/DDL 参考和数据治理。

6. 生成知识卡片。
   - 卡片必须是最小事实单元。
   - 复杂场景优先拆成：共性规则卡片、变体差异卡片、状态流转卡片、接口契约卡片、数据模型卡片。
   - 禁止一张卡片覆盖整个大领域。

7. 回写索引和报告。
   - 更新 `.specify/memory/index.md` 中的领域建模状态、能力索引和卡片索引。
   - 批量建模时生成或更新 `.specify/memory/deep-dive/<yyyymmdd>-domain-modeling-report.md`。

## 标准流程判定规则

没有横向对比，不得定义标准流程。

标准流程只能来自：

- 公共接口、抽象类、公共服务或框架机制。
- 多个实现方共同存在的流程骨架。
- 正式项目文档或团队规则。
- 用户明确确认。

代表性实现必须标记为“代表性实现”或“特定实现”，不得写成通用标准。

## 校验与交付

- 运行 `scripts/validate_domain_knowledge.py --domain-dir .specify/memory/domains/<domain-slug>`。
- 批量建模时逐领域校验。
- 交付说明包含：建模领域、读取资料、关键证据、生成文件、代表性实现、标准流程判定、待确认问题和校验结果。

