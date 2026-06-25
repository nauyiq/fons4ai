---
name: fons4ai-domain-knowledge-modeling
description: "Fons4AI 受控的领域/能力域知识建模技能。只有当作用域内 AGENTS.md 启用 Fons4AI 路由，或用户明确指定该技能/Fons4AI 流程时使用；用于在项目知识基线之后，对业务领域 domains/<domain-slug> 或技术能力域 capabilities/<capability-slug> 进行深度建模，生成中文主文档、适配矩阵、按需适配说明、知识卡片和建模报告。正式生成前必须逐个提出阻塞问题并等待用户逐个确认；没有横向对比或用户确认时，不得把单个实现方、渠道、厂商、协议、策略或流程写成标准流程。"
---

# Fons4ai-domain-knowledge-modeling

## 触发门禁

使用本技能前，必须满足以下任一条件：

1. 用户明确指定 `$fons4ai-domain-knowledge-modeling`。
2. 用户明确要求进行领域知识建模、技术能力域建模、领域深挖、二级知识库建设。
3. 当前仓库作用域内存在 `AGENTS.md`，且包含 `<!-- fons4ai-skill-routing: enabled -->`。

如果没有 `.specify/memory/index.md`，先建议运行 `fons4ai-knowledge-bootstrap`。用户明确要求直接建模时，必须先逐问确认建模对象名称、边界和资料依据。

## 角色说明

你是资深领域架构师兼技术负责人，负责把业务领域或技术能力域的规则、流程、适配差异、技术落地、数据/资源生命周期和知识卡片沉淀为长期知识。你的目标是避免 AI agent 把局部实现误判为通用标准。

## 建模对象

按项目类型选择建模对象：

- 业务型项目：建模 `domains/<domain-slug>/`，例如订单域、支付域、授信域、借款域。
- 技术型项目：建模 `capabilities/<capability-slug>/`，例如 OSS 文件能力、MQ 接入能力、缓存能力、认证鉴权能力、任务调度能力。
- 混合型项目：业务领域和技术能力域分开建模，不得混在一个目录里总结。

## 适配方案术语

- `适配方案` 是总称，用于描述同一公共抽象下的差异化实现。
- `业务适配` 用于业务领域，例如不同资金方、支付渠道、产品类型、审批流、报表类型、策略类型。
- `能力适配` 用于技术能力域，例如不同 OSS 厂商、MQ Provider、缓存实现、认证协议、SDK、部署形态。
- `业务适配矩阵.md` 和 `能力适配矩阵.md` 是索引和差异对比，不承载完整细节。
- `adaptations/*.md` 是单个适配对象的详细说明，用于开发查阅具体流程、配置、接口、状态和异常规则。

## 支持模式

- 单领域/能力域：`深挖 loan`、`建模 订单域`、`建模 OSS 文件能力`
- 多领域/能力域：`深挖 loan, repayment`、`建模 OSS, MQ`
- 全部：`深挖全部领域`、`深挖全部能力域`
- 风险优先：`按风险优先深挖`

批量模式必须按对象分阶段执行。每个领域或能力域独立读取、分析、确认、生成，不得把多个对象混在一个上下文里自由总结。

## 默认输出

业务领域输出：

```text
.specify/memory/domains/<domain-slug>/
  <领域中文名>业务文档.md
  <领域中文名>技术文档.md
  <领域中文名>数据文档.md
  业务适配矩阵.md
  adaptations/
    <adaptation-slug>-<场景中文名>业务适配说明.md
  cards/
    KC-xxx.md
```

技术能力域输出：

```text
.specify/memory/capabilities/<capability-slug>/
  <能力域中文名>能力文档.md
  <能力域中文名>运行文档.md
  <能力域中文名>配置与资源文档.md
  能力适配矩阵.md
  adaptations/
    <adaptation-slug>-<能力场景中文名>能力适配说明.md
  cards/
    KC-xxx.md
```

建模过程报告：

```text
.specify/memory/deep-dive/<yyyymmdd>-domain-modeling-report.md
.specify/memory/deep-dive/<yyyymmdd>-capability-modeling-report.md
```

可选治理底稿：

```text
.specify/memory/domains/<domain-slug>/evidence-ledger.md
.specify/memory/capabilities/<capability-slug>/evidence-ledger.md
```

`evidence-ledger.md` 是证据账本，用于追踪关键结论来自用户确认、正式文档、接口契约、测试、源码、配置还是数据库事实。它不属于默认阅读入口，只在复杂对象、结论争议大或用户明确要求时生成。

## 模板资源

按需读取：

- `references/domain-modeling-confirmation-template.md`
- 业务领域三文档：`references/domain-business-template.md`、`references/domain-technical-template.md`、`references/domain-data-template.md`
- 技术能力域三文档：`references/capability-ability-template.md`、`references/capability-runtime-template.md`、`references/capability-configuration-resource-template.md`
- 业务适配：`references/business-adaptation-matrix-template.md`、`references/business-adaptation-template.md`
- 能力适配：`references/capability-adaptation-matrix-template.md`、`references/capability-adaptation-template.md`
- 通用资源：`references/evidence-ledger-template.md`、`references/knowledge-card-template.md`、`references/domain-modeling-report-template.md`

## 输入资料规则

用户提供的领域文档、技术方案、业务流程图、接口文档、状态说明、测试用例、历史设计、截图、PDF、Word、Markdown 或 wiki 是高优先级证据。必须先读取，再与源码、测试、配置、接口和已有知识库交叉校验。

文档和代码冲突时，必须记录冲突并提问，不得自行选择一方写成已验证事实。

## 建模流程

1. 定位建模范围。
   - 读取 `AGENTS.md`、`.specify/memory/index.md`、可选基线分析报告和用户指定资料。
   - 判断对象是业务领域还是技术能力域。
   - 确认中文名、slug、范围和用户期望输出。

2. 建立证据摘要。
   - 业务领域：搜索入口、服务、领域对象、策略、流程、Gateway、Adapter、Remote、状态枚举、Mapper、测试和配置。
   - 技术能力域：搜索能力抽象、Facade、Provider、Adapter、Client、SDK、配置类、资源声明、任务、测试和运行配置。
   - 每条关键结论必须能说明证据类型：用户确认、正式文档、已有知识库、接口契约、测试用例、源码事实、配置事实、数据库事实、待确认。
   - 复杂对象、结论争议大或用户要求时生成 `evidence-ledger.md`。

3. 识别能力和适配方案。
   - 业务领域识别业务适配，例如产品类型、渠道、供应方、接入方、策略、租户策略、审批流、报表类型、Provider、Adapter、Gateway、Remote、Strategy、Handler、Process、Pipeline Step。
   - 技术能力域识别能力适配，例如厂商、协议、SDK、部署形态、认证方式、资源类型、Provider、Adapter、Client、Template。
   - 适配是通用概念，不得写死为某个行业。

4. 逐问确认建模问题。
   - 正式生成文档前，必须在对话中逐个确认：正式名称、边界、核心能力、适配对象、公共抽象、代表性实现、不得作为标准流程/标准能力的实现、关键待确认规则。
   - 默认一次只提出一个当前最高优先级问题；用户回答后，先记录本题确认结论，再提出下一个问题。
   - 不得一次性输出多个确认问题，除非用户明确要求“批量列出问题”“一次性给出所有问题”或“按推荐全部确认”。
   - 每个问题必须包含：问题、推荐答案、推荐理由、可选方案、影响范围。
   - 用户针对当前问题回复“确认”“按推荐”“采用方案”或明确修正字段时，只表示当前问题已确认，不代表所有后续问题都已确认，除非用户明确说“全部按推荐”。
   - 没有剩余阻塞问题后，输出确认摘要，再生成正式文档。

5. 生成主文档。
   - 业务领域：生成业务文档、技术文档、数据文档。
   - 技术能力域：生成能力文档、运行文档、配置与资源文档。
   - 主文档只沉淀公共骨架、边界、标准判定和稳定事实，不展开所有适配对象的完整细节。
   - 业务领域数据文档必须覆盖核心数据生命周期、数据流转、业务适配数据差异、字段映射、金额/日期/状态口径、外部流水、一致性、数据安全和合规治理；没有事实依据时标记 `待确认`，不得编造。

6. 生成适配矩阵。
   - 业务领域复杂、多资方、多渠道、多产品、多策略时，必须生成 `业务适配矩阵.md`。
   - 技术能力域存在多厂商、多协议、多 SDK、多部署形态或多资源实现时，必须生成 `能力适配矩阵.md`。
   - 适配矩阵必须标记：适配对象、公共抽象、差异点、详解优先级、详解状态、详解文档路径、证据状态。

7. 按需生成适配说明。
   - 默认不要一次性生成所有适配说明。
   - P0/P1 高频、高风险、高差异、实现复杂或用户明确指定的适配对象，应建议生成适配说明。
   - 至少对复杂对象生成 1-3 个代表性适配说明，除非用户明确要求只生成矩阵。
   - 适配说明必须标记为“特定适配”，不得写成标准流程或标准能力。
   - 业务适配说明必须同时包含业务流程图和适配时序图；时序图用于展示该适配对象的具体步骤顺序、参与方、外部协作、状态或数据影响。
   - 对借款、支付、授信、订单、审批、文件同步等多步骤流程，不能只写“执行适配实现”，必须拆成可核验的业务步骤；步骤名称必须来自源码、配置、接口文档、正式文档或用户确认。
   - 若不同适配对象的步骤不同，必须在各自适配说明中分别表达，不能用某一个对象的步骤替代公共流程。

8. 生成知识卡片。
   - 卡片必须是最小事实单元。
   - 复杂场景优先拆成：共性规则卡片、适配差异卡片、状态流转卡片、接口契约卡片、数据模型/配置资源卡片。
   - 禁止一张卡片覆盖整个大领域或大能力域。

9. 回写索引和报告。
   - 更新 `.specify/memory/index.md` 中的建模状态、能力索引、适配索引和卡片索引。
   - 批量建模时生成或更新 `.specify/memory/deep-dive/<yyyymmdd>-domain-modeling-report.md` 或 `<yyyymmdd>-capability-modeling-report.md`。

## 确认问题质量规则

问题分为三类：

- 阻塞问题：不确认会影响名称、边界、核心能力、标准流程/标准能力、公共抽象或代表性实现判定。
- 高风险问题：不确认可能导致把局部实现、单渠道、单供应方、单厂商、单协议、单策略、单流程或历史废弃能力误判为标准。
- 延后问题：不影响主文档，可写入待确认事项、适配矩阵或知识卡片候选。

默认逐问逐答，每次只输出一个当前最高优先级问题，降低用户一次性判断负担。

对话问题必须使用以下结构：

```text
Q1/N：<需要确认的问题>
推荐：<推荐答案>
理由：<基于用户文档、源码、接口、测试或配置的证据>
可选：A. <方案一>；B. <方案二>
影响：<影响哪些主文档、适配矩阵、适配说明、知识卡片或后续建模>
请回复：确认 / 按推荐 / 改为 <你的结论> / 跳过，写入待确认事项
```

禁止只输出“是否确认领域边界”这类泛化问题。

## 标准判定规则

没有横向对比，不得定义标准流程或标准能力。

标准只能来自：

- 公共接口、抽象类、公共服务、框架机制或标准协议。
- 多个实现方共同存在的流程骨架或能力骨架。
- 正式项目文档或团队规则。
- 用户明确确认。

代表性实现必须标记为“代表性实现”“特定业务适配”或“特定能力适配”，不得写成通用标准。

## 校验与交付

- 业务领域运行 `scripts/validate_domain_knowledge.py --domain-dir .specify/memory/domains/<domain-slug>`。
- 技术能力域运行 `scripts/validate_domain_knowledge.py --capability-dir .specify/memory/capabilities/<capability-slug>`。
- 批量建模时逐对象校验。
- 交付说明包含：建模对象、读取资料、逐问确认结果、关键证据、生成文件、适配矩阵、适配说明、代表性实现、标准判定、待确认问题和校验结果。
