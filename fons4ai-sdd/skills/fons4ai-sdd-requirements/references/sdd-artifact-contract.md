# Fons4AI SDD Artifact Contract

## Scope

This contract defines the shared SDD artifact rules for all `fons4ai-sdd-*` skills.
Feature artifacts use `spec/features/<yyyymmdd>/`. Bugfix artifacts use `spec/bugfixes/<yyyymmdd>/<bug中文名>-BUG修复报告.md`. The default project truth sources are `.specify/memory/` and `.specify/sql/`, but projects may declare additional truth sources. Do not require branch hooks or GitHub issue conversion.

## Artifact Responsibilities

- `需求说明书.md` is the business-oriented requirement specification. It records a concise clarification summary, business background and goals, scope, roles and scenarios, requirement list, business rules, simple workflows when useful, lightweight business-data meaning, impact, AC, quality requirements, risks, assumptions, and open items. It should use plain business language and avoid unnecessary professional or technical terminology. It must not expose repository-fact inventories, knowledge-context inventories, modules, classes, tables, columns, DDL paths, MCP details, or technical architecture details.
- `<功能中文名>-技术设计说明书.md` is the technical design specification. It records design summary, architecture and call chain, API/RPC/message contracts, data-model and DDL impact, core logic, DDD-lite/domain-rule landing, state transitions, error/security/transaction/performance decisions, technical decisions, AC mapping, risk/rollback, and concise knowledge-sync impact. It must not repeat requirement-only sections such as clarification summary, requirement scope, roles/scenarios, or business-data wording. It must not expose repository-fact inventories, knowledge-base-fact inventories, or search traces, and it must not replace executable tasks.
- `<功能中文名>-任务规划.md` is the executable task breakdown. It converts `需求说明书.md` and `<功能中文名>-技术设计说明书.md` into task IDs with AC mapping, files, verification, quality checks, and done criteria.
- `changes/CR-xxx.md` is the incremental change record for an existing feature. It records change type, impact analysis, affected artifacts, DDL impact, rollback/regression risks, long-term knowledge impact, and the task IDs appended to `<功能中文名>-任务规划.md`.
- Planning artifacts are not implementation approval; implementation still requires the approval gate below.

## Clarification Approval Gate

Requirements and change planning must close blocking ambiguity before formal artifact generation.

- `使用 SDD`, `继续`, `先生成`, `看一下`, existing artifact files, or a partially inferred plan are not clarification approval.
- `fons4ai-sdd-requirements` must ask the highest-impact requirement question first when blocking ambiguity can change scope, AC, business terms, data meaning, compatibility, security, integration, SDD level, or task breakdown.
- `fons4ai-sdd-change` must ask the highest-impact change question first when blocking ambiguity can change existing feature semantics, AC changes, naming/ownership, public behavior, data model, DDL source, migration, rollback, risk gates, or affected modules.
- While a blocking ambiguity exists, requirements and change skills must not write a formal `需求说明书.md`, formal CR, `<功能中文名>-技术设计说明书.md`, `<功能中文名>-任务规划.md`, or business code.
- If the user explicitly asks for a draft before answering, the artifact may be written only with `文档状态：草案-待确认`; it must not be used by design, task, or implementation skills.
- Formal `需求说明书.md` and CR artifacts must not expose clarification-gate tables, clarification status, or internal question logs. Clarification remains an internal pre-generation workflow.
- Design and task skills must stop if the input `需求说明书.md` or CR is marked `文档状态：草案-待确认`, `阻塞-等待回答`, `草案-含待确认`, `blocking`, or `draft`.

## Project Knowledge

Use `.specify/memory/`, `.specify/sql/`, and `.specify/rules/` as default long-lived project fact sources when they exist. Also respect other project-declared truth sources such as `docs/`, API documents, product documents, custom rule directories, or external knowledge bases.

- `.specify/memory/index.md` is the default memory entrypoint when present.
- Business-system project-level documents are `.specify/memory/项目业务架构文档.md`, `.specify/memory/项目技术架构文档.md`, and `.specify/memory/项目数据架构文档.md`. They are concise global overview documents.
- Technical framework or infrastructure project-level documents are `.specify/memory/项目技术能力架构文档.md`, `.specify/memory/项目运行架构文档.md`, and `.specify/memory/项目配置与资源架构文档.md`. They are concise global overview documents for technical capability projects.
- Domain-level documents live under `.specify/memory/domains/<domain-slug>/` as `<领域中文名>业务文档.md`, `<领域中文名>技术文档.md`, and `<领域中文名>数据文档.md`.
- Capability-level documents live under `.specify/memory/capabilities/<capability-slug>/` as `<能力域中文名>能力文档.md`, `<能力域中文名>运行文档.md`, and `<能力域中文名>配置与资源文档.md`.
- Business adaptations live under `.specify/memory/domains/<domain-slug>/业务适配矩阵.md` and `.specify/memory/domains/<domain-slug>/adaptations/*.md`.
- Capability adaptations live under `.specify/memory/capabilities/<capability-slug>/能力适配矩阵.md` and `.specify/memory/capabilities/<capability-slug>/adaptations/*.md`.
- Knowledge cards live under `.specify/memory/domains/<domain-slug>/cards/` and store fact-level retrievable knowledge: business scenarios, rules, state transitions, technical flows, interface contracts, data models, and governance rules.
- Knowledge facts use lifecycle status: `已验证`, `待确认`, or `已废弃`. Only verified facts should be written as durable truth; planned-only or weakly evidenced facts must remain `待确认`.
- `index.md` is navigation and indexing only. Project-level documents stay concise, domain documents carry detailed context, and knowledge cards carry the smallest retrievable facts.
- `.specify/sql/**/*.sql` stores one DDL SQL file per database-scoped business model. A file may contain multiple strongly related tables only when they belong to the same database/service and cohesive business model.
- SQL knowledge should come from real DDL evidence: configured database MCP query results or existing repository SQL DDL files. Entities, ORM metadata, Mapper interfaces, repository methods, and Java field types may locate candidate models, but must not be used to generate `CREATE TABLE`.
- If multiple database MCP tools or multiple plausible databases are available, ask the user to select the MCP tool/database scope before retrieving DDL unless explicit user input or repository facts identify one unambiguously.
- Generated SQL knowledge files keep database/service, business model, included tables, status, update date, and DDL only. They must not contain MCP/Tool identifiers, query text, repository source paths, or provenance headers such as `Source`, `Migration Script`, or `DDL Evidence`.
- `.specify/rules/` may contain project rules such as `agent运行规则.md`, `代码编写规范.md`, and `sdd团队协作规范.md`.
- `constitution.md`, when present, is governance context and must not be rewritten by SDD feature skills.

Feature artifacts under `spec/features/<yyyymmdd>/` can cite or be constrained by truth-source facts, but should not silently update knowledge sources. If a feature changes long-lived business, technical, data, governance, adaptation, or other source-of-truth facts, record a knowledge impact in the SDD artifacts, but do not create SDD task-planning items for knowledge synchronization. Route verified knowledge updates through `fons4ai-knowledge-summary`, which owns updates to affected domain documents, capability documents, adaptation artifacts, knowledge cards, and `.specify/memory/index.md`.
If a feature changes concrete persistent data models, record `.specify/sql/` impact as well as the affected domain `<领域中文名>数据文档.md`, affected capability `<能力域中文名>配置与资源文档.md` when applicable, and project data index impact.
Incremental CRs must use `长期知识影响` to record source-of-truth impact. They must not create knowledge-sync tasks, knowledge-summary handoff tasks, or `.specify/memory/` editing tasks.

## Context Loading

- Do not bulk-load all of `.specify/memory/`, `.specify/sql/`, `.specify/rules/`, `specs/`, or `docs/` by default.
- First read `AGENTS.md`, the active SDD artifacts, and the directly affected source/test/config files.
- If `.specify/memory/index.md` exists, read it before reading project-level memory documents.
- Use `rg --files` and `rg -n` with feature names, domain names, module names, business objects, table names, API names, error text, `REQ-###`, and `AC-###` to locate relevant cards, domain documents, SQL, rules, and specs before reading.
- Optionally use `scripts/find_relevant_context.py --root <repo-root> <keyword...>` from this skill to get a first-pass candidate list for index, cards, domain memory, SQL, rules, specs, and docs. Treat its output as navigation help, not as verified evidence.
- For S1, read only relevant rules, knowledge cards, domain documents, related SQL files, and affected code paths.
- For S2, expand context around the impacted domain, module, contract, security, transaction, or data model, but still avoid unrelated full-document loading. Cross-domain work may require project-level overview sections.
- For standard-extension work such as adding a payment channel, funding partner, third-party service channel, OSS provider, MQ provider, strategy type, approval flow, or report type, read the relevant adaptation matrix and adaptation detail before designing or implementing. Do not infer a standard flow from a single adaptation object.
- For `.specify/sql/`, prefer `index.md`, domain `<领域中文名>数据文档.md`, capability `<能力域中文名>配置与资源文档.md`, and targeted path search. Read only the database/service and business-model SQL files involved in the work; use `.specify/sql/pending/` when ownership is unknown.
- Full scans are appropriate for knowledge-base initialization, rule generation, explicit audits, or broad refactors, but should still start with a file inventory and evidence matrix.

## Data Model DDL Sync

When SDD work adds, removes, renames, or changes a concrete persistent data model, table, column, index, constraint, relationship, or database-specific default:

- `需求说明书.md` records only the business meaning and user-facing impact of data. Keep this section lightweight. Technical data-model, table, column, and DDL impact belong in `<功能中文名>-技术设计说明书.md`.
- `需求说明书.md` must include a lightweight data-impact check when data may be created, updated, stored, exported, synchronized, reconciled, reported, mapped, or protected. It records business data meaning, source, unit, format, state meaning, and confirmation state, but not table names, column names, DDL paths, modules, classes, or technical architecture.
- If critical data meaning, field mapping, amount unit, date format, state meaning, serial number source, customer identifier, uniqueness, permission, masking, encryption, audit, retention, deletion, or archival rule is unresolved, the requirement artifact must remain `文档状态：草案-待确认` or the skill must ask the user before downstream design.
- `<功能中文名>-技术设计说明书.md` must expand triggered data work into data-design and governance decisions: field-mapping contract, data-flow design, data-model/DDL impact, data security, compliance, idempotency, consistency, and verification strategy.
- Field-mapping contracts are mandatory for file import, API import, data synchronization, reconciliation, reporting, migration, and cross-system data flows. Critical fields such as amount, date, state, serial number, customer identifier, contract number, external identifier, and sensitive fields must not stay `待确认` before implementation.
- Sensitive data scenarios must define applicable handling for transport security, storage encryption, display/log masking, permissions, audit, retention, deletion, and archival. If not applicable, record `不适用，原因`.
- `<功能中文名>-技术设计说明书.md` must name each impacted `.specify/sql/<database_or_service>/<business_model>.sql` file and state whether the action is add, update, rename, or no-op.
- `<功能中文名>-任务规划.md` must include an explicit DDL synchronization task for every impacted SQL file, unless the technical design records a user-approved deferral with owner and reason.
- `fons4ai-sdd-implement` may create or update `.specify/sql/**/*.sql` only when the selected task names the SQL file or when the implementation reveals a necessary schema change and the user approves updating the task/artifact scope.
- Generated SQL knowledge files are documentation artifacts, not migration scripts. Keep migration scripts in the repository's normal migration location when the project has one.
- When an approved implementation changes columns, indexes, constraints, defaults, or relationships of an existing table and the corresponding `.specify/sql/<database_or_service>/<business_model>.sql` already contains confirmed baseline DDL, the plan and task planning must require an executable change DDL script containing the needed `ALTER TABLE` or equivalent statements.
- Prefer the repository's established migration-script location for executable change DDL. If no migration location is established, use `spec/features/<yyyymmdd>/ddl-changes/<change-id>-<database_or_service>-<business_model>.sql`, where `<change-id>` is `INIT` for initial feature work or `CR-xxx` for an incremental change.
- Executable change DDL is generated only during approved implementation, not during requirements, design, task planning, or change planning. It records the operation to execute; `.specify/sql/**/*.sql` separately records the resulting current structure. Like other generated SQL artifacts, it must not contain MCP/Tool identifiers, query text, or source-path/provenance metadata.
- Agents must not directly execute production DDL. Executable DDL is provided for user/DBA review and manual execution, or for the project's established migration process. A DDL execution-confirmation task is complete only after user confirmation or read-only verification proves that the structure is effective.
- Tasks depending on an unexecuted DDL change must not be marked release-ready. Keep the task unchecked or record a blocker in the implementation report until the execution state is confirmed.
- If no repository SQL file exists, query the configured database MCP service for actual DDL. If no MCP DDL and no repository SQL DDL are available, mark SQL evidence as `待确认` and ask for MCP configuration or SQL files instead of fabricating table structure.
- Use `.specify/sql/pending/<business_model>.sql` only when ownership is unknown or the user explicitly requests a pending placeholder.
- Never merge DDL from different databases, service-owned schemas, or physical data sources into one SQL knowledge file, even when the tables belong to the same broad business area.
- Creating or updating SQL knowledge files does not require running a SQL-specific validator by default. Run only the project's current SQL artifact validator or a lightweight format check when the user explicitly requests SQL artifact validation or when diagnosing malformed existing SQL knowledge files.

## Levels

- `S1` is the default for small changes, normal features, and one-module or small multi-module collaboration.
- `S2` is required for cross-core-module changes, database migrations, public API or public contract changes, permission/security changes, cache/MQ/rate-limit/transaction boundaries, compatibility risk, or high rollback cost.
- Keep the classification limited to `S1` and `S2`; small safe changes use concise S1 artifacts.
- S1 artifacts use the minimal complete profile: keep required sections, AC coverage, task quality fields, and verification details, but mark truly absent state transitions, API changes, data changes, migrations, rollback, and diagrams as `不适用，原因` instead of generating speculative content.

## Paths

Use this feature layout:

```text
spec/features/<yyyymmdd>/
  <功能中文名>-需求说明书.md
  <功能中文名>-技术设计说明书.md
  <功能中文名>-任务规划.md
  checklists/
  contracts/
  ddl-changes/
  changes/
  reports/
```

Only create optional folders when they are needed.

## Naming

- `<yyyymmdd>` is the artifact creation date in local project time, for example `20260618`.
- `<功能中文名>` should be concise Chinese, normally 2-12 characters, derived from the feature name or confirmed with the user when ambiguous. The requirement file name must be `<功能中文名>-需求说明书.md`.
- Requirement IDs use `REQ-001`, `REQ-002`, ...
- AC IDs use `AC-001`, `AC-002`, ...
- Task IDs use `T001`, `T002`, ...
- Change records use `CR-001`, `CR-002`, ...
- CR change types use `微调`, `扩展`, `重构`, `数据结构变更`, `契约变更`, or `纯文档修正`.

## Traceability

- Every `REQ-###` in `需求说明书.md` must map to at least one `AC-###` through the requirement summary table or AC text.
- Every AC in `需求说明书.md` must be covered by at least one design decision in `<功能中文名>-技术设计说明书.md`.
- `<功能中文名>-技术设计说明书.md` should preserve REQ context in AC mapping when it materially affects implementation.
- Every implementation task in `<功能中文名>-任务规划.md` must include `AC:`, `Files:`, `Verification:`, `Quality:`, and `Done:`.
- Every task should map to at least one AC ID. If a task is pure setup, use the nearest AC it enables and explain that relationship in `Done:`.
- S2 tasks must include explicit regression and risk-control tasks.
- S2 implementation reports must state whether checklist, rollback, compatibility, and risk-control tasks were closed or explicitly deferred.
- Executable incremental tasks from CRs must be appended to `<功能中文名>-任务规划.md`. CR files may summarize and reference task IDs, but must not be the only location of executable implementation tasks.

## Detailed Document Requirements

- Generated artifact headings and fixed prose should be Chinese-first. Keep file names, IDs, paths, and machine-readable task labels such as `AC:`, `Files:`, `Verification:`, `Quality:`, and `Done:` in English when compatibility requires it.
- New `需求说明书.md` artifacts should use the simplified requirement-spec structure: `## 一句话说明`, `## 需求澄清摘要`, `## 背景与目标`, `## 需求范围`, `## 角色与场景`, `## 需求列表`, `## 业务规则`, `## 业务流程`, `## 业务数据口径`, `## 影响说明`, `## 验收标准`, `## 质量要求`, `## 风险与待确认`, and `## 版本修订记录`.
- New `<功能中文名>-技术设计说明书.md` artifacts must include `## 1. 设计概要`, `## 2. 架构与调用链路`, `## 3. API / RPC / 消息契约设计`, `## 4. 数据模型与 DDL 影响`, `## 5. 核心逻辑设计`, `## 6. 领域建模与业务规则落地`, `## 7. 状态流转设计`, `## 8. 异常、安全、事务与性能`, `## 9. 技术决策`, and `## 10. 验证策略、AC 映射与风险`.
- The business-rule and strategy section must map core rules or policies to modules, domain/application objects, data dependencies, extension points, and verification. Use Mermaid sequence, flow, or state diagrams for important business or strategy flows when facts support them.
- Code sketches in `<功能中文名>-技术设计说明书.md` are design snippets or pseudocode for key rules, validation, status checks, and data transformations. They must be based on repository facts and must not be treated as production code.
- Use Mermaid `sequenceDiagram` for core call chains, `flowchart` for complex decisions, `stateDiagram-v2` for state changes, and `erDiagram` for new tables, relationship changes, or multi-table collaboration when facts support them. A single-column or single-index adjustment may mark the ER diagram as `不适用，原因`.
- State transition and data-model sections may use `不适用，原因` for S1 when genuinely absent. S2 high-risk sections must be filled with concrete facts or explicit deferrals.

## Implementation Approval Gate

- Planning artifacts are not implementation approval: `需求说明书.md`, `<功能中文名>-技术设计说明书.md`, `<功能中文名>-任务规划.md`, and CR files define scope and tasks but do not authorize business-code implementation.
- `<功能中文名>-任务规划.md` and each CR with incremental tasks must contain `## 实现确认门禁`.
- Requirements, design, task, and change skills must stop after writing planning artifacts and must not invoke implementation.
- Implementation approval must come from the user's latest message.
- `fons4ai-sdd-implement` must record approval evidence in the implementation report, quoting or summarizing the latest user message that authorized execution. If that evidence cannot be identified, implementation must stop.
- If the latest user message confirms execution without task IDs, `fons4ai-sdd-implement` executes all unfinished tasks in dependency order.
- If the latest user message names task IDs such as `执行 T001,T002`, only those unfinished tasks are selected.
- Ambiguous messages such as `看看`, `下一步是什么`, or generated planning artifacts alone are not implementation approval.
- Implementation may mark a selected task complete only after its verification passes. DDL execution-confirmation tasks require user confirmation or read-only database verification; blocked or externally pending tasks remain unchecked and must be reported as blockers.
- If implementation reveals missing scope, missing DDL work, changed AC, changed public behavior, or conflicts with design, stop and return to task/change planning instead of expanding implementation scope.

## Editing Rules

- Read existing artifacts before editing them.
- Ask before replacing or materially rewriting existing artifacts.
- Preserve user or prior-agent changes outside the requested scope.
- Do not write business code from requirements, design, task, or change skills.
- Treat truth sources such as `.specify/memory/`, `.specify/sql/`, `.specify/rules/`, and `docs/` as read-only unless the active skill is explicitly responsible for knowledge-base initialization/summary, or a selected SDD task explicitly requires rules or DDL synchronization. SDD task planning must not create knowledge-sync or knowledge-summary handoff tasks.
- `fons4ai-sdd-implement` may record long-term knowledge impact in its report, but must not create knowledge-sync tasks, knowledge-summary handoff tasks, or `.specify/memory/` content. Verified knowledge updates are handled by `fons4ai-knowledge-summary` when explicitly triggered.
