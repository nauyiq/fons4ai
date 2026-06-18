# Fons4AI SDD Artifact Contract

## Scope

This contract defines the shared SDD artifact rules for all `fons4ai-sdd-*` skills.
Feature artifacts use `spec/features/<yyyymmdd>/`. Bugfix artifacts use `specs/bugfixes/`. The default project truth sources are `.specify/memory/` and `.specify/sql/`, but projects may declare additional truth sources. Do not require branch hooks or GitHub issue conversion.

## Artifact Responsibilities

- `需求说明书.md` is the business-oriented requirement specification. It records a concise clarification summary, business background and goals, scope, roles and scenarios, requirement list, business rules, simple workflows when useful, lightweight business-data meaning, impact, AC, quality requirements, risks, assumptions, and open items. It should use plain business language and avoid unnecessary professional or technical terminology. It must not expose repository-fact inventories, knowledge-context inventories, modules, classes, tables, columns, DDL paths, MCP details, or technical architecture details.
- `plan.md` is the technical design specification. It records design goals and scope, architecture, key business-rule and strategy landing, scenario implementation, data flow, DDD-lite decisions, key rule code sketches, state transitions, interface/contract details, data-model and ER design, error handling, transaction and consistency, migration/rollback, AC mapping, a concise knowledge-sync checklist, and verification strategy. It must not expose repository-fact inventories, knowledge-base-fact inventories, or search traces, and it must not replace executable tasks.
- `tasks.md` is the executable task breakdown. It converts `需求说明书.md` and `plan.md` into task IDs with AC mapping, files, verification, quality checks, and done criteria.
- Planning artifacts are not implementation approval; implementation still requires the approval gate below.

## Clarification Approval Gate

Requirements and change planning must close blocking ambiguity before formal artifact generation.

- `使用 SDD`, `继续`, `先生成`, `看一下`, existing artifact files, or a partially inferred plan are not clarification approval.
- `fons4ai-sdd-requirements` must ask the highest-impact requirement question first when blocking ambiguity can change scope, AC, business terms, data meaning, compatibility, security, integration, SDD level, or task breakdown.
- `fons4ai-sdd-change` must ask the highest-impact change question first when blocking ambiguity can change existing feature semantics, AC changes, naming/ownership, public behavior, data model, DDL source, migration, rollback, risk gates, or affected modules.
- While a blocking ambiguity exists, requirements and change skills must not write a formal `需求说明书.md`, formal CR, `plan.md`, `tasks.md`, or business code.
- If the user explicitly asks for a draft before answering, the artifact may be written only with `文档状态：草案-待确认`; it must not be used by design, task, or implementation skills.
- Formal `需求说明书.md` and CR artifacts must not expose clarification-gate tables, clarification status, or internal question logs. Clarification remains an internal pre-generation workflow.
- Design and task skills must stop if the input `需求说明书.md` or CR is marked `文档状态：草案-待确认`, `阻塞-等待回答`, `草案-含待确认`, `blocking`, or `draft`.

## Project Knowledge

Use `.specify/memory/`, `.specify/sql/`, and `.specify/rules/` as default long-lived project fact sources when they exist. Also respect other project-declared truth sources such as `docs/`, API documents, product documents, custom rule directories, or external knowledge bases.

- `.specify/memory/index.md` is the default memory entrypoint when present.
- Project-level `业务架构.md`, `技术架构.md`, and `数据架构.md` are concise global overview documents.
- Domain-level documents live under `.specify/memory/domains/<domain-slug>/` and carry detailed business, technical, and data knowledge for one domain.
- Knowledge cards live under `.specify/memory/domains/<domain-slug>/cards/` and store fact-level retrievable knowledge: business scenarios, rules, state transitions, technical flows, interface contracts, data models, and governance rules.
- `.specify/sql/**/*.sql` stores one DDL SQL file per database-scoped business model. A file may contain multiple strongly related tables only when they belong to the same database/service and cohesive business model.
- SQL knowledge should come from real DDL evidence: configured database MCP query results or existing repository SQL DDL files. Entities, ORM metadata, Mapper interfaces, repository methods, and Java field types may locate candidate models, but must not be used to generate `CREATE TABLE`.
- If multiple database MCP tools or multiple plausible databases are available, ask the user to select the MCP tool/database scope before retrieving DDL unless explicit user input or repository facts identify one unambiguously.
- Generated SQL knowledge files keep database/service, business model, included tables, status, update date, and DDL only. They must not contain MCP/Tool identifiers, query text, repository source paths, or provenance headers such as `Source`, `Migration Script`, or `DDL Evidence`.
- `.specify/rules/` may contain project rules: `code-style-rule.md`, `project-structure-rule.md`, `features-rule.md`, `testing-rule.md`, and `data-ddl-rule.md`.
- `constitution.md`, when present, is governance context and must not be rewritten by SDD feature skills.

Feature artifacts under `spec/features/<yyyymmdd>/` can cite or be constrained by truth-source facts, but should not silently update knowledge sources. If a feature changes long-lived business, technical, data, governance, or other source-of-truth facts, record a knowledge impact and route the synchronization through an explicit documentation update that updates affected domain documents, knowledge cards, and `.specify/memory/index.md`.
If a feature changes concrete persistent data models, record `.specify/sql/` impact as well as the affected domain `数据架构.md` and project data index impact.

## Context Loading

- Do not bulk-load all of `.specify/memory/`, `.specify/sql/`, `.specify/rules/`, `specs/`, or `docs/` by default.
- First read `AGENTS.md`, the active SDD artifacts, and the directly affected source/test/config files.
- If `.specify/memory/index.md` exists, read it before reading project-level memory documents.
- Use `rg --files` and `rg -n` with feature names, domain names, module names, business objects, table names, API names, error text, `REQ-###`, and `AC-###` to locate relevant cards, domain documents, SQL, rules, and specs before reading.
- Optionally use `scripts/find_relevant_context.py --root <repo-root> <keyword...>` from this skill to get a first-pass candidate list for index, cards, domain memory, SQL, rules, specs, and docs. Treat its output as navigation help, not as verified evidence.
- For S1, read only relevant rules, knowledge cards, domain documents, related SQL files, and affected code paths.
- For S2, expand context around the impacted domain, module, contract, security, transaction, or data model, but still avoid unrelated full-document loading. Cross-domain work may require project-level overview sections.
- For `.specify/sql/`, prefer `index.md`, domain `数据架构.md`, and targeted path search. Read only the database/service and business-model SQL files involved in the work; use `.specify/sql/pending/` when ownership is unknown.
- Full scans are appropriate for knowledge-base initialization, rule generation, explicit audits, or broad refactors, but should still start with a file inventory and evidence matrix.

## Data Model DDL Sync

When SDD work adds, removes, renames, or changes a concrete persistent data model, table, column, index, constraint, relationship, or database-specific default:

- `需求说明书.md` records only the business meaning and user-facing impact of data. Keep this section lightweight. Technical data-model, table, column, and DDL impact belong in `plan.md`.
- `plan.md` must name each impacted `.specify/sql/<database_or_service>/<business_model>.sql` file and state whether the action is add, update, rename, or no-op.
- `tasks.md` must include an explicit DDL synchronization task for every impacted SQL file, unless the plan records a user-approved deferral with owner and reason.
- `fons4ai-sdd-implement` may create or update `.specify/sql/**/*.sql` only when the selected task names the SQL file or when the implementation reveals a necessary schema change and the user approves updating the task/artifact scope.
- Generated SQL knowledge files are documentation artifacts, not migration scripts. Keep migration scripts in the repository's normal migration location when the project has one.
- When an approved implementation changes columns, indexes, constraints, defaults, or relationships of an existing table and the corresponding `.specify/sql/<database_or_service>/<business_model>.sql` already contains confirmed baseline DDL, the plan and tasks must require an executable change DDL script containing the needed `ALTER TABLE` or equivalent statements.
- Prefer the repository's established migration-script location for executable change DDL. If no migration location is established, use `spec/features/<yyyymmdd>/ddl-changes/<change-id>-<database_or_service>-<business_model>.sql`, where `<change-id>` is `INIT` for initial feature work or `CR-xxx` for an incremental change.
- Executable change DDL is generated only during approved implementation, not during requirements, design, task planning, or change planning. It records the operation to execute; `.specify/sql/**/*.sql` separately records the resulting current structure. Like other generated SQL artifacts, it must not contain MCP/Tool identifiers, query text, or source-path/provenance metadata.
- If no repository SQL file exists, query the configured database MCP service for actual DDL. If no MCP DDL and no repository SQL DDL are available, mark SQL evidence as `待确认` and ask for MCP configuration or SQL files instead of fabricating table structure.
- Use `.specify/sql/pending/<business_model>.sql` only when ownership is unknown or the user explicitly requests a pending placeholder.
- Never merge DDL from different databases, service-owned schemas, or physical data sources into one SQL knowledge file, even when the tables belong to the same broad business area.
- Creating or updating SQL knowledge files does not require executing `../fons4ai-project-knowledge-base-init/scripts/validate_sql_knowledge.py`. Use that script only when the user explicitly requests SQL artifact validation or when diagnosing malformed existing SQL knowledge files.

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
  plan.md
  tasks.md
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

## Traceability

- Every `REQ-###` in `需求说明书.md` must map to at least one `AC-###` through the requirement summary table or AC text.
- Every AC in `需求说明书.md` must be covered by at least one design decision in `plan.md`.
- `plan.md` should preserve REQ context in AC mapping when it materially affects implementation.
- Every implementation task in `tasks.md` must include `AC:`, `Files:`, `Verification:`, `Quality:`, and `Done:`.
- Every task should map to at least one AC ID. If a task is pure setup, use the nearest AC it enables and explain that relationship in `Done:`.
- S2 tasks must include explicit regression and risk-control tasks.
- S2 implementation reports must state whether checklist, rollback, compatibility, and risk-control tasks were closed or explicitly deferred.

## Detailed Document Requirements

- Generated artifact headings and fixed prose should be Chinese-first. Keep file names, IDs, paths, and machine-readable task labels such as `AC:`, `Files:`, `Verification:`, `Quality:`, and `Done:` in English when compatibility requires it.
- New `需求说明书.md` artifacts should use the simplified requirement-spec structure: `## 一句话说明`, `## 需求澄清摘要`, `## 背景与目标`, `## 需求范围`, `## 角色与场景`, `## 需求列表`, `## 业务规则`, `## 业务流程`, `## 业务数据口径`, `## 影响说明`, `## 验收标准`, `## 质量要求`, `## 风险与待确认`, and `## 版本修订记录`.
- New `plan.md` artifacts must include `## 设计目标与范围`, `## 总体架构设计`, `## 核心业务规则与策略落地`, `## 核心业务场景实现`, `## 数据流设计`, `## 领域建模决策`, `## 关键规则代码片段`, `## 状态流转设计`, `## 接口与契约设计`, `## 数据模型与 ER 设计`, `## 事务与一致性`, `## 异常处理与日志`, `## 工具包与依赖决策`, `## 迁移、兼容与回滚`, `## 验证策略`, `## AC 映射`, `## 知识同步清单`, and `## 风险与待确认事项`.
- The business-rule and strategy section must map core rules or policies to modules, domain/application objects, data dependencies, extension points, and verification. Use Mermaid sequence, flow, or state diagrams for important business or strategy flows when facts support them.
- Code sketches in `plan.md` are design snippets or pseudocode for key rules, validation, status checks, and data transformations. They must be based on repository facts and must not be treated as production code.
- Use Mermaid `sequenceDiagram` for core call chains, `flowchart` for complex decisions, `stateDiagram-v2` for state changes, and `erDiagram` for new tables, relationship changes, or multi-table collaboration when facts support them. A single-column or single-index adjustment may mark the ER diagram as `不适用，原因`.
- State transition and data-model sections may use `不适用，原因` for S1 when genuinely absent. S2 high-risk sections must be filled with concrete facts or explicit deferrals.

## Implementation Approval Gate

- Planning artifacts are not implementation approval: `需求说明书.md`, `plan.md`, `tasks.md`, and CR files define scope and tasks but do not authorize business-code implementation.
- `tasks.md` and each CR with incremental tasks must contain `## 实现确认门禁`.
- Requirements, design, task, and change skills must stop after writing planning artifacts and must not invoke implementation.
- Implementation approval must come from the user's latest message.
- `fons4ai-sdd-implement` must record approval evidence in the implementation report, quoting or summarizing the latest user message that authorized execution. If that evidence cannot be identified, implementation must stop.
- If the latest user message confirms execution without task IDs, `fons4ai-sdd-implement` executes all unfinished tasks in dependency order.
- If the latest user message names task IDs such as `执行 T001,T002`, only those unfinished tasks are selected.
- Ambiguous messages such as `看看`, `下一步是什么`, or generated planning artifacts alone are not implementation approval.

## Editing Rules

- Read existing artifacts before editing them.
- Ask before replacing or materially rewriting existing artifacts.
- Preserve user or prior-agent changes outside the requested scope.
- Do not write business code from requirements, design, task, or change skills.
- Treat truth sources such as `.specify/memory/`, `.specify/sql/`, `.specify/rules/`, and `docs/` as read-only unless the active skill is explicitly responsible for knowledge-base initialization or a selected SDD task explicitly requires knowledge, rules, or DDL synchronization.
