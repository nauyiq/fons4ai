# Fons4AI SDD Artifact Contract

## Scope

This contract defines the shared SDD artifact rules for all `fons4ai-sdd-*` skills.
Feature artifacts use `specs/features/`. Bugfix artifacts use `specs/bugfixes/`. The default project truth sources are `.specify/memory/` and `.specify/sql/`, but projects may declare additional truth sources. Do not require branch hooks or GitHub issue conversion.

## Artifact Responsibilities

- `spec.md` is the requirement summary and acceptance document. It records background, requirement points, business rules, functional overview, workflow overview, impact overview, risk overview, AC, non-functional requirements, and candidate data/domain objects. It must not replace technical design.
- `plan.md` is the detailed technical design. It records repository facts, architecture design, implementation approach, key business-rule and strategy landing, key rule code sketches, state transitions, data structure changes, API/contract details, error handling, transaction and consistency, migration/rollback, AC mapping, and verification strategy. It must not replace executable tasks.
- `tasks.md` is the executable task breakdown. It converts `spec.md` and `plan.md` into task IDs with AC mapping, files, verification, quality checks, and done criteria.
- Planning artifacts are not implementation approval; implementation still requires the approval gate below.

## Project Knowledge

Use `.specify/memory/`, `.specify/sql/`, and `.specify/rules/` as default long-lived project fact sources when they exist. Also respect other project-declared truth sources such as `docs/`, API documents, product documents, custom rule directories, or external knowledge bases.

- `business-architecture.md` stores business domains, roles, processes, objects, and rules.
- `technical-architecture.md` stores module boundaries, layers, integrations, technical constraints, and non-functional decisions.
- `data-architecture.md` stores data domains, core objects, relationships, quality rules, metrics, and data flows.
- `.specify/sql/**/*.sql` stores one DDL SQL file per database-scoped business model. A file may contain multiple strongly related tables only when they belong to the same database/service and cohesive business model.
- Missing migration scripts do not exempt DDL knowledge generation. If a persistent model or table group is known from entities, ORM metadata, mapper SQL, repository code, database configuration, existing SQL, or explicit user facts, create or update the SQL knowledge file and mark incomplete facts as `推断` or `待确认`.
- `.specify/rules/` may contain project rules: `code-style-rule.md`, `project-structure-rule.md`, `features-rule.md`, `testing-rule.md`, and `data-ddl-rule.md`.
- `constitution.md`, when present, is governance context and must not be rewritten by SDD feature skills.

Feature artifacts under `specs/features/` can cite or be constrained by truth-source facts, but should not silently update knowledge sources. If a feature changes long-lived business, technical, data, governance, or other source-of-truth facts, record a knowledge impact and route the synchronization through an explicit documentation update.
If a feature changes concrete persistent data models, record `.specify/sql/` impact as well as `data-architecture.md` impact.

## Context Loading

- Do not bulk-load all of `.specify/memory/`, `.specify/sql/`, `.specify/rules/`, `specs/`, or `docs/` by default.
- First read `AGENTS.md`, the active SDD artifacts, and the directly affected source/test/config files.
- Use `rg --files` and `rg -n` with feature names, module names, business objects, table names, API names, error text, `REQ-###`, and `AC-###` to locate relevant knowledge sections before reading.
- Optionally use `scripts/find_relevant_context.py --root <repo-root> <keyword...>` from this skill to get a first-pass candidate list for memory, rules, SQL, specs, and docs. Treat its output as navigation help, not as verified evidence.
- For S1, read only relevant rules, matching truth-source sections, related SQL files, and affected code paths.
- For S2, expand context around the impacted domain, module, contract, security, transaction, or data model, but still avoid unrelated full-document loading.
- For `.specify/sql/`, prefer `data-architecture.md` index plus targeted path search. Read only the database/service and business-model SQL files involved in the work; use `.specify/sql/pending/` when ownership is unknown.
- Full scans are appropriate for knowledge-base initialization, rule generation, explicit audits, or broad refactors, but should still start with a file inventory and evidence matrix.

## Data Model DDL Sync

When SDD work adds, removes, renames, or changes a concrete persistent data model, table, column, index, constraint, relationship, or database-specific default:

- `spec.md` must record the expected data model or DDL impact when it is known from requirements.
- `plan.md` must name each impacted `.specify/sql/<database_or_service>/<business_model>.sql` file and state whether the action is add, update, rename, or no-op.
- `tasks.md` must include an explicit DDL synchronization task for every impacted SQL file, unless the plan records a user-approved deferral with owner and reason.
- `fons4ai-sdd-implement` may create or update `.specify/sql/**/*.sql` only when the selected task names the SQL file or when the implementation reveals a necessary schema change and the user approves updating the task/artifact scope.
- Generated SQL knowledge files are documentation artifacts, not migration scripts. Keep migration scripts in the repository's normal migration location when the project has one.
- If no migration script exists, derive the SQL knowledge file from the best available code or user evidence. Use `.specify/sql/pending/<business_model>.sql` when database/service ownership is unknown.
- Do not skip SQL knowledge files because evidence is partial. Generate the file with known columns and commented `TODO`/`待确认` lines for unknown fields, indexes, constraints, and relationships.
- Never merge DDL from different databases, service-owned schemas, or physical data sources into one SQL knowledge file, even when the tables belong to the same broad business area.
- After creating or updating SQL knowledge files, run `../fons4ai-project-knowledge-base-init/scripts/validate_sql_knowledge.py --sql-root .specify/sql` when Python is available. If the script cannot run, perform the same header, grouping, status, source, and `CREATE TABLE` checks manually and report the reason.

## Levels

- `S1` is the default for small changes, normal features, and one-module or small multi-module collaboration.
- `S2` is required for cross-core-module changes, database migrations, public API or public contract changes, permission/security changes, cache/MQ/rate-limit/transaction boundaries, compatibility risk, or high rollback cost.
- Keep the classification limited to `S1` and `S2`; small safe changes use concise S1 artifacts.
- S1 artifacts use the minimal complete profile: keep required sections, AC coverage, task quality fields, and verification details, but mark truly absent state transitions, API changes, data changes, migrations, rollback, and diagrams as `不适用，原因` instead of generating speculative content.

## Paths

Use this feature layout:

```text
specs/features/<feature-slug>/
  spec.md
  plan.md
  tasks.md
  checklists/
  contracts/
  changes/
  reports/
```

Only create optional folders when they are needed.

## Naming

- `<feature-slug>` must be lowercase hyphen-case, short, and action-noun oriented.
- Requirement IDs use `REQ-001`, `REQ-002`, ...
- AC IDs use `AC-001`, `AC-002`, ...
- Task IDs use `T001`, `T002`, ...
- Change records use `CR-001`, `CR-002`, ...

## Traceability

- Every `REQ-###` in `spec.md` must map to at least one `AC-###` through the requirement summary table or AC text.
- Every AC in `spec.md` must be covered by at least one design decision in `plan.md`.
- `plan.md` should preserve REQ context in AC mapping when it materially affects implementation.
- Every implementation task in `tasks.md` must include `AC:`, `Files:`, `Verification:`, `Quality:`, and `Done:`.
- Every task should map to at least one AC ID. If a task is pure setup, use the nearest AC it enables and explain that relationship in `Done:`.
- S2 tasks must include explicit regression and risk-control tasks.
- S2 implementation reports must state whether checklist, rollback, compatibility, and risk-control tasks were closed or explicitly deferred.

## Detailed Document Requirements

- Generated artifact headings and fixed prose should be Chinese-first. Keep file names, IDs, paths, and machine-readable task labels such as `AC:`, `Files:`, `Verification:`, `Quality:`, and `Done:` in English when compatibility requires it.
- `spec.md` must include `## 需求概要`, `## 关键业务规则与约束`, `## 功能概览`, and `## 影响面概览`. Legacy English headings are accepted only for existing artifacts.
- `spec.md` should include workflow and risk sections. S1 may use `不适用，原因`; S2 must provide meaningful workflow, risk, data/domain, compatibility, security, and migration hints when applicable.
- `plan.md` must include `## 关键业务规则与策略设计`, `## 关键规则代码片段`, `## 状态流转设计`, `## 数据结构变更`, `## API 与契约细节`, `## 事务与一致性`, and `## 验证策略`. Legacy English headings are accepted only for existing artifacts.
- The business-rule and strategy section must map core rules or policies to modules, domain/application objects, data dependencies, extension points, and verification. Use Mermaid sequence, flow, or state diagrams for important business or strategy flows when facts support them.
- Code sketches in `plan.md` are design snippets or pseudocode for key rules, validation, status checks, and data transformations. They must be based on repository facts and must not be treated as production code.
- State transition and data structure sections may use `不适用，原因` for S1 when genuinely absent. S2 high-risk sections must be filled with concrete facts or explicit deferrals.

## Implementation Approval Gate

- Planning artifacts are not implementation approval: `spec.md`, `plan.md`, `tasks.md`, and CR files define scope and tasks but do not authorize business-code implementation.
- `tasks.md` and each CR with incremental tasks must contain `## 实现确认门禁`. Legacy `## Implementation Approval Gate` is accepted only for existing artifacts.
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
