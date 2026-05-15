# Fons4AI SDD Artifact Contract

## Scope

This contract defines the shared SDD artifact rules for all `fons4ai-sdd-*` skills.
Feature artifacts use `specs/features/`. Bugfix artifacts use `specs/bugfixes/`. The default project truth sources are `.specify/memory/` and `.specify/sql/`, but projects may declare additional truth sources. Do not require branch hooks or GitHub issue conversion.

## Artifact Responsibilities

- `spec.md` is the requirement summary and acceptance document. It records background, requirement points, business rules, functional overview, workflow overview, impact overview, risk overview, AC, non-functional requirements, and candidate data/domain objects. It must not replace technical design.
- `plan.md` is the detailed technical design. It records repository facts, architecture design, implementation approach, key rule code sketches, state transitions, data structure changes, API/contract details, error handling, transaction and consistency, migration/rollback, AC mapping, and verification strategy. It must not replace executable tasks.
- `tasks.md` is the executable task breakdown. It converts `spec.md` and `plan.md` into task IDs with AC mapping, files, verification, quality checks, and done criteria.
- Planning artifacts are not implementation approval; implementation still requires the approval gate below.

## Project Knowledge

Use `.specify/memory/`, `.specify/sql/`, and `.specify/rules/` as default long-lived project fact sources when they exist. Also respect other project-declared truth sources such as `docs/`, API documents, product documents, custom rule directories, or external knowledge bases.

- `business-architecture.md` stores business domains, roles, processes, objects, and rules.
- `technical-architecture.md` stores module boundaries, layers, integrations, technical constraints, and non-functional decisions.
- `data-architecture.md` stores data domains, core objects, relationships, quality rules, metrics, and data flows.
- `.specify/sql/**/*.sql` stores one DDL SQL file per database-scoped business model. A file may contain multiple strongly related tables only when they belong to the same database/service and cohesive business model.
- `.specify/rules/` may contain project rules: `code-style-rule.md`, `project-structure-rule.md`, `features-rule.md`, `testing-rule.md`, and `data-ddl-rule.md`.
- `constitution.md`, when present, is governance context and must not be rewritten by SDD feature skills.

Feature artifacts under `specs/features/` can cite or be constrained by truth-source facts, but should not silently update knowledge sources. If a feature changes long-lived business, technical, data, governance, or other source-of-truth facts, record a knowledge impact and route the synchronization through an explicit documentation update.
If a feature changes concrete persistent data models, record `.specify/sql/` impact as well as `data-architecture.md` impact.

## Data Model DDL Sync

When SDD work adds, removes, renames, or changes a concrete persistent data model, table, column, index, constraint, relationship, or database-specific default:

- `spec.md` must record the expected data model or DDL impact when it is known from requirements.
- `plan.md` must name each impacted `.specify/sql/<database_or_service>/<business_model>.sql` file and state whether the action is add, update, rename, or no-op.
- `tasks.md` must include an explicit DDL synchronization task for every impacted SQL file, unless the plan records a user-approved deferral with owner and reason.
- `fons4ai-sdd-implement` may create or update `.specify/sql/**/*.sql` only when the selected task names the SQL file or when the implementation reveals a necessary schema change and the user approves updating the task/artifact scope.
- Generated SQL knowledge files are documentation artifacts, not migration scripts. Keep migration scripts in the repository's normal migration location when the project has one.
- Never merge DDL from different databases, service-owned schemas, or physical data sources into one SQL knowledge file, even when the tables belong to the same broad business area.

## Levels

- `S1` is the default for small changes, normal features, and one-module or small multi-module collaboration.
- `S2` is required for cross-core-module changes, database migrations, public API or public contract changes, permission/security changes, cache/MQ/rate-limit/transaction boundaries, compatibility risk, or high rollback cost.
- Keep the classification limited to `S1` and `S2`; small safe changes use concise S1 artifacts.

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

- `spec.md` must include `## Requirement Summary`, `## Business Rules and Constraints`, `## Functional Overview`, and `## Impact Overview`.
- `spec.md` should include workflow and risk sections. S1 may use `not applicable, reason`; S2 must provide meaningful workflow, risk, data/domain, compatibility, security, and migration hints when applicable.
- `plan.md` must include `## Key Rule Code Sketches`, `## State Transition Design`, `## Data Structure Changes`, `## API and Contract Details`, `## Transaction and Consistency`, and `## Verification Strategy`.
- Code sketches in `plan.md` are design snippets or pseudocode for key rules, validation, status checks, and data transformations. They must be based on repository facts and must not be treated as production code.
- State transition and data structure sections may use `not applicable, reason` for S1 when genuinely absent. S2 high-risk sections must be filled with concrete facts or explicit deferrals.

## Implementation Approval Gate

- Planning artifacts are not implementation approval: `spec.md`, `plan.md`, `tasks.md`, and CR files define scope and tasks but do not authorize business-code implementation.
- `tasks.md` and each CR with incremental tasks must contain `## Implementation Approval Gate`.
- Requirements, design, task, and change skills must stop after writing planning artifacts and must not invoke implementation.
- Implementation approval must come from the user's latest message.
- If the latest user message confirms execution without task IDs, `fons4ai-sdd-implement` executes all unfinished tasks in dependency order.
- If the latest user message names task IDs such as `执行 T001,T002`, only those unfinished tasks are selected.
- Ambiguous messages such as `看看`, `下一步是什么`, or generated planning artifacts alone are not implementation approval.

## Editing Rules

- Read existing artifacts before editing them.
- Ask before replacing or materially rewriting existing artifacts.
- Preserve user or prior-agent changes outside the requested scope.
- Do not write business code from requirements, design, task, or change skills.
- Treat truth sources such as `.specify/memory/`, `.specify/sql/`, `.specify/rules/`, and `docs/` as read-only unless the active skill is explicitly responsible for knowledge-base initialization or a selected SDD task explicitly requires knowledge, rules, or DDL synchronization.
