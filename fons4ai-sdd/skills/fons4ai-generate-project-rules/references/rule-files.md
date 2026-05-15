# Architect-Grade Rule Files Reference

Use this reference when creating or updating project-level Markdown rule files.
The goal is to generate rules that a senior architect would accept: grounded in facts, explicit about boundaries, and useful during later implementation.

## Default File Set

Generate these files by default:

- `.specify/rules/code-style-rule.md`
- `.specify/rules/project-structure-rule.md`
- `.specify/rules/features-rule.md`
- `.specify/rules/testing-rule.md`
- `.specify/rules/data-ddl-rule.md`

Do not generate an index file, `AGENTS.md`, `.cursorrules`, or Cursor-specific rules unless the user explicitly asks.

## Modes and Evidence Depth

Use Existing Project Mode when the repository already contains build files, source files, tests, or established rules. Rules must cite observed conventions or truth-source facts.

Use New Project Mode when the repository is empty or the user wants rules before implementation. Rules must distinguish:

- `已确认规则`: decisions directly stated by the user or visible in scaffold files.
- `默认建议`: conservative conventions chosen to make future implementation consistent.
- `待补充约定`: decisions that require future code, architecture, or team preference.

Before drafting, build a compact evidence matrix:

| Rule file | Evidence source | Confirmed facts | Defaults | Open questions |
| --- | --- | --- | --- | --- |
| code-style-rule.md |  |  |  |  |

Required evidence sources to inspect when present:

- root build files and module build files;
- major source, resource, and test roots;
- representative classes from each major module;
- existing rule/spec/agent instruction files;
- `.specify/memory/business-architecture.md`, `technical-architecture.md`, `data-architecture.md`, and `constitution.md`;
- `.specify/sql/**/*.sql`;
- test framework, test naming, fixtures, mocks, and regression checks;
- migration scripts, ORM models, mapper XML, entity classes, repositories, and transaction boundaries.

## Shared Document Structure

Each rule file must contain:

- `## 项目事实`: repository facts or user decisions supporting the rules.
- `## 强制规则`: non-negotiable constraints that future implementation must follow.
- `## 推荐规则`: preferred practices that may have justified exceptions.
- `## 禁止事项`: actions that create defects, inconsistency, or governance risk.
- `## 例外机制`: when and how a rule may be bypassed.
- `## 待确认约定`: unknowns that must not be presented as facts.
- `## 验收检查`: checklist for reviewing whether future changes followed the rule.

Key rules should include a trigger condition, execution requirement, and exception path.

## `code-style-rule.md`

Purpose: define source-level coding conventions.

Must cover:

- language, framework, runtime, and annotation conventions;
- package, class, method, field, constant, DTO/VO/BO/entity, enum, and test naming;
- dependency injection, visibility, null handling, validation, and type boundaries;
- comment policy for key logic, domain fields, non-obvious decisions, and public contracts;
- exception handling, business error codes, logging levels, sensitive-data masking, and i18n if present;
- formatting and static analysis only when supported by repository evidence.

Common mistakes to prevent:

- generic style rules that contradict existing code;
- forcing a formatter or library that the repo does not use;
- omitting exception, logging, and sensitive-data constraints.

Acceptance checks:

- every mandatory style rule has evidence or an explicit default label;
- uncertain conventions appear under `待确认约定`;
- examples use project-like names rather than generic placeholders when evidence exists.

## `project-structure-rule.md`

Purpose: define module, package, file placement, and dependency boundaries.

Must cover:

- repository module layout and parent-child build relationships;
- source, resource, generated, migration, and test directory placement;
- package naming and layer boundaries;
- controller/API, service/application, domain, persistence, adapter, config, constants, strategy, utility, and shared module placement;
- dependency direction between modules and layers;
- where `.specify/rules/`, `specs/`, project-local skills, `.specify/memory/`, and `.specify/sql/` belong.

Common mistakes to prevent:

- inventing modules or layers that do not exist;
- allowing lower-level modules to depend on app-level modules;
- mixing infrastructure adapters into domain or application logic without a boundary.

Acceptance checks:

- major modules and directories are named when discoverable;
- dependency direction is explicit;
- missing module decisions are labeled as `待确认约定`.

## `features-rule.md`

Purpose: define how new features and behavior changes should be planned and implemented.

Must cover:

- requirement clarification before design or coding;
- S1/S2 SDD usage, artifact paths, and confirmation gates;
- fact-first repository investigation;
- technical design before non-trivial implementation;
- task breakdown aligned with TDD;
- reuse of existing utilities, components, and local conventions;
- migration, compatibility, rollback, observability, and security considerations;
- knowledge sync for `.specify/memory/`, `.specify/sql/`, `.specify/rules/`, `docs/`, or other truth sources.

Common mistakes to prevent:

- heavyweight process for tiny safe edits;
- direct implementation of ambiguous feature requests;
- changing public behavior or data semantics without SDD change analysis.

Acceptance checks:

- feature workflow explains when to use S1 and S2;
- implementation is gated by approved tasks;
- durable business, technical, data, or governance facts have a knowledge-sync path.

## `testing-rule.md`

Purpose: define automated and manual verification rules.

Must cover:

- test pyramid or practical test layering used by the repository;
- unit, integration, contract/API, persistence, UI, and manual verification expectations where applicable;
- test naming, package placement, fixture data, mocks/stubs/fakes, clock/randomness control, and external dependency isolation;
- RED-GREEN-REFACTOR requirements for behavior changes;
- regression selection, failure triage, flaky test handling, and when manual validation is acceptable;
- commands or build profiles when discoverable.

Common mistakes to prevent:

- treating manual checks as a replacement for feasible automated tests;
- adding broad slow tests when a focused test proves the behavior;
- leaving bug fixes without a reproducible failing signal.

Acceptance checks:

- each implementation task can name a verification method;
- bug fixes include reproduction, root cause, regression, and manual verification;
- unavailable automated tests require a documented reason.

## `data-ddl-rule.md`

Purpose: define data model, persistence, migration, transaction, and DDL knowledge rules.

Must cover:

- persistent model ownership, entity/table naming, field naming, indexes, constraints, lifecycle, and audit fields when discoverable;
- transaction boundaries, consistency, idempotency, concurrency, and rollback expectations;
- migration script location and review expectations when the repo has migrations;
- `.specify/sql/<database_or_service>/<business_model>.sql` DDL knowledge files;
- same database/service plus cohesive business-model grouping;
- mandatory split when tables belong to different databases, service-owned schemas, or physical data sources.

Common mistakes to prevent:

- one table per file when strongly coupled tables belong to the same database and business model;
- merging DDL across databases or service-owned schemas;
- treating `.specify/sql/**/*.sql` as executable migrations;
- changing schema without updating SDD tasks and data architecture knowledge.

Acceptance checks:

- DDL grouping rule is explicit;
- every schema-changing feature has a DDL sync task or approved deferral;
- data architecture indexes generated SQL files when present.
