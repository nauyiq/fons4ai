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
- migration scripts, ORM models, mapper XML, entity classes, repositories, query SQL, database configuration, and transaction boundaries. Migration scripts are strong evidence but not a prerequisite for DDL knowledge generation.

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
- utility package priority: JDK standard library, project utilities/components, already-introduced third-party utility packages such as Hutool, Apache Commons, Guava, then new dependencies;
- dependency addition gates for new utility libraries, including rationale, alternatives, impact, and confirmation requirements;
- readability, method complexity, expressive naming, and duplicate-code control;
- DDD-lite domain expression: rich domain behavior naming, state transition methods, invariant encapsulation, and acceptable anemic-model exceptions;
- dependency injection, visibility, null handling, validation, and type boundaries;
- comment policy for key logic, domain fields, non-obvious decisions, and public contracts;
- exception handling, business error codes, logging levels, sensitive-data masking, and i18n if present;
- formatting and static analysis only when supported by repository evidence.

Common mistakes to prevent:

- generic style rules that contradict existing code;
- forcing a formatter or library that the repo does not use;
- requiring Hutool, Apache Commons, Guava, or any tool library when the project has not introduced or approved it;
- hand-written string, collection, date/time, IO, bean conversion, null-check, or assertion logic when an existing project or approved third-party utility already covers it;
- spreading core domain behavior through setters, controllers, mappers, or application services when a domain object or domain method should own it;
- omitting exception, logging, and sensitive-data constraints.

Acceptance checks:

- every mandatory style rule has evidence or an explicit default label;
- utility and dependency rules follow the existing-first strategy and do not force new dependencies;
- readability, complexity, and duplicate-code checks are explicit;
- DDD-lite expression rules are present without forcing full DDD architecture;
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
- DDD-lite boundary mapping for domain, application, infrastructure, and adapter responsibilities based on the repository's existing structure;
- where `.specify/rules/`, `specs/`, project-local skills, `.specify/memory/`, and `.specify/sql/` belong.

Common mistakes to prevent:

- inventing modules or layers that do not exist;
- forcing a full DDD package hierarchy for a single feature instead of mapping DDD-lite responsibilities to existing structure;
- allowing lower-level modules to depend on app-level modules;
- mixing infrastructure adapters into domain or application logic without a boundary.

Acceptance checks:

- major modules and directories are named when discoverable;
- dependency direction is explicit;
- DDD-lite boundary rules protect domain code from infrastructure concerns while allowing lightweight CRUD exceptions;
- missing module decisions are labeled as `待确认约定`.

## `features-rule.md`

Purpose: define how new features and behavior changes should be planned and implemented.

Must cover:

- requirement clarification before design or coding;
- S1/S2 SDD usage, artifact paths, and confirmation gates;
- fact-first repository investigation with progressive context loading: file inventory, keyword search, targeted truth-source sections, and scoped source/test reads;
- technical design before non-trivial implementation;
- task breakdown aligned with TDD;
- reuse of existing utilities, components, and local conventions;
- DDD-lite implementation rules for business behavior ownership, rich model usage, anemic-model exceptions, application-layer orchestration, and domain-service conditions;
- migration, compatibility, rollback, observability, and security considerations;
- knowledge sync for `.specify/memory/`, `.specify/sql/`, `.specify/rules/`, `docs/`, or other truth sources.

Common mistakes to prevent:

- heavyweight process for tiny safe edits;
- full-loading every `.specify/memory/`, `.specify/sql/`, `.specify/rules/`, `specs/`, or `docs/` file when targeted search would be enough;
- direct implementation of ambiguous feature requests;
- putting core business rules in controllers, mappers, adapters, or long application-service methods without a documented DDD-lite exception;
- changing public behavior or data semantics without SDD change analysis.

Acceptance checks:

- feature workflow explains when to use S1 and S2;
- context sources are targeted and listed, with skipped truth sources explained when relevant;
- implementation is gated by approved tasks;
- core business behavior has a DDD-lite ownership decision or an explicit lightweight exception;
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
- SQL knowledge generation from real DDL evidence: configured database MCP query results or existing repository SQL DDL files;
- user selection before DDL retrieval when multiple MCP tools or plausible databases exist and current facts do not uniquely identify the target;
- SQL artifact privacy: generated `.specify/sql/**/*.sql` files must not store MCP/Tool identifiers, queries, source paths, or provenance headers;
- executable change DDL: when SDD implementation alters an existing table with confirmed original DDL in `.specify/sql/`, require a separate copy-executable `ALTER TABLE` or equivalent script in the established migration location or `specs/features/<feature-slug>/ddl-changes/<change-id>-<database_or_service>-<business_model>.sql`, without MCP/Tool identifiers or provenance metadata;
- stage boundary: design/tasks/CR artifacts plan the executable DDL path and rollback needs, while only approved implementation writes the executable SQL file;
- prohibition on generating `CREATE TABLE` from entities, ORM metadata, mapper interfaces, repositories, Java fields, or code-only guesses;
- `.specify/sql/<database_or_service>/<business_model>.sql` DDL knowledge files;
- `.specify/sql/pending/<business_model>.sql` fallback when database/service ownership is unknown;
- same database/service plus cohesive business-model grouping;
- mandatory split when tables belong to different databases, service-owned schemas, or physical data sources.

Common mistakes to prevent:

- one table per file when strongly coupled tables belong to the same database and business model;
- merging DDL across databases or service-owned schemas;
- skipping SQL knowledge files because no migration script exists;
- presenting inferred columns, indexes, or constraints as confirmed facts;
- treating `.specify/sql/**/*.sql` as executable migrations;
- updating the current-state SQL knowledge file without generating the required executable change DDL for an existing-table structural change;
- changing schema without updating SDD tasks and data architecture knowledge.

Acceptance checks:

- DDL grouping rule is explicit;
- every schema-changing feature has a DDL sync task or approved deferral;
- missing migration scripts still result in SQL knowledge files with `推断` or `待确认` evidence status;
- data architecture indexes generated SQL files when present.
- confirmed existing-table schema changes include an executable change DDL deliverable distinct from the SQL knowledge snapshot.
