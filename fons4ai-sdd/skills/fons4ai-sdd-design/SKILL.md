---
name: fons4ai-sdd-design
description: "Fons4AI gated SDD technical-design workflow. Auto-trigger only when an in-scope AGENTS.md contains '<!-- fons4ai-skill-routing: enabled -->'; otherwise use only when the user explicitly names this skill or asks for the Fons4AI/SDD workflow."
---

# Fons4AI SDD Design

## Activation Gate

Before using this skill, verify at least one condition is true:

1. The user explicitly names this skill, such as `$fons4ai-sdd-design`.
2. The user explicitly asks to use Fons4AI, SDD, or the Fons4AI workflow.
3. The active repository has an in-scope `AGENTS.md` containing `<!-- fons4ai-skill-routing: enabled -->`.

If none is true, do not apply this skill automatically. Continue with normal Codex behavior or ask whether the user wants to enable the Fons4AI workflow.

## Overview

Use this skill after `fons4ai-sdd-requirements` has produced a formal `spec.md` with closed clarification status.
The output is a detailed technical design in `specs/features/<feature-slug>/plan.md`; S2 features may also need `contracts/`, `data-model.md`, or migration notes when the design requires them.

## Required Context

1. Load `../fons4ai-sdd-requirements/references/sdd-artifact-contract.md`, including its context-loading rules.
2. Read `specs/features/<feature-slug>/spec.md` completely and confirm the clarification gate is closed.
3. Search first by feature terms, AC/REQ IDs, modules, domain objects, APIs, tables, and error/risk terms. Do not bulk-read all project rules, memory, SQL, specs, or docs by default.
   - Optionally run `../fons4ai-sdd-requirements/scripts/find_relevant_context.py --root <repo-root> <keyword...>` to get candidate truth-source files before reading.
4. Read `AGENTS.md`, relevant project rules, matching truth-source sections, targeted SQL files, build files, and representative source/test files for affected modules.
5. Use `assets/templates/plan-template.md`.
6. If `plan.md` already exists, read it and ask before replacing or materially rewriting it.

## Workflow

1. Confirm the SDD level and clarification status from `spec.md`.
   - If `spec.md` says `澄清状态：阻塞-等待回答`, `澄清状态：草案-含待确认`, `Clarification Status: blocking`, or `Clarification Status: draft`, stop and route back to `fons4ai-sdd-requirements`.
   - If a legacy `spec.md` has no clarification gate, do a quick ambiguity scan. If blocking requirement ambiguity remains, stop and ask to run `fons4ai-sdd-requirements` before design.
   - If repository facts show the level should be `S2`, upgrade it in the design summary and ask the user before editing `spec.md`.
2. Build a fact base from the repository:
   - Existing modules, layers, package conventions, reusable utilities, components, test style, domain objects, application services, and integration boundaries.
   - Current APIs, data objects, domain rules, state transitions, configs, caches, queues, transactions, permissions, dependencies, utility packages, and extension points relevant to the feature.
   - Relevant long-lived architecture and data facts from `.specify/memory/` and targeted `.specify/sql/` files when available.
   - Any conflict between truth-source and code facts; mark likely stale knowledge explicitly instead of silently overriding it.
3. Design the simplest implementation that satisfies all AC.
   - Prefer existing helpers and patterns.
   - Apply DDD-lite for business behavior: place core business rules, state transitions, validation, and invariants in domain objects or domain methods when the repository structure supports it.
   - Keep application services focused on orchestration, transactions, permissions, external collaboration, and persistence coordination.
   - Do not force full DDD package structures for simple CRUD, read-only queries, thin wrappers, or repositories that do not already support that shape; record the lightweight exception instead.
   - If business rules stay in service/application code, record why: thin logic, existing project style, orchestration-only concern, or unstable domain abstraction.
   - For common utility work such as string, collection, date/time, IO, bean conversion, null-check, assertion, encoding, encryption, or masking, record the intended tool choice.
   - Use this priority for utility decisions: JDK standard library, project utilities/components, already-introduced third-party utilities such as Hutool, Apache Commons, or Guava, then new dependency.
   - Avoid introducing new frameworks, modules, abstractions, or dependencies unless the repository facts justify them.
   - If a new dependency is needed, record rationale, alternatives, impact, and user/design confirmation before implementation.
4. Write a detailed technical design, not just a lightweight plan:
   - Describe architecture design, implementation approach, key business-rule and strategy landing, data flow, affected areas, API/contract details, error handling, transaction and consistency, migration and rollback, and verification strategy.
   - For core business rules, policies, scoring, routing, approval, permission, pricing, status transition, or matching logic, record the technical landing: module, domain/application object, strategy component, data dependency, transaction boundary, extension point, and verification approach.
   - Include Mermaid `sequenceDiagram`, `flowchart`, or `stateDiagram-v2` for important business-rule execution, strategy decisions, or core business flows when facts support it. If facts are partial, write `不适用，原因` or mark uncertain nodes as `待确认`; do not invent actors or systems.
   - Include key rule code sketches for important business rules, validation, status checks, or data transformations. These sketches must be short pseudocode or code-like snippets based on repository facts, existing types, existing utilities, approved dependencies, and DDD-lite domain methods.
   - Do not write final production code in `plan.md`; code sketches explain intent and edge cases for later implementation.
   - Record state transition design with source state, trigger, preconditions, next state, failure handling, and idempotency. Use a table by default; use Mermaid `stateDiagram` only when facts support it.
   - Record data structure changes with fields, types, defaults, indexes, constraints, compatibility, DDL path, migration, and rollback expectations when applicable.
   - When an existing `.specify/sql/<database_or_service>/<business_model>.sql` contains the baseline DDL for a table that will be altered, record the required executable change DDL target: use the repository migration directory when established, otherwise `specs/features/<feature-slug>/ddl-changes/<change-id>-<database_or_service>-<business_model>.sql`. Design names the artifact and expected `ALTER TABLE` intent but does not generate the executable SQL before implementation approval.
5. For S1, use the minimal complete profile: keep all required sections, cover every AC, and keep the design practical but not empty. If there is no state transition, data structure change, API change, migration, rollback, diagram, or rule snippet, write `不适用，原因` in that section instead of fabricating content.
6. For S2, include the additional governance sections that apply:
   - Compatibility and migration impact.
   - Rollback plan.
   - Security/permission analysis.
   - Transaction, cache, MQ, rate-limit, or concurrency risks.
   - Public contract changes under `contracts/` when needed.
   - Data model notes under `data-model.md` when database or persistent schema changes are involved.
   - A concrete DDL sync plan naming every impacted `.specify/sql/<database_or_service>/<business_model>.sql` file for persistent data model additions or changes.
   - For existing-table structural changes backed by an existing SQL knowledge baseline, a concrete executable change DDL plan naming the migration-script path or fallback `ddl-changes/` artifact path, plus forward-change and rollback expectations.
   - DDL knowledge files must be backed by real DDL evidence: configured database MCP query results or existing repository SQL files. Entities, ORM metadata, mapper interfaces, repository methods, and Java field types may locate candidate tables but must not generate `CREATE TABLE`.
   - If multiple candidate database MCP tools or databases could supply DDL and explicit user input or project facts do not uniquely select one, record the ambiguity and require user selection before DDL retrieval.
   - DDL knowledge SQL files must not persist MCP/Tool identifiers, query text, source paths, `Source`, `Migration Script`, or `DDL Evidence` headers.
   - If no MCP DDL and no repository SQL DDL are available, record the SQL source as `待确认` and add a follow-up to configure MCP or provide SQL files instead of fabricating schema details.
   - If database/service ownership is unknown, use `.specify/sql/pending/<business_model>.sql` and mark unresolved schema facts as `推断` or `待确认`.
   - DDL files are grouped by database/service plus cohesive business model. Same-database strongly related tables may share one file; cross-database or cross-service tables must use separate files.
7. Map every AC and relevant REQ to one or more design decisions.
8. Record whether this feature needs knowledge synchronization. For data model additions or changes, `.specify/sql/` synchronization is required unless the user explicitly defers it.

## Output Rules

- Create or update `specs/features/<feature-slug>/plan.md`.
- Generated artifact headings and fixed prose must be Chinese-first. Keep file names, IDs, paths, code identifiers, and technical terms such as `API`, `DDL`, `REQ-001`, and `AC-001` unchanged when needed.
- Create extra S2 artifacts only when they prevent concrete implementation mistakes.
- Do not generate `tasks.md`; leave task breakdown to `fons4ai-sdd-tasks`.
- Do not write business code.
- `plan.md` code snippets are design sketches only and must not become unreviewed production implementation.
- End with generated paths, SDD level, key risks, knowledge impact, and suggested next skill.
