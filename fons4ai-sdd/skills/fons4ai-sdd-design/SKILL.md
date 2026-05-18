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

Use this skill after `fons4ai-sdd-requirements` has produced `spec.md`.
The output is a detailed technical design in `specs/features/<feature-slug>/plan.md`; S2 features may also need `contracts/`, `data-model.md`, or migration notes when the design requires them.

## Required Context

1. Load `../fons4ai-sdd-requirements/references/sdd-artifact-contract.md`.
2. Read `specs/features/<feature-slug>/spec.md` completely.
3. Read project guidance: `AGENTS.md`, relevant project rules under `.specify/rules/` (`code-style-rule.md`, `project-structure-rule.md`, `features-rule.md`, `testing-rule.md`, `data-ddl-rule.md`), `.specify/memory/technical-architecture.md`, `.specify/memory/data-architecture.md`, `.specify/sql/`, `.specify/memory/constitution.md` if present, existing architecture notes under `specs/`, build files, and representative source/test files for affected modules.
4. Use `assets/templates/plan-template.md`.
5. If `plan.md` already exists, read it and ask before replacing or materially rewriting it.

## Workflow

1. Confirm the SDD level from `spec.md`. If repository facts show the level should be `S2`, upgrade it in the design summary and ask the user before editing `spec.md`.
2. Build a fact base from the repository:
   - Existing modules, layers, package conventions, reusable utilities, components, test style, domain objects, application services, and integration boundaries.
   - Current APIs, data objects, domain rules, state transitions, configs, caches, queues, transactions, permissions, dependencies, utility packages, and extension points relevant to the feature.
   - Long-lived architecture and data facts from `.specify/memory/` and `.specify/sql/` when available.
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
   - Describe architecture design, implementation approach, data flow, affected areas, API/contract details, error handling, transaction and consistency, migration and rollback, and verification strategy.
   - Include key rule code sketches for important business rules, validation, status checks, or data transformations. These sketches must be short pseudocode or code-like snippets based on repository facts, existing types, existing utilities, approved dependencies, and DDD-lite domain methods.
   - Do not write final production code in `plan.md`; code sketches explain intent and edge cases for later implementation.
   - Record state transition design with source state, trigger, preconditions, next state, failure handling, and idempotency. Use a table by default; use Mermaid `stateDiagram` only when facts support it.
   - Record data structure changes with fields, types, defaults, indexes, constraints, compatibility, DDL path, migration, and rollback expectations when applicable.
5. For S1, keep the design practical but not empty. If there is no state transition, data structure change, API change, migration, or rule snippet, write `not applicable, reason` in that section.
6. For S2, include the additional governance sections that apply:
   - Compatibility and migration impact.
   - Rollback plan.
   - Security/permission analysis.
   - Transaction, cache, MQ, rate-limit, or concurrency risks.
   - Public contract changes under `contracts/` when needed.
   - Data model notes under `data-model.md` when database or persistent schema changes are involved.
   - A concrete DDL sync plan naming every impacted `.specify/sql/<database_or_service>/<business_model>.sql` file for persistent data model additions or changes.
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
