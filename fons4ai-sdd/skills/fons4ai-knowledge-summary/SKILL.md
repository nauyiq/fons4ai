---
name: fons4ai-knowledge-summary
description: "Fons4AI gated generic knowledge-summary workflow. Auto-trigger only when an in-scope AGENTS.md enables the Fons4AI routing marker; otherwise use only when the user explicitly names this skill or asks for the Fons4AI/SDD workflow. Use to merge verified project facts into architecture memory documents and SQL knowledge with scenario-to-technical traceability."
---

# Fons4AI Knowledge Summary

## Activation Gate

Before using this skill, verify at least one condition is true:

1. The user explicitly names this skill, such as `$fons4ai-knowledge-summary`.
2. The user explicitly asks to use Fons4AI, SDD, or the Fons4AI workflow.
3. The active repository has an in-scope `AGENTS.md` containing `<!-- fons4ai-skill-routing: enabled -->`.

If none is true, do not apply this skill automatically. Continue with normal Codex behavior or ask whether the user wants to enable the Fons4AI workflow.

## Overview

Use this skill to turn scattered project knowledge into durable source-of-truth documents.
The first supported truth source is `.specify/memory/`, but the workflow must not assume this is the only target. Future targets may include `.specify/rules/`, `docs/`, product documents, API catalogs, or other configured knowledge bases.

Common input sources:

- SDD artifacts under `specs/features/<feature-slug>/`: `spec.md`, `plan.md`, `tasks.md`, `changes/`, `reports/`, `contracts/`, and S2 checklists.
- Existing knowledge documents under `.specify/memory/`, `.specify/rules/`, `docs/`, or user-provided paths.
- Verified code changes, tests, migration notes, SQL files, release notes, and explicit user facts.

Common output targets:

- `.specify/memory/business-architecture.md`
- `.specify/memory/technical-architecture.md`
- `.specify/memory/data-architecture.md`
- `.specify/sql/**/*.sql` as database-scoped business-model SQL knowledge
- Other source-of-truth documents named by the user or existing project conventions

## Required Context

1. Identify the requested source artifacts and target knowledge base.
   - If no target is specified, inventory likely truth sources with `rg --files` first, then inspect matching domains, modules, APIs, data models, existing memory documents, SQL knowledge, and repository guidance.
   - If multiple plausible truth sources exist, ask the user to choose before editing.
2. Read existing target documents before writing.
3. Read relevant source artifacts completely enough to distinguish verified facts from plans, assumptions, and open questions.
4. When data model or SQL knowledge is involved, locate relevant `.specify/sql/**/*.sql` files through `data-architecture.md`, path search, table/model names, or SQL references.

## Workflow

1. Confirm summary intent.
   - Determine whether the user wants initialization, incremental merge, conflict cleanup, or post-implementation knowledge sync.
   - Prefer completed reports, checked tasks, passing tests, reviewed change records, and explicit user confirmation as evidence.
   - Treat planned-only items as `待确认` unless the user asks for a planning summary.

2. Classify durable knowledge.
   - Business knowledge: capabilities, actors, scenarios, business objects, detailed processes, decisions, exception paths, rules, statuses, and glossary terms.
   - Technical knowledge: module boundaries, APIs, integrations, extension points, business-rule implementation, domain/application landing, strategy/policy design, transactions, caches, queues, security, observability, and operational constraints.
   - Data knowledge: data domains, persistent objects, relationships, lifecycle, quality rules, metrics, SQL files, and migration constraints.
   - Governance knowledge: principles, coding standards, compatibility rules, review gates, and decision records.

3. Maintain a generic scenario ledger.
   - Extract scenario names from repository facts or explicit user input. Do not treat any domain example as mandatory.
   - For `.specify/memory/`, map business facts to `business-architecture.md`, technical landing to `technical-architecture.md`, and data facts to `data-architecture.md`.
   - Every core scenario added to or retained in business architecture must have a matching technical landing row unless it is explicitly marked `待确认` with a reason.
   - Technical landing must cover entrypoint, orchestration service, domain/strategy object, data access, external collaboration, transaction boundary, exception path, and verification.
   - Add scenario-specific diagrams only when source facts support participants and event ordering; do not use one generic diagram to represent all scenarios.

4. Merge conservatively.
   - Preserve existing structure and user edits.
   - Update the smallest relevant section instead of appending duplicate content.
   - Ask before deleting facts, replacing large sections, or changing governance documents.
   - When old knowledge conflicts with verified new evidence, mark stale content as superseded or replace it only with clear source evidence.
   - Do not promote local debugging notes, abandoned alternatives, transient task details, or unverified implementation guesses into truth sources.

5. Preserve traceability.
   - Add or update a concise `变更记录`, `知识来源`, or equivalent section when the target document has one.
   - Record date, source artifact path, and a one-line summary.
   - Prefer exact source references such as `specs/features/<feature-slug>/reports/<report>.md` over vague descriptions.

6. Handle SQL knowledge when relevant.
   - If data architecture references a concrete persistent model, table group, or user-specified data model, ensure the corresponding `.specify/sql/<database_or_service>/<business_model>.sql` exists and is listed.
   - Prefer `../fons4ai-project-knowledge-base-init/scripts/generate_sql_knowledge.py` for full or broad regeneration.
   - Same-database cohesive business model tables may share one SQL file; cross-database or cross-service tables must remain separate.
   - Migration scripts are evidence, not a prerequisite. If no migration script exists, derive SQL knowledge from entity classes, ORM annotations, mapper XML, repository interfaces, query SQL, database config, existing SQL files, and explicit user facts.
   - Do not invent unsupported DDL as confirmed truth. Mark inferred or incomplete structures as `推断` or `待确认`.
   - Keep SQL `COMMENT` clauses short, business-readable, and free of mojibake. Put source evidence, Java field names, Java types, inferred SQL types, and pending nullability in `-- Field Evidence:` blocks, not inside SQL `COMMENT`.

7. Validate before finishing.
   - Every durable fact added to a truth source must be backed by a source artifact or explicit user fact.
   - No planned-only item should be represented as completed behavior.
   - Target documents should remain concise and non-duplicative.
   - Untouched knowledge documents should not be rewritten.
   - If SQL files were updated, run `../fons4ai-project-knowledge-base-init/scripts/validate_sql_knowledge.py --sql-root .specify/sql --repo-root . --strict-comments` when Python is available.
   - If memory files were updated, run `../fons4ai-project-knowledge-base-init/scripts/validate_memory_knowledge.py --memory-root .specify/memory` when Python is available.

## Output Rules

- Update only the selected knowledge-base documents and explicitly scoped SQL knowledge files.
- When selected sources confirm or introduce persistent models, treat matching `.specify/sql/**/*.sql` files as in scope even if migration scripts do not exist.
- Do not write business code.
- Do not create new SDD requirements, plans, or implementation tasks unless the user explicitly asks.
- Do not invent business, technical, data, or governance facts.
- End with touched knowledge files, source artifacts, summarized facts, deferred gaps, conflicts resolved, validation results, and suggested follow-up.
