---
name: fons4ai-knowledge-summary
description: "Fons4AI gated knowledge-summary workflow. Auto-trigger only when an in-scope AGENTS.md contains '<!-- fons4ai-skill-routing: enabled -->'; otherwise use only when the user explicitly names this skill or asks for the Fons4AI/SDD workflow."
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
The first supported truth source is `.specify/memory/`, but the workflow must not assume that this is the only target. Future targets may include `.specify/rules/`, `docs/`, product documents, API catalogs, or other configured knowledge bases.

Common input sources:

- SDD artifacts under `specs/features/<feature-slug>/`: `spec.md`, `plan.md`, `tasks.md`, `changes/`, `reports/`, `contracts/`, and S2 checklists.
- Existing knowledge documents under `.specify/memory/`, `.specify/rules/`, `docs/`, or user-provided paths.
- Verified code changes, tests, migration notes, DDL files, release notes, and explicit user facts.

Common output targets:

- `.specify/memory/business-architecture.md`
- `.specify/memory/technical-architecture.md`
- `.specify/memory/data-architecture.md`
- `.specify/sql/**/*.sql` as database-scoped business-model DDL knowledge, when explicitly in scope
- `.specify/rules/code-style-rule.md`, `.specify/rules/project-structure-rule.md`, `.specify/rules/features-rule.md`, `.specify/rules/testing-rule.md`, and `.specify/rules/data-ddl-rule.md` as project rule truth sources, when explicitly in scope
- Other source-of-truth documents named by the user or existing project conventions

## Required Context

1. Identify the requested source artifacts and target knowledge base.
   - If no target is specified, inspect `.specify/memory/` first, then `.specify/rules/` five-file project rules, `docs/`, and repository guidance.
   - If multiple plausible truth sources exist, ask the user to choose before editing.
2. Read existing target documents before writing.
3. Read relevant source artifacts completely enough to distinguish verified facts from plans, assumptions, and open questions.
4. When summarizing SDD output, load `../fons4ai-sdd-requirements/references/sdd-artifact-contract.md` if present.
5. When data model or DDL knowledge is involved, read relevant `.specify/sql/**/*.sql` files and normal migration/model files.

## Workflow

1. Confirm summary intent.
   - Determine whether the user wants initialization, incremental merge, conflict cleanup, or post-implementation knowledge sync.
   - Prefer completed reports, checked tasks, passing tests, reviewed change records, and explicit user confirmation as evidence.
   - Treat planned-only items as `待确认` unless the user asks for a planning summary.

2. Classify durable knowledge.
   - Business knowledge: capabilities, actors, scenarios, business objects, rules, processes, statuses, and glossary terms.
   - Technical knowledge: module boundaries, APIs, integrations, extension points, transactions, caches, queues, security, observability, and operational constraints.
   - Data knowledge: data domains, persistent objects, relationships, lifecycle, quality rules, metrics, DDL files, and migration constraints.
   - Governance knowledge: principles, coding standards, compatibility rules, review gates, and decision records.

3. Build a source-to-target map before editing.
   - For `.specify/memory/`, map business facts to `business-architecture.md`, technical facts to `technical-architecture.md`, and data facts to `data-architecture.md`.
   - For other knowledge bases, follow existing headings, document ownership, and repository rules.
   - If the target has no clear structure, propose a compact section plan before writing.

4. Merge conservatively.
   - Preserve existing structure and user edits.
   - Update the smallest relevant section instead of appending duplicate content.
   - Ask before deleting facts, replacing large sections, or changing governance documents.
   - When old knowledge conflicts with verified new evidence, mark stale content as superseded or replace it only with clear source evidence.
   - Do not promote local debugging notes, abandoned alternatives, transient task details, or unverified implementation guesses into the truth source.

5. Preserve traceability.
   - Add or update a concise `变更记录`, `知识来源`, or equivalent section when the target document has one.
   - Record date, source artifact path, and a one-line summary.
   - Prefer exact source references such as `specs/features/<feature-slug>/reports/<report>.md` over vague descriptions.

6. Handle DDL knowledge when relevant.
   - If data architecture references a concrete persistent model or table group, ensure the corresponding `.specify/sql/<database_or_service>/<business_model>.sql` is listed when available.
   - Same-database cohesive business model tables may share one SQL file; cross-database or cross-service tables must remain in separate SQL files.
   - Do not invent DDL. If SQL knowledge is missing, record the gap as `待确认` or recommend a follow-up task.
   - Treat `.specify/sql/**/*.sql` as knowledge files, not as a replacement for project migration scripts.

7. Validate before finishing.
   - Every durable fact added to a truth source must be backed by a source artifact or explicit user fact.
   - No planned-only item should be represented as completed behavior.
   - Target documents should remain concise and non-duplicative.
   - Untouched knowledge documents should not be rewritten.

## Output Rules

- Update only the selected knowledge-base documents and explicitly scoped DDL knowledge files.
- Do not write business code.
- Do not create new SDD requirements, plans, or implementation tasks unless the user explicitly asks.
- Do not invent business, technical, data, or governance facts.
- End with touched knowledge files, source artifacts, summarized facts, deferred gaps, conflicts resolved, and suggested follow-up.
