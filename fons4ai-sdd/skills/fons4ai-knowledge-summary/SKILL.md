---
name: fons4ai-knowledge-summary
description: "Fons4AI gated generic knowledge-summary workflow. Auto-trigger only when an in-scope AGENTS.md enables the Fons4AI routing marker; otherwise use only when the user explicitly names this skill or asks for the Fons4AI/SDD workflow. Use to merge verified project facts into layered architecture memory: .specify/memory/index.md, concise project-level documents, domain-level documents, fact-level knowledge cards, and SQL knowledge."
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
The default truth source is the layered memory model under `.specify/memory/`:

- `.specify/memory/index.md` is the default knowledge entrypoint.
- Project-level `业务架构.md`, `技术架构.md`, and `数据架构.md` are concise global overviews.
- Domain-level documents under `.specify/memory/domains/<domain-slug>/` carry detailed business, technical, and data knowledge.
- Knowledge cards under `.specify/memory/domains/<domain-slug>/cards/` carry fact-level retrieval units.
- `.specify/sql/**/*.sql` stores database-scoped current DDL snapshots.

Future targets may include `.specify/rules/`, `docs/`, product documents, API catalogs, or other configured knowledge bases, but do not assume they replace `.specify/memory/index.md` unless the user says so.

## Required Context

1. Identify the requested source artifacts and target knowledge base.
   - If no target is specified, inventory likely truth sources with `rg --files` first.
   - If `.specify/memory/index.md` exists, read it before reading project-level documents.
   - Read relevant cards and domain documents before reading full project-level architecture documents.
   - If multiple plausible truth sources or domains exist, ask the user to choose before editing.
2. Read existing target documents before writing.
3. Read relevant source artifacts completely enough to distinguish verified facts from plans, assumptions, and open questions.
4. When data model or SQL knowledge is involved, locate relevant `.specify/sql/**/*.sql` files through `index.md`, domain data documents, `数据架构.md`, path search, table/model names, or SQL references.

## Workflow

1. Confirm summary intent.
   - Determine whether the user wants initialization, incremental merge, conflict cleanup, post-implementation knowledge sync, or domain knowledge upgrade.
   - Prefer completed reports, checked tasks, passing tests, reviewed change records, and explicit user confirmation as evidence.
   - Treat planned-only items as `待确认` unless the user asks for a planning summary.

2. Classify durable knowledge.
   - Project-level knowledge: business lines, domain list, cross-domain collaboration, global technical patterns, global data domains, governance constraints.
   - Domain business knowledge: capabilities, actors, scenarios, business objects, processes, decisions, exception paths, rules, statuses, and glossary terms.
   - Domain technical knowledge: module boundaries, APIs, integrations, extension points, business-rule implementation, domain/application landing, strategies, transactions, caches, queues, security, observability, and operational constraints.
   - Domain data knowledge: data objects, relationships, lifecycle, quality rules, metrics, SQL files, and migration constraints.
   - Governance knowledge: principles, coding standards, compatibility rules, review gates, and decision records.

3. Decide target layer.
   - If the fact changes system-wide navigation, cross-domain relationships, global technology, or global data governance, update project-level documents and `index.md`.
   - If the fact belongs to one domain, update `.specify/memory/domains/<domain-slug>/...` and the matching cards.
   - If the fact spans multiple domains, update each affected domain and the cross-domain section in `index.md` or project-level documents.
   - If no suitable domain exists, create a new domain directory only when repository facts or user input justify it, then update `index.md`.

4. Maintain cards as the smallest durable fact unit.
   - Create or update a card for core scenarios, business rules, state transitions, technical flows, interface contracts, data models, and governance rules that future agents should retrieve directly.
   - Use IDs such as `KC-BIZ-001`, `KC-TECH-001`, `KC-DATA-001`, and `KC-GOV-001`.
   - Every card must include `知识编号`, `知识类型`, `所属领域`, `状态`, `来源`, `关联场景`, `关联对象`, `关联代码/接口/SQL`, and `更新日期`.
   - Do not delete or merge cards without confirmation. Prefer setting `状态：已废弃` with a replacement reference when knowledge is superseded.

5. Merge conservatively.
   - Preserve existing structure and user edits.
   - Update the smallest relevant section instead of appending duplicate content.
   - Ask before deleting facts, replacing large sections, or changing governance documents.
   - When old knowledge conflicts with verified new evidence, mark stale content as superseded or replace it only with clear source evidence.
   - Do not promote local debugging notes, abandoned alternatives, transient task details, or unverified implementation guesses into truth sources.

6. Preserve traceability.
   - Add or update concise `变更记录`, `知识来源`, or equivalent sections when target documents have them.
   - Record date, source artifact path, and a one-line summary.
   - Prefer exact source references such as `specs/features/<feature-slug>/reports/<report>.md` over vague descriptions.
   - For cards, keep the source concise and useful; do not paste long evidence excerpts.

7. Handle SQL knowledge when relevant.
   - If data architecture or a data card references a concrete persistent model, table group, or user-specified data model, ensure the corresponding `.specify/sql/<database_or_service>/<business_model>.sql` exists and is listed in the relevant domain data document and `index.md`.
   - Preferred DDL source is a configured database MCP service. Use read-only database MCP queries to retrieve real DDL, then save it under `.specify/sql/<database_or_service>/<business_model>.sql`.
   - If multiple database MCP tools or candidate databases exist and explicit user input or project facts do not identify one target unambiguously, ask the user which MCP tool/database scope to use before querying DDL.
   - Secondary DDL source is existing repository SQL files. If SQL knowledge must be updated, follow the project's SQL knowledge or migration process directly; do not rely on the knowledge-base initialization skill to import SQL files.
   - Same-database cohesive business model tables may share one SQL file; cross-database or cross-service tables must remain separate.
   - Do not generate SQL DDL from entity classes, ORM annotations, Mapper interfaces, repository method names, or inferred Java field types. Code facts may locate candidate tables or business models, but they are not DDL evidence.
   - If neither MCP DDL nor repository SQL DDL is available, mark the SQL source as `待确认` in the relevant domain data document and `index.md`, then ask for MCP configuration or SQL files instead of fabricating `CREATE TABLE`.
   - SQL knowledge files must not contain MCP/Tool identifiers, query text, repository source paths, `Source`, `Migration Script`, or `DDL Evidence` metadata.

8. Validate before finishing.
   - Every durable fact added to a truth source must be backed by a source artifact or explicit user fact.
   - No planned-only item should be represented as completed behavior.
   - Project-level documents should remain concise; domain documents and cards carry details.
   - Untouched knowledge documents should not be rewritten.
   - Updating SQL files does not by itself require running `../fons4ai-project-knowledge-base-init/scripts/validate_sql_knowledge.py`. Run it only when the user explicitly requests SQL artifact validation or when diagnosing malformed existing SQL knowledge files.
   - If memory files were updated, run `../fons4ai-project-knowledge-base-init/scripts/validate_memory_knowledge.py --memory-root .specify/memory` when Python is available.

## Output Rules

- Update only selected knowledge-base documents, domain documents, cards, `index.md`, and explicitly scoped SQL knowledge files.
- When selected sources confirm or introduce persistent models, treat matching `.specify/sql/**/*.sql` files as in scope only when MCP DDL or repository SQL DDL evidence is available, or when the user explicitly requests a pending placeholder.
- Do not write business code.
- Do not create new SDD requirements, plans, or implementation tasks unless the user explicitly asks.
- Do not invent business, technical, data, or governance facts.
- End with touched knowledge files, touched cards, source artifacts, summarized facts, deferred gaps, conflicts resolved, validation results, and suggested follow-up.
