---
name: fons4ai-project-knowledge-base-init
description: "Fons4AI gated generic project knowledge-base initialization workflow. Auto-trigger only when an in-scope AGENTS.md enables the Fons4AI routing marker; otherwise use only when the user explicitly names this skill or asks for the Fons4AI workflow. Use to initialize architecture memory documents and database-scoped SQL knowledge from repository facts."
---

# Fons4AI Project Knowledge Base Init

## Activation Gate

Before using this skill, verify at least one condition is true:

1. The user explicitly names this skill, such as `$fons4ai-project-knowledge-base-init`.
2. The user explicitly asks to use Fons4AI, SDD, or the Fons4AI workflow.
3. The active repository has an in-scope `AGENTS.md` containing `<!-- fons4ai-skill-routing: enabled -->`.

If none is true, do not apply this skill automatically. Continue with normal Codex behavior or ask whether the user wants to enable the Fons4AI workflow.

## Overview

Use this skill to initialize a generic project knowledge base from repository facts and user-provided context.
The default document output location is `.specify/memory/`; SQL knowledge files live under `.specify/sql/`.

Default output files:

- `.specify/memory/business-architecture.md`
- `.specify/memory/technical-architecture.md`
- `.specify/memory/data-architecture.md`
- `.specify/sql/<database_or_service>/<business_model>.sql` for each discovered or user-specified persistent business model group. If the database/service is unknown, use `.specify/sql/pending/<business_model>.sql` until confirmed.

Read the matching template before drafting each document:

- `references/business-architecture-template.md`
- `references/technical-architecture-template.md`
- `references/data-architecture-template.md`

## Workflow

1. Inspect available facts before writing.
   - Build a file inventory with `rg --files`, then inspect project guidance, existing knowledge, build files, module names, ORM models, mapper XML, repository interfaces, API contracts, representative source files, and user-provided context.
   - Do not load every discovered file into context. Read indexes, headings, and representative files first; expand only around confirmed domains, modules, integrations, and persistent models.
   - Prefer observed project facts over generic assumptions. Concrete business names must come from the current repository or explicit user facts, not from this skill.

2. Handle existing target files conservatively.
   - If any target document or SQL file already exists, read it first.
   - Explain whether the work should merge, replace, or append.
   - Ask for confirmation before replacing existing target documents or SQL files unless the user already requested a rebuild.
   - Preserve unrelated `.specify/memory` and `.specify/sql` files; do not delete or rename files without explicit confirmation.

3. Build a scenario ledger before writing memory documents.
   - Identify core business scenarios and high-value capabilities from actual code, docs, specs, APIs, jobs, messages, controllers, services, and data models.
   - Do not hardcode any domain names as mandatory scenarios. Terms such as loan, repayment, payment, refund, order, settlement, approval, cancellation, and reconciliation are examples only.
   - For each scenario, capture trigger, participants, goal, rules, decision points, exception paths, state/data changes, outputs, source evidence, and evidence status.
   - If facts are partial, keep the scenario and mark uncertain fields as `待确认`; do not collapse important scenarios into one-line summaries.

4. Generate architecture documents.
   - Use `business-architecture-template.md` for goals, scope, stakeholders, capabilities, scenario ledger, rule orchestration, business objects, collaboration boundaries, and open questions.
   - Use `technical-architecture-template.md` as the landing document for the business scenario ledger. Every actual core scenario from business architecture must have a technical landing row covering entrypoint, orchestration service, domain/strategy object, data access, integrations, transaction boundary, exception path, and verification.
   - Technical diagrams must be scenario-specific. Include Mermaid `sequenceDiagram`, `flowchart`, or `stateDiagram-v2` when the repository gives enough participants/events. If facts are partial, keep the landing table and mark unknown nodes as `待确认`.
   - Use `data-architecture-template.md` for data goals, domains, objects, relationships, SQL file index, data flows, lifecycle, quality, and risks.

5. Generate SQL knowledge files when persistent models are in scope.
   - Prefer `scripts/generate_sql_knowledge.py --repo-root <repo> --sql-root <repo>/.specify/sql --database <database_or_service>`.
   - Use `--groups <group_a,group_b>` only as an optional filter. The script must not require project-specific group names.
   - Include persistent models backed by Mapper XML/resultMap, DAO/BaseDao, ORM table annotations, repository interfaces, query SQL, mapper-bound entity classes, or explicit table evidence.
   - Exclude Criteria, Key, DTO, Request, Response, and non-persistent fields such as `@TableField(exist = false)`, unless explicit table evidence proves persistence.
   - Treat migration scripts as strong evidence, not a prerequisite. Without real DDL, generate knowledge SQL with `推断` or `待确认` markers.
   - Keep SQL `COMMENT` clauses business-readable and short: use only cleaned field/table meaning. Do not put Java field names, Java types, evidence paths, raw JavaDoc, HTML, `@link`, `@return`, mojibake, or long metadata into SQL `COMMENT`; put evidence in `-- Field Evidence:` blocks instead.

6. Preserve evidence quality.
   - Write Markdown documents in Chinese unless the user explicitly requests another language.
   - Separate `已确认`, `推断`, and `待确认`.
   - Do not invent business rules, technology choices, schemas, integrations, metrics, or ownership.
   - Keep KISS, but do not over-compress critical scenarios, rule orchestration, state changes, exception paths, and data interactions.

7. Validate before finishing.
   - Run `scripts/validate_sql_knowledge.py --sql-root .specify/sql --repo-root . --strict-comments` after SQL files are generated or updated.
   - Run `scripts/validate_memory_knowledge.py --memory-root .specify/memory` after memory documents are generated or updated.
   - Confirm Markdown headings, tables, Mermaid blocks, SQL headers, status values, source evidence, `CREATE TABLE` blocks, scenario-to-technical mapping, and absence of mojibake.
   - Report created or updated paths, key evidence sources, main `待确认` items, and validation results.

## SQL File Contract

Each generated SQL file should use this structure:

```sql
-- Database/Service: <database_or_service>
-- Business Model: <business_model>
-- Tables: <table_1>, <table_2>
-- Source: <repository evidence or 待确认>
-- Status: <已确认 | 推断 | 待确认>
-- Migration Script: <path | none | 待确认>
-- Last Generated: YYYY-MM-DD

CREATE TABLE `<table_name>` (
  `<column_name>` <type> NULL COMMENT '<clean business meaning>'
);

-- Field Evidence:
-- - `<column_name>`; java=<field>; type=<java_type>; source=<evidence>; sql=<inferred_sql_type>; nullable=待确认
```

If no reliable business meaning is available for a column, omit the column `COMMENT` instead of filling it with evidence metadata.

Use one SQL file per database-scoped business model. Multiple strongly related tables may share one file only when they belong to the same database/service and cohesive business model. Cross-database or cross-service tables must remain separate.

## Output Contract

When no path is specified, create the default knowledge base under `.specify/memory/` and SQL knowledge files under `.specify/sql/`.
Do not create extra index files, README files, or example outputs unless the user explicitly asks.
