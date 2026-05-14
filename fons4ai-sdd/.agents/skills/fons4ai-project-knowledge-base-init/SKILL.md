---
name: fons4ai-project-knowledge-base-init
description: "Fons4AI gated knowledge-base initialization workflow. Auto-trigger only when an in-scope AGENTS.md contains '<!-- fons4ai-skill-routing: enabled -->'; otherwise use only when the user explicitly names this skill or asks for the Fons4AI workflow."
---

# Fons4AI Project Knowledge Base Init

## Activation Gate

Before using this skill, verify at least one condition is true:

1. The user explicitly names this skill, such as `$fons4ai-project-knowledge-base-init`.
2. The user explicitly asks to use Fons4AI, SDD, or the Fons4AI workflow.
3. The active repository has an in-scope `AGENTS.md` containing `<!-- fons4ai-skill-routing: enabled -->`.

If none is true, do not apply this skill automatically. Continue with normal Codex behavior or ask whether the user wants to enable the Fons4AI workflow.

## Overview

Use this skill to initialize a project knowledge base from repository facts and user-provided context.
The default document output location is `.specify/memory/`; DDL SQL files live under `.specify/sql/`.

Default output files:

- `.specify/memory/business-architecture.md`
- `.specify/memory/technical-architecture.md`
- `.specify/memory/data-architecture.md`
- `.specify/sql/<table_or_model_name>.sql` for each concrete data model with known DDL

Read the matching template before drafting each document:

- `references/business-architecture-template.md`
- `references/technical-architecture-template.md`
- `references/data-architecture-template.md`

## Workflow

1. Inspect available facts before writing.
   - Read user-provided requirements, product notes, and target output path if provided.
   - Search `.specify/memory/`, `.specify/sql/`, `specs/`, `rules/`, README files, existing architecture notes, build files, module names, database migration files, ORM models, mapper XML, entity classes, and representative source files when code exists.
   - Prefer observed project facts over generic architecture assumptions.

2. Handle existing target files conservatively.
   - If any target document or SQL file already exists, read it first.
   - Explain whether the work should merge, replace, or append.
   - Ask the user for confirmation before modifying existing target documents or SQL files.
   - Preserve unrelated `.specify/memory` files such as `constitution.md`; do not rewrite them unless the user explicitly asks.
   - Preserve unrelated `.specify/sql` files; do not delete or rename DDL files without explicit confirmation.

3. Generate the architecture documents.
   - Create `.specify/memory/` if it does not exist.
   - Use `business-architecture-template.md` for business goals, scope, stakeholders, capabilities, processes, business objects, business rules, collaboration boundaries, and open questions.
   - Use `technical-architecture-template.md` for architecture goals, system view, module boundaries, layers, core technical flows, integrations, non-functional requirements, and risks.
   - Use `data-architecture-template.md` for data goals, data domains, core data objects, data relationships, data models, DDL file index, data flows, data quality, metric definitions, and risks.

4. Generate DDL SQL files when concrete schemas are known.
   - Create `.specify/sql/` if it does not exist.
   - Save each concrete table or data model as a separate SQL file: `.specify/sql/<table_or_model_name>.sql`.
   - Use lowercase snake_case file names that match the table or canonical model name.
   - Include a short SQL header comment with source evidence, model name, status, and last generated date.
   - Do not invent DDL for models that are not supported by repository facts or explicit user input. Record unsupported models as `待确认` in `data-architecture.md` instead.
   - Keep DDL database-specific only when the project has a confirmed database dialect. If the dialect is unknown, write portable SQL and mark dialect assumptions in the header.

5. Preserve evidence quality.
   - Write Markdown documents in Chinese unless the user explicitly requests another language.
   - Separate `已确认`, `推断`, and `待确认` content.
   - Do not invent business rules, technology choices, database schemas, integrations, metrics, or ownership.
   - Mark uncertain content as `待确认` instead of presenting it as fact.
   - Keep KISS: use compact sections, short tables, and diagrams only when they improve verification.

6. Validate before finishing.
   - Confirm all requested documents exist under `.specify/memory/` unless the user provided another explicit path.
   - Confirm every generated DDL file is indexed from `.specify/memory/data-architecture.md`.
   - Confirm Markdown headings, tables, Mermaid blocks, and SQL files are structurally complete.
   - Confirm generated content matches the discovered repository facts.
   - Report created or updated document paths, SQL paths, and the main `待确认` items.

## SQL File Contract

Each generated SQL file should use this structure:

```sql
-- Model: <model or table name>
-- Source: <repository file, user input, or待确认>
-- Status: <已确认 | 推断 | 待确认>
-- Last Generated: YYYY-MM-DD

CREATE TABLE <table_name> (
  id BIGINT PRIMARY KEY
);
```

Use one SQL file per concrete table or canonical data model. Avoid placing multiple unrelated tables in one file.

## Output Contract

When no path is specified, create the default knowledge base under `.specify/memory/` and DDL files under `.specify/sql/`.

Each generated Markdown document must start with:

```markdown
# <文档标题>

> 适用范围：<系统、模块或业务域>
> 生成依据：<用户输入、现有文档、代码或待确认>
> 文档状态：<初稿 | 已评审 | 待补充>
```

Do not create extra index files, README files, or example outputs unless the user explicitly asks.
