---
name: fons4ai-project-knowledge-base-init
description: "Fons4AI gated project knowledge-base initialization workflow. Auto-trigger only when an in-scope AGENTS.md enables the Fons4AI routing marker; otherwise use only when the user explicitly names this skill or asks for the Fons4AI workflow. Use to initialize layered project knowledge: .specify/memory/index.md, concise project-level architecture documents, domain-level architecture documents, and knowledge cards. SQL/DDL facts may be referenced from database MCP or existing repository SQL files, but this skill must not create SQL files."
---

# Fons4AI Project Knowledge Base Init

## Activation Gate

Before using this skill, verify at least one condition is true:

1. The user explicitly names this skill, such as `$fons4ai-project-knowledge-base-init`.
2. The user explicitly asks to use Fons4AI, SDD, or the Fons4AI workflow.
3. The active repository has an in-scope `AGENTS.md` containing `<!-- fons4ai-skill-routing: enabled -->`.

If none is true, do not apply this skill automatically. Continue with normal AI-agent behavior or ask whether the user wants to enable the Fons4AI workflow.

## Role

You are a senior system architect and knowledge architect. Your responsibility is to extract verifiable long-lived knowledge from the codebase, project documents, database facts, and user-provided context, then organize it into a layered project knowledge base.

Your goal is not to merely generate documents. You must identify business domains, core scenarios, technical implementation facts, data relationships, and reusable knowledge cards so future AI agents can load precise context and perform SDD, bugfix, and knowledge-summary work based on verified facts.

## Overview

Use this skill to initialize a layered project knowledge base from repository facts, database MCP facts, existing SQL files, and user-provided context.
Database MCP results and existing repository SQL files are context sources only. This skill must not create, import, copy, or update `.specify/sql/**/*.sql` files.
The default knowledge entrypoint is `.specify/memory/index.md`.

Default output shape:

```text
.specify/memory/
  index.md
  业务架构.md
  技术架构.md
  数据架构.md
  domains/
    <domain-slug>/
      业务架构.md
      技术架构.md
      数据架构.md
      cards/
        KC-BIZ-001-<slug>.md
        KC-TECH-001-<slug>.md
        KC-DATA-001-<slug>.md
```

Project-level architecture documents are concise navigation documents. Domain-level documents carry detailed business, technical, and data knowledge. Knowledge cards are fact-level context units for targeted retrieval.

Read templates as needed:

- `references/memory-index-template.md`
- `references/project-business-architecture-template.md`
- `references/project-technical-architecture-template.md`
- `references/project-data-architecture-template.md`
- `references/domain-business-architecture-template.md`
- `references/domain-technical-architecture-template.md`
- `references/domain-data-architecture-template.md`
- `references/knowledge-card-template.md`

## Workflow

1. Inspect available facts before writing.
   - Build a file inventory with `rg --files`, then inspect project guidance, existing knowledge, build files, module names, database config, repository SQL files, migration directories, API contracts, representative source files, and user-provided context.
   - Check whether local database MCP tools are configured for the target database. When available, use MCP query results only as transient DDL context for data architecture documentation; do not save query results as SQL files.
   - If multiple database MCP tools or candidate databases exist and explicit user input, project configuration, or ownership facts do not identify one target unambiguously, ask the user which MCP tool and database scope to use before retrieving DDL.
   - Do not query multiple candidate databases and merge DDL speculatively.
   - Do not load every discovered file into context. Read indexes, headings, and representative files first; expand only around confirmed business domains, modules, integrations, and persistent models.
   - Prefer observed project facts over generic assumptions. Concrete business names must come from the current repository or explicit user facts, not from this skill.

2. Handle existing target files conservatively.
   - If `.specify/memory/index.md`, project-level documents, domain documents, cards, or related SQL files already exist, read them before planning changes.
   - Explain whether the work should merge, replace, append, or upgrade legacy three-document memory.
   - Ask for confirmation before replacing existing target documents unless the user already requested a rebuild.
   - Preserve unrelated `.specify/memory` and `.specify/sql` files. Do not create, delete, rename, copy, import, or collapse SQL files in this workflow.

3. Identify business domains before writing.
   - Identify domains from actual business language in code, docs, specs, APIs, jobs, messages, controllers, services, data models, and user-provided context.
   - Use domain slugs such as `order`, `payment`, `inventory`, or project-specific equivalents. Do not split domains mechanically by table, Controller, Service, package, or module name.
   - If a larger business line is needed, represent it through the index and domain path convention chosen by the project; do not invent a mandatory hierarchy.
   - If a domain is uncertain, keep it in `index.md` as `待确认` instead of forcing a wrong directory.

4. Build a scenario and evidence ledger.
   - For each domain, capture core scenarios, trigger, participants, goal, rules, decision points, exception paths, state/data changes, outputs, source evidence, and evidence status.
   - Use scenario IDs scoped by domain when possible, such as `BS-ORDER-001`. Keep legacy `BS-001` only for old documents.
   - Do not collapse important scenarios into one-line summaries. If facts are partial, keep the scenario and mark uncertain fields as `待确认`.

5. Generate layered memory documents.
   - Create `.specify/memory/index.md` first with domain index, capability index, cross-domain collaboration, card index, and SQL/DDL reference index.
   - Create concise project-level documents:
     - `业务架构.md`: business lines/domains, global business rules, cross-domain business processes.
     - `技术架构.md`: system modules, public technical patterns, cross-domain calls, global non-functional constraints.
     - `数据架构.md`: data domains, cross-domain data relationships, SQL/DDL reference index, data governance.
   - For every confirmed or useful domain, create domain-level documents:
     - `domains/<domain>/业务架构.md`: domain responsibility, core objects, scenarios, rule orchestration, states, exceptions.
     - `domains/<domain>/技术架构.md`: scenario landing, application orchestration, domain objects, interfaces, transactions, exceptions, tests.
     - `domains/<domain>/数据架构.md`: domain data objects, relationships, SQL/DDL references, data flow, consistency.
   - Create knowledge cards under `domains/<domain>/cards/` for fact-level items that future agents should retrieve directly.
   - Update `index.md` after creating or changing domain documents, cards, or SQL/DDL references.

6. Record SQL/DDL references without generating SQL files.
   - Preferred source: a user-confirmed or unambiguous configured database MCP service. Use read-only database MCP queries to understand real table DDL, then summarize confirmed data facts in data architecture documents and cards.
   - Secondary source: existing repository SQL DDL files, such as migration scripts, schema files, `*.sql` init scripts, or checked-in database DDL. Reference the existing file path when it is useful, but do not copy, import, normalize, or rewrite it into `.specify/sql/`.
   - Do not generate SQL DDL from Java entities, Mapper interfaces, ORM annotations, DTOs, repository method names, query method names, or inferred Java field types. These code facts may help locate candidate business models or table names, but they are not DDL evidence.
   - Do not use any helper script, manual copy, or generated content to import or create SQL files in this workflow.
   - If neither MCP DDL nor repository SQL DDL is available, do not fabricate `CREATE TABLE` content and do not create `.specify/sql/pending/<business_model>.sql`. Record the missing DDL source in the relevant domain data document and project data overview as `待确认`, then ask the user to configure a database MCP service or provide SQL files.
   - Generated Markdown documents and cards must not disclose MCP/Tool names, MCP query text, repository source paths, `Source`, `Migration Script`, or `DDL Evidence` headers. Use source evidence transiently for validation, not as final knowledge content.

7. Preserve evidence quality.
   - Write Markdown documents in Chinese unless the user explicitly requests another language.
   - Separate `已确认`, `推断`, `待确认`, and `已废弃`.
   - Do not invent business rules, technology choices, schemas, integrations, metrics, or ownership.
   - Keep project-level documents concise. Put domain details in domain documents and fact-level retrieval units in knowledge cards.

8. Validate before finishing.
   - Run `scripts/validate_memory_knowledge.py --memory-root .specify/memory` after memory documents are generated or updated.
   - Confirm `index.md`, domain directories, domain three-doc sets, card metadata, scenario-to-technical mapping, SQL references, Markdown fences, and absence of mojibake.
   - Report created or updated Markdown paths, domains discovered, card counts, DDL reference status, main `待确认` items, and validation results. Do not report generated SQL paths because this workflow does not create SQL files.

## Knowledge Card Contract

Each card must include these header fields:

```md
> 知识编号：KC-BIZ-001
> 知识类型：业务场景 | 业务规则 | 状态流转 | 技术流程 | 接口契约 | 数据模型 | 治理规则
> 所属领域：<domain-slug>
> 状态：已确认 | 推断 | 待确认 | 已废弃
> 来源：<spec/report/code/user/docs>
> 关联场景：<BS-xxx | 无>
> 关联对象：<业务对象/模块/API/表 | 无>
> 关联代码/接口/SQL：<path or identifier | 无>
> 更新日期：YYYY-MM-DD
```

Use cards for knowledge that future work should retrieve directly: core scenarios, durable business rules, important state transitions, domain technical flows, API contracts, data models, and governance rules.

## Output Contract

When no path is specified, create the default knowledge base under `.specify/memory/`.
Do not create, import, copy, or update SQL files in `.specify/sql/` as part of this skill.
Do not create extra README files or example outputs unless the user explicitly asks.
