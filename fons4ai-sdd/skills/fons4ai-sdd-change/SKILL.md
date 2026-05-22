---
name: fons4ai-sdd-change
description: "Fons4AI gated SDD change workflow. Auto-trigger only when an in-scope AGENTS.md contains '<!-- fons4ai-skill-routing: enabled -->'; otherwise use only when the user explicitly names this skill or asks for the Fons4AI/SDD workflow."
---

# Fons4AI SDD Change

## Activation Gate

Before using this skill, verify at least one condition is true:

1. The user explicitly names this skill, such as `$fons4ai-sdd-change`.
2. The user explicitly asks to use Fons4AI, SDD, or the Fons4AI workflow.
3. The active repository has an in-scope `AGENTS.md` containing `<!-- fons4ai-skill-routing: enabled -->`.

If none is true, do not apply this skill automatically. Continue with normal Codex behavior or ask whether the user wants to enable the Fons4AI workflow.

## Overview

Use this skill for changes to an existing SDD feature. It performs impact analysis, updates affected SDD artifacts, and creates incremental tasks.
It must clarify blocking change ambiguity before writing a formal CR; existing artifacts do not remove the need to confirm changed business semantics.
It must not write business code; implementation remains the responsibility of `fons4ai-sdd-implement`.

## Required Context

1. Load `../fons4ai-sdd-requirements/references/sdd-artifact-contract.md`, including its context-loading rules.
2. Identify the target `specs/features/<feature-slug>/` directory.
3. Search by change intent, AC IDs, modules, APIs, domain objects, table/model names, and SQL paths before loading truth sources.
   - Optionally run `../fons4ai-sdd-requirements/scripts/find_relevant_context.py --root <repo-root> <keyword...>` to get candidate truth-source files before reading.
4. Read only relevant project rules, matching `.specify/memory/` sections, targeted `.specify/sql/` files, and governance files that affect the change.
5. Read existing `spec.md`, `plan.md`, `tasks.md`, prior `changes/`, reports, and relevant source/test files.
6. Use `assets/templates/change-template.md`.

## Workflow

1. Confirm the existing feature and change intent. If multiple feature directories match, ask the user to choose one.
2. Run the change clarification gate before assigning a final CR ID or modifying artifacts.
   - If a blocking ambiguity exists, stop and ask exactly one highest-impact clarification question. Do not create `CR-xxx.md`, update `spec.md`, update `plan.md`, or append tasks in the same turn.
   - If the user explicitly asks for a draft before answering, create only a draft CR with `澄清状态：草案-含待确认`, mark assumptions as `待确认`, and do not add executable implementation tasks.
   - If all blocking ambiguities are closed, record `澄清状态：已关闭` in the CR and continue.
3. Determine the next CR ID by scanning `changes/CR-*.md`.
4. Classify the change as `S1` or `S2`:
   - S1 for local behavior adjustments or small extensions.
   - S2 for data migrations, public contract changes, permission/security changes, compatibility risk, cross-core-module impact, or high rollback cost.
5. Produce an impact analysis:
   - Requirement changes: added, changed, or removed AC.
   - Design changes: API/data/module/flow impact.
   - Code impact: files likely affected.
   - Test impact: existing tests likely affected and new tests required.
   - Regression risk and rollback needs.
   - Knowledge impact: business, technical, data architecture, governance, other truth-source, or DDL facts that must be synchronized.
   - For any persistent data model addition or change, name each impacted `.specify/sql/<database_or_service>/<business_model>.sql` file, required action, and DDL evidence source: MCP query, repository SQL file, or implementation migration/schema SQL.
   - Keep same-database cohesive business model tables together when useful, but split files for different databases, service-owned schemas, or physical data sources.
6. Ask before modifying existing SDD artifacts. Preserve unaffected content and avoid full rewrites.
7. Update affected docs in place and append a concise change log entry.
8. Create `changes/CR-xxx.md` and add incremental tasks to `tasks.md` or a CR-specific task section. Every incremental task must include `AC:`, `Files:`, `Verification:`, `Quality:`, and `Done:`; implementation tasks must include a DDD-lite/domain-modeling check in `Quality:`.
9. When the change affects persistent data models, add mandatory DDL synchronization tasks for `.specify/sql/**/*.sql`; do not leave SQL knowledge updates as implicit follow-up. Do not infer DDL from entity classes, Mapper interfaces, ORM annotations, or Java field types.
10. Run `../fons4ai-sdd-tasks/scripts/validate_sdd_artifacts.py --change-file <CR-file>` after writing the CR. If DDL knowledge files were created or updated, also run `../fons4ai-project-knowledge-base-init/scripts/validate_sql_knowledge.py --sql-root .specify/sql`. Fix validation failures before reporting success.
11. Stop after change planning. Do not invoke implementation. Tell the user they can reply `执行`, `开始实现`, or `继续执行` to execute all unfinished incremental tasks, or `执行 T001,T002` to specify task IDs.

## Change Clarification Gate

The change clarification gate decides whether the skill may write a formal CR or update existing SDD artifacts.

Blocking ambiguity means any missing or conflicting answer that can change one of these:

- whether the request is a bugfix, requirement change, data semantic change, or technical refactor;
- changed business meaning, terminology, service/package/model naming, ownership, identity, status, lifecycle, or compatibility promise;
- added, removed, or changed AC and the expected observable behavior;
- public API/UI/message/job behavior, permission/security, integration contract, error handling, or rollback expectation;
- data model, table grouping, field naming, DDL evidence source, database/service ownership, migration, or fallback strategy;
- impacted feature directory, affected modules/files, SDD level, risk gates, tests, or knowledge-summary need.

Rules:

1. If at least one blocking ambiguity exists, ask one question and stop. Do not create `CR-xxx.md`, append tasks, rewrite SDD artifacts, or write business code.
2. Ask the highest-impact question first. Prefer 2-5 options with a recommended option when options are known.
3. Treat answers such as `按推荐`, `方案 A`, or a precise short answer as accepted clarification.
4. Do not treat `使用 SDD`, `继续`, `先生成`, `看一下`, or artifact existence as clarification closure.
5. Only write a formal CR when blocking ambiguities are closed, or when the user explicitly asks for a draft with assumptions.
6. A draft CR with unresolved blocking ambiguity must not contain executable implementation tasks and must not be handed to `fons4ai-sdd-implement`.

## Output Rules

- Do not rewrite the feature from scratch unless more than 70% of the feature is impacted; in that case recommend a new feature directory.
- If the change clarification gate is blocked, output only the blocking reason, the single clarification question, recommended options when available, and what will be generated after the answer. Do not write formal artifacts.
- Formal CR files must include `澄清状态：已关闭` and `## 变更澄清门禁`. Draft CRs are allowed only after explicit user request and must include `澄清状态：草案-含待确认`.
- Generated CR headings and fixed prose must be Chinese-first. Keep file names, IDs, paths, and machine-readable task labels `AC:`, `Files:`, `Verification:`, `Quality:`, and `Done:` unchanged.
- Do not write business code.
- Do not delete existing AC, tasks, or docs without explicit user confirmation.
- End with CR path, changed SDD artifacts, new or changed task IDs, SDD level, knowledge impact, implementation approval status `pending`, and this exact execution prompt: `确认执行后默认执行全部未完成任务；如需指定范围，请回复：执行 T001,T002。`
