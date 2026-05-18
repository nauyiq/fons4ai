---
name: fons4ai-sdd-implement
description: "Fons4AI gated SDD implementation workflow. Requires the user's latest message to explicitly confirm implementation; if no task IDs are specified, execute all unfinished tasks."
---

# Fons4AI SDD Implement

## Activation Gate

Before using this skill, verify at least one condition is true:

1. The user explicitly names this skill, such as `$fons4ai-sdd-implement`.
2. The user explicitly asks to use Fons4AI, SDD, or the Fons4AI workflow.
3. The active repository has an in-scope `AGENTS.md` containing `<!-- fons4ai-skill-routing: enabled -->`.

If none is true, do not apply this skill automatically. Continue with normal Codex behavior or ask whether the user wants to enable the Fons4AI workflow.

## Overview

Use this skill only after `tasks.md` exists and the user's latest message explicitly confirms implementation. It executes planned tasks with TDD and updates task status.
All features use the S1 or S2 path.

## Required Context

1. Load `../fons4ai-sdd-requirements/references/sdd-artifact-contract.md`.
2. Read `specs/features/<feature-slug>/tasks.md` first.
3. Read relevant project rules under `.specify/rules/` when present: `code-style-rule.md`, `project-structure-rule.md`, `features-rule.md`, `testing-rule.md`, and `data-ddl-rule.md`.
4. Read `.specify/memory/` and `.specify/sql/` when present for module boundaries, data rules, DDL, and governance constraints relevant to the selected tasks.
5. Load `spec.md`, `plan.md`, or S2 artifacts only when the selected task references them or when the task text is insufficient.
6. Use `assets/templates/implementation-report-template.md` for completion reports.

## Workflow

1. Confirm implementation approval from the user's latest message.
   - Valid approval without task IDs includes `执行`, `开始实现`, `继续执行`, `开始开发`, or an equivalent explicit request to implement. In this case, select all unfinished tasks.
   - Valid approval with task IDs includes patterns such as `执行 T001`, `执行 T001,T002`, or `实现 T003`. In this case, select only the specified unfinished tasks.
   - Ambiguous follow-ups such as `看看`, `下一步是什么`, `继续看`, or a generated `tasks.md` alone are not approval. Stop and ask the user to confirm execution.
2. Identify the requested feature directory. If the feature directory is ambiguous, ask for the exact feature path.
3. Select tasks:
   - If task IDs are named in the latest user message, use exactly those task IDs.
   - If no task IDs are named but implementation is clearly approved, use all unfinished tasks in dependency order.
4. Verify prerequisites:
   - `tasks.md` exists.
   - Selected tasks are unchecked.
   - Dependency tasks are already complete unless the user explicitly approves a different order.
   - The working tree may contain user changes; read affected files before editing and do not revert unrelated work.
5. Before code edits, state the selected tasks and expected file scope. Ask before deleting or materially rewriting existing code.
6. Execute each task with Red-Green-Refactor:
   - RED: write or update the focused test first and confirm it fails for the expected reason.
   - GREEN: implement the smallest code change that passes the test.
   - Before REFACTOR: run a code quality self-check for readability, method length, expressive naming, DDD-lite/domain-modeling fit, duplicate logic, utility/package reuse, dependency gate, exception/logging style, and test readability.
   - For business behavior, prefer rich domain methods for core rules, state transitions, validation, and invariants. Keep application services focused on orchestration, transactions, permissions, external collaboration, and persistence coordination.
   - Do not force full DDD structures for simple CRUD, read-only queries, or thin wrappers; record the lightweight/anemic-model exception when applicable.
   - REFACTOR: improve structure while keeping tests green. Prefer JDK, project utilities/components, and already-introduced third-party utilities such as Hutool, Apache Commons, or Guava before hand-writing common helper logic.
   - If a new dependency appears necessary, stop unless `plan.md` or the user has confirmed rationale, alternatives, and impact.
7. Run the task verification from `tasks.md`, then run the smallest useful regression check.
8. Mark completed tasks as `[x]` in `tasks.md` only after verification passes.
9. When a selected task names `.specify/sql/<database_or_service>/<business_model>.sql`, create or update that SQL file as part of the same task after reading any existing file.
   - Keep one database-scoped cohesive business model per SQL file.
   - Multiple strongly related tables may share one SQL file only when they belong to the same database/service.
   - Never merge DDL from different databases, service-owned schemas, or physical data sources into one SQL file.
   - Preserve source evidence, status, and last generated date in the SQL header.
   - If implementation changes a persistent model but no selected task names the matching database-scoped business-model SQL file, stop and recommend returning to `fons4ai-sdd-tasks` or `fons4ai-sdd-change` to add the DDL sync task.
10. Write a report under `specs/features/<feature-slug>/reports/` summarizing tasks, files, tests, AC coverage, unresolved risks, updated DDL files, S2 gate closure when applicable, and whether knowledge-base or source-of-truth documents need synchronization.

## Output Rules

- Follow the task plan; do not implement unplanned scope.
- Generated implementation reports must use Chinese-first headings and fixed prose. Keep file names, IDs, paths, and technical markers such as `T001`, `AC-001`, `RED`, `GREEN`, and `REFACTOR` unchanged.
- Never treat `spec.md`, `plan.md`, `tasks.md`, or a previous planning response as implementation approval. Approval must come from the latest user message.
- If the plan is wrong or incomplete, stop and recommend returning to `fons4ai-sdd-tasks` or `fons4ai-sdd-change`.
- Treat `.specify/sql/**/*.sql` as required knowledge artifacts for planned persistent data model additions or changes, not as optional follow-up.
- Never skip RED for behavior changes. If automated testing is impossible, record an explicit manual verification reason and steps.
- End with completed task IDs, changed files, verification commands/results, knowledge/DDL sync need, remaining tasks, and suggest `fons4ai-knowledge-summary` when verified changes should be merged into a knowledge base or source-of-truth document.
