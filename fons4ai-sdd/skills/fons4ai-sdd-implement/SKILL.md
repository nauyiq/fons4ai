---
name: fons4ai-sdd-implement
description: "Fons4AI gated SDD implementation workflow. Auto-trigger only when an in-scope AGENTS.md contains '<!-- fons4ai-skill-routing: enabled -->'; otherwise use only when the user explicitly names this skill or asks for the Fons4AI/SDD workflow."
---

# Fons4AI SDD Implement

## Activation Gate

Before using this skill, verify at least one condition is true:

1. The user explicitly names this skill, such as `$fons4ai-sdd-implement`.
2. The user explicitly asks to use Fons4AI, SDD, or the Fons4AI workflow.
3. The active repository has an in-scope `AGENTS.md` containing `<!-- fons4ai-skill-routing: enabled -->`.

If none is true, do not apply this skill automatically. Continue with normal Codex behavior or ask whether the user wants to enable the Fons4AI workflow.

## Overview

Use this skill only after `tasks.md` exists. It executes planned tasks with TDD and updates task status.
All features use the S1 or S2 path.

## Required Context

1. Load `../fons4ai-sdd-requirements/references/sdd-artifact-contract.md`.
2. Read `specs/features/<feature-slug>/tasks.md` first.
3. Read `.specify/memory/` and `.specify/sql/` when present for module boundaries, data rules, DDL, and governance constraints relevant to the selected tasks.
4. Load `spec.md`, `plan.md`, or S2 artifacts only when the selected task references them or when the task text is insufficient.
5. Use `assets/templates/implementation-report-template.md` for completion reports.

## Workflow

1. Identify the requested task, task range, phase, or feature directory. If ambiguous, ask for the exact feature or task ID.
2. Verify prerequisites:
   - `tasks.md` exists.
   - Selected tasks are unchecked.
   - Dependency tasks are already complete unless the user explicitly approves a different order.
   - The working tree may contain user changes; read affected files before editing and do not revert unrelated work.
3. Before code edits, state the selected tasks and expected file scope. Ask before deleting or materially rewriting existing code.
4. Execute each task with Red-Green-Refactor:
   - RED: write or update the focused test first and confirm it fails for the expected reason.
   - GREEN: implement the smallest code change that passes the test.
   - REFACTOR: improve structure while keeping tests green.
5. Run the task verification from `tasks.md`, then run the smallest useful regression check.
6. Mark completed tasks as `[x]` in `tasks.md` only after verification passes.
7. When a selected task names `.specify/sql/<table_or_model_name>.sql`, create or update that SQL file as part of the same task after reading any existing file.
   - Keep one concrete table or canonical data model per SQL file.
   - Preserve source evidence, status, and last generated date in the SQL header.
   - If implementation changes a persistent model but no selected task names the matching SQL file, stop and recommend returning to `fons4ai-sdd-tasks` or `fons4ai-sdd-change` to add the DDL sync task.
8. Write a report under `specs/features/<feature-slug>/reports/` summarizing tasks, files, tests, AC coverage, unresolved risks, updated DDL files, S2 gate closure when applicable, and whether knowledge-base or source-of-truth documents need synchronization.

## Output Rules

- Follow the task plan; do not implement unplanned scope.
- If the plan is wrong or incomplete, stop and recommend returning to `fons4ai-sdd-tasks` or `fons4ai-sdd-change`.
- Treat `.specify/sql/*.sql` as required knowledge artifacts for planned persistent data model additions or changes, not as optional follow-up.
- Never skip RED for behavior changes. If automated testing is impossible, record an explicit manual verification reason and steps.
- End with completed task IDs, changed files, verification commands/results, knowledge/DDL sync need, remaining tasks, and suggest `fons4ai-knowledge-summary` when verified changes should be merged into a knowledge base or source-of-truth document.
