---
name: fons4ai-bugfix-workflow
description: "Fons4AI gated BUG fix workflow. Auto-trigger only when an in-scope AGENTS.md contains '<!-- fons4ai-skill-routing: enabled -->'; otherwise use only when the user explicitly names this skill or asks for the Fons4AI/SDD workflow."
---

# Fons4AI Bugfix Workflow

## Activation Gate

Before using this skill, verify at least one condition is true:

1. The user explicitly names this skill, such as `$fons4ai-bugfix-workflow`.
2. The user explicitly asks to use Fons4AI, SDD, or the Fons4AI workflow.
3. The active repository has an in-scope `AGENTS.md` containing `<!-- fons4ai-skill-routing: enabled -->`.

If none is true, do not apply this skill automatically. Continue with normal Codex behavior or ask whether the user wants to enable the Fons4AI workflow.

## Overview

Use this skill when the user asks to fix a bug, investigate an error, diagnose an exception, repair a regression, or correct behavior that does not match existing expectations.

Default report path:

- `specs/bugfixes/<bug-slug>/bugfix-report.md`

If the user gives an explicit report path, use that path instead.

This skill fixes implementation defects. If the work changes requirements, acceptance criteria, public behavior, data model semantics, or long-lived architecture facts, stop and recommend `fons4ai-sdd-change` before coding.

## Required Context

1. Read `AGENTS.md`, project rules, and relevant truth sources before editing:
   - `.specify/memory/` when present.
   - `.specify/sql/` when data model or DDL knowledge may be involved.
   - Other project-declared knowledge sources such as `docs/`, `rules/`, or API documents.
2. Read existing SDD artifacts for the affected feature when discoverable:
   - `specs/features/<feature-slug>/spec.md`
   - `plan.md`
   - `tasks.md`
   - `changes/`
   - `reports/`
3. Read related source, tests, configuration, logs, and build files before modifying anything.
4. Use `assets/templates/bugfix-report-template.md` for the report.
5. Use `scripts/validate_bugfix_report.py --report <report-path>` after writing the report when Python is available.

## Workflow

1. Collect bug facts.
   - Problem location: page, API, module, job, command, or feature.
   - Expected result and actual result.
   - Reproduction steps, frequency, environment, version/configuration, logs, screenshots, and relevant data.
   - If the user has not supplied executable reproduction information, ask for the missing facts and do not edit code.

2. Reproduce or establish a minimal failing signal.
   - Prefer an automated failing test or command.
   - If the bug can only be reproduced manually, record exact manual steps and observed result.
   - If reproduction fails, stop and report the missing information or the closest observed evidence.

3. Diagnose root cause.
   - Narrow from feature/module to the smallest responsible code path.
   - Compare existing expected behavior from specs, tests, truth sources, or explicit user facts.
   - Distinguish implementation defect from requirement change.
   - If the desired fix changes intended behavior, public contracts, AC, or data semantics, recommend `fons4ai-sdd-change`.

4. Plan the smallest fix.
   - State affected files before editing.
   - Preserve user changes and unrelated files.
   - Ask before deleting logic, rewriting large sections, or making risky migrations.
   - If persistent data models change, ensure `.specify/sql/<table_or_model_name>.sql` is updated or route through `fons4ai-sdd-change` to add a DDL sync task.

5. Fix with Red-Green-Refactor.
   - RED: add or update a focused test that fails for the bug when feasible.
   - GREEN: implement the smallest change that makes the test pass.
   - REFACTOR: keep cleanup local and necessary; avoid unrelated restructuring.
   - If an automated test is not feasible, record why and provide a precise manual verification path.

6. Verify.
   - Run the focused test or command that proves the bug is fixed.
   - Run the smallest useful regression check around the affected area.
   - Always produce manual verification steps with expected results.

7. Write the bugfix report.
   - Create `specs/bugfixes/<bug-slug>/` when using the default path.
   - Copy the report structure from `assets/templates/bugfix-report-template.md`.
   - Fill reproduction, root cause, fix, verification, risk/rollback, knowledge sync, and follow-up fields.
   - Run the report validator when available and fix missing required sections before finishing.

8. Handle knowledge sync.
   - If the fix confirms durable business, technical, data, or governance facts, mark `Knowledge Sync Needed: yes` in the report and suggest `fons4ai-knowledge-summary`.
   - If the fix updates DDL knowledge, list the `.specify/sql/*.sql` files in the report.
   - Do not promote debugging notes or guesses into source-of-truth documents from this skill unless explicitly scoped.

## Hard Gates

- No reproducible signal, no code edit.
- No root-cause hypothesis, no fix.
- No verification result, no task completion.
- No manual verification steps, no final completion.
- No report, no bugfix completion.

## Output Rules

- Do not create or modify SDD requirements, plans, or tasks unless the user explicitly asks; recommend `fons4ai-sdd-change` instead.
- Do not invent expected behavior. Use existing specs, tests, truth sources, or explicit user confirmation.
- End with root cause, changed files, verification commands/results, manual verification steps, report path, knowledge sync need, and follow-up.
