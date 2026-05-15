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
It must not write business code; implementation remains the responsibility of `fons4ai-sdd-implement`.

## Required Context

1. Load `../fons4ai-sdd-requirements/references/sdd-artifact-contract.md`.
2. Identify the target `specs/features/<feature-slug>/` directory.
3. Read relevant project rules under `.specify/rules/` when present: `code-style-rule.md`, `project-structure-rule.md`, `features-rule.md`, `testing-rule.md`, and `data-ddl-rule.md`.
4. Read `.specify/memory/` and `.specify/sql/` when present, especially business, technical, data architecture, DDL, and constitution files.
5. Read existing `spec.md`, `plan.md`, `tasks.md`, prior `changes/`, reports, and relevant source/test files.
6. Use `assets/templates/change-template.md`.

## Workflow

1. Confirm the existing feature and change intent. If multiple feature directories match, ask the user to choose one.
2. Determine the next CR ID by scanning `changes/CR-*.md`.
3. Classify the change as `S1` or `S2`:
   - S1 for local behavior adjustments or small extensions.
   - S2 for data migrations, public contract changes, permission/security changes, compatibility risk, cross-core-module impact, or high rollback cost.
4. Produce an impact analysis:
   - Requirement changes: added, changed, or removed AC.
   - Design changes: API/data/module/flow impact.
   - Code impact: files likely affected.
   - Test impact: existing tests likely affected and new tests required.
   - Regression risk and rollback needs.
   - Knowledge impact: business, technical, data architecture, governance, other truth-source, or DDL facts that must be synchronized.
   - For any persistent data model addition or change, name each impacted `.specify/sql/<database_or_service>/<business_model>.sql` file and required action.
   - Keep same-database cohesive business model tables together when useful, but split files for different databases, service-owned schemas, or physical data sources.
5. Ask before modifying existing SDD artifacts. Preserve unaffected content and avoid full rewrites.
6. Update affected docs in place and append a concise change log entry.
7. Create `changes/CR-xxx.md` and add incremental tasks to `tasks.md` or a CR-specific task section.
8. When the change affects persistent data models, add mandatory DDL synchronization tasks for `.specify/sql/**/*.sql`; do not leave SQL knowledge updates as implicit follow-up.
9. Run `../fons4ai-sdd-tasks/scripts/validate_sdd_artifacts.py --change-file <CR-file>` after writing the CR. Fix validation failures before reporting success.

## Output Rules

- Do not rewrite the feature from scratch unless more than 70% of the feature is impacted; in that case recommend a new feature directory.
- Do not write business code.
- Do not delete existing AC, tasks, or docs without explicit user confirmation.
- End with CR path, changed SDD artifacts, new or changed task IDs, SDD level, knowledge impact, and suggested implementation step.
