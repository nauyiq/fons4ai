---
name: fons4ai-sdd-tasks
description: "Fons4AI gated SDD task-planning workflow. Auto-trigger only when an in-scope AGENTS.md contains '<!-- fons4ai-skill-routing: enabled -->'; otherwise use only when the user explicitly names this skill or asks for the Fons4AI/SDD workflow."
---

# Fons4AI SDD Tasks

## Activation Gate

Before using this skill, verify at least one condition is true:

1. The user explicitly names this skill, such as `$fons4ai-sdd-tasks`.
2. The user explicitly asks to use Fons4AI, SDD, or the Fons4AI workflow.
3. The active repository has an in-scope `AGENTS.md` containing `<!-- fons4ai-skill-routing: enabled -->`.

If none is true, do not apply this skill automatically. Continue with normal Codex behavior or ask whether the user wants to enable the Fons4AI workflow.

## Overview

Use this skill after `fons4ai-sdd-design` has produced `plan.md`.
The output is an executable `tasks.md` for `fons4ai-sdd-implement`.

## Required Context

1. Load `../fons4ai-sdd-requirements/references/sdd-artifact-contract.md`.
2. Read `spec.md`, `plan.md`, and any S2 artifacts in the feature directory.
3. Read `.specify/memory/` and `.specify/sql/` when present to respect long-lived module, data, DDL, and governance boundaries.
4. Use `assets/templates/tasks-template.md`.
5. If `tasks.md` already exists, read it and ask before replacing or materially rewriting it.

## Workflow

1. Confirm `spec.md` and `plan.md` exist. Stop if either is missing.
2. Extract AC IDs, affected modules/files, design decisions, dependencies, and verification expectations.
3. Generate tasks in dependency order:
   - Setup or preparation tasks first.
   - Contract/data/model tasks before services.
   - Core business logic before adapters/UI.
   - Integration and regression tasks after the implementation path is complete.
4. Make every task TDD-ready:
   - Include `AC:` with one or more AC IDs.
   - Include `Files:` with exact expected file paths or file groups.
   - Include `Verification:` with automated test or manual verification steps.
   - Include `Done:` with objective completion criteria.
5. For S2, add explicit tasks for applicable risk controls: migration, rollback, compatibility, permissions, regression, observability, and checklist closure.
6. If `plan.md` declares data model additions or changes, add a mandatory DDL synchronization task for every impacted `.specify/sql/<table_or_model_name>.sql` file.
   - The task must name the exact SQL file.
   - Place it with the related model/migration task, before service-layer tasks that depend on the schema.
   - Verification must confirm the SQL file matches the implemented model and is indexed by `.specify/memory/data-architecture.md` when that document exists.
   - Only mark it as deferred when `plan.md` records user-approved owner/reason.
7. If `plan.md` declares non-DDL knowledge impact, add a documentation synchronization or follow-up task that names the impacted truth-source path.
8. Run `scripts/validate_sdd_artifacts.py --feature-dir <feature-dir>` after writing tasks. This validates AC coverage, DDL mapping, Knowledge Impact, and S2 risk gates. Fix validation failures before reporting success.

## Task Format

Use this shape for each task:

```markdown
- [ ] T001 [P] [US1] Short imperative task title
  - AC: AC-001, AC-002
  - Files: path/to/file.java; path/to/fileTest.java
  - Verification: run the focused test or manual check
  - Done: objective completion rule
```

`[P]` is optional and means the task can run in parallel because it touches different files and has no unfinished dependency. Story labels are optional; use them only when `spec.md` has user-story phases.

## Output Rules

- Create or update only `specs/features/<feature-slug>/tasks.md`.
- Do not write business code.
- Do not mark tasks complete.
- End with total task count, SDD level, parallel groups, validation result, and suggested next skill.
