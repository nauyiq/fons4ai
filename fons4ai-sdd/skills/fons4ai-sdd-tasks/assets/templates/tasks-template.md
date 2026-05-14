# <Feature Name> Tasks

> Feature: `<feature-slug>`
> SDD Level: `S1|S2`
> Source Spec: `specs/features/<feature-slug>/spec.md`
> Source Plan: `specs/features/<feature-slug>/plan.md`
> Status: Draft

## Execution Strategy

- MVP:
- Dependencies:
- Parallel groups:

## Tasks

- [ ] T001 Short imperative task title
  - AC: AC-001
  - Files: path/to/source; path/to/test
  - Verification: command or manual check
  - Done: objective completion rule

## S2 Quality Gates

Use this section only for S2.

- [ ] T999 Regression or rollback gate
  - AC: AC-001
  - Files: path/to/source; path/to/test
  - Verification: command or manual check
  - Done: risk control is verified

## Knowledge and DDL Sync Tasks

Use this section when `plan.md` declares knowledge-source or `.specify/sql/` impact. DDL sync tasks are mandatory for persistent data model additions or changes unless explicitly deferred in `plan.md`.

- [ ] Txxx Sync DDL knowledge file
  - AC: AC-xxx
  - Files: .specify/sql/<table_or_model_name>.sql; .specify/memory/data-architecture.md
  - Verification: confirm the SQL file matches the implemented model/table and is indexed by data-architecture.md when present
  - Done: DDL knowledge update is completed or explicitly deferred with owner/reason

- [ ] Txxx Document knowledge impact
  - AC: AC-xxx
  - Files: <truth-source-path>
  - Verification: confirm the long-lived architecture or data fact is updated by the appropriate documentation workflow
  - Done: knowledge update is completed or explicitly deferred with owner/reason
