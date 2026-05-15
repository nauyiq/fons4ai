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

## Implementation Approval Gate

- Status: pending user approval
- Planning artifacts are not implementation approval.
- After this `tasks.md` is generated, stop and wait for user confirmation before business-code implementation.
- If the user confirms execution without task IDs, execute all unfinished tasks.
- If the user specifies task IDs, for example `执行 T001,T002`, execute only those tasks.

## Tasks

- [ ] T001 Short imperative task title
  - AC: AC-001
  - Files: path/to/source; path/to/test
  - Verification: command or manual check
  - Quality: confirm readability, DDD-lite/domain-modeling check, method size, naming, duplicate-code check, utility reuse, and dependency gate
  - Done: objective completion rule

## S2 Quality Gates

Use this section only for S2.

- [ ] T999 Regression or rollback gate
  - AC: AC-001
  - Files: path/to/source; path/to/test
  - Verification: command or manual check
  - Quality: confirm risk-control code remains readable, respects DDD-lite/domain boundaries, and uses approved utilities/dependencies
  - Done: risk control is verified

## Knowledge and DDL Sync Tasks

Use this section when `plan.md` declares knowledge-source or `.specify/sql/` impact. DDL sync tasks are mandatory for persistent data model additions or changes unless explicitly deferred in `plan.md`.

- [ ] Txxx Sync DDL knowledge file
  - AC: AC-xxx
  - Files: .specify/sql/<database_or_service>/<business_model>.sql; .specify/memory/data-architecture.md
  - Verification: confirm the SQL file matches the implemented same-database business model/table group and is indexed by data-architecture.md when present
  - Quality: confirm generated SQL knowledge is readable, grouped correctly, and avoids duplicate undocumented schema facts
  - Done: DDL knowledge update is completed or explicitly deferred with owner/reason

- [ ] Txxx Document knowledge impact
  - AC: AC-xxx
  - Files: <truth-source-path>
  - Verification: confirm the long-lived architecture or data fact is updated by the appropriate documentation workflow
  - Quality: confirm knowledge text is concise, traceable, non-duplicative, and does not promote unverified assumptions
  - Done: knowledge update is completed or explicitly deferred with owner/reason
