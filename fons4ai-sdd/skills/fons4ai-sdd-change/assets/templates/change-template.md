# CR-xxx <Change Title>

> Feature: `<feature-slug>`
> SDD Level: `S1|S2`
> Status: Draft
> Created: YYYY-MM-DD

## Change Intent

Describe what changes and why.

## Impact Analysis

### Requirements

- Added AC:
- Changed AC:
- Removed AC:

### Design

- API/UI/data/module impact:

### Code

- Existing files likely affected:
- New files likely needed:

### Tests

- Existing tests likely affected:
- New tests required:

### Knowledge Impact

- Business architecture:
- Technical architecture:
- Data architecture:
- Other truth sources:
- SQL DDL files:
- DDL grouping: same database/service + cohesive business model; split by database/service
- SQL DDL action: none | add | update | rename
- Knowledge Sync Needed: no

## Regression and Rollback

- Regression risks:
- Rollback plan:

## Implementation Approval Gate

- Status: pending user approval
- Planning artifacts are not implementation approval.
- After this CR is generated, stop and wait for user confirmation before business-code implementation.
- If the user confirms execution without task IDs, execute all unfinished incremental tasks.
- If the user specifies task IDs, for example `执行 T001,T002`, execute only those tasks.

## Documentation Updates

- `spec.md`:
- `plan.md`:
- `tasks.md`:

## Incremental Tasks

- [ ] Txxx Task title
  - AC: AC-xxx
  - Files:
  - Verification:
  - Quality: confirm readability, DDD-lite/domain-modeling check, method size, naming, duplicate-code check, utility reuse, and dependency gate
  - Done:

- [ ] Txxx Sync DDL knowledge file
  - AC: AC-xxx
  - Files: .specify/sql/<database_or_service>/<business_model>.sql; .specify/memory/data-architecture.md
  - Verification: SQL file matches the changed same-database business model/table group and is indexed by data-architecture.md when present
  - Quality: confirm generated SQL knowledge is readable, grouped correctly, and avoids duplicate undocumented schema facts
  - Done: DDL knowledge update is completed or explicitly deferred with owner/reason
