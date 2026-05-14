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
- SQL DDL action: none | add | update | rename
- Knowledge Sync Needed: no

## Regression and Rollback

- Regression risks:
- Rollback plan:

## Documentation Updates

- `spec.md`:
- `plan.md`:
- `tasks.md`:

## Incremental Tasks

- [ ] Txxx Task title
  - AC: AC-xxx
  - Files:
  - Verification:
  - Done:

- [ ] Txxx Sync DDL knowledge file
  - AC: AC-xxx
  - Files: .specify/sql/<table_or_model_name>.sql; .specify/memory/data-architecture.md
  - Verification: SQL file matches the changed persistent model and is indexed by data-architecture.md when present
  - Done: DDL knowledge update is completed or explicitly deferred with owner/reason
