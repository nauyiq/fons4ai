# <Feature Name> Technical Plan

> Feature: `<feature-slug>`
> SDD Level: `S1|S2`
> Source Spec: `specs/features/<feature-slug>/spec.md`
> Status: Draft

## Repository Facts

- Existing modules:
- Existing patterns:
- Reusable utilities/components:
- Test conventions:
- Representative files inspected:

## Knowledge Facts

- Business architecture:
- Technical architecture:
- Data architecture:
- SQL DDL files:
- Constitution constraints:
- Other truth sources:
- Conflicts between truth-source and code facts:

## Design Summary

Describe the simplest implementation that satisfies the AC and REQ scope.

## Architecture Design

- Target modules/layers:
- Dependency direction:
- Domain/application/infrastructure boundary:
- Reused extension points:
- New or changed abstractions: none | <name, reason>

## Implementation Approach

- Main implementation path:
- Existing code or pattern to reuse:
- New or changed files:
- Backward compatibility approach:
- Implementation notes that affect tasks:

## Domain Modeling Decisions

- DDD-lite applies: yes | no, reason
- Core domain objects:
- Business rules and invariants:
- State transitions:
- Value object candidates:
- Domain service candidates:
- Application-layer orchestration:
- Anemic model exception: none | simple CRUD | read-only query | DTO/projection | compatibility | unstable domain abstraction, reason
- Infrastructure dependency boundary:

## Key Rule Code Sketches

Use short pseudocode or code-like snippets to explain key rules. Do not write final implementation here.

```text
not applicable, reason
```

- Rule source: REQ-001 | AC-001 | repository fact
- Existing types/utilities to reuse:
- Edge cases covered:

## State Transition Design

| State | Trigger/Event | Preconditions | Next State | Failure Handling | Idempotency |
| --- | --- | --- | --- | --- | --- |
| not applicable, reason |  |  |  |  |  |

Diagram: not applicable, reason | Mermaid stateDiagram when facts are known

## Data Flow

- Request/input:
- Validation:
- Domain behavior:
- Persistence:
- External interaction:
- Response/output:

## Utility and Dependency Decisions

- Reusable project utilities/components:
- Existing third-party utilities: none | Hutool | Apache Commons | Guava | other
- Utility choice for string/collection/date-time/IO/bean/null-check/assertion work:
- New dependency needed: no | yes, reason
- Alternatives considered:
- Confirmation for new dependency: n/a | user/design approved | deferred
- Readability and complexity notes:

## Affected Areas

| Area | File/Module | Change Type | Reason |
| --- | --- | --- | --- |
|  |  | add/update/remove |  |

## API and Contract Details

- API/UI/contract change: none | <details>
- Request shape:
- Response shape:
- Compatibility impact:
- Contract files: none | `contracts/<name>.md`

## Data Structure Changes

| Object/Table/DTO | Field | Type | Change | Default/Constraint | Compatibility |
| --- | --- | --- | --- | --- | --- |
| not applicable, reason |  |  |  |  |  |

- Database/service:
- Business model:
- Migration needed: yes/no
- DDL sync required: no | yes
- DDL file action: none | add | update | rename
- DDL file impact: none | `.specify/sql/<database_or_service>/<business_model>.sql`
- DDL sync timing: same implementation task | separate task | deferred with owner/reason

## Error and Exception Handling

- Business errors:
- Validation errors:
- External dependency failures:
- Retry or compensation:
- Logging and sensitive-data masking:

## Transaction and Consistency

- Transaction boundary:
- Consistency model:
- Concurrency or locking:
- Idempotency:
- Cache/MQ/scheduled-task consistency:

## Migration and Rollback Details

- Migration steps: not applicable, reason | <steps>
- Rollback steps:
- Compatibility window:
- Data repair or backfill:
- Operational notes:

## Key Decisions

| Decision | Rationale | Alternatives |
| --- | --- | --- |
|  |  |  |

## AC Mapping

| AC | Requirement | Design Coverage |
| --- | --- | --- |
| AC-001 | REQ-001 |  |

## Risk and Rollback

- Risk:
- Mitigation:
- Rollback:

## Verification Strategy

- Unit tests:
- Integration tests:
- Contract/API tests:
- Persistence or migration checks:
- Manual checks:

## Knowledge Impact

- Business architecture update needed: no
- Technical architecture update needed: no
- Data architecture update needed: no
- Other truth sources update needed: no
- SQL DDL update needed: no
- SQL DDL files:
- DDL grouping rule: same database/service + cohesive business model only; split files for different databases/services
- Knowledge Sync Needed: no
- Notes:

## Open Questions

- Question:
