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

## Knowledge Facts

- Business architecture:
- Technical architecture:
- Data architecture:
- SQL DDL files:
- Constitution constraints:
- Other truth sources:
- Conflicts between truth-source and code facts:

## Design Summary

Describe the simplest implementation that satisfies the AC.

## Affected Areas

- Module/file area:
- Reason:

## Data Flow

Describe request, state, persistence, and response flow.

## API, UI, or Contract Changes

- Change:
- Compatibility impact:

## Data Model Changes

- Change:
- Database/service:
- Business model:
- Tables/models:
- Migration needed: yes/no
- DDL sync required: no | yes
- DDL file action: none | add | update | rename
- DDL file impact: none | `.specify/sql/<database_or_service>/<business_model>.sql`
- DDL sync timing: same implementation task | separate task | deferred with owner/reason

## Key Decisions

| Decision | Rationale | Alternatives |
| --- | --- | --- |
|  |  |  |

## AC Mapping

| AC | Design Coverage |
| --- | --- |
| AC-001 |  |

## Risk and Rollback

- Risk:
- Mitigation:
- Rollback:

## Verification Strategy

- Unit tests:
- Integration tests:
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
