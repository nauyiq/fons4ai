# <Feature Name> Requirements

> Feature: `<feature-slug>`
> SDD Level: `S1|S2`
> Status: Draft
> Created: YYYY-MM-DD
> Source: user request, repository facts, and open questions

## Revision History

| Date | Version | Author/Source | Notes |
| --- | --- | --- | --- |
| YYYY-MM-DD | V1.0.0 | <user/repository/source> | Initial requirement summary |

## Background

Describe the current state, problem, user value, and relevant repository facts.

## Knowledge Context

- Business architecture facts:
- Technical architecture facts:
- Data architecture facts:
- SQL DDL facts:
- Other truth sources:
- Knowledge sync needed: no | yes
- SQL DDL impact: none | add/update `.specify/sql/<database_or_service>/<business_model>.sql`

## Requirement Summary

| Requirement ID | Requirement | Priority | Source | Related AC |
| --- | --- | --- | --- | --- |
| REQ-001 | <specific requirement point> | P0/P1/P2 | <user/fact/source> | AC-001 |

## Business Rules and Constraints

- Permission and security rules:
- Calculation or accounting rules:
- Compatibility and integration constraints:
- Boundary and exception rules:
- Policy or governance constraints:

## Functional Overview

### <Sub-function 1>

- Behavior:
- Input or trigger:
- Output or observable result:
- Failure or boundary behavior:

### <Sub-function 2>

- Behavior:
- Input or trigger:
- Output or observable result:
- Failure or boundary behavior:

## Workflow Overview

- Main flow:
- Alternative flow:
- Failure flow:
- Async or scheduled flow:
- Diagram: not applicable, reason | Mermaid sequence/state diagram when facts are known

## Impact Overview

| Area | Change Type | Candidate Change | Impact | Evidence Status |
| --- | --- | --- | --- | --- |
| Module/API/Data/Config/External System | add/update/remove | <candidate impact> | <scope> | confirmed/inferred/pending |

## Risk Overview

| Risk | Level | Description | Mitigation |
| --- | --- | --- | --- |
| <risk> | high/medium/low | <description> | <mitigation> |

## Clarifications

### Session YYYY-MM-DD

- Q: <question> -> A: <answer>

## Goals

- Goal 1.

## Scope

### In Scope

- Item 1.

### Out of Scope

- Item 1.

## Users and Scenarios

- User or actor:
- Scenario:

## Functional Requirements

- FR-001:

## Acceptance Criteria

- AC-001: Given <context>, when <action>, then <observable result>. Related requirement: REQ-001.

## Non-Functional Requirements

- Performance:
- Security:
- Compatibility:

## Key Data or Domain Objects

- Object:
  - Database/service: none | <database_or_service>
  - Business model: none | <business_model>
  - Persistent tables/models: none | <table_names_or_model_names>
  - DDL sync expected: no | yes, `.specify/sql/<database_or_service>/<business_model>.sql`

## Assumptions

- Assumption 1.

## Open Questions

- Question 1.

## SDD Level Reason

Explain why this feature is S1 or S2.
