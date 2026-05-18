---
name: fons4ai-sdd-requirements
description: "Fons4AI gated SDD requirements workflow. Auto-trigger only when an in-scope AGENTS.md contains '<!-- fons4ai-skill-routing: enabled -->'; otherwise use only when the user explicitly names this skill or asks for the Fons4AI/SDD workflow."
---

# Fons4AI SDD Requirements

## Activation Gate

Before using this skill, verify at least one condition is true:

1. The user explicitly names this skill, such as `$fons4ai-sdd-requirements`.
2. The user explicitly asks to use Fons4AI, SDD, or the Fons4AI workflow.
3. The active repository has an in-scope `AGENTS.md` containing `<!-- fons4ai-skill-routing: enabled -->`.

If none is true, do not apply this skill automatically. Continue with normal Codex behavior or ask whether the user wants to enable the Fons4AI workflow.

## Overview

Use this skill to turn a feature idea into the first SDD artifact: a detailed requirement summary in `specs/features/<feature-slug>/spec.md`.
Feature artifacts are written under `specs/features/`; `.specify/memory/` and `.specify/sql/` are read as long-lived project context when present.

## Required Context

1. Read project guidance before drafting: `AGENTS.md`, project rules under `.specify/rules/` when present (`code-style-rule.md`, `project-structure-rule.md`, `features-rule.md`, `testing-rule.md`, `data-ddl-rule.md`), `.specify/memory/` and `.specify/sql/` if present, existing `specs/`, and relevant source files when the feature touches existing behavior.
2. Load the SDD contract from `references/sdd-artifact-contract.md`.
3. Use `assets/templates/spec-template.md` as the output structure.
4. If a target feature directory or `spec.md` already exists, read it first and ask the user before replacing or materially rewriting it.
5. Prefer facts from `.specify/memory/business-architecture.md` for business domains, roles, processes, business objects, and durable business rules. Prefer `.specify/memory/data-architecture.md` and `.specify/sql/**/*.sql` for confirmed data model and DDL facts. If absent, continue from repository and user facts.

## Workflow

1. Derive or confirm `<feature-slug>` in lowercase hyphen-case. Prefer short action-noun names such as `loan-approval-rule`.
2. Classify the feature:
   - Use `S1` by default for small or normal feature work.
   - Use `S2` for database migrations, public API changes, security/permission changes, cache/MQ/transaction boundaries, cross-core-module work, compatibility risk, or high rollback cost.
   - Record the level and reason in `spec.md`.
3. Perform the structured clarification scan below before finalizing requirements.
4. Build the requirement summary before writing AC:
   - Use `REQ-001`, `REQ-002`, ... for requirement points and business capabilities.
   - Record priority, source, and related AC for every requirement.
   - Include business rules, functional overview, workflow overview, impact overview, and risk overview.
   - Keep `spec.md` focused on requirements, behavior, business rules, workflows, impacts, and acceptance. Do not move detailed technical design into `spec.md`.
5. Write acceptance criteria before technical design:
   - Use `AC-001`, `AC-002`, ... IDs.
   - Use Given-When-Then.
   - Cover normal flow, boundary cases, and failure cases.
   - Keep AC observable from user, API, or system behavior; avoid implementation details.
   - Link each AC to one or more `REQ-###` IDs through the requirement summary table or AC text.
6. Define scope clearly:
   - Include in-scope items.
   - Exclude out-of-scope items.
   - Record assumptions and unresolved questions.
7. For S1, keep details concise but cover requirement summary, key business rules, functional overview, impact overview, and AC. Use `not applicable, reason` where workflow, risk, or data impact is genuinely absent.
8. For S2, include workflow overview, risk overview, data/domain objects, non-functional requirements, compatibility, security, and migration hints in enough detail for design work.
9. Generate Mermaid sequence or state diagrams only when repository/user facts support the participants and flow. Do not invent systems, tables, APIs, or actors to fill the template.
10. Record knowledge impact when the feature appears to add or change long-lived business, technical, data, or governance facts. If it may add or change persistent data models, record expected `.specify/sql/` impact explicitly. Do not update truth sources such as `.specify/memory/`, `.specify/sql/`, `.specify/rules/`, or `docs/` from this skill.
11. For S2, create `checklists/requirements.md` only when it adds real governance value. Checklist items must test requirement quality, not implementation behavior.

## Structured Clarification

Use this process to migrate the useful `speckit-clarify` behavior into the `specs/` workflow.

1. Scan the draft or existing `spec.md` across these categories and mark each internally as Clear, Partial, Missing, Deferred, or Outstanding:
   - Functional scope and behavior: goals, success criteria, out-of-scope items, user roles.
   - Requirement summary and traceability: REQ IDs, priorities, sources, and related AC.
   - Business rules and constraints: permission, security, calculation, compatibility, integration, boundary rules.
   - Functional overview and workflow: sub-functions, main flow, alternate flow, failure flow, async or scheduled flow.
   - Impact and risk overview: candidate modules, APIs, data, config, external systems, risks, mitigations.
   - Domain and data model: entities, identity rules, lifecycle, persistent storage, DDL impact, volume assumptions.
   - Interaction and UX flow: critical journeys, empty/error/loading states, accessibility or localization.
   - Non-functional requirements: performance, reliability, observability, security, privacy, compliance.
   - Integration and dependencies: external services, import/export formats, protocol or version assumptions.
   - Edge cases and failure handling: negative paths, throttling, conflicts, concurrency.
   - Constraints and tradeoffs: technical constraints, rejected alternatives, explicit limitations.
   - Terminology and consistency: canonical terms and avoided synonyms.
   - Completion signals: AC testability and measurable done criteria.
   - Miscellaneous placeholders: TODO, unresolved decisions, vague terms such as "fast", "robust", or "intuitive".
2. Build a prioritized question queue, but ask only questions whose answer materially changes scope, AC, architecture, data modeling, task breakdown, test design, UX, operations, security, or compliance.
3. Ask at most 5 accepted questions per clarification session. Present exactly one question at a time.
4. For option questions, provide 2-5 mutually exclusive options and a recommended option with 1-2 sentences of reasoning. Accept `yes`, `recommended`, or an option letter.
5. For short-answer questions, constrain the answer to 5 words or fewer and provide a suggested answer when the repository facts support one.
6. Stop asking when all critical ambiguity is resolved, the user says to proceed, or the 5-question limit is reached. Record remaining lower-impact gaps under Open Questions or the completion report.

## Integrating Answers

After each accepted answer:

1. Ensure `spec.md` has `## Clarifications`; create `### Session YYYY-MM-DD` for the current date if missing.
2. Append one bullet: `- Q: <question> -> A: <final answer>`.
3. Immediately update the most relevant section:
   - Functional ambiguity -> `功能需求` or `验收标准`.
   - Actor or UX distinction -> `用户与场景`.
   - Data shape -> `关键数据或领域对象`.
   - Non-functional constraint -> `非功能需求`.
   - Failure or boundary behavior -> `验收标准` or `范围`.
   - Terminology conflict -> normalize the term across the spec.
4. Replace contradictory or obsolete text instead of duplicating it.
5. Save `spec.md` after each answer to reduce context-loss risk.
6. Validate after each write:
   - One clarification bullet per accepted answer.
   - No more than 5 accepted questions in the session.
   - No unresolved placeholders the new answer was meant to resolve.
   - No contradictory alternatives remain.
   - Terminology is consistent across touched sections.

## Output Rules

- Create or update only `specs/features/<feature-slug>/spec.md` and, for S2 when needed, `specs/features/<feature-slug>/checklists/requirements.md`.
- Generated artifact headings and fixed prose must be Chinese-first. Keep file names, IDs, paths, and technical markers such as `REQ-001` and `AC-001` unchanged.
- Do not generate `plan.md` or `tasks.md`; leave that to `fons4ai-sdd-design` and `fons4ai-sdd-tasks`.
- Do not write business code.
- End with the feature path, SDD level, number of accepted clarification questions, sections touched, knowledge impact, outstanding or deferred gaps, and suggested next skill.
