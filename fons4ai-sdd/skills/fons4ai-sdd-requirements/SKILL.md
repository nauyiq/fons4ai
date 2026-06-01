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

Use this skill to turn a feature idea into the first SDD artifact: a business-oriented requirement specification in `specs/features/<feature-slug>/spec.md`.
This skill must clarify blocking requirement ambiguity before writing a formal `spec.md`; `use SDD` alone is not permission to guess requirement semantics.
Feature artifacts are written under `specs/features/`; `.specify/memory/` and `.specify/sql/` are read as long-lived project context when present.

## Required Context

1. Read `AGENTS.md` and load the SDD contract from `references/sdd-artifact-contract.md`, including its context-loading rules.
2. Locate context before drafting by searching feature names, business terms, module names, table/model names, API names, and related `REQ-###`/`AC-###` IDs. Do not bulk-read all of `.specify/memory/`, `.specify/sql/`, `.specify/rules/`, or `specs/` by default.
   - Optionally run `scripts/find_relevant_context.py --root <repo-root> <keyword...>` to get candidate truth-source files before reading.
3. Read only relevant project rules, matching truth-source sections, related SQL files, existing SDD artifacts, and source files when the feature touches existing behavior.
4. Use `assets/templates/spec-template.md` as the output structure.
5. If a target feature directory or `spec.md` already exists, read it first and ask the user before replacing or materially rewriting it.
6. Prefer facts from `.specify/memory/business-architecture.md` for business domains, roles, processes, business objects, and durable business rules. Prefer `.specify/memory/data-architecture.md` and targeted `.specify/sql/**/*.sql` files for confirmed data model and DDL facts. If absent, continue from repository and user facts.

## Workflow

1. Derive or confirm `<feature-slug>` in lowercase hyphen-case. Prefer short action-noun names such as `loan-approval-rule`.
2. Classify the feature:
   - Use `S1` by default for small or normal feature work.
   - Use `S2` for database migrations, public API changes, security/permission changes, cache/MQ/transaction boundaries, cross-core-module work, compatibility risk, or high rollback cost.
   - Keep the level and reason as workflow context for the design handoff. Do not expose SDD classification in the business-oriented `spec.md`.
3. Run the clarification gate before writing a formal `spec.md`.
   - If a blocking ambiguity exists, stop and ask exactly one highest-impact clarification question. Do not create or update `spec.md` in the same turn.
   - If the user explicitly asks for a draft before answering, create only a draft `spec.md` with `文档状态：草案-待确认`, mark assumptions as `待确认`, and do not recommend design or task generation.
   - If all blocking ambiguities are closed, create a formal business-oriented `spec.md` without exposing the internal clarification checklist, clarification status, or question log.
4. Build the requirement summary before writing AC:
   - Use `REQ-001`, `REQ-002`, ... for requirement points and business capabilities.
   - Record priority, business scenario, and related AC for every requirement.
   - Include background and goals, business scope, roles and scenarios, business workflow, business rules, business data description, business impact, risks, and acceptance criteria.
   - Keep `spec.md` focused on requirements, behavior, business rules, workflows, impacts, and acceptance. Do not move detailed technical design into `spec.md`.
   - Use business terminology. Do not expose modules, classes, tables, columns, DDL paths, MCP details, or technical architecture facts in `spec.md`.
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
7. For S1, use the minimal complete profile: keep details concise but cover business goals, scope, scenarios, workflow, rules, business data, business impact, REQ/AC mapping, and AC. Use `不适用，原因` where workflow, risk, diagram, or data impact is genuinely absent.
8. For S2, include business workflow, risks, business data, non-functional requirements, compatibility, security, and migration implications in business language and enough detail for design work.
9. Generate Mermaid business flowcharts only when repository/user facts support the participants and flow. Do not invent systems, tables, APIs, or actors to fill the template.
10. Identify knowledge and persistent-data impact internally for the design handoff and completion summary. Do not expose truth-source paths, `.specify/sql/` paths, MCP details, table names, or column names in `spec.md`, and do not update truth sources from this skill.
11. For S2, create `checklists/requirements.md` only when it adds real governance value. Checklist items must test requirement quality, not implementation behavior.

## Clarification Gate

The clarification gate decides whether the skill may write a formal requirements artifact.

Blocking ambiguity means any missing or conflicting answer that can change one of these:

- feature scope, target users, expected behavior, success/failure behavior, or out-of-scope boundaries;
- business terminology, core business objects, ownership, identity, lifecycle, status, or data meaning;
- acceptance criteria, observable behavior, compatibility, security/permission, integration, or non-functional constraints;
- data model, table grouping, field naming, DDL impact, migration, rollback, or source-of-truth updates;
- SDD level (`S1` or `S2`), impacted modules, implementation strategy, test strategy, or task breakdown.

Rules:

1. If at least one blocking ambiguity exists, ask one question and stop. Do not write or update formal `spec.md`, `plan.md`, `tasks.md`, CR, or business code.
2. Ask the highest-impact question first. Prefer a concrete recommendation with 2-5 options when options are known.
3. Treat answers such as `按推荐`, `方案 A`, or a clear short answer as accepted clarification.
4. Do not treat `使用 SDD`, `继续`, `先生成`, `看一下`, or similar ambiguous messages as clarification closure.
5. Only continue to formal `spec.md` when blocking ambiguities are closed, or when the user explicitly says to create a draft with assumptions.
6. A draft with unresolved blocking ambiguity must use `文档状态：草案-待确认` and must not be used as input for `fons4ai-sdd-design` or `fons4ai-sdd-tasks` until the gate is closed.

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

1. Keep the accepted answer in the active conversation context. Do not expose the internal question log in formal artifacts.
2. Update the most relevant business section only when maintaining an explicitly requested draft or updating an existing specification:
   - Functional ambiguity -> `功能需求` or `验收标准`.
   - Actor or UX distinction -> `角色与业务场景`.
   - Business data meaning -> `业务数据说明`.
   - Non-functional constraint -> `非功能要求`.
   - Failure or boundary behavior -> `验收标准` or `业务范围`.
   - Terminology conflict -> normalize the term across the spec.
3. Replace contradictory or obsolete text instead of duplicating it.
4. Do not create a formal `spec.md` until blocking ambiguities are closed. If an explicit draft exists, keep `文档状态：草案-待确认` until the user resolves the gaps.
5. Validate after each draft update:
   - No unresolved placeholders the new answer was meant to resolve.
   - No contradictory alternatives remain.
   - Terminology is consistent across touched sections.

## Output Rules

- Create or update only `specs/features/<feature-slug>/spec.md` and, for S2 when needed, `specs/features/<feature-slug>/checklists/requirements.md`.
- If the clarification gate is blocked, output only the blocking reason, the single clarification question, recommended options when available, and what will be generated after the answer. Do not write formal artifacts.
- Formal `spec.md` must not expose clarification-gate tables, clarification status, question logs, repository-fact inventories, or knowledge-context inventories. Draft specs are allowed only after explicit user request and must include `文档状态：草案-待确认`.
- Generated artifact headings and fixed prose must be Chinese-first. Keep file names, IDs, paths, and technical markers such as `REQ-001` and `AC-001` unchanged.
- Do not generate `plan.md` or `tasks.md`; leave that to `fons4ai-sdd-design` and `fons4ai-sdd-tasks`.
- Do not write business code.
- End with the feature path, SDD level, number of accepted clarification questions, sections touched, knowledge impact, outstanding or deferred gaps, and suggested next skill.
