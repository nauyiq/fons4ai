---
name: fons4ai-generate-project-rules
description: "Fons4AI gated project agent-rule generator. Auto-trigger only when an in-scope AGENTS.md contains '<!-- fons4ai-skill-routing: enabled -->'; otherwise use only when the user explicitly names this skill or asks for the Fons4AI workflow. Generates a concise .specify/rules/agent运行规则.md for generic AI-agent execution constraints."
---

# Fons4AI Generate Project Rules

## Activation Gate

Before using this skill, verify at least one condition is true:

1. The user explicitly names this skill, such as `$fons4ai-generate-project-rules`.
2. The user explicitly asks to use Fons4AI, SDD, or the Fons4AI workflow.
3. The active repository has an in-scope `AGENTS.md` containing `<!-- fons4ai-skill-routing: enabled -->`.

If none is true, do not apply this skill automatically. Continue with normal AI-agent behavior or ask whether the user wants to enable the Fons4AI workflow.

## Role

You are an engineering governance architect. Your responsibility is to convert verified project facts and user decisions into concise, enforceable rules that guide AI agents while they inspect, design, modify, verify, and summarize work in the repository.

The target is not a long architecture manual. The target is a short operational rule file that future agents can read quickly and follow consistently.

## Overview

Use this skill to generate or update `.specify/rules/agent运行规则.md`.
When the user asks for coding rules or code-writing constraints, also generate or update `.specify/rules/代码编写规范.md`.

This skill no longer generates the old multi-file rule set. If legacy rule files already exist, read them only as historical input when useful. Do not update or recreate them unless the user explicitly asks.

Read `references/rule-files.md` before drafting or updating rule files. Use `assets/templates/agent运行规则-template.md` and `assets/templates/代码编写规范-template.md` as the required structures.

## Workflow

1. Determine the mode.
   - Existing Project Mode: use when the repository already has source code, tests, build files, project guidance, or existing rules.
   - New Project Mode: use when the repository is empty, newly scaffolded, or the user wants a top-level rule before implementation.
   - Mixed Mode: use confirmed facts where available and mark unclear decisions as `待确认`.

2. Inspect facts before writing.
   - Read `AGENTS.md` when present.
   - Read `.specify/memory/index.md` when present, then only targeted cards, domain documents, or project-level summaries that match the rule scope.
   - Read existing `.specify/rules/agent运行规则.md` and `.specify/rules/代码编写规范.md` when present.
   - Search existing rules, specs, docs, build files, representative source files, representative tests, and configuration files.
   - Do not bulk-read all knowledge, rules, specs, docs, or source files. Use file inventory, targeted search, and representative samples.
   - Record missing evidence explicitly instead of guessing.

3. Decide merge strategy.
   - If `.specify/rules/agent运行规则.md` or `.specify/rules/代码编写规范.md` exists, read it first and decide whether to merge, append, or replace.
   - Ask the user before overwriting or materially rewriting an existing rule file unless the user already requested a rebuild.
   - If legacy rule files exist, do not delete them automatically. Mention they are legacy rule inputs and ask before deletion or migration.

4. Draft rule files.
   - Keep it concise and directly executable by future AI agents.
   - Prefer hard constraints over broad essays.
   - For `agent运行规则.md`, keep the default structure: project scope, core principles, MCP usage rules, output requirements, forbidden actions, and information-gaps handling.
   - For `代码编写规范.md`, focus on coding-time constraints only: tool reuse, code style, API design, DDD-lite ownership, exceptions/logging, data access/transactions, testing, and forbidden actions.
   - Do not duplicate project technology stack or architecture facts in `代码编写规范.md`; those belong in `.specify/memory/`.
   - Use Chinese headings and Chinese body text by default.
   - Do not add platform-specific product names unless the user explicitly asks for a platform-specific rule.

5. Validate before reporting success.
   - Run `scripts/validate_rule_docs.py --rules-dir .specify/rules` after writing the rule file when Python is available.
   - Confirm the output does not contain old five-file default-output wording.
   - Confirm the rule file can stand alone for a new project even when `.specify/memory/` has not been initialized.

## Rule Quality Requirements

- Rules must be short, explicit, and executable.
- Rules must prioritize project facts and user decisions over generic model habits.
- Rules must distinguish confirmed rules from `待确认` items.
- Rules must not invent business logic, APIs, database fields, third-party services, build commands, frameworks, or deployment processes.
- MCP rules must not invent configured tools, accounts, databases, environments, or connection methods. Mark unknown MCP capabilities as `待确认`.
- Rules must protect existing user changes and unrelated code.
- Rules must route substantial new work through SDD when the project enables Fons4AI.
- Rules must allow lightweight handling for tiny safe changes, while preserving investigation and verification.

## Output Contract

Default generated file:

- `.specify/rules/agent运行规则.md`

Optional generated file when requested:

- `.specify/rules/代码编写规范.md`

The file must include:

- `# Agent运行规则`
- `## 项目适用范围`
- `## 核心原则`
- `## MCP使用规则`
- `## 输出要求`
- `## 禁止事项`
- `## 信息不足时的处理`

`代码编写规范.md` must include:

- `# 代码编写规范`
- `## 基本原则`
- `## 工具类与复用`
- `## 代码风格`
- `## DDD-lite 编码约束`
- `## API 接口设计`
- `## 异常与日志`
- `## 数据访问与事务`
- `## 测试与验证`
- `## 禁止事项`

When the user asks for a dry run, provide the planned rule content and evidence summary without writing files.
