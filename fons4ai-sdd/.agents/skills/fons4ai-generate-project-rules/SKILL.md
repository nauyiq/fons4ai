---
name: fons4ai-generate-project-rules
description: "Fons4AI gated project-rule generator. Auto-trigger only when an in-scope AGENTS.md contains '<!-- fons4ai-skill-routing: enabled -->'; otherwise use only when the user explicitly names this skill or asks for the Fons4AI workflow."
---

# Generate Project Rules

## Activation Gate

Before using this skill, verify at least one condition is true:

1. The user explicitly names this skill, such as `$fons4ai-generate-project-rules`.
2. The user explicitly asks to use Fons4AI, SDD, or the Fons4AI workflow.
3. The active repository has an in-scope `AGENTS.md` containing `<!-- fons4ai-skill-routing: enabled -->`.

If none is true, do not apply this skill automatically. Continue with normal Codex behavior or ask whether the user wants to enable the Fons4AI workflow.
   
## Overview

You MUST consider the user input before proceeding (if not empty).

Use this skill to generate concise, project-specific rules under `rules/` for either an existing project or a new project that needs rules before implementation.
The default output is exactly three Markdown files: `code-style-rule.md`, `project-structure-rule.md`, and `features-rule.md`.

Read `references/rule-files.md` before drafting or updating the rules.

## Workflow

1. Choose the working mode.
   - Existing Project Mode: use when build files, source files, or existing conventions are present.
   - New Project Mode: use when the repository is empty, newly scaffolded, or the user explicitly asks to define rules before implementation.
   - If both apply, prefer Existing Project Mode for discovered facts and use New Project Mode only for missing decisions.

2. Inspect available facts before writing anything.
   - Read root build files such as `pom.xml`, `package.json`, `build.gradle`, or equivalent.
   - Read module build files and list source/resource/test directories.
   - Search existing rule sources: `.specify/memory/`, `.specify/sql/`, `rules/`, `specs/`, `AGENTS.md`, `.cursorrules`, `.cursor/rules/`, and local skill files.
   - Sample representative source files from each major module to learn naming, layering, annotations, logging, exception, and test style.

3. For New Project Mode, collect or infer the minimum required decisions.
   - Confirm project type, primary language, build tool, framework, module strategy, base package or namespace, test strategy, and delivery constraints.
   - Ask the user only for decisions that materially change the rules and cannot be inferred from the prompt or existing files.
   - If the user does not specify a low-risk convention, choose a conservative default and label it as "默认建议".

4. Derive rules from the strongest available source.
   - Prefer existing project conventions, confirmed `.specify/memory/` architecture facts, and confirmed `.specify/sql/` DDL facts over generic best practices.
   - Include user-stated preferences exactly when they are clear.
   - In New Project Mode, separate "已确认规则", "默认建议", and "待补充约定".
   - Do not invent frameworks, directories, tests, CI, or release processes that are not visible in the repository.
   - Mark uncertain conventions as "当前仓库未形成稳定约定" instead of pretending they exist.

5. Handle existing target files conservatively.
   - If any target rule `.md` file already exists, read it first.
   - Explain whether the change should overwrite, merge, or preserve each file.
   - Ask the user for confirmation before editing existing rule files.

6. Generate only the default files unless the user explicitly asks for more.
   - `rules/code-style-rule.md`
   - `rules/project-structure-rule.md`
   - `rules/features-rule.md`

## Writing Rules

- Write in Chinese.
- Keep rules short, concrete, and enforceable.
- Prefer imperative wording such as "必须", "优先", "禁止", and "避免".
- Keep KISS: avoid broad architecture mandates unless the repository already supports them.
- Add just enough rationale to prevent misuse; do not write long tutorials.
- Separate confirmed rules from unresolved gaps.
- In New Project Mode, never present defaults as repository facts.
- Do not modify `AGENTS.md`, `.cursorrules`, or other tool-entry files unless the user explicitly requests it.

## Output Contract

Each rule Markdown file should start with:

```markdown
# <规则标题>

> 适用范围：<简短范围说明>
> 生成依据：<列出主要仓库事实来源>
> 规则状态：<已有项目提炼 | 新项目初始约定 | 混合>
```

Then use compact sections with bullets. Avoid nested bullets unless they clarify a decision boundary.

When the user asks for a dry run, provide the planned file list and summarized rule content without writing files.
