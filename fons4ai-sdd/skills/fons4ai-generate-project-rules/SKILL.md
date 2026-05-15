---
name: fons4ai-generate-project-rules
description: "Fons4AI gated architect-grade project-rule generator. Auto-trigger only when an in-scope AGENTS.md contains '<!-- fons4ai-skill-routing: enabled -->'; otherwise use only when the user explicitly names this skill or asks for the Fons4AI workflow."
---

# Fons4AI Generate Project Rules

## Activation Gate

Before using this skill, verify at least one condition is true:

1. The user explicitly names this skill, such as `$fons4ai-generate-project-rules`.
2. The user explicitly asks to use Fons4AI, SDD, or the Fons4AI workflow.
3. The active repository has an in-scope `AGENTS.md` containing `<!-- fons4ai-skill-routing: enabled -->`.

If none is true, do not apply this skill automatically. Continue with normal Codex behavior or ask whether the user wants to enable the Fons4AI workflow.

## Overview

Use this skill to generate architect-grade, project-specific Markdown rules under `.specify/rules/`.
The rules must be executable, verifiable, traceable to repository facts, and maintainable by future agents or engineers.

Default output files:

- `.specify/rules/code-style-rule.md`
- `.specify/rules/project-structure-rule.md`
- `.specify/rules/features-rule.md`
- `.specify/rules/testing-rule.md`
- `.specify/rules/data-ddl-rule.md`

Read `references/rule-files.md` before drafting or updating rules. Use the templates in `assets/templates/` as the required structure.

## Workflow

1. Choose the working mode.
   - Existing Project Mode: use when build files, source files, tests, or existing conventions are present.
   - New Project Mode: use when the repository is empty, newly scaffolded, or the user explicitly asks to define rules before implementation.
   - Mixed Mode: use existing facts where available and mark missing project decisions as `默认建议` or `待补充约定`.

2. Build a fact inventory before writing.
   - Read root build files such as `pom.xml`, `package.json`, `build.gradle`, or equivalent.
   - Read module build files and list source/resource/test directories.
   - Search `.specify/memory/`, `.specify/sql/`, `.specify/rules/`, `specs/`, `AGENTS.md`, `.cursorrules`, `.cursor/rules/`, legacy/custom rule directories, and local skill files.
   - Sample representative source, configuration, test, migration, entity/model, mapper/repository, API/controller, service, and adapter files from each major module.
   - Record missing evidence explicitly instead of guessing.

3. Create a rule evidence matrix before drafting.
   - For each rule document, list the repository facts or user decisions that justify its strongest rules.
   - Classify each fact as `已确认`, `默认建议`, or `待补充约定`.
   - If a rule cannot be backed by repository evidence or user input, either downgrade it to `默认建议` or place it under `待补充约定`.

4. Draft the five rule documents from templates.
   - Use `assets/templates/code-style-rule-template.md`.
   - Use `assets/templates/project-structure-rule-template.md`.
   - Use `assets/templates/features-rule-template.md`.
   - Use `assets/templates/testing-rule-template.md`.
   - Use `assets/templates/data-ddl-rule-template.md`.
   - Keep each file focused on its responsibility; avoid repeating the same rule across documents unless cross-reference is necessary.

5. Handle existing target files conservatively.
   - If any target rule file already exists, read it first.
   - Explain whether the work should merge, replace, or preserve each file.
   - Ask the user for confirmation before overwriting, deleting, or materially rewriting existing rule files.

6. Validate before reporting success.
   - Confirm all five default rule files are planned or generated unless the user explicitly narrows scope.
   - Ensure each rule document contains project facts, mandatory rules, recommended rules, forbidden practices, exception mechanism, and acceptance checks.
   - Run `scripts/validate_rule_docs.py --rules-dir .specify/rules` after writing actual rule files when Python is available.

## Architect-Grade Rule Requirements

- Write in Chinese.
- Rules are not tutorials, but they must not be slogans. Key rules must state trigger condition, execution requirement, and exception path.
- Prefer enforceable wording: `必须`, `禁止`, `优先`, `仅当`, `除非`, `需要确认`.
- Use repository evidence first, then explicit user decisions, then conservative defaults.
- Separate `强制规则`, `推荐规则`, `禁止事项`, `例外机制`, and `待确认约定`.
- Include concrete examples only when they prevent ambiguity; keep examples short and project-specific.
- Never invent frameworks, modules, CI, deployment, database, migration tools, or release processes that are not visible in the repository or explicitly requested.
- Keep KISS: avoid architecture mandates that the repository does not need or cannot enforce.

## Output Contract

Each rule Markdown file must start with:

```markdown
# <规则标题>

> 适用范围：<系统、模块或业务域>
> 生成依据：<用户输入、仓库事实、知识库或待确认>
> 规则状态：<已有项目提炼 | 新项目初始约定 | 混合>
```

Each file must include:

- `## 项目事实`
- `## 强制规则`
- `## 推荐规则`
- `## 禁止事项`
- `## 例外机制`
- `## 待确认约定`
- `## 验收检查`

When the user asks for a dry run, provide the planned file list, evidence matrix summary, and summarized rule content without writing files.
