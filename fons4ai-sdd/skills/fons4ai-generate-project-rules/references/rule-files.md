# Rule Files Reference

Use this reference when creating or updating project-level Markdown rule files.

## Default File Set

Generate exactly these files by default:

- `rules/code-style-rule.md`
- `rules/project-structure-rule.md`
- `rules/features-rule.md`

When `.specify/memory/` or `.specify/sql/` exists, treat it as a long-lived architecture, data, and DDL knowledge source for rule generation.

Do not generate an index file, `AGENTS.md`, `.cursorrules`, or Cursor-specific rules unless the user explicitly asks.

## Modes

Use Existing Project Mode when the repository already contains build files, source files, tests, or established rules. Rules should cite observed conventions.

Use New Project Mode when the repository is empty or the user wants rules before implementation. Rules should be based on explicit user choices plus conservative defaults.

In New Project Mode, group content with these labels when useful:

- `已确认规则`: decisions directly stated by the user or visible in scaffold files;
- `默认建议`: low-risk conventions chosen to let the project start consistently;
- `待补充约定`: decisions that require future code, architecture, or team preference.

## `code-style-rule.md`

Purpose: define how code should look and behave at the source level.

Include rules for:

- language and framework conventions discovered from the repository;
- initial language and framework conventions selected for a new project;
- naming for packages, classes, methods, fields, constants, records, DTOs, and tests;
- annotation style and dependency injection style;
- comment policy, especially required comments for key logic and class fields;
- exception handling and business error conventions;
- logging level, log message style, and sensitive-data boundaries;
- test naming, test structure, and minimum verification expectations.

Avoid:

- generic style rules that conflict with existing code;
- formatting mandates if no formatter or stable convention exists;
- adding external libraries or tools as requirements without repository evidence.

## `project-structure-rule.md`

Purpose: define module, package, and file placement conventions.

Include rules for:

- repository module layout and parent-child build relationships;
- planned module layout for a new project when the user has confirmed it;
- source, resource, and test directory placement;
- package naming and layer boundaries;
- where configuration, constants, infrastructure adapters, strategies, and shared utilities belong;
- dependency direction between modules;
- where SDD artifacts such as `rules/`, `specs/`, and project-local skills should live.
- where `.specify/memory/` long-lived project knowledge belongs and how feature work should reference or synchronize it.
- where `.specify/sql/` DDL knowledge belongs and how schema changes should synchronize it.

Avoid:

- inventing modules that do not exist;
- presenting a proposed module as existing;
- enforcing clean architecture or DDD terminology unless already visible;
- allowing lower-level modules to depend on app-level modules.

## `features-rule.md`

Purpose: define how new features and behavior changes should be implemented.

Include rules for:

- requirement clarification before design or coding;
- fact-first repository investigation;
- new-project upfront decisions before scaffolding code;
- technical design before implementation for non-trivial changes;
- task breakdown aligned with TDD when implementation is requested;
- reuse of existing utilities, components, and local conventions;
- migration, compatibility, and rollback considerations when public behavior changes;
- verification expectations: unit tests, integration tests, build checks, and manual checks where appropriate;
- documentation updates for rules, specs, or long-lived architecture notes.
- whether a feature changes `.specify/memory/` business, technical, or data architecture facts and how that synchronization should be tracked.
- whether a feature changes `.specify/sql/` DDL files and how schema synchronization should be tracked.

Avoid:

- requiring heavyweight process for tiny safe edits;
- permitting destructive edits without user confirmation;
- treating generated assumptions as confirmed requirements.

## Source Evidence Checklist

Before writing Existing Project Mode rules, inspect enough of the repository to support the content:

- root build file and module build files;
- file tree for major source, resource, and test roots;
- representative classes from each major module;
- existing rule/spec/agent instruction files;
- `.specify/memory/business-architecture.md`, `.specify/memory/technical-architecture.md`, `.specify/memory/data-architecture.md`, and `.specify/memory/constitution.md` when present;
- `.specify/sql/*.sql` when present;
- test framework and naming patterns if tests exist.

If evidence is missing, write a short "待补充约定" section instead of guessing.

Before writing New Project Mode rules, establish at least:

- project goal and application type;
- primary language, runtime, framework, and build tool;
- single-module or multi-module structure;
- base package or namespace;
- unit/integration test expectations;
- documentation or SDD artifacts that should be maintained.
