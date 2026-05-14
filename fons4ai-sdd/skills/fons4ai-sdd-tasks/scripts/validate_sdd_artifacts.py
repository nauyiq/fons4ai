#!/usr/bin/env python3
"""Validate minimal Fons4AI SDD artifact consistency."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


AC_RE = re.compile(r"\bAC-\d{3}\b")
TASK_RE = re.compile(r"^- \[[ xX]\] (T\d{3})(?:\s|$)", re.MULTILINE)
SQL_PATH_RE = re.compile(r"\.specify/sql/[A-Za-z0-9_./-]+\.sql")
S2_RE = re.compile(r"SDD Level:\s*`?S2`?", re.IGNORECASE)
RISK_CONTROL_RE = re.compile(
    r"rollback|compatib|regression|permission|security|observability|migration|risk|checklist|"
    r"回滚|兼容|回归|权限|安全|观测|迁移|风险|检查",
    re.IGNORECASE,
)


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise
    except UnicodeDecodeError:
        return path.read_text()


def task_blocks(tasks_text: str) -> list[tuple[str, str]]:
    matches = list(TASK_RE.finditer(tasks_text))
    blocks: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(tasks_text)
        blocks.append((match.group(1), tasks_text[start:end]))
    return blocks


def validate(feature_dir: Path) -> list[str]:
    errors: list[str] = []
    spec = feature_dir / "spec.md"
    plan = feature_dir / "plan.md"
    tasks = feature_dir / "tasks.md"

    for required in (spec, plan, tasks):
        if not required.exists():
            errors.append(f"Missing required artifact: {required}")

    if errors:
        return errors

    spec_text = read(spec)
    plan_text = read(plan)
    tasks_text = read(tasks)
    all_text = "\n".join((spec_text, plan_text, tasks_text))

    ac_ids = sorted(set(AC_RE.findall(spec_text)))
    if not ac_ids:
        errors.append("spec.md contains no AC-### acceptance criteria IDs")

    for ac_id in ac_ids:
        if ac_id not in plan_text:
            errors.append(f"{ac_id} is not referenced in plan.md")
        if ac_id not in tasks_text:
            errors.append(f"{ac_id} is not referenced in tasks.md")

    plan_requires_sql_sync = (
        re.search(r"SQL DDL update needed:\s*yes", plan_text, re.IGNORECASE)
        or re.search(r"DDL sync required:\s*yes", plan_text, re.IGNORECASE)
        or re.search(r"DDL file action:\s*(add|update|rename)", plan_text, re.IGNORECASE)
    )
    plan_sql_files = sorted(set(SQL_PATH_RE.findall(plan_text)))
    if plan_requires_sql_sync and not plan_sql_files:
        errors.append("plan.md declares DDL sync but names no .specify/sql/*.sql file")
    for sql_file in plan_sql_files:
        if sql_file not in tasks_text:
            errors.append(f"{sql_file} is referenced in plan.md but not in tasks.md")

    if "## Knowledge Impact" not in plan_text:
        errors.append("plan.md is missing '## Knowledge Impact'")
    if re.search(r"Knowledge Sync Needed:\s*yes", plan_text, re.IGNORECASE):
        has_knowledge_task = (
            "Knowledge" in tasks_text
            or "truth-source" in tasks_text
            or ".specify/memory/" in tasks_text
            or "fons4ai-knowledge-summary" in tasks_text
        )
        if not has_knowledge_task:
            errors.append("plan.md declares Knowledge Sync Needed: yes but tasks.md has no knowledge sync task")

    if S2_RE.search(all_text):
        if "## Risk and Rollback" not in plan_text:
            errors.append("S2 plan.md is missing '## Risk and Rollback'")
        if "S2 Quality Gates" not in tasks_text and not RISK_CONTROL_RE.search(tasks_text):
            errors.append("S2 tasks.md must include S2 quality gates or explicit risk-control tasks")

    blocks = task_blocks(tasks_text)
    if not blocks:
        errors.append("tasks.md contains no checklist tasks in '- [ ] T001' format")

    seen: set[str] = set()
    for task_id, block in blocks:
        if task_id in seen:
            errors.append(f"Duplicate task ID: {task_id}")
        seen.add(task_id)
        if not AC_RE.search(block):
            errors.append(f"{task_id} has no AC mapping")
        for label in ("Files:", "Verification:", "Done:"):
            if label not in block:
                errors.append(f"{task_id} is missing '{label}'")

    return errors


def validate_change_file(change_file: Path) -> list[str]:
    errors: list[str] = []
    if not change_file.exists():
        return [f"Missing change artifact: {change_file}"]

    text = read(change_file)
    for heading in ("## Impact Analysis", "### Knowledge Impact", "## Regression and Rollback", "## Incremental Tasks"):
        if heading not in text:
            errors.append(f"{change_file} is missing '{heading}'")

    if not AC_RE.search(text):
        errors.append(f"{change_file} contains no AC-### mapping")
    if not TASK_RE.search(text):
        errors.append(f"{change_file} contains no incremental checklist tasks")

    ddl_action = re.search(r"SQL DDL action:\s*(add|update|rename)", text, re.IGNORECASE)
    if ddl_action:
        sql_files = sorted(set(SQL_PATH_RE.findall(text)))
        if not sql_files:
            errors.append(f"{change_file} declares SQL DDL action but names no .specify/sql/*.sql file")
        if "Sync DDL knowledge file" not in text and "Sync DDL" not in text:
            errors.append(f"{change_file} declares SQL DDL action but has no DDL sync task")

    return errors


def validate_bugfix_report(report: Path) -> list[str]:
    errors: list[str] = []
    if not report.exists():
        return [f"Missing bugfix report: {report}"]

    text = read(report)
    required = (
        "## 问题描述",
        "## 复现步骤",
        "## 根因分析",
        "## 自动化测试",
        "## 手动验证",
        "## 回归验证",
        "## 知识库同步",
    )
    for heading in required:
        if heading not in text:
            errors.append(f"{report} is missing '{heading}'")
    for field in ("回滚方案", "Knowledge Sync Needed", "SQL DDL files"):
        if field not in text:
            errors.append(f"{report} is missing '{field}'")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Fons4AI SDD artifacts")
    parser.add_argument("--feature-dir", help="Path to specs/features/<feature-slug>")
    parser.add_argument("--change-file", help="Path to specs/features/<feature-slug>/changes/CR-xxx.md")
    parser.add_argument("--bugfix-report", help="Path to specs/bugfixes/<bug-slug>/bugfix-report.md")
    args = parser.parse_args()

    selected = [value for value in (args.feature_dir, args.change_file, args.bugfix_report) if value]
    if len(selected) != 1:
        print("ERROR: provide exactly one of --feature-dir, --change-file, or --bugfix-report", file=sys.stderr)
        return 2

    if args.feature_dir:
        target = Path(args.feature_dir).resolve()
        errors = validate(target)
        success = f"OK: {target} SDD artifacts are valid"
    elif args.change_file:
        target = Path(args.change_file).resolve()
        errors = validate_change_file(target)
        success = f"OK: {target} SDD change artifact is valid"
    else:
        target = Path(args.bugfix_report).resolve()
        errors = validate_bugfix_report(target)
        success = f"OK: {target} bugfix report is valid"

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(success)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
