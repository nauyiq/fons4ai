#!/usr/bin/env python3
"""Validate minimal Fons4AI SDD artifact consistency."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


AC_RE = re.compile(r"\bAC-\d{3}\b")
REQ_RE = re.compile(r"\bREQ-\d{3}\b")
TASK_RE = re.compile(r"^- \[[ xX]\] (T\d{3})(?:\s|$)", re.MULTILINE)
SQL_PATH_RE = re.compile(r"\.specify/sql/[A-Za-z0-9_./-]+\.sql")
S2_RE = re.compile(r"(SDD\s*Level|SDD\s*等级)\s*[:：]\s*`?S2`?", re.IGNORECASE)

APPROVAL_GATE_HEADINGS = ("## 实现确认门禁", "## Implementation Approval Gate")
CLARIFICATION_GATE_HEADINGS = ("## 需求澄清门禁", "## Requirement Clarification Gate")
CHANGE_CLARIFICATION_GATE_HEADINGS = ("## 变更澄清门禁", "## Change Clarification Gate")
CLARIFICATION_STATUS_RE = re.compile(r"(澄清状态|Clarification Status)\s*[:：]\s*([^\n\r]+)", re.IGNORECASE)
BLOCKING_CLARIFICATION_RE = re.compile(r"阻塞|草案|blocking|draft", re.IGNORECASE)
CLOSED_CLARIFICATION_RE = re.compile(r"已关闭|closed", re.IGNORECASE)
SPEC_REQUIRED_HEADING_GROUPS = (
    ("需求概要", ("## 需求概要", "## Requirement Summary")),
    ("关键业务规则与约束", ("## 关键业务规则与约束", "## Business Rules and Constraints")),
    ("功能概览", ("## 功能概览", "## Functional Overview")),
    ("影响面概览", ("## 影响面概览", "## Impact Overview")),
)
PLAN_REQUIRED_HEADING_GROUPS = (
    ("关键规则代码片段", ("## 关键规则代码片段", "## Key Rule Code Sketches")),
    ("状态流转设计", ("## 状态流转设计", "## State Transition Design")),
    ("数据结构变更", ("## 数据结构变更", "## Data Structure Changes")),
    ("API 与契约细节", ("## API 与契约细节", "## API and Contract Details")),
    ("事务与一致性", ("## 事务与一致性", "## Transaction and Consistency")),
    ("验证策略", ("## 验证策略", "## Verification Strategy")),
)
PLAN_MODERN_HEADING_GROUPS = (
    ("关键业务规则与策略设计", ("## 关键业务规则与策略设计",)),
)
KNOWLEDGE_IMPACT_HEADINGS = ("## 知识同步影响", "## Knowledge Impact")
RISK_ROLLBACK_HEADINGS = ("## 风险与回滚", "## Risk and Rollback")
S2_QUALITY_GATE_HEADINGS = ("## S2 质量门禁", "## S2 Quality Gates")
CHANGE_REQUIRED_HEADING_GROUPS = (
    ("影响分析", ("## 影响分析", "## Impact Analysis")),
    ("知识同步影响", ("### 知识同步影响", "### Knowledge Impact")),
    ("回归与回滚", ("## 回归与回滚", "## Regression and Rollback")),
    ("实现确认门禁", APPROVAL_GATE_HEADINGS),
    ("增量任务", ("## 增量任务", "## Incremental Tasks")),
)

DOMAIN_QUALITY_RE = re.compile(r"DDD|domain|领域|充血|贫血|业务规则|应用层", re.IGNORECASE)
KNOWLEDGE_OR_DDL_TASK_RE = re.compile(
    r"\.specify/|truth-source|Knowledge|knowledge|知识|DDL|SQL|data-architecture",
    re.IGNORECASE,
)
RISK_CONTROL_RE = re.compile(
    r"rollback|compatib|regression|permission|security|observability|migration|risk|checklist|"
    r"回滚|兼容|回归|权限|安全|观测|迁移|风险|检查|门禁",
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


def heading_match(text: str, heading: str) -> re.Match[str] | None:
    return re.search(rf"^{re.escape(heading)}\s*$", text, re.MULTILINE)


def first_heading(text: str, headings: tuple[str, ...]) -> str | None:
    for heading in headings:
        if heading_match(text, heading):
            return heading
    return None


def has_any_heading(text: str, headings: tuple[str, ...]) -> bool:
    return first_heading(text, headings) is not None


def section_content(text: str, headings: tuple[str, ...]) -> str:
    heading = first_heading(text, headings)
    if not heading:
        return ""
    match = heading_match(text, heading)
    if not match:
        return ""
    content_start = match.end()
    next_heading = re.search(r"^##\s+", text[content_start:], re.MULTILINE)
    if not next_heading:
        return text[content_start:]
    return text[content_start : content_start + next_heading.start()]


def has_section_content(text: str, headings: tuple[str, ...]) -> bool:
    content = section_content(text, headings)
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if set(stripped) <= {"-", "|", ":", "：", " "}:
            continue
        return True
    return False


def validate_required_heading_groups(
    text: str,
    heading_groups: tuple[tuple[str, tuple[str, ...]], ...],
    artifact_name: str,
) -> list[str]:
    errors: list[str] = []
    for display_name, headings in heading_groups:
        if not has_any_heading(text, headings):
            errors.append(f"{artifact_name} is missing section '{display_name}'")
    return errors


def validate_req_ac_mapping(spec_text: str) -> list[str]:
    errors: list[str] = []
    req_ids = sorted(set(REQ_RE.findall(spec_text)))
    for req_id in req_ids:
        mapped = any(req_id in line and AC_RE.search(line) for line in spec_text.splitlines())
        if not mapped:
            errors.append(f"{req_id} has no AC mapping in spec.md")
    return errors


def validate_clarification_gate(
    text: str,
    artifact_name: str,
    gate_headings: tuple[str, ...],
) -> list[str]:
    errors: list[str] = []
    if not has_any_heading(text, gate_headings):
        errors.append(f"{artifact_name} is missing clarification gate section")

    statuses = [match.group(2).strip() for match in CLARIFICATION_STATUS_RE.finditer(text)]
    if not statuses:
        errors.append(f"{artifact_name} is missing clarification status")
        return errors

    if any(BLOCKING_CLARIFICATION_RE.search(status) for status in statuses):
        errors.append(f"{artifact_name} clarification gate is not closed")
    if not any(CLOSED_CLARIFICATION_RE.search(status) for status in statuses):
        errors.append(f"{artifact_name} clarification status must be closed before design, task, or implementation planning")
    return errors


def validate_quality_domain_check(task_id: str, block: str, context: str) -> list[str]:
    if KNOWLEDGE_OR_DDL_TASK_RE.search(block):
        return []
    if DOMAIN_QUALITY_RE.search(block):
        return []
    return [f"{task_id} in {context} Quality is missing DDD-lite/domain-modeling check"]


def plan_declares_sql_sync(plan_text: str) -> bool:
    return bool(
        re.search(r"SQL DDL update needed\s*:\s*yes", plan_text, re.IGNORECASE)
        or re.search(r"DDL sync required\s*:\s*yes", plan_text, re.IGNORECASE)
        or re.search(r"DDL file action\s*:\s*(add|update|rename)", plan_text, re.IGNORECASE)
        or re.search(r"是否需要\s*DDL\s*同步\s*[：:]\s*是", plan_text)
        or re.search(r"DDL\s*文件动作\s*[：:]\s*(新增|更新|重命名)", plan_text)
    )


def validate(feature_dir: Path, strict: bool = False) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    spec = feature_dir / "spec.md"
    plan = feature_dir / "plan.md"
    tasks = feature_dir / "tasks.md"

    for required in (spec, plan, tasks):
        if not required.exists():
            errors.append(f"Missing required artifact: {required}")

    if errors:
        return errors, warnings

    spec_text = read(spec)
    plan_text = read(plan)
    tasks_text = read(tasks)
    all_text = "\n".join((spec_text, plan_text, tasks_text))

    ac_ids = sorted(set(AC_RE.findall(spec_text)))
    if not ac_ids:
        errors.append("spec.md contains no AC-### acceptance criteria IDs")

    errors.extend(validate_required_heading_groups(spec_text, SPEC_REQUIRED_HEADING_GROUPS, "spec.md"))
    clarification_errors = validate_clarification_gate(spec_text, "spec.md", CLARIFICATION_GATE_HEADINGS)
    if strict:
        errors.extend(clarification_errors)
    else:
        warnings.extend(
            f"{message}; legacy-compatible mode allows this, but close requirements clarification before new design/tasks"
            for message in clarification_errors
        )
    errors.extend(validate_req_ac_mapping(spec_text))
    errors.extend(validate_required_heading_groups(plan_text, PLAN_REQUIRED_HEADING_GROUPS, "plan.md"))
    modern_plan_errors = validate_required_heading_groups(plan_text, PLAN_MODERN_HEADING_GROUPS, "plan.md")
    if strict:
        errors.extend(modern_plan_errors)
    else:
        warnings.extend(
            f"{message}; legacy-compatible mode allows this, but update plan.md before new implementation"
            for message in modern_plan_errors
        )

    for ac_id in ac_ids:
        if ac_id not in plan_text:
            errors.append(f"{ac_id} is not referenced in plan.md")
        if ac_id not in tasks_text:
            errors.append(f"{ac_id} is not referenced in tasks.md")

    plan_sql_files = sorted(set(SQL_PATH_RE.findall(plan_text)))
    if plan_declares_sql_sync(plan_text) and not plan_sql_files:
        errors.append("plan.md declares DDL sync but names no .specify/sql/**/*.sql file")
    for sql_file in plan_sql_files:
        if sql_file not in tasks_text:
            errors.append(f"{sql_file} is referenced in plan.md but not in tasks.md")

    if not has_any_heading(plan_text, KNOWLEDGE_IMPACT_HEADINGS):
        errors.append("plan.md is missing knowledge impact section")
    if not has_any_heading(tasks_text, APPROVAL_GATE_HEADINGS):
        errors.append("tasks.md is missing implementation approval gate section")
    if re.search(r"Knowledge Sync Needed\s*:\s*yes", plan_text, re.IGNORECASE):
        has_knowledge_task = (
            "Knowledge" in tasks_text
            or "知识" in tasks_text
            or "truth-source" in tasks_text
            or ".specify/memory/" in tasks_text
            or "fons4ai-knowledge-summary" in tasks_text
        )
        if not has_knowledge_task:
            errors.append("plan.md declares Knowledge Sync Needed: yes but tasks.md has no knowledge sync task")

    if S2_RE.search(all_text):
        if not has_any_heading(plan_text, RISK_ROLLBACK_HEADINGS):
            errors.append("S2 plan.md is missing risk and rollback section")
        for display_name, headings in PLAN_REQUIRED_HEADING_GROUPS + PLAN_MODERN_HEADING_GROUPS:
            if has_any_heading(plan_text, headings) and not has_section_content(plan_text, headings):
                errors.append(f"S2 plan.md section '{display_name}' has no content")
        has_s2_quality_gate = has_any_heading(tasks_text, S2_QUALITY_GATE_HEADINGS)
        if not has_s2_quality_gate and not RISK_CONTROL_RE.search(tasks_text):
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
        for label in ("Files:", "Verification:", "Quality:", "Done:"):
            if label not in block:
                errors.append(f"{task_id} is missing '{label}'")
        errors.extend(validate_quality_domain_check(task_id, block, "tasks.md"))

    return errors, warnings


def validate_change_file(change_file: Path) -> list[str]:
    errors: list[str] = []
    if not change_file.exists():
        return [f"Missing change artifact: {change_file}"]

    text = read(change_file)
    errors.extend(validate_required_heading_groups(text, CHANGE_REQUIRED_HEADING_GROUPS, str(change_file)))
    errors.extend(validate_clarification_gate(text, str(change_file), CHANGE_CLARIFICATION_GATE_HEADINGS))

    if not AC_RE.search(text):
        errors.append(f"{change_file} contains no AC-### mapping")
    if not TASK_RE.search(text):
        errors.append(f"{change_file} contains no incremental checklist tasks")

    for task_id, block in task_blocks(text):
        for label in ("Files:", "Verification:", "Quality:", "Done:"):
            if label not in block:
                errors.append(f"{task_id} in {change_file} is missing '{label}'")
        errors.extend(validate_quality_domain_check(task_id, block, str(change_file)))

    ddl_action = (
        re.search(r"SQL DDL action\s*:\s*(add|update|rename)", text, re.IGNORECASE)
        or re.search(r"SQL\s*DDL\s*动作\s*[：:]\s*(新增|更新|重命名)", text)
    )
    if ddl_action:
        sql_files = sorted(set(SQL_PATH_RE.findall(text)))
        if not sql_files:
            errors.append(f"{change_file} declares SQL DDL action but names no .specify/sql/**/*.sql file")
        if not re.search(r"(Sync\s+DDL|同步\s*DDL)", text, re.IGNORECASE):
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
    parser.add_argument("--strict", action="store_true", help="Fail modern SDD section omissions instead of warning")
    args = parser.parse_args()

    selected = [value for value in (args.feature_dir, args.change_file, args.bugfix_report) if value]
    if len(selected) != 1:
        print("ERROR: provide exactly one of --feature-dir, --change-file, or --bugfix-report", file=sys.stderr)
        return 2

    if args.feature_dir:
        target = Path(args.feature_dir).resolve()
        errors, warnings = validate(target, strict=args.strict)
        success = f"OK: {target} SDD artifacts are valid"
    elif args.change_file:
        target = Path(args.change_file).resolve()
        errors = validate_change_file(target)
        warnings = []
        success = f"OK: {target} SDD change artifact is valid"
    else:
        target = Path(args.bugfix_report).resolve()
        errors = validate_bugfix_report(target)
        warnings = []
        success = f"OK: {target} bugfix report is valid"

    for warning in warnings:
        print(f"WARN: {warning}", file=sys.stderr)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(success)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
