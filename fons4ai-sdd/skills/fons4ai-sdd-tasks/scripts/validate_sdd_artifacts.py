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
EXECUTABLE_DDL_PATH_RE = re.compile(
    r"\.?[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*\.sql",
    re.IGNORECASE,
)
S2_RE = re.compile(r"(SDD\s*Level|SDD\s*等级)\s*[:：]\s*`?S2`?", re.IGNORECASE)

APPROVAL_GATE_HEADINGS = ("## 实现确认门禁", "## Implementation Approval Gate")
DOCUMENT_STATUS_RE = re.compile(r"(文档状态|Document Status)\s*[:：]\s*([^\n\r]+)", re.IGNORECASE)
CLARIFICATION_STATUS_RE = re.compile(r"(澄清状态|Clarification Status)\s*[:：]\s*([^\n\r]+)", re.IGNORECASE)
BLOCKING_ARTIFACT_RE = re.compile(r"草案-待确认|草案-含待确认|阻塞|blocking|draft", re.IGNORECASE)
SPEC_REQUIRED_HEADING_GROUPS = (
    ("背景与目标", ("## 背景与目标", "## 背景与问题")),
    ("业务范围", ("## 业务范围", "## 范围")),
    ("角色与业务场景", ("## 角色与业务场景", "## 用户与场景")),
    ("业务流程", ("## 业务流程", "## 流程概览")),
    ("业务规则", ("## 业务规则", "## 关键业务规则与约束")),
    ("功能需求", ("## 功能需求", "## 需求概要", "## Functional Overview")),
    ("业务数据说明", ("## 业务数据说明", "## 关键数据或领域对象")),
    ("业务影响", ("## 业务影响", "## 影响面概览", "## Impact Overview")),
    ("验收标准", ("## 验收标准", "## Acceptance Criteria")),
    ("非功能要求", ("## 非功能要求", "## 非功能需求")),
    ("风险、假设与待确认事项", ("## 风险、假设与待确认事项", "## 风险概览", "## 假设", "## 待确认问题")),
    ("版本修订记录", ("## 版本修订记录", "## Revision History")),
)
PLAN_REQUIRED_HEADING_GROUPS = (
    ("设计目标与范围", ("## 设计目标与范围", "## 设计摘要")),
    ("总体架构设计", ("## 总体架构设计", "## 架构设计")),
    ("核心业务规则与策略落地", ("## 核心业务规则与策略落地", "## 关键业务规则与策略设计")),
    ("核心业务场景实现", ("## 核心业务场景实现", "### 核心业务方案落地")),
    ("数据流设计", ("## 数据流设计", "## 数据流")),
    ("领域建模决策", ("## 领域建模决策",)),
    ("关键规则代码片段", ("## 关键规则代码片段", "## Key Rule Code Sketches")),
    ("状态流转设计", ("## 状态流转设计", "## State Transition Design")),
    ("接口与契约设计", ("## 接口与契约设计", "## API 与契约细节", "## API and Contract Details")),
    ("数据模型与 ER 设计", ("## 数据模型与 ER 设计", "## 数据结构变更", "## Data Structure Changes")),
    ("事务与一致性", ("## 事务与一致性", "## Transaction and Consistency")),
    ("异常处理与日志", ("## 异常处理与日志", "## 错误与异常处理")),
    ("工具包与依赖决策", ("## 工具包与依赖决策",)),
    ("迁移、兼容与回滚", ("## 迁移、兼容与回滚", "## 迁移与回滚细节")),
    ("验证策略", ("## 验证策略", "## Verification Strategy")),
    ("AC 映射", ("## AC 映射",)),
    ("知识同步清单", ("## 知识同步清单", "## 知识同步影响", "## Knowledge Impact")),
    ("风险与待确认事项", ("## 风险与待确认事项", "## 风险与回滚", "## Risk and Rollback")),
)
KNOWLEDGE_IMPACT_HEADINGS = ("## 知识同步清单", "## 知识同步影响", "## Knowledge Impact")
RISK_ROLLBACK_HEADINGS = ("## 风险与待确认事项", "## 风险与回滚", "## Risk and Rollback")
S2_QUALITY_GATE_HEADINGS = ("## S2 质量门禁", "## S2 Quality Gates")
CHANGE_REQUIRED_HEADING_GROUPS = (
    ("影响分析", ("## 影响分析", "## Impact Analysis")),
    ("知识同步清单", ("### 知识同步清单", "### 知识同步影响", "### Knowledge Impact")),
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


def validate_artifact_readiness(text: str, artifact_name: str) -> list[str]:
    errors: list[str] = []
    statuses = [match.group(2).strip() for match in DOCUMENT_STATUS_RE.finditer(text)]
    legacy_statuses = [match.group(2).strip() for match in CLARIFICATION_STATUS_RE.finditer(text)]
    if any(BLOCKING_ARTIFACT_RE.search(status) for status in statuses + legacy_statuses):
        errors.append(f"{artifact_name} is a draft or has unresolved clarification and cannot enter downstream planning")
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


def declares_existing_table_change_with_baseline(text: str) -> bool:
    return bool(
        re.search(r"Existing\s+table.*baseline\s+DDL\s*:\s*(yes|confirmed)", text, re.IGNORECASE)
        or re.search(r"存量表原始\s*DDL\s*[：:]\s*已存在", text)
        or re.search(r"是否为存量表结构变更\s*[：:]\s*是[，,]?\s*原始\s*DDL\s*已存在", text)
    )


def executable_ddl_paths(text: str) -> list[str]:
    return sorted(
        {
            path
            for path in EXECUTABLE_DDL_PATH_RE.findall(text)
            if not path.startswith(".specify/sql/")
        }
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
    errors.extend(validate_artifact_readiness(spec_text, "spec.md"))
    errors.extend(validate_req_ac_mapping(spec_text))
    errors.extend(validate_required_heading_groups(plan_text, PLAN_REQUIRED_HEADING_GROUPS, "plan.md"))

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

    if declares_existing_table_change_with_baseline(plan_text):
        executable_ddl_files = executable_ddl_paths(plan_text)
        if not executable_ddl_files:
            errors.append("plan.md declares an existing-table change with baseline DDL but names no executable change DDL file")
        for ddl_file in executable_ddl_files:
            if ddl_file not in tasks_text:
                errors.append(f"{ddl_file} is referenced as executable change DDL in plan.md but not in tasks.md")
        if not re.search(r"(执行型变更\s*DDL|Executable\s+change\s+DDL|ALTER\s+TABLE)", tasks_text, re.IGNORECASE):
            errors.append("tasks.md has no executable change DDL task for the existing-table structural change")

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
        for display_name, headings in PLAN_REQUIRED_HEADING_GROUPS:
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
    errors.extend(validate_artifact_readiness(text, str(change_file)))

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
    if declares_existing_table_change_with_baseline(text):
        executable_ddl_files = executable_ddl_paths(text)
        if not executable_ddl_files:
            errors.append(f"{change_file} declares an existing-table change with baseline DDL but names no executable change DDL file")
        if not re.search(r"(执行型变更\s*DDL|Executable\s+change\s+DDL|ALTER\s+TABLE)", text, re.IGNORECASE):
            errors.append(f"{change_file} has no executable change DDL task for the existing-table structural change")

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
