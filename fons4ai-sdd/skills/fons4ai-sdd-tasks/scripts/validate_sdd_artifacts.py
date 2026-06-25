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

APPROVAL_GATE_HEADINGS = ("## 2. 实现确认门禁", "## 12. 实现确认门禁", "## 实现确认门禁", "## Implementation Approval Gate")
DOCUMENT_STATUS_RE = re.compile(r"(文档状态|Document Status)\s*[:：]\s*([^\n\r]+)", re.IGNORECASE)
BLOCKING_ARTIFACT_RE = re.compile(r"草案-待确认|草案-含待确认|阻塞|blocking|draft", re.IGNORECASE)
SPEC_REQUIRED_HEADING_GROUPS = (
    ("一句话说明", ("## 一句话说明",)),
    ("需求澄清摘要", ("## 需求澄清摘要",)),
    ("背景与目标", ("## 背景与目标",)),
    ("需求范围", ("## 需求范围",)),
    ("角色与场景", ("## 角色与场景",)),
    ("需求列表", ("## 需求列表",)),
    ("业务规则", ("## 业务规则",)),
    ("业务流程", ("## 业务流程",)),
    ("业务数据口径", ("## 业务数据口径",)),
    ("影响说明", ("## 影响说明",)),
    ("验收标准", ("## 验收标准",)),
    ("质量要求", ("## 质量要求",)),
    ("风险与待确认", ("## 风险与待确认",)),
    ("版本修订记录", ("## 版本修订记录",)),
)
PLAN_REQUIRED_HEADING_GROUPS = (
    ("设计概要", ("## 1. 设计概要", "## 设计概要")),
    ("架构与调用链路", ("## 2. 架构与调用链路", "## 架构与调用链路")),
    ("API / RPC / 消息契约设计", ("## 3. API / RPC / 消息契约设计", "## API / RPC / 消息契约设计")),
    ("数据模型与 DDL 影响", ("## 4. 数据模型与 DDL 影响", "## 数据模型与 DDL 影响")),
    ("核心逻辑设计", ("## 5. 核心逻辑设计", "## 核心逻辑设计")),
    ("领域建模与业务规则落地", ("## 6. 领域建模与业务规则落地", "## 领域建模与业务规则落地")),
    ("状态流转设计", ("## 7. 状态流转设计", "## 状态流转设计")),
    ("异常、安全、事务与性能", ("## 8. 异常、安全、事务与性能", "## 异常、安全、事务与性能")),
    ("技术决策", ("## 9. 技术决策", "## 技术决策")),
    ("验证策略、AC 映射与风险", ("## 10. 验证策略、AC 映射与风险", "## 验证策略、AC 映射与风险")),
)
KNOWLEDGE_IMPACT_HEADINGS = ("### 10.4 知识同步影响", "## 知识同步清单", "## 知识同步影响", "## Knowledge Impact")
RISK_ROLLBACK_HEADINGS = ("### 10.3 风险与回滚", "## 10. 验证策略、AC 映射与风险", "## 风险与回滚", "## Risk and Rollback")
S2_QUALITY_GATE_HEADINGS = ("## S2 质量门禁", "## S2 Quality Gates")
DATA_IMPACT_HEADINGS = ("### 数据影响判断", "## 数据影响判断", "### Data Impact")
FIELD_MAPPING_HEADINGS = ("### 4.2 字段映射契约", "### 字段映射契约", "## 字段映射契约", "### Field Mapping Contract")
DATA_FLOW_HEADINGS = ("### 4.3 数据流设计", "### 数据流设计", "## 数据流设计", "### Data Flow Design")
DATA_SECURITY_HEADINGS = ("### 4.4 数据安全与合规设计", "### 数据安全与合规设计", "## 数据安全与合规设计", "### Data Security")
CHANGE_REQUIRED_HEADING_GROUPS = (
    ("变更摘要", ("## 1. 变更摘要", "## 变更摘要")),
    ("影响范围", ("## 4. 影响范围", "## 影响范围", "## 影响分析", "## Impact Analysis")),
    ("需求与 AC 变化", ("## 5. 需求与 AC 变化", "## 需求与 AC 变化", "### 需求影响")),
    ("技术设计影响", ("## 6. 技术设计影响", "## 技术设计影响", "### 设计影响")),
    ("数据结构与 DDL 影响", ("## 7. 数据结构与 DDL 影响", "## 数据结构与 DDL 影响")),
    ("回归与回滚", ("## 8. 回归与回滚", "## 回归与回滚", "## Regression and Rollback")),
    ("长期知识影响", ("## 9. 长期知识影响", "## 长期知识影响", "### 知识同步清单", "### 知识同步影响", "### Knowledge Impact")),
    ("文档更新", ("## 10. 文档更新", "## 文档更新")),
    ("实现确认门禁", APPROVAL_GATE_HEADINGS),
    ("增量任务", ("## 11. 增量任务", "## 增量任务", "## Incremental Tasks")),
)

CHANGE_TYPE_RE = re.compile(r"(变更类型|Change Type)\s*[:：]\s*(微调|扩展|重构|数据结构变更|契约变更|纯文档修正|tweak|extension|refactor)", re.IGNORECASE)

DOMAIN_QUALITY_RE = re.compile(r"DDD|domain|领域|充血|贫血|业务规则|应用层", re.IGNORECASE)
KNOWLEDGE_OR_DDL_TASK_RE = re.compile(
    r"\.specify/|truth-source|Knowledge|knowledge|知识|DDL|SQL|数据文档|数据架构|配置与资源文档",
    re.IGNORECASE,
)
RISK_CONTROL_RE = re.compile(
    r"rollback|compatib|regression|permission|security|observability|migration|risk|checklist|"
    r"回滚|兼容|回归|权限|安全|观测|迁移|风险|检查|门禁",
    re.IGNORECASE,
)
DATA_RISK_RE = re.compile(
    r"字段映射|数据流|外部数据|入库|出库|同步|对账|报表|迁移|金额|日期|状态|流水号|客户标识|敏感数据|脱敏|加密|审计|保留期限|"
    r"field\s+mapping|data\s+flow|import|export|sync|reconciliation|report|migration|amount|date|status|serial|sensitive",
    re.IGNORECASE,
)
DATA_VERIFICATION_RE = re.compile(
    r"字段映射|数据口径|样例数据|目标数据|金额单位|日期格式|状态口径|流水号|客户标识|敏感数据|日志脱敏|数据安全|合规|"
    r"field\s+mapping|sample\s+data|target\s+data|data\s+security",
    re.IGNORECASE,
)


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise
    except UnicodeDecodeError:
        return path.read_text()


def requirement_artifact_paths(feature_dir: Path) -> list[Path]:
    return sorted(feature_dir.glob("*-需求说明书.md"))


def technical_design_artifact_paths(feature_dir: Path) -> list[Path]:
    return sorted(feature_dir.glob("*-技术设计说明书.md"))


def task_planning_artifact_paths(feature_dir: Path) -> list[Path]:
    return sorted(feature_dir.glob("*-任务规划.md"))


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
            errors.append(f"{req_id} has no AC mapping in requirement artifact")
    return errors


def validate_artifact_readiness(text: str, artifact_name: str) -> list[str]:
    errors: list[str] = []
    statuses = [match.group(2).strip() for match in DOCUMENT_STATUS_RE.finditer(text)]
    if any(BLOCKING_ARTIFACT_RE.search(status) for status in statuses):
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
    requirement_files = requirement_artifact_paths(feature_dir)
    if not requirement_files:
        errors.append(f"Missing required artifact: {feature_dir / '<功能中文名>-需求说明书.md'}")
        spec = feature_dir / "<功能中文名>-需求说明书.md"
    elif len(requirement_files) > 1:
        errors.append(f"Multiple requirement artifacts found in {feature_dir}; keep exactly one *-需求说明书.md")
        spec = requirement_files[0]
    else:
        spec = requirement_files[0]
    design_files = technical_design_artifact_paths(feature_dir)
    if not design_files:
        errors.append(f"Missing required artifact: {feature_dir / '<功能中文名>-技术设计说明书.md'}")
        design = feature_dir / "<功能中文名>-技术设计说明书.md"
    elif len(design_files) > 1:
        errors.append(f"Multiple technical design artifacts found in {feature_dir}; keep exactly one *-技术设计说明书.md")
        design = design_files[0]
    else:
        design = design_files[0]
    task_files = task_planning_artifact_paths(feature_dir)
    if not task_files:
        errors.append(f"Missing required artifact: {feature_dir / '<功能中文名>-任务规划.md'}")
        tasks = feature_dir / "<功能中文名>-任务规划.md"
    elif len(task_files) > 1:
        errors.append(f"Multiple task planning artifacts found in {feature_dir}; keep exactly one *-任务规划.md")
        tasks = task_files[0]
    else:
        tasks = task_files[0]

    if errors:
        return errors, warnings

    spec_text = read(spec)
    design_text = read(design)
    tasks_text = read(tasks)
    all_text = "\n".join((spec_text, design_text, tasks_text))

    ac_ids = sorted(set(AC_RE.findall(spec_text)))
    if not ac_ids:
        errors.append("requirement artifact contains no AC-### acceptance criteria IDs")

    errors.extend(validate_required_heading_groups(spec_text, SPEC_REQUIRED_HEADING_GROUPS, str(spec.name)))
    errors.extend(validate_artifact_readiness(spec_text, str(spec.name)))
    errors.extend(validate_req_ac_mapping(spec_text))
    errors.extend(validate_required_heading_groups(design_text, PLAN_REQUIRED_HEADING_GROUPS, str(design.name)))
    if not has_any_heading(spec_text, DATA_IMPACT_HEADINGS):
        errors.append(f"{spec.name} is missing data impact check section")
    for display_name, headings in (
        ("字段映射契约", FIELD_MAPPING_HEADINGS),
        ("数据流设计", DATA_FLOW_HEADINGS),
        ("数据安全与合规设计", DATA_SECURITY_HEADINGS),
    ):
        if not has_any_heading(design_text, headings):
            errors.append(f"{design.name} is missing data design section '{display_name}'")

    for ac_id in ac_ids:
        if ac_id not in design_text:
            errors.append(f"{ac_id} is not referenced in {design.name}")
        if ac_id not in tasks_text:
            errors.append(f"{ac_id} is not referenced in {tasks.name}")

    plan_sql_files = sorted(set(SQL_PATH_RE.findall(design_text)))
    if plan_declares_sql_sync(design_text) and not plan_sql_files:
        errors.append(f"{design.name} declares DDL sync but names no .specify/sql/**/*.sql file")
    for sql_file in plan_sql_files:
        if sql_file not in tasks_text:
            errors.append(f"{sql_file} is referenced in {design.name} but not in {tasks.name}")

    if declares_existing_table_change_with_baseline(design_text):
        executable_ddl_files = executable_ddl_paths(design_text)
        if not executable_ddl_files:
            errors.append(f"{design.name} declares an existing-table change with baseline DDL but names no executable change DDL file")
        for ddl_file in executable_ddl_files:
            if ddl_file not in tasks_text:
                errors.append(f"{ddl_file} is referenced as executable change DDL in {design.name} but not in {tasks.name}")
        if not re.search(r"(执行型变更\s*DDL|Executable\s+change\s+DDL|ALTER\s+TABLE)", tasks_text, re.IGNORECASE):
            errors.append(f"{tasks.name} has no executable change DDL task for the existing-table structural change")

    if DATA_RISK_RE.search(design_text) and not DATA_VERIFICATION_RE.search(tasks_text):
        errors.append(f"{tasks.name} has no data verification task for triggered data design/governance risk")

    if not has_any_heading(design_text, KNOWLEDGE_IMPACT_HEADINGS):
        errors.append(f"{design.name} is missing knowledge impact section")
    if not has_any_heading(tasks_text, APPROVAL_GATE_HEADINGS):
        errors.append(f"{tasks.name} is missing implementation approval gate section")

    if S2_RE.search(all_text):
        if not has_any_heading(design_text, RISK_ROLLBACK_HEADINGS):
            errors.append(f"S2 {design.name} is missing risk and rollback section")
        for display_name, headings in PLAN_REQUIRED_HEADING_GROUPS:
            if has_any_heading(design_text, headings) and not has_section_content(design_text, headings):
                errors.append(f"S2 {design.name} section '{display_name}' has no content")
        has_s2_quality_gate = has_any_heading(tasks_text, S2_QUALITY_GATE_HEADINGS)
        if not has_s2_quality_gate and not RISK_CONTROL_RE.search(tasks_text):
            errors.append(f"S2 {tasks.name} must include S2 quality gates or explicit risk-control tasks")

    blocks = task_blocks(tasks_text)
    if not blocks:
        errors.append(f"{tasks.name} contains no checklist tasks in '- [ ] T001' format")

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
        errors.extend(validate_quality_domain_check(task_id, block, str(tasks.name)))

    return errors, warnings


def validate_change_file(change_file: Path) -> list[str]:
    errors: list[str] = []
    if not change_file.exists():
        return [f"Missing change artifact: {change_file}"]

    text = read(change_file)
    errors.extend(validate_required_heading_groups(text, CHANGE_REQUIRED_HEADING_GROUPS, str(change_file)))
    errors.extend(validate_artifact_readiness(text, str(change_file)))

    if not CHANGE_TYPE_RE.search(text):
        errors.append(f"{change_file} is missing change type field")
    if BLOCKING_ARTIFACT_RE.search(text) and TASK_RE.search(text):
        errors.append(f"{change_file} is a draft or blocked CR but contains executable incremental tasks")
    if re.search(r"^- \[[ xX]\] T\d{3}.*(知识同步|知识汇总)", text, re.MULTILINE):
        errors.append(f"{change_file} must not create knowledge-sync or knowledge-summary handoff tasks")

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
        if not re.search(r"(执行型\s*DDL|执行型变更\s*DDL|Executable\s+change\s+DDL|ALTER\s+TABLE)", text, re.IGNORECASE):
            errors.append(f"{change_file} declares SQL DDL action but has no executable DDL draft task")
        if not re.search(r"(确认\s*DDL\s*执行状态|DDL\s*执行确认|read-only verification|只读.*验证)", text, re.IGNORECASE):
            errors.append(f"{change_file} declares SQL DDL action but has no DDL execution-confirmation task")
        if not re.search(r"(同步\s*SQL\s*当前结构快照|SQL\s*当前结构快照|\\.specify/sql/)", text, re.IGNORECASE):
            errors.append(f"{change_file} declares SQL DDL action but has no SQL current-structure snapshot task")
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
    parser.add_argument("--feature-dir", help="Path to spec/features/<yyyymmdd> containing *-任务规划.md")
    parser.add_argument("--change-file", help="Path to spec/features/<yyyymmdd>/changes/CR-xxx.md")
    parser.add_argument("--bugfix-report", help="Path to spec/bugfixes/<yyyymmdd>/<bug中文名>-BUG修复报告.md")
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
