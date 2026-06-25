#!/usr/bin/env python3
"""Validate Fons4AI agent running rule document."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REQUIRED_FILE = "agent运行规则.md"
OPTIONAL_CODE_RULE_FILE = "代码编写规范.md"
OPTIONAL_SDD_RULE_FILE = "sdd团队协作规范.md"

REQUIRED_HEADINGS = (
    "# Agent运行规则",
    "## 项目适用范围",
    "## 核心原则",
    "## MCP与外部工具使用规则",
    "## 输出要求",
    "## 禁止事项",
    "## 信息不足时的处理",
)

REQUIRED_TERMS = (
    "修改代码前必须先理解需求",
    "优先复用已有代码",
    "优先遵循项目规范",
    "不允许修改与当前需求无关",
    "不允许引入新的技术框架",
    "不允许删除核心业务逻辑",
    "不允许修改数据库结构",
    "MCP",
    "只读",
    "个人本机 MCP",
    "猜测业务逻辑",
    "编造接口",
    "编造数据库字段",
    "编造第三方 API",
    "信息不足",
)

FORBIDDEN_DEFAULT_MCP_TABLE_TERMS = (
    "| 能力场景 | 首选 MCP |",
    "<MCP 名称或待确认>",
    "能力清单填写 `待确认`",
)

CODE_RULE_HEADINGS = (
    "# 代码编写规范",
    "## 基本原则",
    "## 工具类与复用",
    "## 代码风格",
    "## DDD-lite 编码约束",
    "## API 接口设计",
    "## 异常与日志",
    "## 数据访问与事务",
    "## 数据映射与安全",
    "## 测试与验证",
    "## 禁止事项",
)

CODE_RULE_TERMS = (
    "工具类",
    "已有代码",
    "代码风格",
    "API 接口设计",
    "DDD-lite",
    "异常",
    "日志",
    "数据访问",
    "事务",
    "字段映射",
    "金额",
    "敏感数据",
    "日志",
    "测试",
)

SDD_RULE_HEADINGS = (
    "# SDD团队协作规范",
    "## 1. SDD使用准入规则",
    "## 2. SDD产物质量标准",
    "## 3. 角色与评审门禁",
    "## 4. 评审结论",
    "## 5. 标准扩展场景与专业工作流复用",
    "## 6. 标准样例库",
    "## 7. SDD完成定义",
)

SDD_RULE_TERMS = (
    "全新业务功能",
    "已有功能业务变更",
    "S2",
    "需求说明书合格标准",
    "技术设计说明书合格标准",
    "任务规划合格标准",
    "数据影响判断",
    "字段映射",
    "数据流",
    "样例验证",
    "脱敏",
    "主要评审人",
    "有条件通过",
    "专业工作流",
    "标准扩展场景",
    ".specify/examples/sdd",
    "fons4ai-knowledge-summary",
)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text()


def validate_rule_file(path: Path) -> list[str]:
    errors: list[str] = []
    text = read_text(path)

    for heading in REQUIRED_HEADINGS:
        if heading not in text:
            errors.append(f"{path.name} missing heading: {heading}")

    for term in REQUIRED_TERMS:
        if term not in text:
            errors.append(f"{path.name} missing required rule term: {term}")

    for term in FORBIDDEN_DEFAULT_MCP_TABLE_TERMS:
        if term in text:
            errors.append(f"{path.name} contains default MCP table placeholder: {term}")

    if len(text.strip()) < 500:
        errors.append(f"{path.name} is too short for a useful agent rule")

    return errors


def validate_code_rule_file(path: Path) -> list[str]:
    errors: list[str] = []
    text = read_text(path)

    for heading in CODE_RULE_HEADINGS:
        if heading not in text:
            errors.append(f"{path.name} missing heading: {heading}")

    for term in CODE_RULE_TERMS:
        if term not in text:
            errors.append(f"{path.name} missing required coding term: {term}")

    forbidden_knowledge_sections = ("## 项目技术栈", "## 项目事实", "## 模块结构")
    for heading in forbidden_knowledge_sections:
        if heading in text:
            errors.append(f"{path.name} must not duplicate knowledge-base section: {heading}")

    if len(text.strip()) < 800:
        errors.append(f"{path.name} is too short for a useful coding rule")

    return errors


def validate_sdd_rule_file(path: Path) -> list[str]:
    errors: list[str] = []
    text = read_text(path)

    for heading in SDD_RULE_HEADINGS:
        if heading not in text:
            errors.append(f"{path.name} missing heading: {heading}")

    for term in SDD_RULE_TERMS:
        if term not in text:
            errors.append(f"{path.name} missing required SDD governance term: {term}")

    forbidden_sections = ("## 具体需求", "## 具体技术方案", "## 真实项目事实")
    for heading in forbidden_sections:
        if heading in text:
            errors.append(f"{path.name} must not duplicate feature-specific content: {heading}")

    if len(text.strip()) < 1000:
        errors.append(f"{path.name} is too short for a useful SDD team rule")

    return errors


def validate_rules_dir(rules_dir: Path) -> list[str]:
    errors: list[str] = []
    rule_path = rules_dir / REQUIRED_FILE

    if not rule_path.exists():
        errors.append(f"Missing required rule file: {rule_path}")
        return errors

    errors.extend(validate_rule_file(rule_path))

    code_rule_path = rules_dir / OPTIONAL_CODE_RULE_FILE
    if code_rule_path.exists():
        errors.extend(validate_code_rule_file(code_rule_path))

    sdd_rule_path = rules_dir / OPTIONAL_SDD_RULE_FILE
    if sdd_rule_path.exists():
        errors.extend(validate_sdd_rule_file(sdd_rule_path))

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Fons4AI agent running rule document")
    parser.add_argument("--rules-dir", default=".specify/rules", help="Directory containing generated rule markdown files")
    args = parser.parse_args()

    rules_dir = Path(args.rules_dir).resolve()
    if not rules_dir.exists():
        print(f"ERROR: rules directory does not exist: {rules_dir}", file=sys.stderr)
        return 1

    errors = validate_rules_dir(rules_dir)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"OK: {rules_dir / REQUIRED_FILE} is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
