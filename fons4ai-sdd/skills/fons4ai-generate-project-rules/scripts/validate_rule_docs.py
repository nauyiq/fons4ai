#!/usr/bin/env python3
"""Validate Fons4AI agent running rule document."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REQUIRED_FILE = "agent运行规则.md"
OPTIONAL_CODE_RULE_FILE = "代码编写规范.md"

REQUIRED_HEADINGS = (
    "# Agent运行规则",
    "## 项目适用范围",
    "## 核心原则",
    "## MCP使用规则",
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
    "猜测业务逻辑",
    "编造接口",
    "编造数据库字段",
    "编造第三方 API",
    "信息不足",
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
    "测试",
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

    for extra_path in sorted(rules_dir.glob("*.md")):
        if extra_path.name not in (REQUIRED_FILE, OPTIONAL_CODE_RULE_FILE):
            errors.append(
                f"Unexpected extra rule file: {extra_path}. Allowed outputs are {REQUIRED_FILE} and {OPTIONAL_CODE_RULE_FILE}."
            )

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
