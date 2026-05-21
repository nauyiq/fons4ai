#!/usr/bin/env python3
"""Validate Fons4AI project rule documents."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REQUIRED_FILES = (
    "code-style-rule.md",
    "project-structure-rule.md",
    "features-rule.md",
    "testing-rule.md",
    "data-ddl-rule.md",
)

REQUIRED_HEADINGS = (
    "## 项目事实",
    "## 强制规则",
    "## 推荐规则",
    "## 禁止事项",
    "## 例外机制",
    "## 待确认约定",
    "## 验收检查",
)

REQUIRED_FRONTMATTER_LINES = (
    "> 适用范围：",
    "> 生成依据：",
    "> 规则状态：",
)

LEGACY_THREE_FILE_MARKERS = (
    "三件套",
    "exactly three",
    "只生成 3",
    "仅生成 3",
    "默认只生成 3",
    "code-style-rule.md、project-structure-rule.md、features-rule.md",
)

CODE_STYLE_REQUIRED_TERMS = (
    ("工具包", "utility package usage"),
    ("依赖", "dependency gate"),
    ("可读性", "readability rule"),
)

CODE_STYLE_COMPLEXITY_TERMS = ("复杂度", "重复代码")

DDD_LITE_REQUIRED_TERMS = (
    "DDD-lite",
    "充血模型",
    "领域行为",
    "应用层",
)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text()


def validate_file(path: Path) -> list[str]:
    errors: list[str] = []
    text = read_text(path)

    for line in REQUIRED_FRONTMATTER_LINES:
        if line not in text:
            errors.append(f"{path.name} missing header line: {line}")

    for heading in REQUIRED_HEADINGS:
        if heading not in text:
            errors.append(f"{path.name} missing section: {heading}")

    lower_text = text.lower()
    for marker in LEGACY_THREE_FILE_MARKERS:
        if marker.lower() in lower_text:
            errors.append(f"{path.name} contains legacy three-file marker: {marker}")

    if len(text.strip()) < 600:
        errors.append(f"{path.name} is too short for architect-grade rules")

    return errors


def validate_rules_dir(rules_dir: Path) -> list[str]:
    errors: list[str] = []

    for file_name in REQUIRED_FILES:
        path = rules_dir / file_name
        if not path.exists():
            errors.append(f"Missing required rule file: {path}")
            continue
        errors.extend(validate_file(path))

    data_rule = rules_dir / "data-ddl-rule.md"
    if data_rule.exists():
        data_text = read_text(data_rule)
        if ".specify/sql/<database_or_service>/<business_model>.sql" not in data_text:
            errors.append("data-ddl-rule.md missing database-scoped DDL path rule")
        if ".specify/sql/pending/<business_model>.sql" not in data_text:
            errors.append("data-ddl-rule.md missing pending DDL path rule for unknown database/service")
        if "跨库" not in data_text and "不同数据库" not in data_text:
            errors.append("data-ddl-rule.md missing cross-database split rule")
        if "没有迁移脚本" not in data_text and "无迁移脚本" not in data_text:
            errors.append("data-ddl-rule.md missing no-migration-script SQL generation rule")
        if "推断" not in data_text or "待确认" not in data_text:
            errors.append("data-ddl-rule.md missing inferred/pending evidence state rule")

    code_style_rule = rules_dir / "code-style-rule.md"
    if code_style_rule.exists():
        code_text = read_text(code_style_rule)
        for term, label in CODE_STYLE_REQUIRED_TERMS:
            if term not in code_text:
                errors.append(f"code-style-rule.md missing {label}: {term}")
        if not any(term in code_text for term in CODE_STYLE_COMPLEXITY_TERMS):
            errors.append("code-style-rule.md missing complexity or duplicate-code rule")

    ddd_text = "\n".join(
        read_text(rules_dir / file_name)
        for file_name in ("code-style-rule.md", "project-structure-rule.md", "features-rule.md")
        if (rules_dir / file_name).exists()
    )
    for term in DDD_LITE_REQUIRED_TERMS:
        if term not in ddd_text:
            errors.append(f"rule documents missing DDD-lite term: {term}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Fons4AI project rule documents")
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

    print(f"OK: {rules_dir} rule documents are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
