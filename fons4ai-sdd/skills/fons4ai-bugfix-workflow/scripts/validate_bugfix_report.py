#!/usr/bin/env python3
"""Validate minimal Fons4AI bugfix report completeness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REQUIRED_HEADINGS = (
    "## 问题描述",
    "## 复现步骤",
    "## 复现环境",
    "## 根因分析",
    "## 修复方案",
    "## 自动化测试",
    "## 手动验证",
    "## 回归验证",
    "## 知识库同步",
)

REQUIRED_FIELDS = (
    "回滚方案",
    "Knowledge Sync Needed",
    "SQL DDL files",
    "Suggested follow-up",
)


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text()


def validate(report: Path) -> list[str]:
    if not report.exists():
        return [f"Missing bugfix report: {report}"]

    text = read(report)
    errors: list[str] = []
    for heading in REQUIRED_HEADINGS:
        if heading not in text:
            errors.append(f"Missing heading: {heading}")
    for field in REQUIRED_FIELDS:
        if field not in text:
            errors.append(f"Missing field: {field}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Fons4AI bugfix report")
    parser.add_argument("--report", required=True, help="Path to spec/bugfixes/<yyyymmdd>/<bug中文名>-BUG修复报告.md")
    args = parser.parse_args()

    report = Path(args.report).resolve()
    errors = validate(report)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"OK: {report} bugfix report is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
