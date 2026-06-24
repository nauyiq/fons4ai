#!/usr/bin/env python3
"""Validate Fons4AI project knowledge bootstrap artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REQUIRED_INDEX_FILE = "index.md"
BUSINESS_ROOT_FILES = ("项目业务架构文档.md", "项目技术架构文档.md", "项目数据架构文档.md")
TECH_ROOT_FILES = ("项目技术能力架构文档.md", "项目运行架构文档.md", "项目配置与资源架构文档.md")
INDEX_SECTIONS = (
    "## 3. 领域索引",
    "## 4. 核心业务能力索引",
    "## 5. 高风险领域与建模优先级",
    "## 7. 已完成确认项",
)
TECH_INDEX_SECTIONS = (
    "## 3. 技术能力域索引",
    "## 4. 核心平台能力索引",
    "## 5. 高风险技术能力与建模优先级",
    "## 7. 已完成确认项",
)
DATA_SECTIONS = (
    "## 2. 核心业务对象",
    "## 3. 业务对象生命周期",
    "## 4. 项目级 ER 关系",
    "## 5. 跨领域数据流",
    "## 6. 数据所有权与事实源",
    "## 7. 一致性与补偿风险",
)
CONFIG_RESOURCE_SECTIONS = (
    "## 2. 核心配置对象",
    "## 3. 运行时资源对象",
    "## 4. 配置优先级",
    "## 5. 配置所有权与变更风险",
    "## 6. 治理规则",
)


def read(path: Path, errors: list[str]) -> str:
    if not path.exists():
        errors.append(f"{path} does not exist")
        return ""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        errors.append(f"{path} is not valid UTF-8: {exc}")
        return ""
    if not text.lstrip().startswith("# "):
        errors.append(f"{path} must start with a level-1 title")
    if text.count("```") % 2:
        errors.append(f"{path} has unbalanced Markdown fences")
    return text


def validate(memory_root: Path) -> list[str]:
    errors: list[str] = []
    index = read(memory_root / REQUIRED_INDEX_FILE, errors)

    business_exists = all((memory_root / name).exists() for name in BUSINESS_ROOT_FILES)
    tech_exists = all((memory_root / name).exists() for name in TECH_ROOT_FILES)
    if not business_exists and not tech_exists:
        errors.append(
            f"{memory_root} must contain either business root files {BUSINESS_ROOT_FILES} "
            f"or technical root files {TECH_ROOT_FILES}"
        )

    if business_exists:
        texts = {name: read(memory_root / name, errors) for name in BUSINESS_ROOT_FILES}
        for section in INDEX_SECTIONS:
            if index and section not in index:
                errors.append(f"{memory_root / REQUIRED_INDEX_FILE} is missing required section: {section}")
        data_text = texts.get("项目数据架构文档.md", "")
        for section in DATA_SECTIONS:
            if data_text and section not in data_text:
                errors.append(f"{memory_root / '项目数据架构文档.md'} is missing required section: {section}")

    if tech_exists:
        texts = {name: read(memory_root / name, errors) for name in TECH_ROOT_FILES}
        for section in TECH_INDEX_SECTIONS:
            if index and section not in index:
                errors.append(f"{memory_root / REQUIRED_INDEX_FILE} is missing required section: {section}")
        config_text = texts.get("项目配置与资源架构文档.md", "")
        for section in CONFIG_RESOURCE_SECTIONS:
            if config_text and section not in config_text:
                errors.append(f"{memory_root / '项目配置与资源架构文档.md'} is missing required section: {section}")

    bootstrap_dir = memory_root / "bootstrap"
    if bootstrap_dir.exists():
        for report in sorted(bootstrap_dir.glob("*基线分析报告.md")):
            read(report, errors)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Fons4AI knowledge bootstrap artifacts")
    parser.add_argument("--memory-root", default=".specify/memory")
    args = parser.parse_args()
    errors = validate(Path(args.memory_root))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("OK: validated knowledge bootstrap artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
