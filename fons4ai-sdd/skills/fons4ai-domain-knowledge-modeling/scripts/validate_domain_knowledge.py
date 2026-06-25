#!/usr/bin/env python3
"""Validate Fons4AI domain/capability knowledge modeling artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DOMAIN_REQUIRED_SECTIONS = {
    "business": ("## 5. 业务适配矩阵", "## 6. 公共抽象与标准流程判定", "## 8. 状态流转"),
    "technical": ("## 4. 业务适配技术落地矩阵", "## 5. 公共抽象", "## 6. 代表性实现"),
    "data": ("## 2. 业务对象生命周期", "## 3. 数据关系 ER 图", "## 4. 业务适配数据差异", "## 8. 数据安全与合规"),
}

CAPABILITY_REQUIRED_SECTIONS = {
    "ability": ("## 5. 公共抽象与标准能力判定", "## 6. 能力规则"),
    "runtime": ("## 2. 场景运行落地", "## 4. 运行治理"),
    "resource": ("## 2. 配置项", "## 4. 能力适配资源差异"),
}

BUSINESS_ADAPTATION_REQUIRED_SECTIONS = (
    "## 2. 业务流程",
    "## 3. 适配时序图",
    "## 4. 关键业务规则",
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


def find_doc(root: Path, suffix: str) -> Path | None:
    matches = sorted(root.glob(f"*{suffix}.md"))
    return matches[0] if matches else None


def check_sections(path: Path, text: str, sections: tuple[str, ...], errors: list[str]) -> None:
    for section in sections:
        if section not in text:
            errors.append(f"{path} is missing required section: {section}")


def check_optional_markdown(path: Path, errors: list[str]) -> None:
    if path.exists():
        read(path, errors)


def validate_business_adaptations(domain_dir: Path, errors: list[str]) -> None:
    adaptations_dir = domain_dir / "adaptations"
    if not adaptations_dir.exists():
        return
    for path in sorted(adaptations_dir.glob("*.md")):
        text = read(path, errors)
        if not text:
            continue
        check_sections(path, text, BUSINESS_ADAPTATION_REQUIRED_SECTIONS, errors)
        if "sequenceDiagram" not in text and "不适用，原因" not in text:
            errors.append(f"{path} must include a sequenceDiagram or explicit 不适用，原因 in 适配时序图")


def validate_domain(domain_dir: Path) -> list[str]:
    errors: list[str] = []
    docs = {
        "business": find_doc(domain_dir, "业务文档"),
        "technical": find_doc(domain_dir, "技术文档"),
        "data": find_doc(domain_dir, "数据文档"),
    }
    labels = {"business": "业务文档", "technical": "技术文档", "data": "数据文档"}
    for kind, path in docs.items():
        if path is None:
            errors.append(f"{domain_dir} missing *{labels[kind]}.md")
            continue
        text = read(path, errors)
        if text:
            check_sections(path, text, DOMAIN_REQUIRED_SECTIONS[kind], errors)

    matrix = domain_dir / "业务适配矩阵.md"
    if not matrix.exists():
        errors.append(f"{domain_dir} missing 业务适配矩阵.md")
    else:
        read(matrix, errors)

    check_optional_markdown(domain_dir / "evidence-ledger.md", errors)
    validate_business_adaptations(domain_dir, errors)
    return errors


def validate_capability(capability_dir: Path) -> list[str]:
    errors: list[str] = []
    docs = {
        "ability": find_doc(capability_dir, "能力文档"),
        "runtime": find_doc(capability_dir, "运行文档"),
        "resource": find_doc(capability_dir, "配置与资源文档"),
    }
    labels = {"ability": "能力文档", "runtime": "运行文档", "resource": "配置与资源文档"}
    for kind, path in docs.items():
        if path is None:
            errors.append(f"{capability_dir} missing *{labels[kind]}.md")
            continue
        text = read(path, errors)
        if text:
            check_sections(path, text, CAPABILITY_REQUIRED_SECTIONS[kind], errors)

    matrix = capability_dir / "能力适配矩阵.md"
    if not matrix.exists():
        errors.append(f"{capability_dir} missing 能力适配矩阵.md")
    else:
        read(matrix, errors)

    check_optional_markdown(capability_dir / "evidence-ledger.md", errors)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Fons4AI domain/capability knowledge artifacts")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--domain-dir")
    group.add_argument("--capability-dir")
    args = parser.parse_args()

    if args.domain_dir:
        errors = validate_domain(Path(args.domain_dir))
    else:
        errors = validate_capability(Path(args.capability_dir))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("OK: validated knowledge modeling artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
