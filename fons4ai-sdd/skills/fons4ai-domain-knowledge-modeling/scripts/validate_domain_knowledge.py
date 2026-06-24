#!/usr/bin/env python3
"""Validate Fons4AI domain knowledge modeling artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REQUIRED_SECTIONS = {
    "business": ("## 5. 业务能力变体矩阵", "## 6. 公共抽象与标准流程判定", "## 8. 状态流转"),
    "technical": ("## 4. 能力变体技术落地矩阵", "## 5. 公共抽象", "## 6. 代表性实现"),
    "data": ("## 2. 业务对象生命周期", "## 3. 数据关系 ER 图", "## 4. 业务能力变体数据差异"),
}


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


def find_doc(domain_dir: Path, suffix: str) -> Path | None:
    matches = sorted(domain_dir.glob(f"*{suffix}.md"))
    return matches[0] if matches else None


def check_sections(path: Path, text: str, sections: tuple[str, ...], errors: list[str]) -> None:
    for section in sections:
        if section not in text:
            errors.append(f"{path} is missing required section: {section}")


def validate(domain_dir: Path) -> list[str]:
    errors: list[str] = []
    docs = {
        "business": find_doc(domain_dir, "业务文档"),
        "technical": find_doc(domain_dir, "技术文档"),
        "data": find_doc(domain_dir, "数据文档"),
    }
    for kind, path in docs.items():
        if path is None:
            errors.append(f"{domain_dir} missing *{ {'business':'业务文档','technical':'技术文档','data':'数据文档'}[kind] }.md")
            continue
        text = read(path, errors)
        if text:
            check_sections(path, text, REQUIRED_SECTIONS[kind], errors)
    ledger = domain_dir / "evidence-ledger.md"
    matrix = domain_dir / "variant-matrix.md"
    if ledger.exists():
        read(ledger, errors)
    if matrix.exists():
        read(matrix, errors)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Fons4AI domain knowledge modeling artifacts")
    parser.add_argument("--domain-dir", required=True)
    args = parser.parse_args()
    errors = validate(Path(args.domain_dir))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("OK: validated domain knowledge artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
