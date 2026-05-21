#!/usr/bin/env python3
"""Validate Fons4AI memory knowledge documents."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


BAD_TEXT_PATTERNS = tuple(s.encode("utf-8").decode("unicode_escape") for s in (r"\ufffd", r"\u9225", r"\u9239", r"\u93ba\u3126", r"\u5bf0\u5477", r"\u5bb8\u8336", r"\u93c2\u56e8", r"\u740c\u3126"))
REQUIRED_FILES = ("business-architecture.md", "technical-architecture.md", "data-architecture.md")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def scenario_ids(business: str) -> set[str]:
    ids = set(re.findall(r"\|\s*(BS-\d+)\s*\|", business))
    return ids


def validate(memory_root: Path) -> list[str]:
    errors: list[str] = []
    docs: dict[str, str] = {}
    for name in REQUIRED_FILES:
        path = memory_root / name
        if not path.exists():
            errors.append(f"{path} does not exist")
            continue
        try:
            text = read(path)
        except UnicodeDecodeError as exc:
            errors.append(f"{path} is not valid UTF-8: {exc}")
            continue
        docs[name] = text
        if not text.lstrip().startswith("# "):
            errors.append(f"{path} must start with a level-1 title")
        if text.count("```") % 2 != 0:
            errors.append(f"{path} has unbalanced Markdown fences")
        if not re.search(r"知识来源|变更记录|生成依据", text):
            errors.append(f"{path} must include source/change-record semantics")
        for pattern in BAD_TEXT_PATTERNS:
            if pattern in text:
                errors.append(f"{path} contains mojibake pattern: {pattern}")

    business = docs.get("business-architecture.md", "")
    technical = docs.get("technical-architecture.md", "")
    if business and technical:
        ids = scenario_ids(business)
        if ids and not re.search(r"技术落地|场景.*技术|业务场景到技术", technical):
            errors.append("technical architecture must include scenario-to-technical landing semantics")
        missing = sorted(sid for sid in ids if sid not in technical)
        if missing:
            errors.append("technical architecture missing business scenario id(s): " + ", ".join(missing))

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Fons4AI memory knowledge documents")
    parser.add_argument("--memory-root", default=".specify/memory")
    args = parser.parse_args()
    errors = validate(Path(args.memory_root).resolve())
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("OK: validated memory knowledge documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
