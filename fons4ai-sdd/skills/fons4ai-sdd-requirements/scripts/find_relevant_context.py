#!/usr/bin/env python3
"""Suggest targeted Fons4AI context files for a task."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_DIRS = (
    ".specify/memory",
    ".specify/sql",
    ".specify/rules",
    "specs",
    "docs",
)
TEXT_SUFFIXES = {
    ".md",
    ".sql",
    ".yaml",
    ".yml",
    ".json",
    ".xml",
    ".properties",
    ".java",
    ".kt",
    ".py",
    ".ts",
    ".js",
}
MAX_BYTES = 1_000_000


@dataclass
class Hit:
    path: Path
    score: int = 0
    terms: set[str] = field(default_factory=set)
    lines: list[str] = field(default_factory=list)


def read_text(path: Path) -> str:
    data = path.read_bytes()[:MAX_BYTES]
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def iter_candidate_files(root: Path, include_source: bool) -> list[Path]:
    dirs = [root / item for item in DEFAULT_DIRS]
    if include_source:
        dirs.extend(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("."))

    seen: set[Path] = set()
    files: list[Path] = []
    for directory in dirs:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not path.is_file() or path in seen:
                continue
            if path.suffix.lower() in TEXT_SUFFIXES:
                seen.add(path)
                files.append(path)
    return files


def classify(path: Path) -> str:
    text = str(path).replace("\\", "/")
    if "/.specify/memory/" in text:
        return "memory"
    if "/.specify/sql/" in text:
        return "sql"
    if "/.specify/rules/" in text:
        return "rules"
    if "/specs/" in text:
        return "specs"
    if "/docs/" in text:
        return "docs"
    return "source"


def find_hits(root: Path, keywords: list[str], include_source: bool) -> list[Hit]:
    patterns = [(term, re.compile(re.escape(term), re.IGNORECASE)) for term in keywords if term.strip()]
    hits: list[Hit] = []
    for path in iter_candidate_files(root, include_source):
        text = read_text(path)
        hit = Hit(path=path)
        lines = text.splitlines()
        for term, pattern in patterns:
            matches = list(pattern.finditer(text))
            if not matches:
                continue
            hit.score += len(matches)
            hit.terms.add(term)
            for line_no, line in enumerate(lines, start=1):
                if pattern.search(line):
                    trimmed = line.strip()
                    if trimmed:
                        hit.lines.append(f"L{line_no}: {trimmed[:160]}")
                    if len(hit.lines) >= 3:
                        break
        if hit.score:
            hits.append(hit)
    return sorted(hits, key=lambda item: (-item.score, classify(item.path), str(item.path)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Find relevant Fons4AI context files")
    parser.add_argument("keywords", nargs="*", help="Feature, module, API, table, object, error, REQ, or AC keywords")
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--max-results", type=int, default=20, help="Maximum files to print")
    parser.add_argument("--include-source", action="store_true", help="Also scan source-like project files")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not args.keywords:
        print("ERROR: provide at least one keyword")
        return 2

    hits = find_hits(root, args.keywords, args.include_source)
    if not hits:
        print("No relevant context files found.")
        return 0

    print("Recommended context files:")
    for hit in hits[: args.max_results]:
        relative = hit.path.resolve().relative_to(root)
        terms = ", ".join(sorted(hit.terms))
        print(f"- [{classify(hit.path)}] {relative} score={hit.score} terms={terms}")
        for line in hit.lines[:3]:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
