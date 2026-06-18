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
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass
class Hit:
    path: Path
    score: int = 0
    terms: set[str] = field(default_factory=set)
    lines: list[str] = field(default_factory=list)
    reason: str = ""


def read_text(path: Path) -> str:
    data = path.read_bytes()[:MAX_BYTES]
    for encoding in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
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


def normalized(path: Path) -> str:
    return str(path).replace("\\", "/")


def classify(path: Path) -> str:
    text = normalized(path)
    if text.endswith("/.specify/memory/index.md"):
        return "project-memory"
    if "/.specify/memory/domains/" in text and "/cards/" in text:
        return "knowledge-card"
    if "/.specify/memory/domains/" in text:
        return "domain-memory"
    if "/.specify/memory/" in text:
        return "project-memory"
    if "/.specify/sql/" in text:
        return "sql"
    if "/.specify/rules/" in text:
        return "rules"
    if "/specs/" in text:
        return "specs"
    if "/docs/" in text:
        return "docs"
    return "source"


def priority(path: Path) -> int:
    kind = classify(path)
    text = normalized(path)
    if text.endswith("/.specify/memory/index.md"):
        return 0
    order = {
        "knowledge-card": 1,
        "domain-memory": 2,
        "sql": 3,
        "rules": 3,
        "specs": 3,
        "project-memory": 4,
        "docs": 4,
        "source": 5,
    }
    return order.get(kind, 9)


def match_keywords(text: str, lines: list[str], patterns: list[tuple[str, re.Pattern[str]]]) -> tuple[int, set[str], list[str]]:
    score = 0
    terms: set[str] = set()
    snippets: list[str] = []
    for term, pattern in patterns:
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        score += len(matches)
        terms.add(term)
        for line_no, line in enumerate(lines, start=1):
            if pattern.search(line):
                trimmed = line.strip()
                if trimmed:
                    snippets.append(f"L{line_no}: {trimmed[:160]}")
                if len(snippets) >= 3:
                    break
    return score, terms, snippets


def expand_keywords(keywords: list[str]) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()
    for keyword in keywords:
        term = keyword.strip()
        if not term:
            continue
        candidates = [term]
        if CJK_RE.search(term) and len(term) > 2:
            candidates.extend(term[index : index + 2] for index in range(len(term) - 1))
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                expanded.append(candidate)
    return expanded


def find_hits(root: Path, keywords: list[str], include_source: bool) -> list[Hit]:
    patterns = [(term, re.compile(re.escape(term), re.IGNORECASE)) for term in expand_keywords(keywords)]
    hits: list[Hit] = []
    index_path = root / ".specify" / "memory" / "index.md"
    if index_path.exists():
        hits.append(Hit(path=index_path, score=1, reason="default knowledge entrypoint"))

    seen = {index_path.resolve()} if index_path.exists() else set()
    for path in iter_candidate_files(root, include_source):
        resolved = path.resolve()
        if resolved in seen:
            continue
        text = read_text(path)
        lines = text.splitlines()
        score, terms, snippets = match_keywords(text, lines, patterns)
        if not score:
            continue
        kind = classify(path)
        if kind == "knowledge-card":
            score += 8
        elif kind == "domain-memory":
            score += 5
        elif kind in {"sql", "rules", "specs"}:
            score += 3
        hit = Hit(path=path, score=score, terms=terms, lines=snippets)
        hit.reason = f"matched {kind}"
        hits.append(hit)
    return sorted(hits, key=lambda item: (priority(item.path), -item.score, str(item.path)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Find relevant Fons4AI context files")
    parser.add_argument("keywords", nargs="*", help="Feature, module, API, table, object, error, REQ, AC, or domain keywords")
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
    print("Read order: index.md -> knowledge cards -> domain memory -> SQL/rules/specs -> project memory -> source")
    for hit in hits[: args.max_results]:
        relative = hit.path.resolve().relative_to(root)
        terms = ", ".join(sorted(hit.terms)) if hit.terms else "entrypoint"
        reason = f" reason={hit.reason}" if hit.reason else ""
        print(f"- [{classify(hit.path)}] {relative} score={hit.score} terms={terms}{reason}")
        for line in hit.lines[:3]:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
