#!/usr/bin/env python3
"""Validate Fons4AI SQL knowledge files under .specify/sql."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


REQUIRED_HEADERS = ("Database/Service", "Business Model", "Tables", "Status", "Last Generated")
FORBIDDEN_METADATA_HEADERS = (
    "Source",
    "DDL Source",
    "Origin",
    "Original File",
    "Migration Script",
    "Repository SQL File",
    "DDL Evidence",
    "Evidence",
    "Query",
    "Tool",
    "MCP Tool",
    "MCP Server",
)
VALID_STATUS = {"已确认", "推断", "待确认"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CREATE_TABLE_RE = re.compile(r"\bCREATE\s+TABLE\b|TODO:\s*CREATE\s+TABLE", re.IGNORECASE)
BAD_TEXT_PATTERNS = tuple(s.encode("utf-8").decode("unicode_escape") for s in (r"\ufffd", r"\u9225", r"\u9239", r"\u93ba\u3126", r"\u5bf0\u5477", r"\u5bb8\u8336", r"\u93c2\u56e8", r"\u740c\u3126"))
TOOL_METADATA_RE = re.compile(r"^\s*--.*(?:\bMCP\b|\bTOOL\b|mcp__)", re.IGNORECASE | re.MULTILINE)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def header_value(text: str, name: str) -> str | None:
    match = re.search(rf"^\s*--\s*{re.escape(name)}\s*:\s*(.+?)\s*$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def comment_values(text: str) -> list[str]:
    return re.findall(r"COMMENT\s+'((?:[^']|'')*)'", text, flags=re.IGNORECASE)


def create_blocks(text: str) -> list[str]:
    return re.findall(r"CREATE\s+TABLE\s+`?[\w.]+`?\s*\((.*?)\)\s*COMMENT", text, flags=re.IGNORECASE | re.DOTALL)


def validate_sql_file(path: Path, sql_root: Path | None = None, strict_comments: bool = False) -> list[str]:
    errors: list[str] = []
    try:
        text = read_text(path)
    except UnicodeDecodeError as exc:
        return [f"{path} is not valid UTF-8: {exc}"]

    if path.suffix.lower() != ".sql":
        return errors

    relative = path
    if sql_root is not None:
        try:
            relative = path.resolve().relative_to(sql_root.resolve())
        except ValueError:
            relative = path
    parts = relative.parts
    if len(parts) < 2:
        errors.append(f"{path} must be grouped as <database_or_service>/<business_model>.sql")
    if len(parts) > 2 and parts[0] == "pending":
        errors.append(f"{path} should use .specify/sql/pending/<business_model>.sql, not nested pending paths")

    for header in REQUIRED_HEADERS:
        if not header_value(text, header):
            errors.append(f"{path} missing SQL header: -- {header}: <value>")
    for header in FORBIDDEN_METADATA_HEADERS:
        if re.search(rf"^\s*--\s*{re.escape(header)}\s*:", text, re.IGNORECASE | re.MULTILINE):
            errors.append(f"{path} must not contain provenance header: -- {header}:")
    if TOOL_METADATA_RE.search(text):
        errors.append(f"{path} must not contain MCP/Tool metadata comments")

    status = header_value(text, "Status")
    if status and status not in VALID_STATUS:
        errors.append(f"{path} has invalid Status '{status}', expected one of {sorted(VALID_STATUS)}")

    last_generated = header_value(text, "Last Generated")
    if last_generated and not DATE_RE.match(last_generated):
        errors.append(f"{path} Last Generated must use YYYY-MM-DD")

    tables = header_value(text, "Tables") or ""
    if tables in {"<table_1>, <table_2>", "<tables>", "TODO", "待确认"}:
        errors.append(f"{path} Tables header must list known tables or explicit placeholder comments in body")

    if not CREATE_TABLE_RE.search(text):
        errors.append(f"{path} must contain CREATE TABLE or TODO: CREATE TABLE")

    if status in {"推断", "待确认"} and not re.search(r"推断|待确认|TODO", text):
        errors.append(f"{path} has inferred/pending status but no inline 推断/待确认/TODO evidence markers")

    for pattern in BAD_TEXT_PATTERNS:
        if pattern in text:
            errors.append(f"{path} contains mojibake pattern: {pattern}")

    for block in create_blocks(text):
        columns: list[str] = []
        for line in block.splitlines():
            match = re.match(r"\s*`([^`]+)`\s+", line)
            if match:
                col = match.group(1).lower()
                if col in columns:
                    errors.append(f"{path} has duplicate column `{match.group(1)}`")
                columns.append(col)

    if strict_comments:
        for comment in comment_values(text):
            clean = comment.replace("''", "'")
            if len(clean) > 260:
                errors.append(f"{path} has overlong COMMENT ({len(clean)} chars)")
            if any(token in clean for token in ("@link", "@return", "<p>", "</", "/*", "*/")):
                errors.append(f"{path} has raw source markup in COMMENT: {clean[:80]}")
            if any(pattern in clean for pattern in BAD_TEXT_PATTERNS):
                errors.append(f"{path} has mojibake in COMMENT: {clean[:80]}")
            if any(token in clean for token in ("Java字段=", "Java类型=", "来源=", "类型/长度/可空性待确认")):
                errors.append(f"{path} has evidence metadata inside SQL COMMENT: {clean[:80]}")

    return errors


def collect_sql_files(sql_root: Path, explicit_files: list[str]) -> list[Path]:
    if explicit_files:
        return [Path(file_name) for file_name in explicit_files]
    return sorted(sql_root.rglob("*.sql")) if sql_root.exists() else []


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Fons4AI SQL knowledge files")
    parser.add_argument("--sql-root", default=".specify/sql", help="SQL knowledge root directory")
    parser.add_argument("--repo-root", default=".", help="Repository root, reserved for future evidence checks")
    parser.add_argument("--file", action="append", default=[], help="Specific SQL file to validate")
    parser.add_argument("--strict-comments", action="store_true", help="Reject mojibake, source markup, and overlong comments")
    args = parser.parse_args()

    sql_root = Path(args.sql_root).resolve()
    sql_files = collect_sql_files(sql_root, args.file)
    if not sql_files:
        print(f"ERROR: no SQL files found under {sql_root}", file=sys.stderr)
        return 1

    errors: list[str] = []
    for sql_file in sql_files:
        errors.extend(validate_sql_file(sql_file.resolve(), sql_root if not args.file else None, args.strict_comments))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"OK: validated {len(sql_files)} SQL knowledge file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
