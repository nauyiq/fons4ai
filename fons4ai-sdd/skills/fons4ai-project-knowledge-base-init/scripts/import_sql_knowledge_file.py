#!/usr/bin/env python3
"""Import an existing SQL DDL file into .specify/sql with Fons4AI headers."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path


REQUIRED_HEADERS = (
    "Database/Service",
    "Business Model",
    "Tables",
    "Source",
    "Status",
    "Migration Script",
    "Last Generated",
)
VALID_STATUS = {"已确认", "推断", "待确认"}


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def relative(path: Path, root: Path | None) -> str:
    if root is None:
        return str(path).replace("\\", "/")
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def header_value(text: str, name: str) -> str | None:
    match = re.search(rf"^\s*--\s*{re.escape(name)}\s*:\s*(.+?)\s*$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def has_contract_header(text: str) -> bool:
    return all(header_value(text, header) for header in REQUIRED_HEADERS)


def table_names(text: str) -> list[str]:
    names: list[str] = []
    pattern = re.compile(
        r"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"]?([\w.]+)[`\"]?",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        raw = match.group(1)
        names.append(raw.split(".")[-1].strip("`\""))
    return sorted(set(names))


def build_header(
    *,
    database: str,
    business_model: str,
    tables: list[str],
    source: str,
    status: str,
    migration_script: str,
) -> str:
    db_value = "待确认" if database == "pending" else database
    today = dt.date.today().isoformat()
    return "\n".join(
        (
            f"-- Database/Service: {db_value}",
            f"-- Business Model: {business_model}",
            f"-- Tables: {', '.join(tables)}",
            f"-- Source: repository SQL file: {source}",
            f"-- Status: {status}",
            f"-- Migration Script: {migration_script}",
            f"-- Last Generated: {today}",
            "",
            "",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Import a repository SQL DDL file into .specify/sql")
    parser.add_argument("--source", required=True, help="Existing repository SQL DDL file")
    parser.add_argument("--sql-root", default=".specify/sql", help="SQL knowledge root")
    parser.add_argument("--database", required=True, help="Database/service directory, or pending")
    parser.add_argument("--business-model", required=True, help="Business model SQL file name without .sql")
    parser.add_argument("--repo-root", default=".", help="Repository root for relative source evidence")
    parser.add_argument("--status", default="已确认", choices=sorted(VALID_STATUS))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing target file")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    if not source.exists() or source.suffix.lower() != ".sql":
        print(f"ERROR: source SQL file not found or not .sql: {source}", file=sys.stderr)
        return 1

    text = read_text(source).strip()
    tables = table_names(text)
    if not tables:
        print(f"ERROR: source SQL file contains no CREATE TABLE statement: {source}", file=sys.stderr)
        return 1

    sql_root = Path(args.sql_root).resolve()
    database = args.database.strip().lower().replace("\\", "/").strip("/")
    business_model = args.business_model.strip().lower().replace("\\", "/").strip("/")
    if "/" in database or "/" in business_model or not database or not business_model:
        print("ERROR: database and business-model must be single path segments", file=sys.stderr)
        return 1

    target = sql_root / database / f"{business_model}.sql"
    if target.exists() and not args.overwrite:
        print(f"ERROR: target exists; pass --overwrite to replace: {target}", file=sys.stderr)
        return 1

    repo_root = Path(args.repo_root).resolve() if args.repo_root else None
    if has_contract_header(text):
        output = text + "\n"
    else:
        source_ref = relative(source, repo_root)
        header = build_header(
            database=database,
            business_model=business_model,
            tables=tables,
            source=source_ref,
            status=args.status,
            migration_script=source_ref,
        )
        output = header + text + "\n"

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(output, encoding="utf-8")
    print(f"OK: imported {source} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
