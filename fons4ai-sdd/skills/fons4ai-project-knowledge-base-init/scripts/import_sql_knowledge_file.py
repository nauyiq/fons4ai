#!/usr/bin/env python3
"""Import existing SQL DDL into .specify/sql without provenance metadata."""

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
    "Status",
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


def header_value(text: str, name: str) -> str | None:
    match = re.search(rf"^\s*--\s*{re.escape(name)}\s*:\s*(.+?)\s*$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def has_contract_header(text: str) -> bool:
    return all(header_value(text, header) for header in REQUIRED_HEADERS)


def strip_provenance_metadata(text: str) -> str:
    """Remove legacy Fons4AI provenance metadata from SQL artifact content."""
    lines = text.splitlines()
    output: list[str] = []
    skipping_evidence = False
    for line in lines:
        if re.match(
            r"^\s*--\s*(?:Source|DDL Source|Origin|Original File|Migration Script|"
            r"Repository SQL File|DDL Evidence|Evidence|Query|Tool|MCP Tool|MCP Server)\s*:",
            line,
            re.IGNORECASE,
        ):
            skipping_evidence = bool(
                re.match(r"^\s*--\s*(?:DDL\s+)?Evidence\s*:", line, re.IGNORECASE)
            )
            continue
        if skipping_evidence and re.match(r"^\s*--\s*-\s+", line):
            continue
        skipping_evidence = False
        output.append(line)
    return "\n".join(output).strip()


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
    status: str,
) -> str:
    db_value = "待确认" if database == "pending" else database
    today = dt.date.today().isoformat()
    return "\n".join(
        (
            f"-- Database/Service: {db_value}",
            f"-- Business Model: {business_model}",
            f"-- Tables: {', '.join(tables)}",
            f"-- Status: {status}",
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
    parser.add_argument("--repo-root", default=".", help="Accepted for compatibility; not written into SQL output")
    parser.add_argument("--status", default="已确认", choices=sorted(VALID_STATUS))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing target file")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    if not source.exists() or source.suffix.lower() != ".sql":
        print(f"ERROR: source SQL file not found or not .sql: {source}", file=sys.stderr)
        return 1

    text = strip_provenance_metadata(read_text(source))
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

    if has_contract_header(text):
        output = text + "\n"
    else:
        header = build_header(
            database=database,
            business_model=business_model,
            tables=tables,
            status=args.status,
        )
        output = header + text + "\n"

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(output, encoding="utf-8")
    print(f"OK: imported {source} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
