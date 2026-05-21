#!/usr/bin/env python3
"""Generate generic Fons4AI SQL knowledge files from Java entities and MyBatis metadata."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET


BAD_TEXT_PATTERNS = tuple(s.encode("utf-8").decode("unicode_escape") for s in (r"\ufffd", r"\u9225", r"\u9239", r"\u93ba\u3126", r"\u5bf0\u5477", r"\u5bb8\u8336", r"\u93c2\u56e8", r"\u740c\u3126", r"\u4e36", r"\u9286"))
EXCLUDED_NAME_SUFFIXES = ("Criteria", "Key", "DTO", "Dto", "Request", "Response", "VO", "Vo", "View")
DEFAULT_STATUS = "推断"


@dataclass
class FieldInfo:
    property: str
    column: str
    java_type: str = "String"
    jdbc_type: str = ""
    source: str = "推断"
    note: str = ""
    pk: bool = False


@dataclass
class ModelInfo:
    table: str
    class_name: str
    group: str
    source_files: set[str] = field(default_factory=set)
    fields: list[FieldInfo] = field(default_factory=list)
    status: str = DEFAULT_STATUS


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def snake(name: str) -> str:
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def class_name(fqcn: str) -> str:
    return fqcn.rsplit(".", 1)[-1].split("$", 1)[0]


def clean_note(text: str) -> str:
    cleaned_lines: list[str] = []
    for raw in text.splitlines():
        line = re.sub(r"^\s*/\*\*?", "", raw)
        line = re.sub(r"\*/\s*$", "", line)
        line = re.sub(r"^\s*\*\s?", "", line)
        line = re.sub(r"^\s*//\s?", "", line)
        line = line.strip()
        if line:
            cleaned_lines.append(line)
    text = " ".join(cleaned_lines)
    text = re.sub(r"\{@link[^}]+}", "", text)
    text = re.sub(r"@\w+(?:\s+\S+)?", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"(^|\s)\*\s+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" ;,，。")
    if not text or any(p in text for p in BAD_TEXT_PATTERNS):
        return ""
    if len(text) > 80:
        text = text[:80].rstrip() + "..."
    return text


def should_exclude_class(name: str, has_table: bool = False) -> bool:
    if has_table:
        return False
    return name.endswith(EXCLUDED_NAME_SUFFIXES)


def parse_java_fields(path: Path) -> tuple[str, str | None, list[FieldInfo]]:
    text = read_text(path)
    cls_match = re.search(r"\bclass\s+(\w+)", text)
    cls = cls_match.group(1) if cls_match else path.stem
    table_match = re.search(r"@TableName\s*\(\s*(?:value\s*=\s*)?\"([^\"]+)\"", text)
    table = table_match.group(1) if table_match else None
    fields: list[FieldInfo] = []
    lines = text.splitlines()
    pending_comment: list[str] = []
    skip_next_exist_false = False
    pk_next = False
    column_next: str | None = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("/**") or stripped.startswith("*") or stripped.startswith("//"):
            pending_comment.append(stripped)
            continue
        if "@TableField" in stripped:
            if re.search(r"exist\s*=\s*false", stripped):
                skip_next_exist_false = True
            col = re.search(r"@TableField\s*\(\s*(?:value\s*=\s*)?\"([^\"]+)\"", stripped)
            if col:
                column_next = col.group(1)
            continue
        if "@TableId" in stripped:
            pk_next = True
            col = re.search(r"@TableId\s*\(\s*(?:value\s*=\s*)?\"([^\"]+)\"", stripped)
            if col:
                column_next = col.group(1)
            continue
        m = re.search(r"\bprivate\s+(?:static\s+|final\s+)?([\w<>?, ]+)\s+(\w+)\s*(?:=.*)?;", stripped)
        if not m:
            if stripped and not stripped.startswith("@"):
                pending_comment = []
            continue
        java_type, prop = m.group(1).strip(), m.group(2)
        if prop == "serialVersionUID" or re.search(r"\b(static|final|transient)\b", stripped):
            pending_comment = []
            skip_next_exist_false = False
            pk_next = False
            column_next = None
            continue
        inline = stripped.split("//", 1)[1] if "//" in stripped else ""
        note = clean_note("\n".join(pending_comment + ([inline] if inline else [])))
        pending_comment = []
        if skip_next_exist_false:
            skip_next_exist_false = False
            pk_next = False
            column_next = None
            continue
        fields.append(
            FieldInfo(
                property=prop,
                column=column_next or snake(prop),
                java_type=java_type,
                source="实体字段",
                note=note,
                pk=pk_next,
            )
        )
        pk_next = False
        column_next = None
    return cls, table, fields


def build_java_index(repo_root: Path) -> dict[str, tuple[Path, str | None, list[FieldInfo]]]:
    index: dict[str, tuple[Path, str | None, list[FieldInfo]]] = {}
    for path in repo_root.rglob("*.java"):
        if any(part in {"target", ".settings", ".git"} for part in path.parts):
            continue
        cls, table, fields = parse_java_fields(path)
        if fields:
            index[cls] = (path, table, fields)
    return index


def parse_mapper(path: Path, repo_root: Path, java_index: dict[str, tuple[Path, str | None, list[FieldInfo]]]) -> list[ModelInfo]:
    try:
        root = ET.fromstring(read_text(path))
    except ET.ParseError:
        return []
    text = read_text(path)
    insert_match = re.search(r"\binsert\s+into\s+([`\"\[]?[\w.]+[`\"\]]?)", text, flags=re.IGNORECASE)
    table_from_insert = insert_match.group(1).strip("`\"[]").split(".")[-1] if insert_match else ""
    group = infer_group(path, repo_root, table_from_insert, "")
    result: list[ModelInfo] = []
    for rm in root.iter("resultMap"):
        typ = rm.attrib.get("type", "")
        cls = class_name(typ)
        if not cls or should_exclude_class(cls):
            continue
        table = table_from_insert
        if not table and cls in java_index:
            table = java_index[cls][1] or snake(cls)
        if not table:
            continue
        fields: list[FieldInfo] = []
        for child in rm:
            if child.tag not in {"id", "result"}:
                continue
            column = child.attrib.get("column", "")
            prop = child.attrib.get("property", "")
            if not column or not prop:
                continue
            fields.append(
                FieldInfo(
                    property=prop,
                    column=column,
                    java_type=infer_java_type_from_jdbc(child.attrib.get("jdbcType", "")),
                    jdbc_type=child.attrib.get("jdbcType", ""),
                    source="Mapper XML",
                    pk=child.tag == "id",
                )
            )
        model = ModelInfo(table=table, class_name=cls, group=group)
        model.source_files.add(rel(path, repo_root))
        if cls in java_index:
            jpath, _, jfields = java_index[cls]
            model.source_files.add(rel(jpath, repo_root))
            fields = merge_fields(jfields, fields)
        model.fields = fields
        result.append(model)
    return result


def parse_table_annotated(repo_root: Path, java_index: dict[str, tuple[Path, str | None, list[FieldInfo]]]) -> list[ModelInfo]:
    result: list[ModelInfo] = []
    for cls, (path, table, fields) in java_index.items():
        if not table or should_exclude_class(cls, has_table=True):
            continue
        group = infer_group(path, repo_root, table, cls)
        model = ModelInfo(table=table, class_name=cls, group=group, fields=fields)
        model.source_files.add(rel(path, repo_root))
        result.append(model)
    return result


def parse_ws_bound(repo_root: Path, java_index: dict[str, tuple[Path, str | None, list[FieldInfo]]]) -> list[ModelInfo]:
    result: list[ModelInfo] = []
    for path in repo_root.rglob("*Mapper.java"):
        if "\\dao\\ws\\" not in str(path) and "/dao/ws/" not in str(path).replace("\\", "/"):
            continue
        text = read_text(path)
        entity_match = re.search(r"@Entity\s+([\w.]+)", text)
        imports = re.findall(r"import\s+([\w.]+);", text)
        candidates = [class_name(entity_match.group(1))] if entity_match else []
        candidates.extend(class_name(i) for i in imports if ".entity.ws." in i)
        for cls in dict.fromkeys(candidates):
            if cls not in java_index or should_exclude_class(cls):
                continue
            jpath, table, fields = java_index[cls]
            table_name = table or snake(cls)
            model = ModelInfo(table=table_name, class_name=cls, group="ws", fields=fields)
            model.source_files.update({rel(path, repo_root), rel(jpath, repo_root)})
            result.append(model)
    return result


def infer_group(path: Path, repo_root: Path, table: str, cls: str) -> str:
    parts = [p.lower() for p in path.parts]
    if "mybatis" in parts:
        idx = parts.index("mybatis")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if "ws" in parts:
        return "ws"
    if table:
        t = table.lower()
        for prefix in ("credit", "loan", "repay", "third", "sys", "oper", "cap", "stat"):
            if t.startswith(prefix + "_") or t.startswith("t_" + prefix + "_"):
                return {"oper": "pub", "stat": "cap"}.get(prefix, prefix)
    if cls:
        return snake(cls).split("_", 1)[0]
    return "pending"


def infer_java_type_from_jdbc(jdbc: str) -> str:
    j = jdbc.upper()
    if j in {"INTEGER", "INT", "SMALLINT", "TINYINT"}:
        return "Integer"
    if j in {"BIGINT", "LONG"}:
        return "Long"
    if j in {"DECIMAL", "NUMERIC", "DOUBLE", "FLOAT"}:
        return "BigDecimal"
    if j in {"TIMESTAMP", "DATE", "TIME"}:
        return "Date"
    if j in {"BIT", "BOOLEAN"}:
        return "Boolean"
    return "String"


def sql_type(field: FieldInfo) -> str:
    jdbc = field.jdbc_type.upper()
    if jdbc in {"CHAR", "VARCHAR", "LONGVARCHAR", "NVARCHAR"}:
        return "VARCHAR(255)"
    if jdbc in {"INTEGER", "INT", "SMALLINT", "TINYINT"}:
        return "INT"
    if jdbc == "BIGINT":
        return "BIGINT"
    if jdbc in {"DECIMAL", "NUMERIC"}:
        return "DECIMAL(18,2)"
    if jdbc in {"TIMESTAMP", "DATETIME"}:
        return "DATETIME"
    if jdbc == "DATE":
        return "DATE"
    if jdbc in {"BIT", "BOOLEAN"}:
        return "TINYINT(1)"
    jt = field.java_type
    if "BigDecimal" in jt:
        return "DECIMAL(18,2)"
    if any(x in jt for x in ("Integer", "int")):
        return "INT"
    if any(x in jt for x in ("Long", "long")):
        return "BIGINT"
    if any(x in jt for x in ("Date", "LocalDateTime", "Timestamp")):
        return "DATETIME"
    if any(x in jt for x in ("Boolean", "boolean")):
        return "TINYINT(1)"
    return "VARCHAR(255)"


def merge_fields(entity_fields: list[FieldInfo], mapper_fields: list[FieldInfo]) -> list[FieldInfo]:
    by_prop = {f.property: FieldInfo(**f.__dict__) for f in entity_fields}
    ordered: list[FieldInfo] = []
    seen: set[str] = set()
    for mf in mapper_fields:
        base = by_prop.get(mf.property)
        if base:
            base.column = mf.column
            base.jdbc_type = mf.jdbc_type
            base.source = "Mapper XML + 实体字段"
            base.pk = base.pk or mf.pk
            field = base
        else:
            field = mf
        key = field.column.lower()
        if key not in seen:
            ordered.append(field)
            seen.add(key)
    for ef in entity_fields:
        key = ef.column.lower()
        if key not in seen:
            ordered.append(ef)
            seen.add(key)
    return ordered


def merge_models(models: Iterable[ModelInfo]) -> list[ModelInfo]:
    merged: dict[tuple[str, str], ModelInfo] = {}
    for model in models:
        key = (model.group, model.table.lower())
        if key not in merged:
            merged[key] = model
            continue
        current = merged[key]
        current.source_files.update(model.source_files)
        current.fields = merge_fields(current.fields, model.fields)
        if not current.class_name and model.class_name:
            current.class_name = model.class_name
    return sorted(merged.values(), key=lambda m: (m.group, m.table.lower()))


def esc(text: str) -> str:
    return text.replace("'", "''")


def render_group(database: str, group: str, models: list[ModelInfo]) -> str:
    date = _dt.date.today().isoformat()
    tables = ", ".join(m.table for m in models) or "待确认"
    sources = sorted({s for m in models for s in m.source_files})
    lines = [
        f"-- Database/Service: {database}",
        f"-- Business Model: {group}",
        f"-- Tables: {tables}",
        f"-- Source: {'; '.join(sources) if sources else '待确认'}",
        f"-- Status: {DEFAULT_STATUS}",
        "-- Migration Script: none",
        f"-- Last Generated: {date}",
        "-- Note: 本文件是 SQL 知识库草案；字段来自实体/Mapper/DAO 证据，类型、长度、可空性、索引、唯一约束和外键需以真实 DDL 校准。",
        "",
    ]
    for model in models:
        lines.append(f"CREATE TABLE `{model.table}` (")
        column_lines = []
        for field in model.fields:
            col_def = f"  `{field.column}` {sql_type(field)} NULL"
            if field.note:
                col_def += f" COMMENT '{esc(field.note)}'"
            column_lines.append(col_def)
        pks = [f.column for f in model.fields if f.pk]
        if pks:
            column_lines.append("  PRIMARY KEY (" + ", ".join(f"`{p}`" for p in pks) + ")")
        else:
            column_lines.append("  -- TODO: 主键待确认")
        for idx, line in enumerate(column_lines):
            suffix = "," if idx < len(column_lines) - 1 else ""
            lines.append(line + suffix)
        source = "; ".join(sorted(model.source_files)) or "待确认"
        lines.append(f") COMMENT='{esc(model.class_name or model.table)} 持久化模型；DDL待确认';")
        lines.append("")
        lines.append("-- Field Evidence:")
        for field in model.fields:
            evidence = [
                f"`{field.column}`",
                f"java={field.property}",
                f"type={field.java_type}",
                f"source={field.source}",
                f"sql={sql_type(field)}",
            ]
            if field.jdbc_type:
                evidence.append(f"jdbc={field.jdbc_type}")
            if field.pk:
                evidence.append("pk=是")
            evidence.append("nullable=待确认")
            lines.append("-- - " + "; ".join(evidence))
        lines.append(f"-- Source: {source}")
        lines.append(f"-- TODO: 校准 `{model.table}` 的真实字段类型、长度、NOT NULL、默认值、索引、唯一约束、外键和表注释。")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def collect(repo_root: Path) -> list[ModelInfo]:
    java_index = build_java_index(repo_root)
    models: list[ModelInfo] = []
    for xml in repo_root.rglob("*.xml"):
        if any(part in {"target", ".settings", ".git"} for part in xml.parts):
            continue
        if "mybatis" in [p.lower() for p in xml.parts]:
            models.extend(parse_mapper(xml, repo_root, java_index))
    models.extend(parse_table_annotated(repo_root, java_index))
    models.extend(parse_ws_bound(repo_root, java_index))
    return merge_models(models)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Fons4AI SQL knowledge files")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--sql-root", required=True)
    parser.add_argument("--database", default="pending")
    parser.add_argument("--groups", default="", help="Optional comma-separated group filter")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    sql_root = Path(args.sql_root).resolve()
    filters = {g.strip() for g in args.groups.split(",") if g.strip()}
    models = [m for m in collect(repo_root) if not filters or m.group in filters]
    by_group: dict[str, list[ModelInfo]] = defaultdict(list)
    for model in models:
        by_group[model.group].append(model)
    if args.dry_run:
        print(json.dumps({g: [m.table for m in ms] for g, ms in sorted(by_group.items())}, ensure_ascii=False, indent=2))
        return 0
    for group, group_models in sorted(by_group.items()):
        out_dir = sql_root / (args.database if args.database else "pending")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{group}.sql").write_text(render_group(args.database, group, group_models), encoding="utf-8")
    print(f"OK: generated {sum(len(v) for v in by_group.values())} table model(s) in {len(by_group)} group(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
