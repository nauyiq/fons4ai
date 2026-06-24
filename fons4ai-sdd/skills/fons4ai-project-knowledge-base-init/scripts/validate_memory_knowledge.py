#!/usr/bin/env python3
"""Validate Fons4AI layered memory knowledge documents."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


BAD_TEXT_PATTERNS = tuple(
    s.encode("utf-8").decode("unicode_escape")
    for s in (r"\ufffd", r"\u9225", r"\u9239", r"\u93ba\u3126", r"\u5bf0\u5477", r"\u5bb8\u8336", r"\u93c2\u56e8", r"\u740c\u3126")
)
PROJECT_FILES = ("index.md", "业务架构.md", "技术架构.md", "数据架构.md")
DOMAIN_FILES = ("业务架构.md", "技术架构.md", "数据架构.md")
CARD_REQUIRED_FIELDS = (
    "知识编号",
    "知识类型",
    "所属领域",
    "状态",
    "来源",
    "可信度说明",
    "关联能力",
    "关联变体",
    "关联场景",
    "关联对象",
    "关联代码/接口/SQL",
    "更新日期",
)
CARD_ALLOWED_TYPES = {"业务场景", "业务规则", "业务变体", "状态流转", "技术流程", "接口契约", "数据模型", "治理规则"}
# Keep 已确认 for legacy documents; new templates should emit 已验证 / 待确认 / 已废弃.
CARD_ALLOWED_STATUS = {"已验证", "待确认", "已废弃", "已确认"}
SQL_PATH_RE = re.compile(r"\.specify/sql/[A-Za-z0-9_./-]+\.sql")
DOMAIN_PATH_RE = re.compile(r"(?:\.specify/memory/)?domains/([A-Za-z0-9_-]+)(?:/|$)")
CARD_PATH_RE = re.compile(r"\.specify/memory/domains/[A-Za-z0-9_-]+/cards/[A-Za-z0-9_.-]+\.md")
SCENARIO_RE = re.compile(r"\bBS-(?:[A-Z0-9]+-)?\d{3}\b")
HEADER_FIELD_RE = re.compile(r"^>\s*([^：:]+)\s*[：:]\s*(.+?)\s*$", re.MULTILINE)
DOMAIN_SLUG_IN_LABEL_RE = re.compile(r"[（(]\s*([A-Za-z0-9_-]+)\s*[）)]")
DOMAIN_REQUIRED_SECTIONS = {
    "业务架构.md": ("## 4. 核心业务场景", "## 5. 业务能力变体矩阵", "## 6. 共性规则与差异规则"),
    "技术架构.md": ("## 2. 场景技术落地", "## 5. 能力变体技术落地矩阵"),
    "数据架构.md": ("## 3. 业务对象生命周期", "## 4. 业务能力变体数据差异"),
}
INDEX_REQUIRED_SECTIONS = ("## 4. 核心能力索引", "## 5. 业务能力变体索引")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def repo_root_for(memory_root: Path) -> Path:
    resolved = memory_root.resolve()
    if resolved.name == "memory" and resolved.parent.name == ".specify":
        return resolved.parent.parent
    return resolved.parent


def check_markdown(path: Path, text: str, errors: list[str]) -> None:
    if not text.lstrip().startswith("# "):
        errors.append(f"{path} must start with a level-1 title")
    if text.count("```") % 2 != 0:
        errors.append(f"{path} has unbalanced Markdown fences")
    for pattern in BAD_TEXT_PATTERNS:
        if pattern in text:
            errors.append(f"{path} contains mojibake pattern: {pattern}")
    if "推断" in text:
        errors.append(f"{path} must not use 推断 as a knowledge status; use 待确认 with a credibility note")


def validate_sections(path: Path, text: str, sections: tuple[str, ...], errors: list[str]) -> None:
    for section in sections:
        if section not in text:
            errors.append(f"{path} is missing required section: {section}")


def read_required(path: Path, errors: list[str]) -> str:
    if not path.exists():
        errors.append(f"{path} does not exist")
        return ""
    try:
        text = read(path)
    except UnicodeDecodeError as exc:
        errors.append(f"{path} is not valid UTF-8: {exc}")
        return ""
    check_markdown(path, text, errors)
    return text


def referenced_domains(index_text: str) -> set[str]:
    return {match.group(1) for match in DOMAIN_PATH_RE.finditer(index_text)}


def scenario_ids(text: str) -> set[str]:
    return set(SCENARIO_RE.findall(text))


def card_fields(text: str) -> dict[str, str]:
    return {match.group(1).strip(): match.group(2).strip() for match in HEADER_FIELD_RE.finditer(text)}


def domain_slug_from_label(label: str) -> str:
    label = label.strip()
    match = DOMAIN_SLUG_IN_LABEL_RE.search(label)
    if match:
        return match.group(1)
    return label


def resolve_repo_path(repo_root: Path, path_text: str) -> Path:
    clean = path_text.strip("`").replace("\\", "/")
    if clean.startswith(".specify/"):
        return repo_root / clean
    return repo_root / clean


def validate_sql_refs(path: Path, text: str, repo_root: Path, errors: list[str]) -> None:
    for sql_ref in sorted(set(SQL_PATH_RE.findall(text))):
        if resolve_repo_path(repo_root, sql_ref).exists():
            continue
        if "待确认" not in text:
            errors.append(f"{path} references missing SQL file without pending marker: {sql_ref}")


def validate_card_refs(path: Path, text: str, repo_root: Path, errors: list[str]) -> None:
    for card_ref in sorted(set(CARD_PATH_RE.findall(text))):
        if resolve_repo_path(repo_root, card_ref).exists():
            continue
        if "待确认" not in text:
            errors.append(f"{path} references missing knowledge card without pending marker: {card_ref}")


def validate_card(path: Path, expected_domain: str, repo_root: Path) -> list[str]:
    errors: list[str] = []
    text = read_required(path, errors)
    if not text:
        return errors
    fields = card_fields(text)
    for field in CARD_REQUIRED_FIELDS:
        if not fields.get(field):
            errors.append(f"{path} card is missing header field '{field}'")
    knowledge_type = fields.get("知识类型", "").split("|", 1)[0].strip()
    if knowledge_type and knowledge_type not in CARD_ALLOWED_TYPES:
        errors.append(f"{path} has unsupported 知识类型: {knowledge_type}")
    status = fields.get("状态", "").split("|", 1)[0].strip()
    if status and status not in CARD_ALLOWED_STATUS:
        errors.append(f"{path} has unsupported 状态: {status}")
    if status == "已废弃" and not re.search(r"替代知识|废弃原因|替代", text):
        errors.append(f"{path} is 已废弃 but has no replacement or deprecation reason")
    if fields.get("状态", "").startswith("已验证") and not fields.get("来源", "").strip():
        errors.append(f"{path} is 已验证 but has no 来源")
    if status == "待确认" and not fields.get("可信度说明", "").strip():
        errors.append(f"{path} is 待确认 but has no 可信度说明")
    if fields.get("知识类型", "").startswith("业务变体") and not fields.get("关联能力", "").strip():
        errors.append(f"{path} is 业务变体 but has no 关联能力")
    domain = fields.get("所属领域", "")
    domain_slug = domain_slug_from_label(domain)
    if domain_slug and domain_slug != expected_domain:
        errors.append(f"{path} 所属领域 must reference slug '{expected_domain}', got '{domain}'")
    validate_sql_refs(path, text, repo_root, errors)
    validate_card_refs(path, text, repo_root, errors)
    return errors


def validate_domain(domain_dir: Path, repo_root: Path) -> list[str]:
    errors: list[str] = []
    docs: dict[str, str] = {}
    for file_name in DOMAIN_FILES:
        text = read_required(domain_dir / file_name, errors)
        if text:
            docs[file_name] = text
            validate_sections(domain_dir / file_name, text, DOMAIN_REQUIRED_SECTIONS.get(file_name, ()), errors)

    business = docs.get("业务架构.md", "")
    technical = docs.get("技术架构.md", "")
    if business and technical:
        ids = scenario_ids(business)
        missing = sorted(sid for sid in ids if sid not in technical)
        if missing:
            errors.append(f"{domain_dir} technical architecture missing business scenario id(s): {', '.join(missing)}")

    for file_name, text in docs.items():
        validate_sql_refs(domain_dir / file_name, text, repo_root, errors)
        validate_card_refs(domain_dir / file_name, text, repo_root, errors)

    cards_dir = domain_dir / "cards"
    if cards_dir.exists():
        for card in sorted(cards_dir.rglob("*.md")):
            errors.extend(validate_card(card, domain_dir.name, repo_root))
    return errors


def validate(memory_root: Path) -> list[str]:
    errors: list[str] = []
    repo_root = repo_root_for(memory_root)
    docs: dict[str, str] = {}
    for name in PROJECT_FILES:
        text = read_required(memory_root / name, errors)
        if text:
            docs[name] = text

    index_text = docs.get("index.md", "")
    if index_text:
        validate_sections(memory_root / "index.md", index_text, INDEX_REQUIRED_SECTIONS, errors)
    domains = referenced_domains(index_text)
    domains_dir = memory_root / "domains"
    validated_domains: set[str] = set()
    for domain in sorted(domains):
        domain_dir = domains_dir / domain
        if not domain_dir.exists():
            errors.append(f"index.md references missing domain directory: {domain_dir}")
            continue
        errors.extend(validate_domain(domain_dir, repo_root))
        validated_domains.add(domain)

    if domains_dir.exists():
        for domain_dir in sorted(path for path in domains_dir.iterdir() if path.is_dir()):
            if domain_dir.name in validated_domains:
                continue
            if domain_dir.name not in domains:
                errors.append(f"{domain_dir} exists but is not referenced by index.md")
            errors.extend(validate_domain(domain_dir, repo_root))

    for name, text in docs.items():
        validate_sql_refs(memory_root / name, text, repo_root, errors)
        validate_card_refs(memory_root / name, text, repo_root, errors)

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Fons4AI layered memory knowledge documents")
    parser.add_argument("--memory-root", default=".specify/memory")
    args = parser.parse_args()
    errors = validate(Path(args.memory_root).resolve())
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("OK: validated layered memory knowledge documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
