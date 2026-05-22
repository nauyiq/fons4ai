#!/usr/bin/env python3
"""Deprecated: Fons4AI no longer infers SQL DDL from source code metadata."""

from __future__ import annotations

import sys


MESSAGE = """ERROR: generate_sql_knowledge.py is deprecated.

Fons4AI SQL knowledge files must be created from real DDL evidence:
- configured database MCP query results, or
- existing repository SQL DDL files imported with import_sql_knowledge_file.py.

Do not generate CREATE TABLE statements from Java entities, Mapper interfaces,
ORM annotations, repository method names, or inferred Java field types.
"""


def main() -> int:
    print(MESSAGE.strip(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
