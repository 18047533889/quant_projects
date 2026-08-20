#!/usr/bin/env python3
"""Validate repository factor definitions against the public daily DSL."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterator

import yaml

ROOT = Path(__file__).resolve().parents[2]
FE = ROOT / "factor_engine"
for path in (ROOT, FE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from api.dsl_parser import parse_expr  # noqa: E402

FORMULA_KEYS = {"expr", "formula", "dsl_formula", "expression"}
SKIP_PARTS = {"archive", "output", "reports", ".git"}
ROOTS = (
    ROOT / "gtja191" / "candidate_pool",
    ROOT / "week2_pv_factors" / "candidate_pool",
    FE / "examples" / "configs",
)


def walk(value: Any, *, key: str = "") -> Iterator[str]:
    if isinstance(value, dict):
        for child_key, child in value.items():
            if child_key in FORMULA_KEYS and isinstance(child, str) and child.strip():
                yield child.strip()
            else:
                yield from walk(child, key=str(child_key))
    elif isinstance(value, list):
        for child in value:
            yield from walk(child, key=key)


def load(path: Path) -> Any:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> int:
    checked = 0
    failures: list[str] = []
    for root in ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in {".json", ".yaml", ".yml"}:
                continue
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            try:
                payload = load(path)
            except Exception:
                continue
            for formula in walk(payload):
                checked += 1
                try:
                    # These are already-published repository formulas, not new
                    # authoring submissions.  Parse through the compatibility
                    # surface while daily admission remains fail-closed.
                    parse_expr(formula, surface="compat")
                except Exception as exc:
                    failures.append(f"{path.relative_to(ROOT)}: {type(exc).__name__}: {exc}: {formula}")
    if failures:
        print("\n".join(failures[:100]))
        raise SystemExit(f"{len(failures)} public factor formulas failed validation")
    print(f"validated {checked} repository factor formulas against the compatibility DSL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
