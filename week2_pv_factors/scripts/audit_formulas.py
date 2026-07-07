#!/usr/bin/env python3
"""Week2 因子 DSL 审计：parse_expr + 算子白名单 + 汇报对照。"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lib.dsl_validate import validate_formula  # noqa: E402
from lib.paths import resolve_factor_engine_root  # noqa: E402

FIELDS = {"open", "high", "low", "close", "volume"}


def extract_calls(formula: str) -> set[str]:
    ops: set[str] = set()
    for node in ast.walk(ast.parse(formula, mode="eval")):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            ops.add(node.func.id)
    return ops


def load_allowlist() -> set[str]:
    fe = resolve_factor_engine_root()
    if fe is None:
        return set()
    if str(fe) not in sys.path:
        sys.path.insert(0, str(fe))
    from api.operator_registry import build_dsl_allowlist

    return set(build_dsl_allowlist().keys())


def main() -> int:
    catalog = json.loads(
        (PACKAGE_ROOT / "source" / "week2_factors_catalog.json").read_text(encoding="utf-8")
    )
    allow = {k.lower() for k in load_allowlist()}
    all_ops: set[str] = set()
    parse_fail = []
    allow_fail = []

    for item in catalog["factors"]:
        formula = item["formula"]
        ok, msg = validate_formula(formula)
        if not ok:
            parse_fail.append((item["id"], msg))
        calls = extract_calls(formula)
        all_ops |= calls
        for op in calls:
            if op.lower() not in allow:
                allow_fail.append((item["id"], op))

    print("=" * 60)
    print("Week2 DSL 审计")
    print("=" * 60)
    print(f"因子数: {len(catalog['factors'])}")
    print(f"parse_expr 通过: {len(catalog['factors']) - len(parse_fail)}")
    print(f"算子种类: {len(all_ops)} -> {sorted(all_ops)}")
    print(f"白名单外算子: {len(allow_fail)}")
    if parse_fail:
        for fid, msg in parse_fail:
            print(f"  PARSE FAIL {fid}: {msg}")
    if allow_fail:
        for fid, op in allow_fail:
            print(f"  ALLOW FAIL {fid}: {op}")

    audit = json.loads(
        (PACKAGE_ROOT / "source" / "audit_report.json").read_text(encoding="utf-8")
    )
    print(f"\n汇报对照: {audit['summary']}")
    for item in audit["checks"]:
        flag = "⚠" if item["status"] != "ok" else "✓"
        print(f"  {flag} {item['id']}: {item['status']} — {item['note']}")

    block = bool(parse_fail or allow_fail)
    print("=" * 60)
    print("VERDICT:", "PASS" if not block else "BLOCKED")
    return 1 if block else 0


if __name__ == "__main__":
    raise SystemExit(main())
