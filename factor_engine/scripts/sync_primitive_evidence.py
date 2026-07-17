#!/usr/bin/env python3
"""同步 parity case 注册表到 ``evidence/primitive_case_registry.json``（不授予 verified）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]


def _bootstrap() -> None:
    root = str(FE_ROOT.parent)
    fe = str(FE_ROOT)
    for p in (root, fe):
        if p not in sys.path:
            sys.path.insert(0, p)
    from cleaned_operators import load_all
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()


def _collect_cases() -> dict:
    from tests.backend_parity.evidence_case_registry import build_case_registry_payload, merge_case_lists
    from tests.backend_parity.duckdb_ieee_edge_cases import DUCKDB_IEEE_INF_CASES, DUCKDB_IEEE_NAN_CASES
    from tests.backend_parity.test_p0_edge_cases_triple_parity import DUCKDB_EDGE_CASES, EDGE_CASES
    from tests.backend_parity.test_polars_long_no_pandas_path import NO_PANDAS_CASES
    from tests.backend_parity.test_production_core_triple_parity import DUCKDB_CASES, MEMORY_CASES
    from tests.backend_parity.test_production_safe_bulk_parity import DUCKDB_BULK_CASES, POLARS_BULK_CASES
    from cleaned_operators.registry import OperatorRegistry

    active = set(OperatorRegistry._operators)

    def exact_active(*case_groups):
        # Case entries are ``(canonical, builder)`` tuples. Certification never
        # travels through aliases, so retain only exact active canonical names.
        return [
            (name, builder)
            for name, builder in merge_case_lists(*case_groups)
            if name in active
        ]

    payload = build_case_registry_payload(
        polars_reference=exact_active(MEMORY_CASES, POLARS_BULK_CASES),
        polars_edge=exact_active(EDGE_CASES),
        duckdb_reference=exact_active(DUCKDB_CASES, DUCKDB_BULK_CASES),
        duckdb_edge=exact_active(DUCKDB_EDGE_CASES),
        duckdb_null_edge=exact_active(DUCKDB_EDGE_CASES),
        duckdb_nan_edge=exact_active(DUCKDB_IEEE_NAN_CASES),
        duckdb_inf_edge=exact_active(DUCKDB_IEEE_INF_CASES),
        no_fallback=exact_active(NO_PANDAS_CASES),
    )
    tracked_keys = (
        "polars_reference_parity", "polars_edge_verified",
        "duckdb_reference_parity", "duckdb_edge_verified",
        "no_fallback_verified",
    )
    if any(not payload.get(key) for key in tracked_keys):
        empty = [key for key in tracked_keys if not payload.get(key)]
        raise RuntimeError(f"active evidence case sets unexpectedly empty: {empty}")
    payload["active_registry_count"] = len(active)
    payload["exact_canonical_only"] = True
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    _bootstrap()
    payload = _collect_cases()
    count = payload.pop("_six_way_count", 0)
    out = FE_ROOT / "evidence" / "primitive_case_registry.json"
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if not out.is_file():
            print("missing primitive_case_registry.json", file=sys.stderr)
            return 1
        ref = json.loads(out.read_text(encoding="utf-8"))
        if ref != payload:
            print("primitive_case_registry.json 过期", file=sys.stderr)
            return 1
        print(f"case registry fresh ({count} six-way case-list intersection)")
        print("NOTE: case registry ≠ verified；须运行 certify_primitive_evidence.py")
        return 0
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} (six-way case-list: {count})")
    print("NOTE: 未授予 verified；须 pytest 通过后运行 certify_primitive_evidence.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
