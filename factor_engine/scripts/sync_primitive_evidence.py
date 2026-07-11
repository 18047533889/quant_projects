#!/usr/bin/env python3
"""从 parity case 注册表同步 ``evidence/primitive_verified.json``。"""
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
    from tests.backend_parity.evidence_case_registry import build_evidence_payload, merge_case_lists
    from tests.backend_parity.duckdb_ieee_edge_cases import DUCKDB_IEEE_INF_CASES, DUCKDB_IEEE_NAN_CASES
    from tests.backend_parity.test_p0_edge_cases_triple_parity import DUCKDB_EDGE_CASES, EDGE_CASES
    from tests.backend_parity.test_polars_long_no_pandas_path import NO_PANDAS_CASES
    from tests.backend_parity.test_production_core_triple_parity import DUCKDB_CASES, MEMORY_CASES
    from tests.backend_parity.test_production_safe_bulk_parity import DUCKDB_BULK_CASES, POLARS_BULK_CASES

    polars_reference = merge_case_lists(MEMORY_CASES, POLARS_BULK_CASES)
    duckdb_reference = merge_case_lists(DUCKDB_CASES, DUCKDB_BULK_CASES)
    polars_edge = merge_case_lists(EDGE_CASES)
    duckdb_edge = merge_case_lists(DUCKDB_EDGE_CASES)
    no_fallback = merge_case_lists(NO_PANDAS_CASES)

    existing = {}
    path = FE_ROOT / "evidence" / "primitive_verified.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
    operators_meta = existing.get("operators") or {}

    return build_evidence_payload(
        polars_reference=polars_reference,
        polars_edge=polars_edge,
        duckdb_reference=duckdb_reference,
        duckdb_edge=duckdb_edge,
        duckdb_null_edge=duckdb_edge,
        duckdb_nan_edge=merge_case_lists(DUCKDB_IEEE_NAN_CASES),
        duckdb_inf_edge=merge_case_lists(DUCKDB_IEEE_INF_CASES),
        no_fallback=no_fallback,
        operators_meta=operators_meta,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    _bootstrap()
    payload = _collect_cases()
    count = payload.pop("_six_way_count", 0)
    out = FE_ROOT / "evidence" / "primitive_verified.json"
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if not out.is_file():
            print("missing primitive_verified.json", file=sys.stderr)
            return 1
        ref = json.loads(out.read_text(encoding="utf-8"))
        if ref != payload:
            print("primitive_verified.json 过期", file=sys.stderr)
            return 1
        print(f"evidence fresh ({count} six-way certified)")
        return 0
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} (six-way certified: {count})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
