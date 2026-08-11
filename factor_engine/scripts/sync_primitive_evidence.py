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
    from tests.operators.test_production_convergence import STRICT_PERIOD_CASES
    from cleaned_operators.operator_surface import DAILY_CANONICALS
    from cleaned_operators.registry import OperatorRegistry

    evidence_canonicals = set(DAILY_CANONICALS) | {"protected_div"}

    def daily_canonical_cases(cases):
        """Normalize aliases and exclude non-production cases from evidence."""
        normalized = {}
        for name, builder in cases:
            canonical = OperatorRegistry._aliases.get(name, name)
            if canonical in evidence_canonicals:
                normalized.setdefault(canonical, builder)
        return sorted(normalized.items())

    # R38 evidence-regen blocker: strict-period primitives are DECLARED only in
    # fields that have real executed tests.  ``test_production_convergence.py``
    # runs ``test_strict_period_polars_*`` for 6 canonicals (no
    # quarter_from_cumulative/ttm_from_cumulative polars parity case) and
    # ``test_strict_period_duckdb_*`` for all 7.  There is NO strict-period edge
    # or no-pandas-fallback test, so declaring them in those fields would be an
    # over-declaration that makes the six-way intersection forever unattainable
    # (edge/no_fallback have no executed cases to certify them).
    from tests.operators.test_production_convergence import STRICT_PERIOD_CASES as _SPC
    _SP_POLARS = {(name, fn) for name, fn in _SPC if name != "quarter_from_cumulative" and name != "ttm_from_cumulative"}
    polars_reference = daily_canonical_cases(merge_case_lists(MEMORY_CASES, POLARS_BULK_CASES, _SP_POLARS))
    duckdb_reference = daily_canonical_cases(merge_case_lists(DUCKDB_CASES, DUCKDB_BULK_CASES, STRICT_PERIOD_CASES))
    polars_edge = daily_canonical_cases(merge_case_lists(EDGE_CASES))
    duckdb_edge = daily_canonical_cases(merge_case_lists(DUCKDB_EDGE_CASES))
    no_fallback = daily_canonical_cases(merge_case_lists(NO_PANDAS_CASES))

    return build_case_registry_payload(
        polars_reference=polars_reference,
        polars_edge=polars_edge,
        duckdb_reference=duckdb_reference,
        duckdb_edge=duckdb_edge,
        duckdb_null_edge=duckdb_edge,
        duckdb_nan_edge=daily_canonical_cases(merge_case_lists(DUCKDB_IEEE_NAN_CASES)),
        duckdb_inf_edge=daily_canonical_cases(merge_case_lists(DUCKDB_IEEE_INF_CASES)),
        no_fallback=no_fallback,
    )


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
