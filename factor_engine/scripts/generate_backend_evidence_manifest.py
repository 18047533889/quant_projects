#!/usr/bin/env python3
"""Generate the canonical backend evidence manifest from runtime facts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FE_ROOT))


def build() -> dict:
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry
    from cleaned_operators.operator_surface import classify_canonical, surface_summary
    from cleaned_operators.operator_policy import POLARS_PARITY_VERIFIED
    from backend.primitive_evidence import DUCKDB_REAL_SQL_VERIFIED, POLARS_REFERENCE_PARITY_VERIFIED
    from backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS, effective_sql_production_safe

    load_all()
    catalog = OperatorRegistry.catalog()
    operators = {}
    for canonical in sorted(catalog):
        backends = set(OperatorRegistry.backends_for(canonical))
        polars_meta = dict(((catalog[canonical].get("backend_meta") or {}).get("polars") or {}))
        execution_kind = polars_meta.get("execution_kind", "unsupported")
        polars_implemented = "polars" in backends
        polars_reference = polars_implemented and (
            canonical in POLARS_REFERENCE_PARITY_VERIFIED or canonical in POLARS_PARITY_VERIFIED
        )
        # Edge certification has no independent legacy source yet.  Fail closed:
        # only explicit edge evidence can promote this field in future.
        polars_edge = False
        polars_no_fallback = polars_implemented and execution_kind in {"expression_native", "long_native"}
        duckdb_implemented = canonical in SQL_IMPLEMENTED_CANONICALS
        duckdb_reference = canonical in DUCKDB_REAL_SQL_VERIFIED
        duckdb_edge = duckdb_reference
        operators[canonical] = {
            "surface": classify_canonical(canonical),
            "pandas_runtime": "pandas_numpy" in backends,
            "polars_implemented": polars_implemented,
            "polars_execution_kind": execution_kind,
            "polars_reference_parity": polars_reference,
            "polars_edge_parity": polars_edge,
            "polars_no_fallback": polars_no_fallback,
            "polars_production_safe": polars_reference and polars_edge and polars_no_fallback,
            "duckdb_implemented": duckdb_implemented,
            "duckdb_real_sql_tested": duckdb_reference,
            "duckdb_reference_parity": duckdb_reference,
            "duckdb_edge_parity": duckdb_edge,
            "duckdb_production_safe": duckdb_edge and effective_sql_production_safe(canonical),
        }
    return {
        "schema_version": 1,
        "surface_counts": surface_summary(operators),
        "operators": operators,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FE_ROOT / "docs" / "backend_evidence_manifest.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = build()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.check:
        if not args.out.is_file() or args.out.read_text(encoding="utf-8") != rendered:
            print(f"stale backend evidence manifest: {args.out}", file=sys.stderr)
            return 1
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
