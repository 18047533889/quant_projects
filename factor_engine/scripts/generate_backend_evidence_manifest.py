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
    from backend.primitive_evidence import (
        DUCKDB_EDGE_VERIFIED,
        DUCKDB_REAL_SQL_VERIFIED,
        DUCKDB_REFERENCE_PARITY_VERIFIED,
        POLARS_EDGE_VERIFIED,
        POLARS_NO_FALLBACK_VERIFIED,
        POLARS_REFERENCE_PARITY_VERIFIED,
    )
    from backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS, effective_sql_production_safe
    from backend.sql_pushdown.sql_registry import register_sql_backends

    load_all()
    register_sql_backends()
    catalog = OperatorRegistry.catalog()
    operators = {}
    for canonical in sorted(catalog):
        backends = set(OperatorRegistry.backends_for(canonical))
        polars_meta = dict(((catalog[canonical].get("backend_meta") or {}).get("polars") or {}))
        execution_kind = polars_meta.get("execution_kind", "unsupported")
        polars_implemented = "polars" in backends
        polars_reference = polars_implemented and canonical in POLARS_REFERENCE_PARITY_VERIFIED
        polars_edge = polars_implemented and canonical in POLARS_EDGE_VERIFIED
        polars_no_fallback = (
            polars_implemented
            and canonical in POLARS_NO_FALLBACK_VERIFIED
            and execution_kind in {"expression_native", "long_native"}
        )
        duckdb_implemented = canonical in SQL_IMPLEMENTED_CANONICALS
        duckdb_reference = canonical in DUCKDB_REFERENCE_PARITY_VERIFIED
        duckdb_real_sql = canonical in DUCKDB_REAL_SQL_VERIFIED
        duckdb_edge = canonical in DUCKDB_EDGE_VERIFIED
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
            "duckdb_real_sql_tested": duckdb_real_sql,
            "duckdb_reference_parity": duckdb_reference,
            "duckdb_edge_parity": duckdb_edge,
            "duckdb_production_safe": (
                duckdb_reference and duckdb_real_sql and duckdb_edge
                and effective_sql_production_safe(canonical)
            ),
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
