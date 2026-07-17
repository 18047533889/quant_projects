#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate backend capability artifacts from the final runtime registry."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FACTOR_ENGINE = ROOT / "factor_engine"
if str(FACTOR_ENGINE) not in sys.path:
    sys.path.insert(0, str(FACTOR_ENGINE))

from cleaned_operators import load_all  # noqa: E402
from backend.active_capabilities import (  # noqa: E402
    backend_contract_errors,
    capability_matrix,
)


def _yes(value: bool) -> str:
    return "yes" if value else "no"


def main() -> None:
    load_all()
    errors = backend_contract_errors()
    if errors:
        raise SystemExit("backend capability contract errors:\n- " + "\n- ".join(errors))

    matrix = capability_matrix()
    rows = list(matrix.values())
    surface_counts = Counter(row.surface for row in rows)

    payload = {
        "schema_version": "factor_engine.backend_capabilities.v2",
        "canonical_count": len(rows),
        "surface_counts": dict(sorted(surface_counts.items())),
        "definitions": {
            "registered": "runtime backend object exists",
            "polars_no_pandas_bridge": "source is not a Pandas conversion bridge",
            "polars_expression_native": "audited Polars expression or long-plan implementation; no Python rolling callback/full-panel NumPy",
            "parity_verified": "named backend has current canonical reference parity evidence",
            "production_safe": "routing may select this backend automatically",
        },
        "operators": {name: item.to_dict() for name, item in sorted(matrix.items())},
    }
    docs = FACTOR_ENGINE / "docs"
    (docs / "backend_capabilities.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    totals = {
        "pandas": sum(row.pandas_runtime for row in rows),
        "polars_registered": sum(row.polars_registered for row in rows),
        "polars_non_bridge": sum(row.polars_no_pandas_bridge for row in rows),
        "polars_expression_native": sum(row.polars_expression_native for row in rows),
        "polars_verified": sum(row.polars_parity_verified for row in rows),
        "polars_safe": sum(row.polars_production_safe for row in rows),
        "duckdb_implemented": sum(row.duckdb_emitter_implemented for row in rows),
        "duckdb_verified": sum(row.duckdb_reference_parity for row in rows),
        "duckdb_safe": sum(row.duckdb_production_safe for row in rows),
        "clickhouse_safe": sum(row.clickhouse_production_safe for row in rows),
    }

    header = [
        "# Backend capability matrix",
        "",
        "> Generated from the final `OperatorRegistry` and exact active canonical names.",
        "> Registration, non-bridge implementation, expression-native implementation, parity evidence and production routing are distinct states.",
        "",
        "## Summary",
        "",
        f"- active canonical: **{len(rows)}**",
        *[f"- {surface} surface: **{count}**" for surface, count in sorted(surface_counts.items())],
        f"- Pandas runtime: **{totals['pandas']}**",
        f"- Polars registered: **{totals['polars_registered']}**",
        f"- Polars non-bridge: **{totals['polars_non_bridge']}**",
        f"- Polars expression-native: **{totals['polars_expression_native']}**",
        f"- Polars parity verified: **{totals['polars_verified']}**",
        f"- Polars production safe: **{totals['polars_safe']}**",
        f"- DuckDB emitter implemented: **{totals['duckdb_implemented']}**",
        f"- DuckDB reference parity: **{totals['duckdb_verified']}**",
        f"- DuckDB production safe: **{totals['duckdb_safe']}**",
        f"- ClickHouse production safe: **{totals['clickhouse_safe']}**",
        "",
        "## Matrix",
        "",
        "| canonical | surface | pandas | polars | no bridge | expression native | polars verified | polars safe | duckdb implemented | duckdb verified | duckdb safe | polars source |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    body = [
        "| " + " | ".join([
            row.canonical,
            row.surface,
            _yes(row.pandas_runtime),
            _yes(row.polars_registered),
            _yes(row.polars_no_pandas_bridge),
            _yes(row.polars_expression_native),
            _yes(row.polars_parity_verified),
            _yes(row.polars_production_safe),
            _yes(row.duckdb_emitter_implemented),
            _yes(row.duckdb_reference_parity),
            _yes(row.duckdb_production_safe),
            row.polars_source or "-",
        ]) + " |"
        for row in rows
    ]
    (docs / "backend_coverage.md").write_text("\n".join(header + body) + "\n", encoding="utf-8")

    sql_rows = [row for row in rows if row.duckdb_emitter_implemented]
    sql_doc = [
        "# SQL pushdown coverage",
        "",
        "> Generated from exact active runtime canonicals. Historical aliases do not inherit certification.",
        "> ClickHouse production certification remains separate and is never inferred from DuckDB.",
        "",
        f"- active SQL emitter implementations: **{len(sql_rows)}**",
        f"- DuckDB reference parity: **{totals['duckdb_verified']}**",
        f"- DuckDB production safe: **{totals['duckdb_safe']}**",
        "",
        "| canonical | implemented | reference parity | production safe |",
        "|---|---:|---:|---:|",
    ]
    sql_doc.extend(
        f"| {row.canonical} | yes | {_yes(row.duckdb_reference_parity)} | {_yes(row.duckdb_production_safe)} |"
        for row in sql_rows
    )
    (docs / "sql_pushdown_coverage.md").write_text("\n".join(sql_doc) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
