#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate backend coverage from the final runtime registry.

This replaces historical hand-maintained totals.  Registered, parity-verified
and production-safe are deliberately reported as separate states.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FACTOR_ENGINE = ROOT / "factor_engine"
if str(FACTOR_ENGINE) not in sys.path:
    sys.path.insert(0, str(FACTOR_ENGINE))

from cleaned_operators import load_all  # noqa: E402
from cleaned_operators.registry import OperatorRegistry  # noqa: E402


def _yes(value: bool) -> str:
    return "yes" if value else "no"


def main() -> None:
    load_all()
    from cleaned_operators.operator_policy import POLARS_PARITY_VERIFIED, POLARS_PRODUCTION_SAFE
    from cleaned_operators.operator_surface import classify_canonical
    from backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS, SQL_PARITY_VERIFIED_CANONICALS, SQL_PRODUCTION_SAFE_CANONICALS

    catalog = OperatorRegistry.catalog()
    active = sorted(catalog)
    rows = []
    for name in active:
        item = catalog[name]
        backends = set(item.get("backends") or [])
        polars_meta = ((item.get("backend_meta") or {}).get("polars") or {})
        polars_native = "polars" in backends and polars_meta.get("execution_kind") == "expression_native"
        polars_verified = "polars" in backends and name in POLARS_PARITY_VERIFIED
        polars_safe = polars_native and polars_verified and name in POLARS_PRODUCTION_SAFE
        sql_verified = name in SQL_PARITY_VERIFIED_CANONICALS
        sql_safe = sql_verified and name in SQL_PRODUCTION_SAFE_CANONICALS
        rows.append((
            name,
            classify_canonical(name),
            "pandas_numpy" in backends,
            "polars" in backends,
            polars_native,
            polars_verified,
            polars_safe,
            name in SQL_IMPLEMENTED_CANONICALS,
            sql_verified,
            sql_safe,
        ))

    daily = sum(row[1] == "daily" for row in rows)
    polars_registered = sum(row[3] for row in rows)
    polars_native = sum(row[4] for row in rows)
    polars_verified = sum(row[5] for row in rows)
    polars_safe = sum(row[6] for row in rows)
    sql_implemented = sum(row[7] for row in rows)
    sql_verified = sum(row[8] for row in rows)
    sql_safe = sum(row[9] for row in rows)
    dual_safe = sum(row[6] and row[9] for row in rows)

    header = [
        "# Backend capability matrix",
        "",
        "> Generated from the final `OperatorRegistry`; static policy sets are intersected with active canonicals.",
        "> `implemented` does not mean parity verified; `verified` does not mean production-safe.",
        "",
        "## Summary",
        "",
        f"- active canonical: **{len(rows)}**",
        f"- daily surface: **{daily}**",
        f"- Polars registered: **{polars_registered}**",
        f"- Polars non-bridge implementation: **{polars_native}**",
        f"- Polars parity verified: **{polars_verified}**",
        f"- Polars production safe: **{polars_safe}**",
        f"- DuckDB SQL emitter implemented: **{sql_implemented}**",
        f"- DuckDB parity verified: **{sql_verified}**",
        f"- DuckDB production safe: **{sql_safe}**",
        f"- Polars and DuckDB both production safe: **{dual_safe}**",
        "",
        "## Matrix",
        "",
        "| canonical | surface | pandas | polars | polars native | polars verified | polars safe | duckdb implemented | duckdb verified | duckdb safe |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    body = [
        "| " + " | ".join([
            name, surface, *(_yes(value) for value in values)
        ]) + " |"
        for name, surface, *values in rows
    ]
    backend_doc = "\n".join(header + body) + "\n"
    (FACTOR_ENGINE / "docs" / "backend_coverage.md").write_text(backend_doc, encoding="utf-8")

    sql_rows = [row for row in rows if row[7]]
    sql_doc = [
        "# SQL pushdown coverage",
        "",
        "> Generated from active runtime canonicals. SQL means DuckDB emitter support unless a dialect-specific certification says otherwise.",
        "> ClickHouse production certification remains separate and is not inferred from DuckDB.",
        "",
        f"- active SQL emitter implementations: **{len(sql_rows)}**",
        f"- DuckDB parity verified: **{sql_verified}**",
        f"- DuckDB production safe: **{sql_safe}**",
        "",
        "| canonical | implemented | parity verified | production safe |",
        "|---|---:|---:|---:|",
    ]
    sql_doc.extend(
        f"| {name} | yes | {_yes(verified)} | {_yes(safe)} |"
        for name, _surface, _pd, _pl, _native, _pv, _ps, _impl, verified, safe in sql_rows
    )
    (FACTOR_ENGINE / "docs" / "sql_pushdown_coverage.md").write_text("\n".join(sql_doc) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
