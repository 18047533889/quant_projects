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

from factor_engine.cleaned_operators import load_all  # noqa: E402
from factor_engine.cleaned_operators.registry import OperatorRegistry  # noqa: E402


def _yes(value: bool) -> str:
    return "yes" if value else "no"


def main() -> None:
    load_all()
    from factor_engine.backend.sql_pushdown.sql_registry import register_sql_backends

    register_sql_backends()
    from factor_engine.cleaned_operators.operator_surface import classify_canonical
    from factor_engine.backend.operator_capability import capability_for

    catalog = OperatorRegistry.catalog()
    active = sorted(catalog)
    rows = []
    for name in active:
        item = catalog[name]
        backends = set(item.get("backends") or [])
        polars_meta = ((item.get("backend_meta") or {}).get("polars") or {})
        execution_kind = str(polars_meta.get("execution_kind", "unsupported"))
        polars = capability_for(name, "polars")
        duckdb = capability_for(name, "duckdb_sql")
        polars_verified = polars.status in {"parity_verified", "production_safe"}
        polars_safe = polars.status == "production_safe"
        sql_verified = duckdb.status in {"parity_verified", "production_safe"}
        sql_safe = duckdb.status == "production_safe"
        rows.append((
            name,
            classify_canonical(name),
            "pandas_numpy" in backends,
            "polars" in backends,
            execution_kind,
            polars_verified,
            polars_safe,
            duckdb.status != "unsupported",
            sql_verified,
            sql_safe,
            polars.supports_lazy,
            polars.supports_streaming,
            polars.materializes_full_panel,
        ))

    daily = sum(row[1] == "daily" for row in rows)
    polars_registered = sum(row[3] for row in rows)
    polars_native = sum(row[4] in {"expression_native", "polars_eager_native"} for row in rows)
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
        "| canonical | surface | pandas | polars | execution kind | polars verified | polars safe | lazy | streaming | full-panel | duckdb implemented | duckdb verified | duckdb safe |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    body = [
        "| " + " | ".join((
            name, surface, _yes(pd_ok), _yes(pl_ok), execution_kind,
            _yes(pl_verified), _yes(pl_safe), _yes(lazy), _yes(streaming),
            _yes(full_panel), _yes(sql_impl), _yes(sql_verified), _yes(sql_safe),
        )) + " |"
        for (
            name, surface, pd_ok, pl_ok, execution_kind, pl_verified, pl_safe,
            sql_impl, sql_verified, sql_safe, lazy, streaming, full_panel,
        ) in rows
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
        for name, _surface, _pd, _pl, _kind, _pv, _ps, _impl, verified, safe, _lazy, _streaming, _full in sql_rows
    )
    (FACTOR_ENGINE / "docs" / "sql_pushdown_coverage.md").write_text("\n".join(sql_doc) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
