# -*- coding: utf-8 -*-
"""DuckDB SQL semantic corrections from the full operator audit."""
from __future__ import annotations

from typing import Any


_APPLIED = False


def _period_ordinal_sql(value: str) -> str:
    """DuckDB expression matching the strict Pandas/Polars period parser."""
    text = f"TRIM(CAST({value} AS VARCHAR))"
    upper = f"UPPER({text})"
    raw = f"TRY_CAST({text} AS BIGINT)"
    year_q = f"TRY_CAST(REGEXP_EXTRACT({upper}, '^([0-9]{{4}})\\s*Q\\s*([1-4])$', 1) AS BIGINT)"
    quarter_q = f"TRY_CAST(REGEXP_EXTRACT({upper}, '^([0-9]{{4}})\\s*Q\\s*([1-4])$', 2) AS BIGINT)"
    year_y0q = f"TRY_CAST(REGEXP_EXTRACT({upper}, '^([0-9]{{4}})0([1-4])$', 1) AS BIGINT)"
    quarter_y0q = f"TRY_CAST(REGEXP_EXTRACT({upper}, '^([0-9]{{4}})0([1-4])$', 2) AS BIGINT)"
    year_yq = f"TRY_CAST(REGEXP_EXTRACT({upper}, '^([0-9]{{4}})([1-4])$', 1) AS BIGINT)"
    quarter_yq = f"TRY_CAST(REGEXP_EXTRACT({upper}, '^([0-9]{{4}})([1-4])$', 2) AS BIGINT)"
    compact_year = f"TRY_CAST(SUBSTR({text}, 1, 4) AS BIGINT)"
    compact_month = f"TRY_CAST(SUBSTR({text}, 5, 2) AS BIGINT)"
    separated_year = f"TRY_CAST(REGEXP_EXTRACT({text}, '^([0-9]{{4}})[-/]([0-9]{{1,2}})', 1) AS BIGINT)"
    separated_month = f"TRY_CAST(REGEXP_EXTRACT({text}, '^([0-9]{{4}})[-/]([0-9]{{1,2}})', 2) AS BIGINT)"
    return (
        "CASE "
        f"WHEN {value} IS NULL THEN NULL "
        f"WHEN REGEXP_MATCHES({upper}, '^([0-9]{{4}})\\s*Q\\s*([1-4])$') "
        f"THEN {year_q} * 4 + {quarter_q} - 1 "
        f"WHEN REGEXP_MATCHES({upper}, '^([0-9]{{4}})0([1-4])$') "
        f"THEN {year_y0q} * 4 + {quarter_y0q} - 1 "
        f"WHEN REGEXP_MATCHES({upper}, '^([0-9]{{4}})([1-4])$') "
        f"THEN {year_yq} * 4 + {quarter_yq} - 1 "
        f"WHEN REGEXP_MATCHES({text}, '^[0-9]{{8}}$') "
        f"AND {compact_month} BETWEEN 1 AND 12 "
        f"THEN {compact_year} * 4 + FLOOR(({compact_month} - 1) / 3) "
        f"WHEN REGEXP_MATCHES({text}, '^[0-9]{{4}}[-/][0-9]{{1,2}}([-/][0-9]{{1,2}})?') "
        f"AND {separated_month} BETWEEN 1 AND 12 "
        f"THEN {separated_year} * 4 + FLOOR(({separated_month} - 1) / 3) "
        f"WHEN {raw} IS NOT NULL AND ABS({raw}) < 100000 THEN {raw} "
        "ELSE NULL END"
    )


def _exact_nonnegative_int(value: Any, name: str) -> int:
    out = int(value)
    if out < 0 or float(value) != float(out):
        raise ValueError(f"{name} must be a non-negative integer")
    return out


def apply_sql_audit_fixes() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from backend.sql_pushdown import emitter

    original = emitter._compile_layer_impl

    def audited_compile_layer_impl(node, *, dialect):
        op = emitter._resolve_canonical(node.op)
        if op != "period_lag":
            return original(node, dialect=dialect)
        if dialect != emitter.SqlDialect.DUCKDB or len(node.inputs) < 2:
            return None

        value = emitter._compile_layer(node.inputs[0], dialect=dialect)
        period = emitter._compile_layer(node.inputs[1], dialect=dialect)
        if value is None or period is None:
            return None

        raw_periods = node.attrs.get(
            "periods", emitter._raw_literal(node, 2, 1)
        )
        periods = _exact_nonnegative_int(raw_periods, "periods")
        policy = str(
            node.attrs.get(
                "revision_policy",
                emitter._raw_literal(node, 3, "latest_available"),
            )
        ).lower()
        if policy not in {"latest_available", "first_available"}:
            raise ValueError(
                "revision_policy must be latest_available or first_available"
            )

        joined = (
            "SELECT x.ts, x.inst, x._v AS xv, p._v AS pid, "
            f"{_period_ordinal_sql('p._v')} AS ordinal "
            f"FROM ({value.sql}) x "
            f"LEFT JOIN ({period.sql}) p USING (ts, inst)"
        )
        aggregate = "arg_min" if policy == "first_available" else "arg_max"
        valid = emitter._duckdb_valid("h.xv")
        sql = (
            "SELECT r.ts, r.inst, "
            "CASE WHEN r.ordinal IS NULL THEN NULL ELSE ("
            f"SELECT {aggregate}(h.xv, h.ts) FROM ({joined}) h "
            "WHERE h.inst = r.inst AND h.ts <= r.ts "
            f"AND h.ordinal = r.ordinal - {periods} AND {valid}"
            ") END AS _v "
            f"FROM ({joined}) r"
        )
        return emitter._Layer(sql, has_inst_window=True)

    emitter._compile_layer_impl = audited_compile_layer_impl
    _APPLIED = True
