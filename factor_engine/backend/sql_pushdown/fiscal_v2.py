# -*- coding: utf-8 -*-
"""Revision-aware exact-fiscal-ordinal DuckDB lowering.

The legacy SQL implementation ordered periods by first observation.  That is
incorrect for out-of-order filings and revisions.  These lowerings match the
strict Pandas/Polars contract: target periods are selected by fiscal ordinal and
only values visible at the current timestamp may be used.
"""
from __future__ import annotations

from typing import Any

_APPLIED = False
_ORIGINAL_COMPILE_LAYER = None


def _raw(node, key: str, input_index: int, default: Any) -> Any:
    attrs = node.attrs or {}
    if key in attrs and attrs[key] is not None:
        return attrs[key]
    if input_index < len(node.inputs):
        child = node.inputs[input_index]
        if child.op == "literal":
            return child.attrs.get("value", default)
    return default


def _ordinal_sql(value: str) -> str:
    text = f"UPPER(TRIM(CAST({value} AS VARCHAR)))"
    year = f"TRY_CAST(regexp_extract({text}, '(\\d{{4}})', 1) AS BIGINT)"
    quarter = f"TRY_CAST(regexp_extract({text}, '([1-4])$', 1) AS BIGINT)"
    numeric = f"TRY_CAST({value} AS BIGINT)"
    date_value = f"TRY_CAST({value} AS DATE)"
    encoded_year = f"FLOOR(({numeric}) / 10)"
    encoded_quarter = f"MOD(({numeric}), 10)"
    return (
        "CASE "
        f"WHEN {year} >= 1000 AND {quarter} BETWEEN 1 AND 4 "
        f"THEN {year} * 4 + {quarter} - 1 "
        f"WHEN {date_value} IS NOT NULL "
        f"THEN YEAR({date_value}) * 4 + FLOOR((MONTH({date_value}) - 1) / 3) "
        f"WHEN {encoded_year} >= 1000 AND {encoded_quarter} BETWEEN 1 AND 4 "
        f"THEN {encoded_year} * 4 + {encoded_quarter} - 1 "
        f"ELSE {numeric} END"
    )


def _revision_policy(node, input_index: int) -> str:
    policy = str(_raw(node, "revision_policy", input_index, "latest_available")).lower()
    if policy not in {"latest_available", "first_available"}:
        raise ValueError("revision_policy must be latest_available or first_available")
    return policy


def _compile_period_lag(node, dialect):
    from factor_engine.backend.sql_pushdown import emitter

    if dialect != emitter.SqlDialect.DUCKDB or len(node.inputs) < 2:
        return None
    value = emitter._compile_layer(node.inputs[0], dialect=dialect)
    period = emitter._compile_layer(node.inputs[1], dialect=dialect)
    if value is None or period is None:
        return None
    periods = int(_raw(node, "periods", 2, 1))
    if periods < 0:
        return None
    policy = _revision_policy(node, 3)
    aggregate = "arg_min" if policy == "first_available" else "arg_max"
    joined = (
        f"SELECT x.ts, x.inst, x._v AS xv, p._v AS pid, "
        f"{_ordinal_sql('p._v')} AS pord "
        f"FROM ({value.sql}) x LEFT JOIN ({period.sql}) p USING (ts, inst)"
    )
    valid_history = emitter._duckdb_valid("h.xv")
    return emitter._Layer(
        f"SELECT r.ts, r.inst, CASE WHEN r.pord IS NULL THEN NULL ELSE ("
        f"SELECT {aggregate}(h.xv, h.ts) FROM ({joined}) h "
        f"WHERE h.inst = r.inst AND h.ts <= r.ts "
        f"AND h.pord = r.pord - {periods} AND {valid_history}"
        f") END AS _v FROM ({joined}) r",
        has_inst_window=True,
    )


def _lag_node(node, periods: int, policy: str):
    from factor_engine.planner.logical_plan import PlanNode

    return PlanNode(
        op="period_lag",
        inputs=[node.inputs[0], node.inputs[1]],
        attrs={"periods": periods, "revision_policy": policy},
    )


def _compile_change_cagr_yoy(node, dialect):
    from factor_engine.backend.sql_pushdown import emitter

    op = node.op
    if dialect != emitter.SqlDialect.DUCKDB or len(node.inputs) < 2:
        return None
    current = emitter._compile_layer(node.inputs[0], dialect=dialect)
    default_periods = 4 if op == "yoy_by_period" else 1
    periods = int(_raw(node, "periods", 2, default_periods))
    if periods < 1:
        return None
    revision_policy_index = {"period_change": 5, "period_cagr": 6, "yoy_by_period": 5}[op]
    revision_policy = _revision_policy(node, revision_policy_index)
    previous = emitter._compile_layer(
        _lag_node(node, periods, revision_policy), dialect=dialect
    )
    if current is None or previous is None:
        return None
    joined = (
        f"SELECT x.ts, x.inst, x._v AS xv, p._v AS pv "
        f"FROM ({current.sql}) x LEFT JOIN ({previous.sql}) p USING (ts, inst)"
    )
    valid = f"({emitter._duckdb_valid('xv')} AND {emitter._duckdb_valid('pv')})"
    if op == "period_change":
        mode = str(_raw(node, "mode", 3, "absolute")).lower()
        if mode == "absolute":
            expr = "xv - pv"
        elif mode == "ratio":
            valid += " AND ABS(pv) > 1e-12"
            expr = "xv / pv - 1.0"
        elif mode == "log":
            valid += " AND ABS(pv) > 1e-12 AND xv / pv > 0"
            expr = "LN(xv / pv)"
        else:
            return None
    elif op == "period_cagr":
        ppy = int(_raw(node, "periods_per_year", 3, 4))
        if ppy < 1:
            return None
        sign_policy = str(_raw(node, "sign_policy", 4, "strict")).lower()
        if sign_policy == "strict":
            valid += " AND xv > 0 AND pv > 0"
            ratio = "xv / pv"
        elif sign_policy == "absolute":
            valid += " AND ABS(pv) > 1e-12"
            ratio = "ABS(xv) / ABS(pv)"
        else:
            return None
        expr = f"POW({ratio}, {float(ppy) / float(periods)}) - 1.0"
    else:
        denominator = str(_raw(node, "denominator", 3, "signed")).lower()
        if denominator not in {"signed", "absolute"}:
            return None
        valid += " AND ABS(pv) > 1e-12"
        denom = "pv" if denominator == "signed" else "ABS(pv)"
        expr = f"(xv - pv) / {denom}"
    return emitter._Layer(
        f"SELECT ts, inst, CASE WHEN {valid} THEN {expr} ELSE NULL END AS _v "
        f"FROM ({joined}) fp",
        has_inst_window=True,
    )


def _compile_average_ttm(node, dialect):
    from factor_engine.backend.sql_pushdown import emitter

    op = node.op
    if dialect != emitter.SqlDialect.DUCKDB or len(node.inputs) < 2:
        return None
    default = 2 if op == "period_average" else 4
    count = int(_raw(node, "periods", 2, default))
    if count < 1:
        return None
    revision_policy = _revision_policy(node, 4)
    layers = [emitter._compile_layer(node.inputs[0], dialect=dialect)]
    layers.extend(
        emitter._compile_layer(
            _lag_node(node, lag, revision_policy), dialect=dialect
        )
        for lag in range(1, count)
    )
    if any(layer is None for layer in layers):
        return None
    aliases = [f"v{i}" for i in range(count)]
    sql = f"SELECT b.ts, b.inst, b._v AS {aliases[0]} FROM ({layers[0].sql}) b"
    for index in range(1, count):
        sql = (
            f"SELECT j.*, p._v AS {aliases[index]} FROM ({sql}) j "
            f"LEFT JOIN ({layers[index].sql}) p USING (ts, inst)"
        )
    valid = " AND ".join(emitter._duckdb_valid(alias) for alias in aliases)
    total = " + ".join(aliases)
    expr = f"({total}) / {float(count)}" if op == "period_average" else f"({total})"
    return emitter._Layer(
        f"SELECT ts, inst, CASE WHEN {valid} THEN {expr} ELSE NULL END AS _v "
        f"FROM ({sql}) fs",
        has_inst_window=True,
    )


def _compile_quarter(node, dialect):
    from factor_engine.backend.sql_pushdown import emitter
    from factor_engine.planner.logical_plan import PlanNode

    if dialect != emitter.SqlDialect.DUCKDB or len(node.inputs) < 3:
        return None
    revision_policy = _revision_policy(node, 3)
    current = emitter._compile_layer(node.inputs[0], dialect=dialect)
    period = emitter._compile_layer(node.inputs[1], dialect=dialect)
    quarter = emitter._compile_layer(node.inputs[2], dialect=dialect)
    previous = emitter._compile_layer(
        PlanNode(
            op="period_lag",
            inputs=[node.inputs[0], node.inputs[1]],
            attrs={"periods": 1, "revision_policy": revision_policy},
        ),
        dialect=dialect,
    )
    if any(layer is None for layer in (current, period, quarter, previous)):
        return None
    joined = (
        f"SELECT x.ts, x.inst, x._v AS xv, q._v AS qv, p._v AS pv, "
        f"{_ordinal_sql('pid._v')} AS pord "
        f"FROM ({current.sql}) x LEFT JOIN ({period.sql}) pid USING (ts, inst) "
        f"LEFT JOIN ({quarter.sql}) q USING (ts, inst) "
        f"LEFT JOIN ({previous.sql}) p USING (ts, inst)"
    )
    expr = (
        "CASE WHEN qv = 1 THEN xv "
        "WHEN qv BETWEEN 2 AND 4 AND pv IS NOT NULL THEN xv - pv ELSE NULL END"
    )
    valid = f"{emitter._duckdb_valid('xv')} AND pord IS NOT NULL"
    return emitter._Layer(
        f"SELECT ts, inst, CASE WHEN {valid} THEN {expr} ELSE NULL END AS _v "
        f"FROM ({joined}) fq",
        has_inst_window=True,
    )


def _compile_ttm_cumulative(node, dialect):
    from factor_engine.backend.sql_pushdown import emitter
    from factor_engine.planner.logical_plan import PlanNode

    if dialect != emitter.SqlDialect.DUCKDB or len(node.inputs) < 3:
        return None
    policy = _revision_policy(node, 3)
    quarterly = PlanNode(
        op="quarter_from_cumulative",
        inputs=list(node.inputs[:3]),
        attrs={"revision_policy": policy},
    )
    return emitter._compile_layer(
        PlanNode(
            op="ttm_from_quarterly",
            inputs=[quarterly, node.inputs[1]],
            attrs={
                "periods": 4,
                "require_consecutive": True,
                "revision_policy": policy,
            },
        ),
        dialect=dialect,
    )


def _compile_fiscal(node, dialect):
    if node.op == "period_lag":
        return _compile_period_lag(node, dialect)
    if node.op in {"period_change", "period_cagr", "yoy_by_period"}:
        return _compile_change_cagr_yoy(node, dialect)
    if node.op in {"period_average", "ttm_from_quarterly"}:
        return _compile_average_ttm(node, dialect)
    if node.op == "quarter_from_cumulative":
        return _compile_quarter(node, dialect)
    if node.op == "ttm_from_cumulative":
        return _compile_ttm_cumulative(node, dialect)
    return None


def apply_fiscal_sql_v2() -> None:
    global _APPLIED, _ORIGINAL_COMPILE_LAYER
    if _APPLIED:
        return
    from factor_engine.backend.sql_pushdown import emitter

    _ORIGINAL_COMPILE_LAYER = emitter._compile_layer

    def compile_layer(node, *, dialect):
        if node.op in {
            "period_lag",
            "period_change",
            "period_average",
            "period_cagr",
            "quarter_from_cumulative",
            "ttm_from_quarterly",
            "ttm_from_cumulative",
            "yoy_by_period",
        }:
            return _compile_fiscal(node, dialect)
        return _ORIGINAL_COMPILE_LAYER(node, dialect=dialect)

    emitter._compile_layer = compile_layer
    _APPLIED = True
