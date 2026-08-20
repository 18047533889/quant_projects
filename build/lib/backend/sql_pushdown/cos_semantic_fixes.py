# -*- coding: utf-8 -*-
"""Late SQL semantic fixes shared with audited Pandas/Polars operators."""
from __future__ import annotations

from backend.sql_pushdown import emitter

_INSTALLED = False


def install_sql_semantic_fixes() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    original = emitter._compile_layer_impl

    def fixed(node, *, dialect):
        op = emitter._resolve_canonical(node.op)
        if op == "cs_bucket" and dialect == emitter.SqlDialect.DUCKDB:
            inner = emitter._compile_layer(node.inputs[0], dialect=dialect)
            if inner is None:
                return None
            buckets = int(node.attrs.get("buckets", emitter._raw_literal(node, 1, 10)))
            ascending = bool(node.attrs.get("ascending", emitter._raw_literal(node, 2, True)))
            if buckets < 1:
                return None
            direction = "ASC" if ascending else "DESC"
            valid = emitter._duckdb_valid("_v")
            average_rank = "(_rank + (_ties - 1) / 2.0)"
            rank01 = f"(({average_rank} - 1.0) / NULLIF(_n - 1.0, 0))"
            bucket = f"LEAST({buckets}.0, GREATEST(1.0, FLOOR(({rank01}) * {buckets}) + 1.0))"
            single_bucket = min(float(buckets), max(1.0, float(int(0.5 * buckets) + 1)))
            sql = (
                f"SELECT ts, inst, CASE "
                f"WHEN NOT ({valid}) THEN NULL "
                f"WHEN _n = 1 THEN {single_bucket} "
                f"ELSE {bucket} END AS _v FROM ("
                f"SELECT ts, inst, _v, "
                f"RANK() OVER (PARTITION BY ts ORDER BY CASE WHEN {valid} THEN _v END {direction} NULLS LAST) AS _rank, "
                f"COUNT(*) OVER (PARTITION BY ts, _v) AS _ties, "
                f"SUM(CASE WHEN {valid} THEN 1 ELSE 0 END) OVER (PARTITION BY ts) AS _n "
                f"FROM ({inner.sql}) b0"
                f") b1"
            )
            return emitter._Layer(sql, has_ts_partition=True)
        return original(node, dialect=dialect)

    emitter._compile_layer_impl = fixed
    _INSTALLED = True


__all__ = ["install_sql_semantic_fixes"]
