# -*- coding: utf-8 -*-
"""PlanNode → DuckDB SQL 编译（长表语义，短期 MVP 子集）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cleaned_operators.registry import OperatorRegistry
from planner.logical_plan import PlanNode

# 可通过 SQL 下推的 canonical（其余走 Python fallback）
SQL_CAPABLE_OPS: frozenset[str] = frozenset(
    {
        "column",
        "literal",
        "add",
        "subtract",
        "multiply",
        "divide",
        "neg",
        "abs",
        "log",
        "sqrt",
        "ts_mean",
        "ts_delay",
        "ts_delta",
        "ts_std",
        "ts_sum",
        "rank",
        "zscore",
    }
)


@dataclass(frozen=True)
class CompiledSql:
    query: str
    read_datasets: tuple[str, ...]
    referenced_columns: frozenset[str]


def _resolve_canonical(op: str) -> str:
    return OperatorRegistry._aliases.get(op, op)


def _window_int(node: PlanNode, default: int = 20) -> int:
    attrs = node.attrs
    for key in ("d", "window", "n", "periods"):
        if key in attrs and attrs[key] is not None:
            return max(int(attrs[key]), 1)
    if node.inputs and node.inputs[0].op == "literal":
        return max(int(node.inputs[0].attrs.get("value", default)), 1)
    return default


def _quote_col(name: str) -> str:
    safe = name.replace('"', '""')
    return f'"{safe}"'


@dataclass
class _Layer:
    sql: str
    has_inst_window: bool = False
    has_ts_partition: bool = False


def _compile_layer(node: PlanNode) -> _Layer | None:
    op = _resolve_canonical(node.op)

    if op == "column":
        col = node.attrs.get("name")
        if not col:
            return None
        c = _quote_col(str(col))
        return _Layer(
            f"SELECT ts, inst, {c} AS _v FROM base",
            has_inst_window=False,
            has_ts_partition=False,
        )

    if op == "literal":
        val = node.attrs.get("value")
        if isinstance(val, bool):
            lit = "TRUE" if val else "FALSE"
        elif val is None:
            lit = "NULL"
        else:
            lit = str(float(val) if isinstance(val, (int, float)) else repr(val))
        return _Layer(
            f"SELECT ts, inst, {lit} AS _v FROM base",
            has_inst_window=False,
            has_ts_partition=False,
        )

    if op in {"add", "subtract", "multiply", "divide"}:
        if len(node.inputs) != 2:
            return None
        left = _compile_layer(node.inputs[0])
        right = _compile_layer(node.inputs[1])
        if left is None or right is None:
            return None
        op_map = {
            "add": "+",
            "subtract": "-",
            "multiply": "*",
            "divide": "/",
        }
        sym = op_map[op]
        return _Layer(
            f"SELECT l.ts, l.inst, (l._v {sym} r._v) AS _v "
            f"FROM ({left.sql}) l INNER JOIN ({right.sql}) r "
            f"USING (ts, inst)",
            has_inst_window=left.has_inst_window or right.has_inst_window,
            has_ts_partition=left.has_ts_partition or right.has_ts_partition,
        )

    if op == "neg":
        inner = _compile_layer(node.inputs[0])
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, (-_v) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op in {"abs", "log", "sqrt"}:
        inner = _compile_layer(node.inputs[0])
        if inner is None:
            return None
        fn = {"abs": "abs", "log": "ln", "sqrt": "sqrt"}[op]
        return _Layer(
            f"SELECT ts, inst, {fn}(_v) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "ts_mean":
        inner = _compile_layer(node.inputs[0])
        if inner is None:
            return None
        w = _window_int(node)
        return _Layer(
            f"SELECT ts, inst, "
            f"AVG(_v) OVER (PARTITION BY inst ORDER BY ts "
            f"ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
            has_ts_partition=False,
        )

    if op == "ts_std":
        inner = _compile_layer(node.inputs[0])
        if inner is None:
            return None
        w = _window_int(node)
        return _Layer(
            f"SELECT ts, inst, "
            f"STDDEV_SAMP(_v) OVER (PARTITION BY inst ORDER BY ts "
            f"ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
            has_ts_partition=False,
        )

    if op == "ts_sum":
        inner = _compile_layer(node.inputs[0])
        if inner is None:
            return None
        w = _window_int(node)
        return _Layer(
            f"SELECT ts, inst, "
            f"SUM(_v) OVER (PARTITION BY inst ORDER BY ts "
            f"ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
            has_ts_partition=False,
        )

    if op == "ts_delay":
        inner = _compile_layer(node.inputs[0])
        if inner is None:
            return None
        lag = _window_int(node, default=1)
        return _Layer(
            f"SELECT ts, inst, "
            f"LAG(_v, {lag}) OVER (PARTITION BY inst ORDER BY ts) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
            has_ts_partition=False,
        )

    if op == "ts_delta":
        inner = _compile_layer(node.inputs[0])
        if inner is None:
            return None
        lag = _window_int(node, default=1)
        return _Layer(
            f"SELECT ts, inst, "
            f"(_v - LAG(_v, {lag}) OVER (PARTITION BY inst ORDER BY ts)) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
            has_ts_partition=False,
        )

    if op == "rank":
        inner = _compile_layer(node.inputs[0])
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, "
            f"(RANK() OVER (PARTITION BY ts ORDER BY _v) - 1.0) "
            f"/ GREATEST(COUNT(*) OVER (PARTITION BY ts) - 1, 1) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "zscore":
        inner = _compile_layer(node.inputs[0])
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, "
            f"(_v - AVG(_v) OVER (PARTITION BY ts)) "
            f"/ NULLIF(STDDEV_SAMP(_v) OVER (PARTITION BY ts), 0) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    return None


def _collect_columns(node: PlanNode, out: set[str]) -> None:
    if node.op == "column":
        name = node.attrs.get("name")
        if name:
            out.add(str(name))
    for child in node.inputs:
        _collect_columns(child, out)


def plan_is_sql_capable(plan: PlanNode) -> bool:
    """计划树是否全部由 SQL_CAPABLE_OPS 组成。"""
    canon = _resolve_canonical(plan.op)
    if canon not in SQL_CAPABLE_OPS:
        return False
    return all(plan_is_sql_capable(c) for c in plan.inputs)


def compile_plan_to_sql(
    plan: PlanNode,
    *,
    dataset: str,
    time_column: str,
    instrument_column: str,
) -> CompiledSql | None:
    """编译为 DuckDB SQL；失败返回 None。"""
    if not plan_is_sql_capable(plan):
        return None

    layer = _compile_layer(plan)
    if layer is None:
        return None

    cols: set[str] = set()
    _collect_columns(plan, cols)
    if not cols:
        return None

    col_list = ", ".join(_quote_col(c) for c in sorted(cols))
    base = (
        f"base AS (SELECT {_quote_col(time_column)} AS ts, "
        f"{_quote_col(instrument_column)} AS inst, {col_list} FROM {dataset})"
    )
    query = (
        f"WITH {base} "
        f"SELECT ts, inst, _v AS value FROM ({layer.sql}) result "
        f"ORDER BY ts, inst"
    )
    return CompiledSql(
        query=query,
        read_datasets=(dataset,),
        referenced_columns=frozenset(cols),
    )
