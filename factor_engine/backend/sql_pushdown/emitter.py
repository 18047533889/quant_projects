# -*- coding: utf-8 -*-
"""PlanNode → SQL 编译（长表语义；支持 DuckDB / ClickHouse 方言）。"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from cleaned_operators.registry import OperatorRegistry
from planner.logical_plan import PlanNode


class SqlDialect(str, Enum):
    DUCKDB = "duckdb"
    CLICKHOUSE = "clickhouse"


# 可通过 SQL 下推的 canonical（与 sql_registry 同步）
from backend.sql_pushdown.sql_registry import SQL_CAPABLE_CANONICALS as SQL_CAPABLE_OPS  # noqa: E402


@dataclass(frozen=True)
class SqlPushdownFilter:
    """下推至 base CTE 的过滤条件（与 DataAccessSource 对齐）。"""

    time_column: str | None = None
    start: Any = None
    end: Any = None
    instrument_column: str | None = None
    instruments: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompiledSql:
    query: str
    read_datasets: tuple[str, ...]
    referenced_columns: frozenset[str]
    dialect: SqlDialect = SqlDialect.DUCKDB
    table: str | None = None  # ClickHouse 直读表名


def _resolve_canonical(op: str) -> str:
    return OperatorRegistry._aliases.get(op, op)


def _window_int(node: PlanNode, default: int = 20) -> int:
    attrs = node.attrs
    for key in ("d", "window", "n", "periods"):
        if key in attrs and attrs[key] is not None:
            return max(int(attrs[key]), 1)
    for child in node.inputs:
        if child.op == "literal":
            val = child.attrs.get("value")
            if val is not None:
                try:
                    return max(int(val), 1)
                except (TypeError, ValueError):
                    pass
    return default


def _float_attr(node: PlanNode, *keys: str, default: float) -> float:
    for key in keys:
        if key in node.attrs and node.attrs[key] is not None:
            return float(node.attrs[key])
    return default


def _quote_ident(name: str) -> str:
    safe = name.replace('"', '""')
    return f'"{safe}"'


def _sql_literal(val: Any) -> str:
    if val is None:
        return "NULL"
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, (int, float)):
        return str(val)
    s = str(val).replace("'", "''")
    return f"'{s}'"


def _dialect_fn(dialect: SqlDialect, name: str) -> str:
    if dialect == SqlDialect.CLICKHOUSE:
        mapping = {
            "stddev": "stddevSamp",
            "ln": "log",
            "greatest": "greatest",
            "least": "least",
            "nullif": "nullIf",
            "abs": "abs",
            "exp": "exp",
            "sqrt": "sqrt",
        }
        return mapping.get(name, name)
    mapping = {
        "stddev": "STDDEV_SAMP",
        "ln": "ln",
        "greatest": "GREATEST",
        "least": "LEAST",
        "nullif": "NULLIF",
        "abs": "abs",
        "exp": "exp",
        "sqrt": "sqrt",
    }
    return mapping.get(name, name)


@dataclass
class _Layer:
    sql: str
    has_inst_window: bool = False
    has_ts_partition: bool = False


def _inst_window(dialect: SqlDialect, w: int, agg: str, inner_sql: str) -> str:
    return (
        f"SELECT ts, inst, "
        f"{agg}(_v) OVER (PARTITION BY inst ORDER BY ts "
        f"ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW) AS _v "
        f"FROM ({inner_sql}) t"
    )


def _compile_layer(node: PlanNode, *, dialect: SqlDialect) -> _Layer | None:
    op = _resolve_canonical(node.op)
    std = _dialect_fn(dialect, "stddev")
    ln = _dialect_fn(dialect, "ln")
    g = _dialect_fn(dialect, "greatest")
    l = _dialect_fn(dialect, "least")
    nf = _dialect_fn(dialect, "nullif")

    if op == "column":
        col = node.attrs.get("name")
        if not col:
            return None
        c = _quote_ident(str(col))
        return _Layer(f"SELECT ts, inst, {c} AS _v FROM base")

    if op == "literal":
        val = node.attrs.get("value")
        lit = _sql_literal(val)
        return _Layer(f"SELECT ts, inst, {lit} AS _v FROM base")

    if op in {"add", "subtract", "multiply", "divide"}:
        if len(node.inputs) != 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        sym = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}[op]
        return _Layer(
            f"SELECT l.ts, l.inst, (l._v {sym} r._v) AS _v "
            f"FROM ({left.sql}) l INNER JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=left.has_inst_window or right.has_inst_window,
            has_ts_partition=left.has_ts_partition or right.has_ts_partition,
        )

    if op == "neg":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, (-_v) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "abs":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        fn = _dialect_fn(dialect, "abs")
        return _Layer(
            f"SELECT ts, inst, {fn}(_v) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "log":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, {ln}(_v) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "exp":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        fn = _dialect_fn(dialect, "exp")
        return _Layer(
            f"SELECT ts, inst, {fn}(_v) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "sqrt":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        fn = _dialect_fn(dialect, "sqrt")
        return _Layer(
            f"SELECT ts, inst, {fn}(_v) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "clip":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        lo = _float_attr(node, "lo", "min", default=-3.0)
        hi = _float_attr(node, "max", "hi", default=3.0)
        return _Layer(
            f"SELECT ts, inst, {g}({lo}, {l}({hi}, _v)) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "ts_mean":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        return _Layer(
            _inst_window(dialect, w, "AVG", inner.sql),
            has_inst_window=True,
        )

    if op == "ts_std":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        return _Layer(
            _inst_window(dialect, w, std, inner.sql),
            has_inst_window=True,
        )

    if op == "ts_sum":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        return _Layer(
            _inst_window(dialect, w, "SUM", inner.sql),
            has_inst_window=True,
        )

    if op == "ts_max":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        return _Layer(
            _inst_window(dialect, w, "MAX", inner.sql),
            has_inst_window=True,
        )

    if op == "ts_min":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        return _Layer(
            _inst_window(dialect, w, "MIN", inner.sql),
            has_inst_window=True,
        )

    if op == "ts_zscore":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        over = (
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        return _Layer(
            f"SELECT ts, inst, "
            f"(_v - AVG(_v) OVER ({over})) / {nf}({std}(_v) OVER ({over}), 0) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_delay":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        lag = _window_int(node, default=1)
        return _Layer(
            f"SELECT ts, inst, "
            f"LAG(_v, {lag}) OVER (PARTITION BY inst ORDER BY ts) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_delta":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        lag = _window_int(node, default=1)
        return _Layer(
            f"SELECT ts, inst, "
            f"(_v - LAG(_v, {lag}) OVER (PARTITION BY inst ORDER BY ts)) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_pct":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        lag = _window_int(node, default=1)
        return _Layer(
            f"SELECT ts, inst, "
            f"(_v / {nf}(LAG(_v, {lag}) OVER (PARTITION BY inst ORDER BY ts), 0) - 1.0) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "rank":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        # 对齐 pandas DataFrame.rank(pct=True, axis=1)：rank / count
        return _Layer(
            f"SELECT ts, inst, "
            f"(RANK() OVER (PARTITION BY ts ORDER BY _v) * 1.0 "
            f"/ COUNT(*) OVER (PARTITION BY ts)) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "zscore":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, "
            f"(_v - AVG(_v) OVER (PARTITION BY ts)) "
            f"/ {nf}({std}(_v) OVER (PARTITION BY ts), 0) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "cs_demean":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, "
            f"(_v - AVG(_v) OVER (PARTITION BY ts)) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "scale":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        to_val = _float_attr(node, "to", default=1.0)
        abs_fn = _dialect_fn(dialect, "abs")
        return _Layer(
            f"SELECT ts, inst, "
            f"(_v * ({to_val} / {g}(SUM({abs_fn}(_v)) OVER (PARTITION BY ts), 1))) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "group_neutralize":
        if len(node.inputs) < 1:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        if len(node.inputs) >= 2:
            grp = _compile_layer(node.inputs[1], dialect=dialect)
            if grp is None:
                return None
            return _Layer(
                f"SELECT x.ts, x.inst, (x._v - AVG(x._v) OVER (PARTITION BY x.ts, g._v)) AS _v "
                f"FROM ({inner.sql}) x "
                f"INNER JOIN ({grp.sql}) g USING (ts, inst)",
                has_inst_window=inner.has_inst_window or grp.has_inst_window,
                has_ts_partition=True,
            )
        return _Layer(
            f"SELECT ts, inst, (_v - AVG(_v) OVER (PARTITION BY ts)) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "ts_corr":
        if len(node.inputs) != 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        w = _window_int(node)
        over = (
            f"PARTITION BY l.inst ORDER BY l.ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        corr_fn = "corr" if dialect == SqlDialect.DUCKDB else "corrStable"
        return _Layer(
            f"SELECT l.ts, l.inst, {corr_fn}(l._v, r._v) OVER ({over}) AS _v "
            f"FROM ({left.sql}) l INNER JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "where" or op == "if_else":
        if len(node.inputs) != 3:
            return None
        cond = _compile_layer(node.inputs[0], dialect=dialect)
        a = _compile_layer(node.inputs[1], dialect=dialect)
        b = _compile_layer(node.inputs[2], dialect=dialect)
        if cond is None or a is None or b is None:
            return None
        return _Layer(
            f"SELECT c.ts, c.inst, "
            f"CASE WHEN c._v THEN a._v ELSE b._v END AS _v "
            f"FROM ({cond.sql}) c "
            f"INNER JOIN ({a.sql}) a USING (ts, inst) "
            f"INNER JOIN ({b.sql}) b USING (ts, inst)",
            has_inst_window=cond.has_inst_window or a.has_inst_window or b.has_inst_window,
            has_ts_partition=cond.has_ts_partition or a.has_ts_partition or b.has_ts_partition,
        )

    if op in {"group_rank", "group_mean", "group_zscore"}:
        if not node.inputs:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        std = _dialect_fn(dialect, "stddev")
        nf = _dialect_fn(dialect, "nullif")
        if len(node.inputs) >= 2:
            grp = _compile_layer(node.inputs[1], dialect=dialect)
            if grp is None:
                return None
            part = "PARTITION BY x.ts, g._v"
            join = (
                f"FROM ({inner.sql}) x INNER JOIN ({grp.sql}) g USING (ts, inst)"
            )
        else:
            part = "PARTITION BY x.ts"
            join = f"FROM ({inner.sql}) x"
        if op == "group_mean":
            expr = f"AVG(x._v) OVER ({part})"
        elif op == "group_zscore":
            expr = f"(x._v - AVG(x._v) OVER ({part})) / {nf}({std}(x._v) OVER ({part}), 0)"
        else:
            expr = (
                f"(RANK() OVER ({part} ORDER BY x._v) * 1.0 "
                f"/ COUNT(*) OVER ({part}))"
            )
        return _Layer(
            f"SELECT x.ts, x.inst, {expr} AS _v {join}",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "ts_median":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        med = "median" if dialect == SqlDialect.DUCKDB else "median"
        return _Layer(
            _inst_window(dialect, w, med, inner.sql),
            has_inst_window=True,
        )

    if op == "ts_var":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        var_fn = "VAR_SAMP" if dialect == SqlDialect.DUCKDB else "varSamp"
        return _Layer(
            _inst_window(dialect, w, var_fn, inner.sql),
            has_inst_window=True,
        )

    if op == "winsorize":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        a = _float_attr(node, "a", "p", default=0.05)
        lo = a
        hi = 1.0 - a
        g = _dialect_fn(dialect, "greatest")
        l = _dialect_fn(dialect, "least")
        return _Layer(
            f"SELECT ts, inst, "
            f"{g}(quantile_cont(_v, {lo}) OVER (PARTITION BY ts), "
            f"{l}(quantile_cont(_v, {hi}) OVER (PARTITION BY ts), _v)) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "group_winsorize":
        if not node.inputs:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        a = _float_attr(node, "a", "p", default=0.05)
        lo = a
        hi = 1.0 - a
        g = _dialect_fn(dialect, "greatest")
        l = _dialect_fn(dialect, "least")
        if len(node.inputs) >= 2:
            grp = _compile_layer(node.inputs[1], dialect=dialect)
            if grp is None:
                return None
            part = "PARTITION BY x.ts, g._v"
            join = f"FROM ({inner.sql}) x INNER JOIN ({grp.sql}) g USING (ts, inst)"
        else:
            part = "PARTITION BY x.ts"
            join = f"FROM ({inner.sql}) x"
        return _Layer(
            f"SELECT x.ts, x.inst, "
            f"{g}(quantile_cont(x._v, {lo}) OVER ({part}), "
            f"{l}(quantile_cont(x._v, {hi}) OVER ({part}), x._v)) AS _v "
            f"{join}",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "ts_beta":
        if len(node.inputs) != 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        w = _window_int(node)
        over = (
            f"PARTITION BY l.inst ORDER BY l.ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        if dialect == SqlDialect.DUCKDB:
            cov_fn = "covar_samp"
            var_fn = "var_samp"
        else:
            cov_fn = "covarSamp"
            var_fn = "varSamp"
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"SELECT l.ts, l.inst, "
            f"({cov_fn}(l._v, r._v) OVER ({over})) / "
            f"{nf}({var_fn}(r._v) OVER ({over}), 0) AS _v "
            f"FROM ({left.sql}) l INNER JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "ts_mad":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        med = "median" if dialect == SqlDialect.DUCKDB else "median"
        over = (
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        return _Layer(
            f"SELECT ts, inst, "
            f"AVG(ABS(_v - med)) OVER ({over}) AS _v "
            f"FROM ("
            f"SELECT ts, inst, _v, {med}(_v) OVER ({over}) AS med "
            f"FROM ({inner.sql}) t0"
            f") t",
            has_inst_window=True,
        )

    if op == "ts_ema":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        alpha = 2.0 / (float(w) + 1.0)
        over = (
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        if dialect == SqlDialect.CLICKHOUSE:
            return _Layer(
                f"SELECT ts, inst, "
                f"exponentialMovingAverage(_v, {alpha}) OVER (PARTITION BY inst ORDER BY ts) AS _v "
                f"FROM ({inner.sql}) t",
                has_inst_window=True,
            )
        decay = 1.0 - alpha
        return _Layer(
            f"SELECT ts, inst, "
            f"SUM(_v * POW({decay}, rn)) OVER ({over}) / "
            f"NULLIF(SUM(POW({decay}, rn)) OVER ({over}), 0) AS _v "
            f"FROM ("
            f"SELECT ts, inst, _v, "
            f"(COUNT(*) OVER ({over}) - 1 - "
            f"(ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts) - "
            f"MIN(ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts)) OVER ({over}))) AS rn "
            f"FROM ({inner.sql}) t0"
            f") t",
            has_inst_window=True,
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
    """计划树是否全部由 SQL backend 支持（委托 sql_registry）。"""
    from backend.sql_pushdown.sql_registry import is_sql_capable

    return is_sql_capable(plan)


def _build_filter_clause(
    filt: SqlPushdownFilter | None,
    *,
    dialect: SqlDialect,
) -> str:
    if filt is None:
        return ""
    parts: list[str] = []
    if filt.time_column and filt.start is not None:
        tc = _quote_ident(filt.time_column)
        parts.append(f"{tc} >= {_sql_literal(filt.start)}")
    if filt.time_column and filt.end is not None:
        tc = _quote_ident(filt.time_column)
        parts.append(f"{tc} <= {_sql_literal(filt.end)}")
    if filt.instrument_column and filt.instruments:
        ic = _quote_ident(filt.instrument_column)
        insts = ", ".join(_sql_literal(x) for x in filt.instruments)
        parts.append(f"{ic} IN ({insts})")
    if not parts:
        return ""
    return " WHERE " + " AND ".join(parts)


def _build_base_cte(
    *,
    source_from: str,
    time_column: str,
    instrument_column: str,
    columns: Sequence[str],
    filt: SqlPushdownFilter | None,
    dialect: SqlDialect,
) -> str:
    col_list = ", ".join(_quote_ident(c) for c in sorted(columns))
    where = _build_filter_clause(filt, dialect=dialect)
    return (
        f"base AS (SELECT {_quote_ident(time_column)} AS ts, "
        f"{_quote_ident(instrument_column)} AS inst, {col_list} "
        f"FROM {source_from}{where})"
    )


def compile_plan_to_sql(
    plan: PlanNode,
    *,
    dataset: str | None = None,
    table: str | None = None,
    time_column: str,
    instrument_column: str,
    filt: SqlPushdownFilter | None = None,
    dialect: SqlDialect = SqlDialect.DUCKDB,
) -> CompiledSql | None:
    """编译为 SQL；DuckDB 用 registry 数据集名，ClickHouse 用物理表名。"""
    if not plan_is_sql_capable(plan):
        return None

    layer = _compile_layer(plan, dialect=dialect)
    if layer is None:
        return None

    cols: set[str] = set()
    _collect_columns(plan, cols)
    if not cols:
        return None

    source_from = table if dialect == SqlDialect.CLICKHOUSE else (dataset or table)
    if not source_from:
        return None

    push_filter = filt or SqlPushdownFilter(
        time_column=time_column,
        instrument_column=instrument_column,
    )
    base = _build_base_cte(
        source_from=source_from,
        time_column=time_column,
        instrument_column=instrument_column,
        columns=sorted(cols),
        filt=push_filter,
        dialect=dialect,
    )
    query = (
        f"WITH {base} "
        f"SELECT ts, inst, _v AS value FROM ({layer.sql}) result "
        f"ORDER BY ts, inst"
    )
    read_datasets = (dataset,) if dataset and dialect == SqlDialect.DUCKDB else tuple()
    return CompiledSql(
        query=query,
        read_datasets=read_datasets,
        referenced_columns=frozenset(cols),
        dialect=dialect,
        table=table if dialect == SqlDialect.CLICKHOUSE else None,
    )


@dataclass(frozen=True)
class BatchCompiledSql:
    """多子树单条 SQL（WITH CSE 批执行）。"""

    query: str
    read_datasets: tuple[str, ...]
    referenced_columns: frozenset[str]
    column_aliases: tuple[tuple[str, str], ...]  # (sid, sql_alias)
    dialect: SqlDialect = SqlDialect.DUCKDB
    table: str | None = None


def compile_plans_batch_to_sql(
    plans: dict[str, PlanNode],
    *,
    dataset: str | None = None,
    table: str | None = None,
    time_column: str,
    instrument_column: str,
    filt: SqlPushdownFilter | None = None,
    dialect: SqlDialect = SqlDialect.DUCKDB,
) -> BatchCompiledSql | None:
    """将多个 SQL 可编译子树合并为一条 WITH 查询（共享 base CTE）。"""
    if not plans:
        return None
    if len(plans) == 1:
        sid, plan = next(iter(plans.items()))
        single = compile_plan_to_sql(
            plan,
            dataset=dataset,
            table=table,
            time_column=time_column,
            instrument_column=instrument_column,
            filt=filt,
            dialect=dialect,
        )
        if single is None:
            return None
        return BatchCompiledSql(
            query=single.query,
            read_datasets=single.read_datasets,
            referenced_columns=single.referenced_columns,
            column_aliases=((sid, "value"),),
            dialect=single.dialect,
            table=single.table,
        )

    cols: set[str] = set()
    layers: list[tuple[str, str, _Layer]] = []
    for sid, plan in plans.items():
        if not plan_is_sql_capable(plan):
            return None
        layer = _compile_layer(plan, dialect=dialect)
        if layer is None:
            return None
        _collect_columns(plan, cols)
        alias = f"v_{len(layers)}"
        layers.append((sid, alias, layer))

    if not cols:
        return None

    source_from = table if dialect == SqlDialect.CLICKHOUSE else (dataset or table)
    if not source_from:
        return None

    push_filter = filt or SqlPushdownFilter(
        time_column=time_column,
        instrument_column=instrument_column,
    )
    base = _build_base_cte(
        source_from=source_from,
        time_column=time_column,
        instrument_column=instrument_column,
        columns=sorted(cols),
        filt=push_filter,
        dialect=dialect,
    )

    sub_ctes: list[str] = []
    alias_map: list[tuple[str, str]] = []
    for sid, col_alias, layer in layers:
        sub_name = f"sub_{col_alias}"
        sub_ctes.append(
            f"{sub_name} AS (SELECT ts, inst, _v AS {col_alias} FROM ({layer.sql}) t)"
        )
        alias_map.append((sid, col_alias))

    first = f"sub_v_0"
    join_from = first
    select_cols = [f"{first}.ts", f"{first}.inst"]
    for _sid, col_alias in alias_map:
        sub_name = f"sub_{col_alias}"
        select_cols.append(f"{sub_name}.{col_alias}")
        if sub_name != first:
            join_from += f" INNER JOIN {sub_name} USING (ts, inst)"

    with_body = ", ".join([base] + sub_ctes)
    query = (
        f"WITH {with_body} "
        f"SELECT {', '.join(select_cols)} FROM {join_from} "
        f"ORDER BY {first}.ts, {first}.inst"
    )
    read_datasets = (dataset,) if dataset and dialect == SqlDialect.DUCKDB else tuple()
    return BatchCompiledSql(
        query=query,
        read_datasets=read_datasets,
        referenced_columns=frozenset(cols),
        column_aliases=tuple(alias_map),
        dialect=dialect,
        table=table if dialect == SqlDialect.CLICKHOUSE else None,
    )
