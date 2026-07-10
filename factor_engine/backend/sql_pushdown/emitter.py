# -*- coding: utf-8 -*-
"""PlanNode → SQL 编译（长表语义；支持 DuckDB / ClickHouse 方言）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from contextvars import ContextVar
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


def _int_attr(node: PlanNode, *keys: str, default: int) -> int:
    for key in keys:
        if key in node.attrs and node.attrs[key] is not None:
            return max(int(node.attrs[key]), 1)
    return default


def _literal_positional(node: PlanNode, index: int, *, default: float | None = None) -> float | None:
    """读取 positional literal 参数（``clip(x, lo, hi)`` 等）。"""
    pos = index + 1
    if pos >= len(node.inputs):
        return default
    child = node.inputs[pos]
    if child.op != "literal":
        return default
    val = child.attrs.get("value")
    if val is None:
        return default
    if val == "zero":
        return 0.0
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    return default


def _clip_bounds(node: PlanNode, *, default_lo: float = -3.0, default_hi: float = 3.0) -> tuple[float, float]:
    lo = _float_attr(node, "lo", "min", default=default_lo)
    hi = _float_attr(node, "max", "hi", default=default_hi)
    pos_lo = _literal_positional(node, 0)
    pos_hi = _literal_positional(node, 1)
    if pos_lo is not None:
        lo = pos_lo
    if pos_hi is not None:
        hi = pos_hi
    return lo, hi


def _truthy_sql(value_col: str) -> str:
    """浮点条件真值（对齐 pandas ``bool(0.0)==False``）。"""
    return f"({value_col} IS NOT NULL AND {value_col} <> 0)"


def _const_fill_value(node: PlanNode, *, default: float | None = None) -> float | None:
    """解析常量填充值（``fillna_const`` / ``nan_to_num`` / ``fillna(..., 0)``）。"""
    if "num" in node.attrs and node.attrs["num"] is not None:
        return float(node.attrs["num"])
    for key in ("value", "fill_value", "const", "c", "method"):
        if key in node.attrs and node.attrs[key] is not None:
            raw = node.attrs[key]
            if raw == "zero":
                return 0.0
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                return float(raw)
    pos = _literal_positional(node, 0)
    if pos is not None:
        return pos
    if len(node.inputs) >= 2 and node.inputs[1].op == "literal":
        raw = node.inputs[1].attrs.get("value")
        if raw == "zero":
            return 0.0
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return float(raw)
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


def _quantile_over(
    dialect: SqlDialect,
    col: str,
    p: float,
    partition: str,
) -> str:
    """窗口分位数：DuckDB ``quantile_cont`` / ClickHouse ``quantileExact``。"""
    if dialect == SqlDialect.CLICKHOUSE:
        return f"quantileExact({p})({col}) OVER ({partition})"
    return f"quantile_cont({col}, {p}) OVER ({partition})"


def _dialect_fn(dialect: SqlDialect, name: str) -> str:
    if dialect == SqlDialect.CLICKHOUSE:
        mapping = {
            "stddev": "stddevSamp",
            "stddev_pop": "stddevPop",
            "ln": "log",
            "greatest": "greatest",
            "least": "least",
            "nullif": "nullIf",
            "abs": "abs",
            "sign": "sign",
            "exp": "exp",
            "sqrt": "sqrt",
        }
        return mapping.get(name, name)
    mapping = {
        "stddev": "STDDEV_SAMP",
        "stddev_pop": "STDDEV_POP",
        "ln": "ln",
        "greatest": "GREATEST",
        "least": "LEAST",
        "nullif": "NULLIF",
        "abs": "abs",
        "sign": "sign",
        "exp": "exp",
        "sqrt": "sqrt",
    }
    return mapping.get(name, name)


@dataclass
class _Layer:
    sql: str
    has_inst_window: bool = False
    has_ts_partition: bool = False


@dataclass
class _SqlCompileMemo:
    """DAG-level CTE 去重：同一 structural_key 只编译一次。"""

    use_counts: dict[str, int]
    refs: dict[str, str] = field(default_factory=dict)
    bodies: list[tuple[str, str]] = field(default_factory=list)
    _counter: int = 0


_sql_memo_ctx: ContextVar[_SqlCompileMemo | None] = ContextVar("_sql_memo_ctx", default=None)


def _structural_use_counts(plan: PlanNode) -> dict[str, int]:
    from planner.plan_hash import structural_key

    counts: dict[str, int] = {}

    def walk(n: PlanNode) -> None:
        key = structural_key(n)
        counts[key] = counts.get(key, 0) + 1
        for child in n.inputs:
            walk(child)

    walk(plan)
    return counts


def _inst_window(dialect: SqlDialect, w: int, agg: str, inner_sql: str) -> str:
    return (
        f"SELECT ts, inst, "
        f"{agg}(_v) OVER (PARTITION BY inst ORDER BY ts "
        f"ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW) AS _v "
        f"FROM ({inner_sql}) t"
    )


def _wilder_ewm_over_inst(inner_sql: str, w: int, *, dialect: SqlDialect, value_col: str = "_v") -> str:
    """Wilder 平滑（alpha=1/w），与 pandas ``ewm(alpha=1/w, adjust=False)`` 对齐。"""
    alpha = 1.0 / float(max(w, 1))
    decay = 1.0 - alpha
    over = (
        f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
    )
    if dialect == SqlDialect.CLICKHOUSE:
        return (
            f"SELECT ts, inst, "
            f"exponentialMovingAverage({value_col}, {alpha}) OVER (PARTITION BY inst ORDER BY ts) AS _v "
            f"FROM ({inner_sql}) t"
        )
    return (
        f"SELECT ts, inst, "
        f"SUM({value_col} * POW({decay}, rn)) OVER ({over}) / "
        f"NULLIF(SUM(POW({decay}, rn)) OVER ({over}), 0) AS _v "
        f"FROM ("
        f"SELECT ts, inst, {value_col}, "
        f"(COUNT(*) OVER ({over}) - 1 - "
        f"(ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts) - "
        f"MIN(ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts)) OVER ({over}))) AS rn "
        f"FROM ({inner_sql}) t0"
        f") t"
    )


def _rsi_wilder_sql(close_sql: str, w: int, *, dialect: SqlDialect) -> str:
    g = _dialect_fn(dialect, "greatest")
    abs_fn = _dialect_fn(dialect, "abs")
    delta = f"(_v - LAG(_v, 1) OVER (PARTITION BY inst ORDER BY ts))"
    if dialect == SqlDialect.CLICKHOUSE:
        gain_expr = f"if(isNull({delta}), NULL, {g}(0, {delta}))"
        loss_expr = f"if(isNull({delta}), NULL, {g}(0, -({delta})))"
    else:
        gain_expr = f"CASE WHEN {delta} IS NULL THEN NULL ELSE {g}(0, {delta}) END"
        loss_expr = f"CASE WHEN {delta} IS NULL THEN NULL ELSE {g}(0, -({delta})) END"
    gain_sql = f"SELECT ts, inst, {gain_expr} AS _v FROM ({close_sql}) c0"
    loss_sql = f"SELECT ts, inst, {loss_expr} AS _v FROM ({close_sql}) c1"
    avg_gain = _wilder_ewm_over_inst(gain_sql, w, dialect=dialect)
    avg_loss = _wilder_ewm_over_inst(loss_sql, w, dialect=dialect)
    if dialect == SqlDialect.CLICKHOUSE:
        rs = "g.gain / nullIf(l.loss, 0)"
        core = f"100 - (100 / (1 + {rs}))"
        return (
            f"SELECT c.ts, c.inst, "
            f"if(l.loss = 0 AND g.gain > 0, 100, "
            f"if(g.gain = 0 AND l.loss > 0, 0, "
            f"if(g.gain = 0 AND l.loss = 0, 50, {core}))) AS _v "
            f"FROM ({close_sql}) c "
            f"INNER JOIN (SELECT ts, inst, _v AS gain FROM ({avg_gain}) g0) g USING (ts, inst) "
            f"INNER JOIN (SELECT ts, inst, _v AS loss FROM ({avg_loss}) l0) l USING (ts, inst)"
        )
    return (
        f"SELECT c.ts, c.inst, "
        f"CASE "
        f"WHEN l.loss = 0 AND g.gain > 0 THEN 100 "
        f"WHEN g.gain = 0 AND l.loss > 0 THEN 0 "
        f"WHEN g.gain = 0 AND l.loss = 0 THEN 50 "
        f"ELSE 100 - (100 / (1 + g.gain / NULLIF(l.loss, 0))) END AS _v "
        f"FROM ({close_sql}) c "
        f"INNER JOIN (SELECT ts, inst, _v AS gain FROM ({avg_gain}) g0) g USING (ts, inst) "
        f"INNER JOIN (SELECT ts, inst, _v AS loss FROM ({avg_loss}) l0) l USING (ts, inst)"
    )


def _atr_wilder_sql(
    high_sql: str,
    low_sql: str,
    close_sql: str,
    w: int,
    *,
    dialect: SqlDialect,
) -> str:
    abs_fn = _dialect_fn(dialect, "abs")
    g = _dialect_fn(dialect, "greatest")
    lag_close = f"LAG(c._v, 1) OVER (PARTITION BY h.inst ORDER BY h.ts)"
    if dialect == SqlDialect.CLICKHOUSE:
        tr_expr = (
            f"if(isNull({lag_close}), NULL, "
            f"{g}(h._v - l._v, {abs_fn}(h._v - {lag_close}), {abs_fn}(l._v - {lag_close})))"
        )
    else:
        tr_expr = (
            f"CASE WHEN {lag_close} IS NULL THEN NULL "
            f"ELSE {g}(h._v - l._v, {abs_fn}(h._v - {lag_close}), {abs_fn}(l._v - {lag_close})) END"
        )
    tr_sql = (
        f"SELECT h.ts, h.inst, {tr_expr} AS _v "
        f"FROM ({high_sql}) h "
        f"INNER JOIN ({low_sql}) l USING (ts, inst) "
        f"INNER JOIN ({close_sql}) c USING (ts, inst)"
    )
    return _wilder_ewm_over_inst(tr_sql, w, dialect=dialect)


def _ffill_over_inst(inner_sql: str, *, dialect: SqlDialect) -> str:
    if dialect == SqlDialect.CLICKHOUSE:
        return (
            f"SELECT ts, inst, "
            f"anyLast(if(isNotNull(_v), _v, NULL)) OVER ("
            f"PARTITION BY inst ORDER BY ts "
            f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS _v "
            f"FROM ({inner_sql}) t"
        )
    return (
        f"SELECT ts, inst, "
        f"LAST_VALUE(_v IGNORE NULLS) OVER ("
        f"PARTITION BY inst ORDER BY ts "
        f"ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS _v "
        f"FROM ({inner_sql}) t"
    )


def _bfill_over_inst(inner_sql: str, *, dialect: SqlDialect) -> str:
    """bfill 在因子链路中为因果算子（不引用未来值），SQL 层透传 inner。"""
    return inner_sql


def _pct_rank_frac(*, value_col: str, partition: str, dialect: SqlDialect) -> str:
    """百分位 rank 分数（0-1）；不处理 NULL。"""
    cnt = f"COUNT({value_col}) OVER ({partition})"
    if dialect == SqlDialect.CLICKHOUSE:
        return (
            f"toFloat64(RANK() OVER ({partition} ORDER BY {value_col})) "
            f"/ nullIf({cnt}, 0)"
        )
    return (
        f"(RANK() OVER ({partition} ORDER BY {value_col} NULLS LAST) * 1.0 "
        f"/ NULLIF({cnt}, 0))"
    )


def _pct_rank_expr(*, value_col: str, partition: str, dialect: SqlDialect) -> str:
    """百分位 rank 表达式；NaN/NULL 保持缺失，分母仅计非空。"""
    frac = _pct_rank_frac(value_col=value_col, partition=partition, dialect=dialect)
    if dialect == SqlDialect.CLICKHOUSE:
        return f"if(isNull({value_col}), NULL, {frac})"
    return f"CASE WHEN {value_col} IS NULL THEN NULL ELSE {frac} END"


def _cs_rank_01_expr(*, value_col: str, partition: str, dialect: SqlDialect) -> str:
    """截面 0-1 排名，对齐 ``cs_rank_01``（单有效值行 → 全行 0.5，含 NULL 格）。"""
    order = f"{partition} ORDER BY {value_col}"
    if dialect != SqlDialect.CLICKHOUSE:
        order = f"{partition} ORDER BY {value_col} NULLS LAST"
    r = f"RANK() OVER ({order})"
    cnt = f"COUNT({value_col}) OVER ({partition})"
    if dialect == SqlDialect.CLICKHOUSE:
        return (
            f"multiIf({cnt} <= 1, 0.5, isNull({value_col}), NULL, "
            f"({r} - 1) / nullIf({cnt} - 1, 0))"
        )
    return (
        f"CASE WHEN {cnt} <= 1 THEN 0.5 "
        f"WHEN {value_col} IS NULL THEN NULL "
        f"ELSE ({r} - 1.0) / NULLIF({cnt} - 1, 0) END"
    )


def _cs_rank_01_sql(inner_sql: str, *, partition: str, dialect: SqlDialect) -> str:
    expr = _cs_rank_01_expr(value_col="_v", partition=partition, dialect=dialect)
    return f"SELECT ts, inst, {expr} AS _v FROM ({inner_sql}) t"


def _cs_pct_rank_sql(inner_sql: str, *, partition: str, dialect: SqlDialect) -> str:
    """截面百分位 rank；NaN 保持 NULL（``rank_pct`` / ``cs_pct_rank`` 语义）。"""
    expr = _pct_rank_expr(value_col="_v", partition=partition, dialect=dialect)
    return f"SELECT ts, inst, {expr} AS _v FROM ({inner_sql}) t"


def _group_zscore_expr(
    *,
    value_col: str,
    partition: str,
    dialect: SqlDialect,
    canon: str = "group_zscore",
) -> str:
    """组内 zscore；零/缺失标准差时输出语义见 ``numeric_semantics``。"""
    from backend.numeric_semantics import sql_stddev_fn_key, zscore_zero_std_fill

    std_fn = _dialect_fn(dialect, sql_stddev_fn_key(canon))
    nf = _dialect_fn(dialect, "nullif")
    zero_fill = zscore_zero_std_fill(canon)
    zero_sql = "0" if zero_fill == 0.0 else ("NULL" if zero_fill is None else "NaN")
    avg = f"AVG({value_col}) OVER ({partition})"
    stdv = f"{std_fn}({value_col}) OVER ({partition})"
    return (
        f"CASE WHEN {value_col} IS NULL THEN NULL "
        f"WHEN {stdv} IS NULL OR {stdv} = 0 THEN {zero_sql} "
        f"ELSE ({value_col} - {avg}) / {nf}({stdv}, 0) END"
    )


def _group_decay_linear_expr(*, value_col: str, partition: str, dialect: SqlDialect) -> str:
    """组内按排名线性衰减权重 × 原值（对齐 pandas group_decay_linear）。"""
    nf = _dialect_fn(dialect, "nullif")
    cnt = f"COUNT({value_col}) OVER ({partition})"
    denom = f"{nf}({cnt} * ({cnt} + 1.0) / 2.0, 0)"
    if dialect == SqlDialect.CLICKHOUSE:
        rank = f"toFloat64(RANK() OVER ({partition} ORDER BY {value_col}))"
    else:
        rank = f"RANK() OVER ({partition} ORDER BY {value_col} NULLS LAST)"
    return (
        f"CASE WHEN {value_col} IS NULL THEN NULL "
        f"ELSE {value_col} * ({rank} * 1.0 / {denom}) END"
    )


def _group_minmax_expr(*, value_col: str, partition: str, dialect: SqlDialect) -> str:
    """组内 [0,1] min-max；零区间输出 0.5（对齐 pandas group_normalize）。"""
    nf = _dialect_fn(dialect, "nullif")
    lo = f"MIN({value_col}) OVER ({partition})"
    hi = f"MAX({value_col}) OVER ({partition})"
    span = f"{nf}({hi} - {lo}, 0)"
    return (
        f"CASE WHEN {value_col} IS NULL THEN NULL "
        f"WHEN {span} IS NULL THEN 0.5 "
        f"ELSE ({value_col} - {lo}) / {span} END"
    )


def _cs_ols_components(
    dialect: SqlDialect,
    *,
    y_col: str,
    x_col: str,
    partition: str,
) -> tuple[str, str, str]:
    """截面 OLS y ~ x + const：返回 (beta_expr, alpha_expr, n_valid_expr)。"""
    if dialect == SqlDialect.DUCKDB:
        cov_fn = "covar_samp"
        var_fn = "var_samp"
    else:
        cov_fn = "covarSamp"
        var_fn = "varSamp"
    nf = _dialect_fn(dialect, "nullif")
    mean_y = f"AVG({y_col}) OVER ({partition})"
    mean_x = f"AVG({x_col}) OVER ({partition})"
    cov_xy = f"{cov_fn}({y_col}, {x_col}) OVER ({partition})"
    var_x = f"{var_fn}({x_col}) OVER ({partition})"
    beta = f"({cov_xy}) / {nf}({var_x}, 0)"
    alpha = f"({mean_y}) - ({beta}) * ({mean_x})"
    n_valid = (
        f"COUNT(CASE WHEN {y_col} IS NOT NULL AND {x_col} IS NOT NULL "
        f"THEN 1 END) OVER ({partition})"
    )
    return beta, alpha, n_valid


def _int_attr(node: PlanNode, *keys: str, input_index: int | None = None, default: int = 0) -> int:
    for key in keys:
        if key in node.attrs and node.attrs[key] is not None:
            return int(node.attrs[key])
    if input_index is not None:
        pos = _literal_positional(node, input_index, default=float(default))
        if pos is not None:
            return int(pos)
    return default


def _linear_decay_over_inst(w: int, inner_sql: str, *, dialect: SqlDialect) -> str:
    """线性衰减加权均值：最近观测权重 ``w``，最早为 ``1``（对齐 ``ts_decay_linear``）。"""
    num_parts: list[str] = []
    den_parts: list[str] = []
    for lag in range(w):
        weight = w - lag
        if lag == 0:
            v = "_v"
        else:
            v = f"LAG(_v, {lag}) OVER (PARTITION BY inst ORDER BY ts)"
        num_parts.append(f"CASE WHEN {v} IS NOT NULL THEN {weight}.0 * {v} ELSE 0 END")
        den_parts.append(f"CASE WHEN {v} IS NOT NULL THEN {weight}.0 ELSE 0 END")
    num = " + ".join(num_parts)
    den = f"{_dialect_fn(dialect, 'nullif')}(" + " + ".join(den_parts) + ", 0)"
    return f"SELECT ts, inst, ({num}) / {den} AS _v FROM ({inner_sql}) t"


def _scalar_from_plan(node: PlanNode, *, default: float | None = None) -> float | None:
    for key in ("value", "fill_value", "const", "c"):
        if key in node.attrs and node.attrs[key] is not None:
            return float(node.attrs[key])
    if len(node.inputs) >= 2 and node.inputs[1].op == "literal":
        val = node.inputs[1].attrs.get("value")
        if val is not None:
            return float(val)
    return default


def _compare_binary_sql(op: str, left_sql: str, right_sql: str, *, dialect: SqlDialect) -> str:
    sym = {"gt": ">", "lt": "<", "eq": "=", "ge": ">=", "le": "<=", "ne": "<>"}[op]
    if dialect == SqlDialect.CLICKHOUSE and op == "ne":
        sym = "!="
    return (
        f"SELECT l.ts, l.inst, "
        f"CASE WHEN l._v IS NULL OR r._v IS NULL THEN NULL "
        f"WHEN l._v {sym} r._v THEN 1.0 ELSE 0.0 END AS _v "
        f"FROM ({left_sql}) l INNER JOIN ({right_sql}) r USING (ts, inst)"
    )


def _rolling_ols_partition(w: int, *, prefix: str = "l") -> str:
    return (
        f"PARTITION BY {prefix}.inst ORDER BY {prefix}.ts "
        f"ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
    )


def _inst_cum_agg(dialect: SqlDialect, agg: str, inner_sql: str) -> str:
    over = "PARTITION BY inst ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
    return (
        f"SELECT ts, inst, "
        f"{agg}(_v) OVER ({over}) AS _v "
        f"FROM ({inner_sql}) t"
    )


def _cs_broadcast_agg(agg: str, inner_sql: str) -> str:
    return (
        f"SELECT ts, inst, "
        f"{agg}(_v) OVER (PARTITION BY ts) AS _v "
        f"FROM ({inner_sql}) t"
    )


def _cs_mad_sql(inner_sql: str, *, dialect: SqlDialect) -> str:
    """截面 MAD：median(|x - median(x)|) 广播到各行。"""
    med_fn = "median"
    return (
        f"SELECT ts, inst, "
        f"{med_fn}(abs(_v - med)) OVER (PARTITION BY ts) AS _v "
        f"FROM ("
        f"SELECT ts, inst, _v, {med_fn}(_v) OVER (PARTITION BY ts) AS med "
        f"FROM ({inner_sql}) t0"
        f") t"
    )


def _cs_mad_zscore_sql(inner_sql: str, *, dialect: SqlDialect) -> str:
    """MAD 稳健 zscore：(x - median) / MAD；MAD=0 → NULL。"""
    med_fn = "median"
    if dialect == SqlDialect.CLICKHOUSE:
        core = (
            f"if(isNull(mad) OR mad = 0, NULL, (_v - med) / mad)"
        )
    else:
        core = (
            f"CASE WHEN mad IS NULL OR mad = 0 THEN NULL "
            f"ELSE (_v - med) / mad END"
        )
    return (
        f"SELECT ts, inst, {core} AS _v "
        f"FROM ("
        f"SELECT ts, inst, _v, med, "
        f"{med_fn}(abs(_v - med)) OVER (PARTITION BY ts) AS mad "
        f"FROM ("
        f"SELECT ts, inst, _v, {med_fn}(_v) OVER (PARTITION BY ts) AS med "
        f"FROM ({inner_sql}) t0"
        f") t1"
        f") t"
    )


def _cum_delta_sql(inner_sql: str, *, dialect: SqlDialect) -> str:
    over = "PARTITION BY inst ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
    first_v = f"FIRST_VALUE(_v IGNORE NULLS) OVER ({over})"
    if dialect == SqlDialect.CLICKHOUSE:
        first_v = f"first_value(_v) IGNORE NULLS OVER ({over})"
        return (
            f"SELECT ts, inst, "
            f"if(isNull(_v) OR isNull({first_v}), NULL, _v - {first_v}) AS _v "
            f"FROM ({inner_sql}) t"
        )
    return (
        f"SELECT ts, inst, "
        f"CASE WHEN _v IS NULL OR {first_v} IS NULL THEN NULL "
        f"ELSE _v - {first_v} END AS _v "
        f"FROM ({inner_sql}) t"
    )


def _ewm_weighted_moment_sql(
    inner_sql: str,
    w: int,
    *,
    dialect: SqlDialect,
    sqrt: bool,
) -> str:
    """指数衰减窗口矩（对齐 ``ewm(span, adjust=False)`` 的有限窗近似）。"""
    alpha = 2.0 / (float(w) + 1.0)
    decay = 1.0 - alpha
    over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
    nf = _dialect_fn(dialect, "nullif")
    g = _dialect_fn(dialect, "greatest")
    sqrt_fn = _dialect_fn(dialect, "sqrt")
    mean = (
        f"SUM(_v * POW({decay}, rn)) OVER ({over}) / "
        f"{nf}(SUM(POW({decay}, rn)) OVER ({over}), 0)"
    )
    mean2 = (
        f"SUM(_v * _v * POW({decay}, rn)) OVER ({over}) / "
        f"{nf}(SUM(POW({decay}, rn)) OVER ({over}), 0)"
    )
    var = f"{g}(0, {mean2} - ({mean}) * ({mean}))"
    val = f"{sqrt_fn}({var})" if sqrt else var
    return (
        f"SELECT ts, inst, {val} AS _v FROM ("
        f"SELECT ts, inst, _v, "
        f"(COUNT(*) OVER ({over}) - 1 - "
        f"(ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts) - "
        f"MIN(ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts)) OVER ({over}))) AS rn "
        f"FROM ({inner_sql}) t0"
        f") t"
    )


def _ewm_cov_corr_sql(
    left_sql: str,
    right_sql: str,
    span: int,
    *,
    dialect: SqlDialect,
    corr: bool,
) -> str:
    """二元 EWM 协方差/相关系数（``adjust=False`` 有限窗近似）。"""
    w = span
    alpha = 2.0 / (float(w) + 1.0)
    decay = 1.0 - alpha
    over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
    nf = _dialect_fn(dialect, "nullif")
    g = _dialect_fn(dialect, "greatest")
    sqrt_fn = _dialect_fn(dialect, "sqrt")
    joined = (
        f"SELECT l.ts, l.inst, l._v AS xv, r._v AS yv, "
        f"(COUNT(*) OVER ({over}) - 1 - "
        f"(ROW_NUMBER() OVER (PARTITION BY l.inst ORDER BY l.ts) - "
        f"MIN(ROW_NUMBER() OVER (PARTITION BY l.inst ORDER BY l.ts)) OVER ({over}))) AS rn "
        f"FROM ({left_sql}) l INNER JOIN ({right_sql}) r USING (ts, inst)"
    )
    wsum = f"SUM(POW({decay}, rn)) OVER ({over})"
    mx = f"SUM(xv * POW({decay}, rn)) OVER ({over}) / {nf}({wsum}, 0)"
    my = f"SUM(yv * POW({decay}, rn)) OVER ({over}) / {nf}({wsum}, 0)"
    mxy = f"SUM(xv * yv * POW({decay}, rn)) OVER ({over}) / {nf}({wsum}, 0)"
    mxx = f"SUM(xv * xv * POW({decay}, rn)) OVER ({over}) / {nf}({wsum}, 0)"
    myy = f"SUM(yv * yv * POW({decay}, rn)) OVER ({over}) / {nf}({wsum}, 0)"
    cov = f"({mxy}) - ({mx}) * ({my})"
    if corr:
        varx = f"{g}(0, {mxx} - ({mx}) * ({mx}))"
        vary = f"{g}(0, {myy} - ({my}) * ({my}))"
        val = f"{cov} / {nf}({sqrt_fn}({varx}) * {sqrt_fn}({vary}), 0)"
    else:
        val = cov
    return f"SELECT ts, inst, {val} AS _v FROM ({joined}) t"


def _rolling_slope_over_inst(w: int, inner_sql: str, *, dialect: SqlDialect) -> str:
    """滚动 OLS 斜率（对齐 ``Slope(x, w)`` / ``rolling_time_slope``）。"""
    t_mean = (w - 1) / 2.0
    denom = sum((i - t_mean) ** 2 for i in range(w))
    num_parts: list[str] = []
    for lag in range(w):
        idx = w - 1 - lag
        weight = (idx - t_mean) / denom if denom else 0.0
        v = "_v" if lag == 0 else f"LAG(_v, {lag}) OVER (PARTITION BY inst ORDER BY ts)"
        num_parts.append(f"CASE WHEN {v} IS NOT NULL THEN ({weight}) * {v} ELSE 0 END")
    num = " + ".join(num_parts)
    return f"SELECT ts, inst, ({num}) AS _v FROM ({inner_sql}) t"


def _ts_argext_sql(inner_sql: str, w: int, *, dialect: SqlDialect, pick: str) -> str:
    """滚动 argmax/argmin 位置（0=窗口内最早 bar，对齐 pandas ``rolling_argmax``）。"""
    over = "PARTITION BY inst ORDER BY ts"
    win = f"{over} ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
    ext_fn = "MAX" if pick == "max" else "MIN"
    cases: list[str] = []
    for lag in range(w - 1, -1, -1):
        val = "_v" if lag == 0 else f"LAG(_v, {lag}) OVER ({over})"
        cases.append(
            f"WHEN {val} IS NOT NULL AND w_ext IS NOT NULL AND ABS({val} - w_ext) < 1e-9 "
            f"THEN (w_cnt - 1.0 - {lag})"
        )
    case_expr = "CASE " + " ".join(cases) + " ELSE 0.0 END"
    return (
        f"SELECT ts, inst, "
        f"CASE WHEN w_ext IS NULL THEN 0.0 ELSE ({case_expr}) END AS _v "
        f"FROM ("
        f"SELECT ts, inst, _v, "
        f"{ext_fn}(_v) OVER ({win}) AS w_ext, "
        f"COUNT(_v) OVER ({win}) AS w_cnt "
        f"FROM ({inner_sql}) t0"
        f") t"
    )


def _compile_layer_impl(node: PlanNode, *, dialect: SqlDialect) -> _Layer | None:
    op = _resolve_canonical(node.op)
    if op == "rolling_beta":
        op = "ts_beta"
    std = _dialect_fn(dialect, "stddev")
    ln = _dialect_fn(dialect, "ln")
    g = _dialect_fn(dialect, "greatest")
    l = _dialect_fn(dialect, "least")
    nf = _dialect_fn(dialect, "nullif")

    if op == "WMA":
        wma_node = PlanNode(op="ts_decay_linear", inputs=list(node.inputs), attrs=dict(node.attrs))
        return _compile_layer(wma_node, dialect=dialect)

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

    if op in {"add", "subtract", "multiply", "divide", "maximum", "minimum"}:
        if len(node.inputs) != 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        if op == "maximum":
            fn = _dialect_fn(dialect, "greatest")
            expr = (
                f"CASE WHEN l._v IS NULL OR r._v IS NULL THEN NULL "
                f"ELSE {fn}(l._v, r._v) END"
            )
        elif op == "minimum":
            fn = _dialect_fn(dialect, "least")
            expr = (
                f"CASE WHEN l._v IS NULL OR r._v IS NULL THEN NULL "
                f"ELSE {fn}(l._v, r._v) END"
            )
        else:
            sym = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}[op]
            expr = f"(l._v {sym} r._v)"
        return _Layer(
            f"SELECT l.ts, l.inst, {expr} AS _v "
            f"FROM ({left.sql}) l INNER JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=left.has_inst_window or right.has_inst_window,
            has_ts_partition=left.has_ts_partition or right.has_ts_partition,
        )

    if op == "protected_div":
        if len(node.inputs) != 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        from backend.numeric_semantics import protected_div_default, protected_epsilon_default

        eps = _float_attr(node, "epsilon", default=protected_epsilon_default())
        default = _float_attr(node, "default", default=protected_div_default())
        abs_fn = _dialect_fn(dialect, "abs")
        coalesce_fn = "coalesce" if dialect == SqlDialect.CLICKHOUSE else "COALESCE"
        lit = _sql_literal(default)
        return _Layer(
            f"SELECT l.ts, l.inst, "
            f"{coalesce_fn}(CASE WHEN l._v IS NULL OR r._v IS NULL THEN NULL "
            f"WHEN {abs_fn}(r._v) <= {eps} THEN NULL "
            f"ELSE l._v / r._v END, {lit}) AS _v "
            f"FROM ({left.sql}) l INNER JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=left.has_inst_window or right.has_inst_window,
            has_ts_partition=left.has_ts_partition or right.has_ts_partition,
        )

    if op == "protected_log":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        eps = _float_attr(node, "epsilon", default=1e-12)
        lit = _sql_literal(eps)
        return _Layer(
            f"SELECT ts, inst, "
            f"{ln}(CASE WHEN _v IS NULL OR _v <= {lit} THEN {lit} ELSE _v END) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "protected_sqrt":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        fn = _dialect_fn(dialect, "sqrt")
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL THEN NULL ELSE {fn}({g}(_v, 0)) END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
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

    if op == "sign":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        fn = _dialect_fn(dialect, "sign")
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
        lo, hi = _clip_bounds(node)
        clip = f"{g}({lo}, {l}({hi}, _v))"
        return _Layer(
            f"SELECT ts, inst, CASE WHEN _v IS NULL THEN NULL ELSE {clip} END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "is_nan":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        if dialect == SqlDialect.CLICKHOUSE:
            pred = "isNaN(_v) OR _v IS NULL"
        else:
            pred = "_v IS NULL OR isnan(_v)"
        return _Layer(
            f"SELECT ts, inst, CASE WHEN {pred} THEN 1.0 ELSE 0.0 END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "is_finite":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        finite_fn = "isFinite" if dialect == SqlDialect.CLICKHOUSE else "isfinite"
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NOT NULL AND {finite_fn}(_v) THEN 1.0 ELSE 0.0 END AS _v "
            f"FROM ({inner.sql}) t",
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

    if op == "ts_sharpe":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        ann = _float_attr(node, "ann_factor", default=252.0)
        sqrt_af = ann**0.5
        over = (
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE "
            f"WHEN {std}(_v) OVER ({over}) = 0 AND AVG(_v) OVER ({over}) > 0 THEN 1e308 "
            f"WHEN {std}(_v) OVER ({over}) = 0 THEN 0 "
            f"ELSE AVG(_v) OVER ({over}) / {nf}({std}(_v) OVER ({over}), 0) "
            f"END * {sqrt_af} AS _v "
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
        return _Layer(
            _cs_rank_01_sql(inner.sql, partition="PARTITION BY ts", dialect=dialect),
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op in {"rank_pct", "cs_pct_rank"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _cs_pct_rank_sql(inner.sql, partition="PARTITION BY ts", dialect=dialect),
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op in {"cs_quantile", "c_percentile"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        p = _float_attr(node, "p", default=0.5)
        pos_p = _literal_positional(node, 1)
        if pos_p is not None:
            p = pos_p
        qexpr = _quantile_over(dialect, "_v", p, "PARTITION BY ts")
        return _Layer(
            f"SELECT ts, inst, {qexpr} AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "log_returns":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        lag = f"LAG(_v) OVER (PARTITION BY inst ORDER BY ts)"
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL OR {lag} IS NULL OR {lag} = 0 THEN NULL "
            f"ELSE {ln}(_v / {lag}) END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "volatility":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node, default=20)
        sqrt_fn = _dialect_fn(dialect, "sqrt")
        scale = f"{sqrt_fn}(252)"
        return _Layer(
            f"SELECT ts, inst, "
            f"{std}(_v) OVER (PARTITION BY inst ORDER BY ts "
            f"ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW) * {scale} AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "vwap":
        if len(node.inputs) < 2:
            return None
        price = _compile_layer(node.inputs[0], dialect=dialect)
        vol = _compile_layer(node.inputs[1], dialect=dialect)
        if price is None or vol is None:
            return None
        w = _window_int(node, default=20)
        over = (
            f"PARTITION BY p.inst ORDER BY p.ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        return _Layer(
            f"SELECT p.ts, p.inst, "
            f"SUM(p._v * v._v) OVER ({over}) / "
            f"{nf}(SUM(v._v) OVER ({over}), 0) AS _v "
            f"FROM ({price.sql}) p INNER JOIN ({vol.sql}) v USING (ts, inst)",
            has_inst_window=True,
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

    if op == "normalize":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        part = "PARTITION BY ts"
        lo = f"MIN(_v) OVER ({part})"
        hi = f"MAX(_v) OVER ({part})"
        span = f"{nf}({hi} - {lo}, 0)"
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL THEN NULL ELSE (_v - {lo}) / {span} END AS _v "
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

    if op == "cs_mad":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _cs_mad_sql(inner.sql, dialect=dialect),
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "cs_mad_zscore":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _cs_mad_zscore_sql(inner.sql, dialect=dialect),
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "c_mean":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _cs_broadcast_agg("AVG", inner.sql),
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "c_sum":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _cs_broadcast_agg("SUM", inner.sql),
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "c_count":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _cs_broadcast_agg("COUNT", inner.sql),
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "c_std":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        std_fn = _dialect_fn(dialect, "stddev")
        return _Layer(
            _cs_broadcast_agg(std_fn, inner.sql),
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op in {"cs_resid", "cs_regression"}:
        if len(node.inputs) < 2:
            return None
        y_layer = _compile_layer(node.inputs[0], dialect=dialect)
        x_layer = _compile_layer(node.inputs[1], dialect=dialect)
        if y_layer is None or x_layer is None:
            return None
        part = "PARTITION BY y.ts"
        beta, alpha, n_valid = _cs_ols_components(
            dialect, y_col="y._v", x_col="x._v", partition=part
        )
        fit = f"({alpha}) + ({beta}) * x._v"
        mode = 0 if op == "cs_resid" else _int_attr(node, "mode", input_index=2, default=0)
        if mode == 1:
            core = beta
        elif mode == 2:
            core = fit
        else:
            core = f"y._v - ({fit})"
        expr = (
            f"CASE WHEN y._v IS NULL OR x._v IS NULL THEN NULL "
            f"WHEN {n_valid} < 3 THEN NULL "
            f"ELSE {core} END"
        )
        return _Layer(
            f"SELECT y.ts, y.inst, {expr} AS _v "
            f"FROM ({y_layer.sql}) y INNER JOIN ({x_layer.sql}) x USING (ts, inst)",
            has_inst_window=y_layer.has_inst_window or x_layer.has_inst_window,
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
        if len(node.inputs) < 2:
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

    if op == "ts_autocorr":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        lag = _int_attr(node, "lag", default=1)
        over = (
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        corr_fn = "corr" if dialect == SqlDialect.DUCKDB else "corrStable"
        return _Layer(
            f"SELECT ts, inst, {corr_fn}(curr_v, lagged_v) OVER ({over}) AS _v "
            f"FROM ("
            f"SELECT ts, inst, _v AS curr_v, "
            f"LAG(_v, {lag}) OVER (PARTITION BY inst ORDER BY ts) AS lagged_v "
            f"FROM ({inner.sql}) inner0"
            f") aligned",
            has_inst_window=True,
        )

    if op == "RSI_WILDER":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = max(_window_int(node, default=14), 2)
        return _Layer(
            _rsi_wilder_sql(inner.sql, w, dialect=dialect),
            has_inst_window=True,
        )

    if op == "ATR_WILDER":
        if len(node.inputs) < 3:
            return None
        high = _compile_layer(node.inputs[0], dialect=dialect)
        low = _compile_layer(node.inputs[1], dialect=dialect)
        close = _compile_layer(node.inputs[2], dialect=dialect)
        if high is None or low is None or close is None:
            return None
        w = max(_window_int(node, default=14), 2)
        return _Layer(
            _atr_wilder_sql(high.sql, low.sql, close.sql, w, dialect=dialect),
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
            f"CASE WHEN {_truthy_sql('c._v')} THEN a._v ELSE b._v END AS _v "
            f"FROM ({cond.sql}) c "
            f"INNER JOIN ({a.sql}) a USING (ts, inst) "
            f"INNER JOIN ({b.sql}) b USING (ts, inst)",
            has_inst_window=cond.has_inst_window or a.has_inst_window or b.has_inst_window,
            has_ts_partition=cond.has_ts_partition or a.has_ts_partition or b.has_ts_partition,
        )

    if op in {"group_rank", "group_mean", "group_zscore", "group_normalize", "group_std"}:
        if not node.inputs:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
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
        elif op == "group_std":
            std_fn = _dialect_fn(dialect, "stddev")
            expr = f"{std_fn}(x._v) OVER ({part})"
        elif op == "group_zscore":
            expr = _group_zscore_expr(value_col="x._v", partition=part, dialect=dialect)
        elif op == "group_normalize":
            expr = _group_minmax_expr(value_col="x._v", partition=part, dialect=dialect)
        else:
            expr = _pct_rank_expr(value_col="x._v", partition=part, dialect=dialect)
        return _Layer(
            f"SELECT x.ts, x.inst, {expr} AS _v {join}",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "group_percentile":
        if not node.inputs:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        p = _float_attr(node, "p", default=0.5)
        pos_p = _literal_positional(node, 1)
        if pos_p is not None:
            p = pos_p
        if len(node.inputs) >= 2:
            grp = _compile_layer(node.inputs[1], dialect=dialect)
            if grp is None:
                return None
            part = "PARTITION BY x.ts, g._v"
            join = f"FROM ({inner.sql}) x INNER JOIN ({grp.sql}) g USING (ts, inst)"
        else:
            part = "PARTITION BY x.ts"
            join = f"FROM ({inner.sql}) x"
        rank_frac = _pct_rank_frac(value_col="x._v", partition=part, dialect=dialect)
        expr = (
            f"CASE WHEN x._v IS NULL THEN 0.0 "
            f"WHEN ({rank_frac}) <= {p} THEN 1.0 ELSE 0.0 END"
        )
        return _Layer(
            f"SELECT x.ts, x.inst, {expr} AS _v {join}",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "group_decay_linear":
        if not node.inputs:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        if len(node.inputs) >= 2:
            grp = _compile_layer(node.inputs[1], dialect=dialect)
            if grp is None:
                return None
            part = "PARTITION BY x.ts, g._v"
            join = f"FROM ({inner.sql}) x INNER JOIN ({grp.sql}) g USING (ts, inst)"
        else:
            part = "PARTITION BY x.ts"
            join = f"FROM ({inner.sql}) x"
        expr = _group_decay_linear_expr(value_col="x._v", partition=part, dialect=dialect)
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
        q_lo = _quantile_over(dialect, "_v", lo, "PARTITION BY ts")
        q_hi = _quantile_over(dialect, "_v", hi, "PARTITION BY ts")
        clip = f"{g}({q_lo}, {l}({q_hi}, _v))"
        return _Layer(
            f"SELECT ts, inst, CASE WHEN _v IS NULL THEN NULL ELSE {clip} END AS _v "
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
        q_lo = _quantile_over(dialect, "x._v", lo, part)
        q_hi = _quantile_over(dialect, "x._v", hi, part)
        clip = f"{g}({q_lo}, {l}({q_hi}, x._v))"
        return _Layer(
            f"SELECT x.ts, x.inst, "
            f"CASE WHEN x._v IS NULL THEN NULL ELSE {clip} END AS _v "
            f"{join}",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "ts_beta":
        if len(node.inputs) < 2:
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

    if op == "ewm_mean":
        span = _window_int(node, default=20)
        ema_node = PlanNode(
            op="ts_ema",
            inputs=list(node.inputs),
            attrs={**node.attrs, "d": span, "window": span, "span": span},
        )
        return _compile_layer(ema_node, dialect=dialect)

    if op == "ts_rank":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        over = (
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        return _Layer(
            _cs_pct_rank_sql(inner.sql, partition=over, dialect=dialect),
            has_inst_window=True,
        )

    if op == "ffill":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _ffill_over_inst(inner.sql, dialect=dialect),
            has_inst_window=True,
        )

    if op == "bfill":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _bfill_over_inst(inner.sql, dialect=dialect),
            has_inst_window=inner.has_inst_window,
        )

    if op in {"fillna_const", "nan_to_num", "fillna"}:
        if not node.inputs:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        if op == "nan_to_num":
            const = _const_fill_value(node, default=0.0)
        elif op == "fillna":
            const = _const_fill_value(node, default=None)
            if const is None:
                return None
        else:
            const = _const_fill_value(node)
            if const is None:
                const = _scalar_from_plan(node, default=0.0)
        lit = _sql_literal(const)
        coalesce = "coalesce" if dialect == SqlDialect.CLICKHOUSE else "COALESCE"
        return _Layer(
            f"SELECT ts, inst, {coalesce}(_v, {lit}) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "coalesce":
        if len(node.inputs) < 2:
            return None
        layers: list[_Layer] = []
        for inp in node.inputs:
            layer = _compile_layer(inp, dialect=dialect)
            if layer is None:
                return None
            layers.append(layer)
        aliases = [f"t{i}" for i in range(len(layers))]
        coalesce_fn = "coalesce" if dialect == SqlDialect.CLICKHOUSE else "COALESCE"
        cols = ", ".join(f"{alias}._v" for alias in aliases)
        join = f"FROM ({layers[0].sql}) {aliases[0]}"
        for alias, layer in zip(aliases[1:], layers[1:], strict=True):
            join += f" INNER JOIN ({layer.sql}) {alias} USING (ts, inst)"
        return _Layer(
            f"SELECT {aliases[0]}.ts, {aliases[0]}.inst, "
            f"{coalesce_fn}({cols}) AS _v {join}",
            has_inst_window=any(layer.has_inst_window for layer in layers),
            has_ts_partition=any(layer.has_ts_partition for layer in layers),
        )

    if op == "ts_decay_linear":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        return _Layer(
            _linear_decay_over_inst(w, inner.sql, dialect=dialect),
            has_inst_window=True,
        )

    if op == "power":
        if len(node.inputs) != 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        return _Layer(
            f"SELECT l.ts, l.inst, POW(l._v, r._v) AS _v "
            f"FROM ({left.sql}) l INNER JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=left.has_inst_window or right.has_inst_window,
            has_ts_partition=left.has_ts_partition or right.has_ts_partition,
        )

    if op in {"gt", "lt", "eq", "ge", "le", "ne"}:
        if len(node.inputs) != 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        return _Layer(
            _compare_binary_sql(op, left.sql, right.sql, dialect=dialect),
            has_inst_window=left.has_inst_window or right.has_inst_window,
            has_ts_partition=left.has_ts_partition or right.has_ts_partition,
        )

    if op == "and_":
        if len(node.inputs) != 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        return _Layer(
            f"SELECT l.ts, l.inst, "
            f"CASE WHEN l._v IS NULL OR r._v IS NULL THEN NULL "
            f"WHEN {_truthy_sql('l._v')} AND {_truthy_sql('r._v')} THEN 1.0 ELSE 0.0 END AS _v "
            f"FROM ({left.sql}) l INNER JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=left.has_inst_window or right.has_inst_window,
            has_ts_partition=left.has_ts_partition or right.has_ts_partition,
        )

    if op == "or_":
        if len(node.inputs) != 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        return _Layer(
            f"SELECT l.ts, l.inst, "
            f"CASE WHEN l._v IS NULL AND r._v IS NULL THEN NULL "
            f"WHEN {_truthy_sql('l._v')} OR {_truthy_sql('r._v')} THEN 1.0 ELSE 0.0 END AS _v "
            f"FROM ({left.sql}) l INNER JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=left.has_inst_window or right.has_inst_window,
            has_ts_partition=left.has_ts_partition or right.has_ts_partition,
        )

    if op == "not_":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL THEN 0.0 "
            f"WHEN {_truthy_sql('_v')} THEN 0.0 ELSE 1.0 END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "ts_cov":
        if len(node.inputs) < 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        w = _window_int(node)
        over = _rolling_ols_partition(w, prefix="l")
        cov_fn = "covar_samp" if dialect == SqlDialect.DUCKDB else "covarSamp"
        return _Layer(
            f"SELECT l.ts, l.inst, {cov_fn}(l._v, r._v) OVER ({over}) AS _v "
            f"FROM ({left.sql}) l INNER JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "ts_quantile":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        p = _float_attr(node, "q", "p", default=0.5)
        pos_p = _literal_positional(node, 1)
        if pos_p is not None:
            p = pos_p
        over = (
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        qexpr = _quantile_over(dialect, "_v", p, over)
        return _Layer(
            f"SELECT ts, inst, {qexpr} AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_product":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        over = (
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        eps = 1e-12
        return _Layer(
            f"SELECT ts, inst, "
            f"EXP(SUM(CASE WHEN _v IS NULL OR _v <= {eps} THEN NULL ELSE {ln}(_v) END) "
            f"OVER ({over})) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_skew":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        over = (
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        skew_fn = "skewness" if dialect == SqlDialect.DUCKDB else "skewSamp"
        return _Layer(
            f"SELECT ts, inst, {skew_fn}(_v) OVER ({over}) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_regression":
        if len(node.inputs) < 2:
            return None
        y_layer = _compile_layer(node.inputs[0], dialect=dialect)
        x_layer = _compile_layer(node.inputs[1], dialect=dialect)
        if y_layer is None or x_layer is None:
            return None
        w = _window_int(node)
        over = _rolling_ols_partition(w, prefix="y")
        beta, alpha, n_valid = _cs_ols_components(
            dialect, y_col="y._v", x_col="x._v", partition=over
        )
        fit = f"({alpha}) + ({beta}) * x._v"
        raw_ret = node.attrs.get("retval", node.attrs.get("mode", "slope"))
        retval = str(raw_ret).lower() if raw_ret is not None else "slope"
        if retval in {"1", "intercept", "alpha"}:
            core = alpha
        elif retval in {"2", "fit", "predict", "prediction"}:
            core = fit
        elif retval in {"0", "resid", "residual", "residuals"}:
            core = f"y._v - ({fit})"
        else:
            core = beta
        expr = (
            f"CASE WHEN y._v IS NULL OR x._v IS NULL THEN NULL "
            f"WHEN {n_valid} < 3 THEN NULL "
            f"ELSE {core} END"
        )
        return _Layer(
            f"SELECT y.ts, y.inst, {expr} AS _v "
            f"FROM ({y_layer.sql}) y INNER JOIN ({x_layer.sql}) x USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "cum_sum":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _inst_cum_agg(dialect, "SUM", inner.sql),
            has_inst_window=True,
        )

    if op == "cum_max":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _inst_cum_agg(dialect, "MAX", inner.sql),
            has_inst_window=True,
        )

    if op == "cum_min":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _inst_cum_agg(dialect, "MIN", inner.sql),
            has_inst_window=True,
        )

    if op == "cum_prod":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        prod_fn = "product"
        return _Layer(
            _inst_cum_agg(dialect, prod_fn, inner.sql),
            has_inst_window=True,
        )

    if op == "cum_delta":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _cum_delta_sql(inner.sql, dialect=dialect),
            has_inst_window=True,
        )

    if op == "expanding_mean":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _inst_cum_agg(dialect, "AVG", inner.sql),
            has_inst_window=True,
        )

    if op == "expanding_sum":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _inst_cum_agg(dialect, "SUM", inner.sql),
            has_inst_window=True,
        )

    if op in {"cum_std", "expanding_std"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        std_fn = _dialect_fn(dialect, "stddev")
        return _Layer(
            _inst_cum_agg(dialect, std_fn, inner.sql),
            has_inst_window=True,
        )

    if op == "count":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _inst_cum_agg(dialect, "COUNT", inner.sql),
            has_inst_window=True,
        )

    if op == "floor":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        fn = "floor"
        return _Layer(
            f"SELECT ts, inst, {fn}(_v) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "ceil":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        fn = "ceil"
        return _Layer(
            f"SELECT ts, inst, {fn}(_v) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "inverse":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL OR _v = 0 THEN NULL ELSE (1.0 / _v) END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "log_abs":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        abs_fn = _dialect_fn(dialect, "abs")
        ln_fn = _dialect_fn(dialect, "ln")
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL THEN NULL ELSE {ln_fn}({abs_fn}(_v)) END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "signed_log":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        abs_fn = _dialect_fn(dialect, "abs")
        ln_fn = _dialect_fn(dialect, "ln")
        sign_fn = _dialect_fn(dialect, "sign")
        eps = 1e-10
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL THEN NULL "
            f"ELSE {sign_fn}(_v) * {ln_fn}({abs_fn}(_v) + {eps}) END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "signed_sqrt":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        abs_fn = _dialect_fn(dialect, "abs")
        sqrt_fn = _dialect_fn(dialect, "sqrt")
        sign_fn = _dialect_fn(dialect, "sign")
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL THEN NULL "
            f"ELSE {sign_fn}(_v) * {sqrt_fn}({abs_fn}(_v)) END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "ewm_std":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        span = _window_int(node, default=20)
        return _Layer(
            _ewm_weighted_moment_sql(inner.sql, span, dialect=dialect, sqrt=True),
            has_inst_window=True,
        )

    if op == "ewm_var":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        span = _window_int(node, default=20)
        return _Layer(
            _ewm_weighted_moment_sql(inner.sql, span, dialect=dialect, sqrt=False),
            has_inst_window=True,
        )

    if op == "Slope":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        return _Layer(
            _rolling_slope_over_inst(w, inner.sql, dialect=dialect),
            has_inst_window=True,
        )

    if op == "ts_argmax":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        return _Layer(
            _ts_argext_sql(inner.sql, w, dialect=dialect, pick="max"),
            has_inst_window=True,
        )

    if op == "ts_argmin":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        return _Layer(
            _ts_argext_sql(inner.sql, w, dialect=dialect, pick="min"),
            has_inst_window=True,
        )

    if op in {"ewm_cov", "ewm_corr"}:
        if len(node.inputs) < 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        span = _window_int(node, default=20)
        return _Layer(
            _ewm_cov_corr_sql(
                left.sql,
                right.sql,
                span,
                dialect=dialect,
                corr=(op == "ewm_corr"),
            ),
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


def _duckdb_dataset_ref(dataset: str) -> str:
    """DuckDB sql() 要求 ``{{dataset}}`` 占位符。"""
    return f"{{{{{dataset}}}}}"


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


def _compile_layer(node: PlanNode, *, dialect: SqlDialect) -> _Layer | None:
    """带 optional CTE memo 的编译入口（递归经此函数以共享子树）。"""
    memo = _sql_memo_ctx.get()
    if memo is not None:
        from planner.plan_hash import structural_key

        key = structural_key(node)
        if key in memo.refs:
            return _Layer(f"SELECT ts, inst, _v FROM {memo.refs[key]}")
    layer = _compile_layer_impl(node, dialect=dialect)
    if layer is None or memo is None:
        return layer
    from planner.plan_hash import structural_key

    key = structural_key(node)
    canon = _resolve_canonical(node.op)
    if canon in {"column", "literal"} or memo.use_counts.get(key, 0) < 2:
        return layer
    if key not in memo.refs:
        memo._counter += 1
        name = f"s{memo._counter}"
        memo.refs[key] = name
        memo.bodies.append((name, layer.sql))
    return _Layer(f"SELECT ts, inst, _v FROM {memo.refs[key]}")


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

    use_counts = _structural_use_counts(plan)
    memo = _SqlCompileMemo(use_counts=use_counts) if any(c > 1 for c in use_counts.values()) else None
    token = _sql_memo_ctx.set(memo)
    try:
        layer = _compile_layer(plan, dialect=dialect)
    finally:
        _sql_memo_ctx.reset(token)
    if layer is None:
        return None

    cols: set[str] = set()
    _collect_columns(plan, cols)
    if not cols:
        return None

    source_from = table if dialect == SqlDialect.CLICKHOUSE else (dataset or table)
    if source_from and dialect == SqlDialect.DUCKDB and dataset:
        source_from = _duckdb_dataset_ref(dataset)
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
    cte_parts = [base]
    if memo and memo.bodies:
        for name, sql in memo.bodies:
            cte_parts.append(f"{name} AS (SELECT ts, inst, _v FROM ({sql}) t)")
    with_body = ", ".join(cte_parts)
    query = (
        f"WITH {with_body} "
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
    if source_from and dialect == SqlDialect.DUCKDB and dataset:
        source_from = _duckdb_dataset_ref(dataset)
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
