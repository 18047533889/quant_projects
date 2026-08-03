# -*- coding: utf-8 -*-
"""PlanNode → SQL 编译器（长表语义；支持 DuckDB / ClickHouse 方言）。

将逻辑计划树编译为 ``(ts, inst, _v)`` 长表 SQL，含 base CTE、子树 CSE 去重与
方言函数映射。对外入口为 ``compile_plan_to_sql`` / ``compile_plans_batch_to_sql``，
能力判断委托 ``plan_is_sql_capable``。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from contextvars import ContextVar
from typing import Any, Sequence

from cleaned_operators.registry import OperatorRegistry
from planner.logical_plan import PlanNode


class SqlDialect(str, Enum):
    """SQL 下推目标方言：DuckDB 数据集或 ClickHouse 物理表。"""
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
    """单因子 SQL 编译结果：查询文本、引用列集与数据源定位信息。"""
    query: str
    read_datasets: tuple[str, ...]
    referenced_columns: frozenset[str]
    dialect: SqlDialect = SqlDialect.DUCKDB
    table: str | None = None  # ClickHouse 直读表名


def _resolve_canonical(op: str) -> str:
    """将算子别名解析为 canonical 名称。"""
    resolved = OperatorRegistry._aliases.get(op, op)
    from backend.sql_tiers import SQL_IMPLEMENTED_CANONICALS

    if op in SQL_IMPLEMENTED_CANONICALS and resolved not in SQL_IMPLEMENTED_CANONICALS:
        return op
    return resolved


def _window_int(node: PlanNode, default: int = 3) -> int:
    """从 attrs 或子 literal 输入解析窗口长度（与 PolarsLong 共用）。"""
    return _window_spec(node, default=default).size


def _window_spec(node: PlanNode, default: int = 3):
    """完整 WindowSpec。"""
    from backend.plan_params import PlanParamError, window_spec_from_plan_node

    spec = window_spec_from_plan_node(node, default=default)
    if spec.closed != "right":
        raise PlanParamError(f"closed={spec.closed!r} 暂未支持，仅 right")
    return spec


def _float_attr(node: PlanNode, *keys: str, default: float) -> float:
    """从 attrs 中按候选键读取有限浮点参数。"""
    from backend.plan_params import PlanParamError, parse_finite_float, parse_unit_interval

    for key in keys:
        if key in (node.attrs or {}) and node.attrs[key] is not None:
            raw = node.attrs[key]
            if key in {"p", "q", "lo", "hi", "min_pct", "max_pct", "lower", "upper"}:
                try:
                    return parse_unit_interval(raw, label=key)
                except PlanParamError:
                    return default
            try:
                need_pos = key in {"epsilon", "eps", "to", "ann_factor"}
                return parse_finite_float(raw, label=key, gt=0.0 if need_pos else None)
            except PlanParamError:
                return default
    return default


def _int_attr(node: PlanNode, *keys: str, default: int) -> int:
    """从 attrs 中按候选键读取整数参数（至少为 1）。"""
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
    """解析 clip 算子的上下界（attrs 与 positional literal 合并）。"""
    lo = _float_attr(node, "lo", "min", default=default_lo)
    hi = _float_attr(node, "max", "hi", default=default_hi)
    pos_lo = _literal_positional(node, 0)
    pos_hi = _literal_positional(node, 1)
    if pos_lo is not None:
        lo = pos_lo
    if pos_hi is not None:
        hi = pos_hi
    return lo, hi


def _truthy_sql(value_col: str, *, dialect: SqlDialect | None = None) -> str:
    """浮点条件真值：NULL/NaN/0 → false。"""
    from backend.logical_semantics import truthy_sql

    is_ch = dialect == SqlDialect.CLICKHOUSE if dialect is not None else False
    return truthy_sql(value_col, dialect_is_clickhouse=is_ch)


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
    """双引号转义 SQL 标识符。"""
    safe = name.replace('"', '""')
    return f'"{safe}"'


def _sql_literal(val: Any) -> str:
    """将 Python 值转为 SQL 字面量字符串。"""
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
    """将逻辑函数名映射为方言特定的 SQL 函数名。"""
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
    """编译中间层：子查询 SQL 及窗口/partition 语义标记。"""
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
    """统计计划树中各 structural_key 的出现次数（用于 CTE 去重）。"""
    from planner.plan_hash import structural_key

    counts: dict[str, int] = {}

    def walk(n: PlanNode) -> None:
        key = structural_key(n)
        counts[key] = counts.get(key, 0) + 1
        for child in n.inputs:
            walk(child)

    walk(plan)
    return counts


def _inst_window(
    dialect: SqlDialect,
    w: int,
    agg: str,
    inner_sql: str,
    *,
    min_periods: int = 1,
) -> str:
    """按 instrument 分区的固定长度滚动窗口聚合 SQL（含 min_periods 门槛）。"""
    over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
    cnt = f"COUNT(_v) OVER ({over})"
    rolled = f"{agg}(_v) OVER ({over})"
    if min_periods <= 1:
        body = rolled
    else:
        body = f"CASE WHEN {cnt} < {min_periods} THEN NULL ELSE {rolled} END"
    return f"SELECT ts, inst, {body} AS _v FROM ({inner_sql}) t"


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
    """Wilder RSI 指标 SQL（对齐 pandas ``RSI_WILDER``）。"""
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
    """Wilder ATR 指标 SQL（对齐 pandas ``ATR_WILDER``）。"""
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
    """按 instrument 前向填充 NULL（``ffill`` 语义）。"""
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


def _average_rank_frac_correlated(
    *,
    row_value_col: str,
    partition_keys: list[str],
    numbered_sql: str,
    row_alias: str,
    dialect: SqlDialect,
    exclude_nan: bool = True,
    descending: bool = False,
) -> str:
    """pandas ``rank(method='average', pct=True)`` 相关子查询表达式。"""
    from backend.stat_valid import row_stat_invalid_sql, stat_valid_sql

    match = " AND ".join(f"p.{k} = {row_alias}.{k}" for k in partition_keys)
    valid = stat_valid_sql("p._v", dialect=dialect, exclude_nan=exclude_nan)
    invalid_row = row_stat_invalid_sql(row_value_col, dialect=dialect, exclude_nan=exclude_nan)
    rank_cmp = ">=" if descending else "<="
    cnt_subq = (
        f"(SELECT COUNT(*) FILTER (WHERE {valid}) "
        f"FROM ({numbered_sql}) p WHERE {match})"
    )
    if dialect == SqlDialect.CLICKHOUSE:
        valid_ch = valid.replace("p._v", "p._v")
        return (
            f"multiIf("
            f"{invalid_row}, NULL, "
            f"{cnt_subq} = 0, NULL, "
            f"("
            f"SELECT if(s.cnt = 0, NULL, (s.cnt_le - (s.cnt_eq - 1) / 2.0) / s.cnt) "
            f"FROM ("
            f"SELECT "
            f"countIf({valid_ch}) AS cnt, "
            f"countIf({valid_ch} AND p._v {rank_cmp} {row_value_col}) AS cnt_le, "
            f"countIf({valid_ch} AND p._v = {row_value_col}) AS cnt_eq "
            f"FROM ({numbered_sql}) p WHERE {match}"
            f") s"
            f")"
            f")"
        )
    return (
        f"CASE WHEN {invalid_row} THEN NULL "
        f"WHEN {cnt_subq} = 0 THEN NULL "
        f"ELSE ("
        f"SELECT CASE WHEN s.cnt = 0 THEN NULL "
        f"ELSE (s.cnt_le - (s.cnt_eq - 1) / 2.0) / s.cnt END "
        f"FROM ("
        f"SELECT "
        f"COUNT(*) FILTER (WHERE {valid}) AS cnt, "
        f"COUNT(*) FILTER (WHERE {valid} AND p._v {rank_cmp} {row_value_col}) AS cnt_le, "
        f"COUNT(*) FILTER (WHERE {valid} AND p._v = {row_value_col}) AS cnt_eq "
        f"FROM ({numbered_sql}) p WHERE {match}"
        f") s"
        f") END"
    )


def _average_rank_01_correlated(
    *,
    row_value_col: str,
    partition_keys: list[str],
    numbered_sql: str,
    row_alias: str,
    dialect: SqlDialect,
    exclude_nan: bool = True,
) -> str:
    """截面 0-1 average rank，对齐 ``cs_rank_01``。"""
    from backend.stat_valid import row_stat_invalid_sql, stat_valid_sql

    match = " AND ".join(f"p.{k} = {row_alias}.{k}" for k in partition_keys)
    valid = stat_valid_sql("p._v", dialect=dialect, exclude_nan=exclude_nan)
    invalid_row = row_stat_invalid_sql(row_value_col, dialect=dialect, exclude_nan=exclude_nan)
    cnt_subq = (
        f"(SELECT COUNT(*) FILTER (WHERE {valid}) "
        f"FROM ({numbered_sql}) p WHERE {match})"
    )
    if dialect == SqlDialect.CLICKHOUSE:
        valid_ch = valid
        return (
            f"multiIf("
            f"{invalid_row}, NULL, "
            f"{cnt_subq} = 0, NULL, "
            f"{cnt_subq} = 1, 0.5, "
            f"("
            f"SELECT (s.cnt_le - (s.cnt_eq - 1) / 2.0 - 1) / nullIf(s.cnt - 1, 0) "
            f"FROM ("
            f"SELECT "
            f"countIf({valid_ch}) AS cnt, "
            f"countIf({valid_ch} AND p._v <= {row_value_col}) AS cnt_le, "
            f"countIf({valid_ch} AND p._v = {row_value_col}) AS cnt_eq "
            f"FROM ({numbered_sql}) p WHERE {match}"
            f") s"
            f")"
            f")"
        )
    return (
        f"CASE WHEN {invalid_row} THEN NULL "
        f"WHEN {cnt_subq} = 0 THEN NULL "
        f"WHEN {cnt_subq} = 1 THEN 0.5 "
        f"ELSE ("
        f"SELECT (s.cnt_le - (s.cnt_eq - 1) / 2.0 - 1.0) / NULLIF(s.cnt - 1, 0) "
        f"FROM ("
        f"SELECT "
        f"COUNT(*) FILTER (WHERE {valid}) AS cnt, "
        f"COUNT(*) FILTER (WHERE {valid} AND p._v <= {row_value_col}) AS cnt_le, "
        f"COUNT(*) FILTER (WHERE {valid} AND p._v = {row_value_col}) AS cnt_eq "
        f"FROM ({numbered_sql}) p WHERE {match}"
        f") s"
        f") END"
    )


def _cs_average_rank_pct_sql(
    inner_sql: str,
    *,
    partition_keys: list[str],
    value_col: str = "_v",
    select_out: str = "ts, inst",
    dialect: SqlDialect,
) -> str:
    """截面/分组 average-rank 百分位 SQL。"""
    keys_csv = ", ".join(partition_keys)
    numbered = (
        f"SELECT {select_out}, {value_col} AS _v"
        + (f", {keys_csv}" if keys_csv else "")
        + f" FROM ({inner_sql}) t0"
    )
    frac = _average_rank_frac_correlated(
        row_value_col="b._v",
        partition_keys=partition_keys,
        numbered_sql=numbered,
        row_alias="b",
        dialect=dialect,
    )
    return f"SELECT {select_out}, {frac} AS _v FROM ({numbered}) b"


def _cs_average_rank_01_sql(
    inner_sql: str,
    *,
    partition_keys: list[str],
    value_col: str = "_v",
    select_out: str = "ts, inst",
    dialect: SqlDialect,
) -> str:
    """截面/分组 0-1 average rank SQL。"""
    keys_csv = ", ".join(partition_keys)
    numbered = (
        f"SELECT {select_out}, {value_col} AS _v"
        + (f", {keys_csv}" if keys_csv else "")
        + f" FROM ({inner_sql}) t0"
    )
    expr = _average_rank_01_correlated(
        row_value_col="b._v",
        partition_keys=partition_keys,
        numbered_sql=numbered,
        row_alias="b",
        dialect=dialect,
    )
    return f"SELECT {select_out}, {expr} AS _v FROM ({numbered}) b"


def _group_partition_wrap(
    inner_sql: str,
    grp_sql: str | None,
    *,
    value_alias: str = "_v",
) -> tuple[str, list[str]]:
    """分组 join 包装：返回 (wrapped_sql, partition_keys)。"""
    if grp_sql is not None:
        wrapped = (
            f"SELECT x.ts, x.inst, x._v AS {value_alias}, g._v AS _grp "
            f"FROM ({inner_sql}) x LEFT JOIN ({grp_sql}) g USING (ts, inst)"
        )
        return wrapped, ["ts", "_grp"]
    wrapped = f"SELECT ts, inst, _v AS {value_alias}, 1.0 AS _grp FROM ({inner_sql}) x"
    return wrapped, ["ts", "_grp"]


def _cs_rank_01_sql(inner_sql: str, *, partition: str, dialect: SqlDialect) -> str:
    """截面 0-1 排名 SQL 包装（average rank）。"""
    keys = [c.strip().split(".")[-1] for c in partition.replace("PARTITION BY", "").split(",") if c.strip()]
    return _cs_average_rank_01_sql(
        inner_sql,
        partition_keys=keys,
        dialect=dialect,
    )


def _cs_pct_rank_sql(inner_sql: str, *, partition: str, dialect: SqlDialect) -> str:
    """截面百分位 rank；NaN 保持 NULL（``rank_pct`` / ``cs_pct_rank`` 语义）。"""
    keys = [c.strip().split(".")[-1] for c in partition.replace("PARTITION BY", "").split(",") if c.strip()]
    return _cs_average_rank_pct_sql(
        inner_sql,
        partition_keys=keys,
        dialect=dialect,
    )


def _ts_pct_rank_sql(
    inner_sql: str,
    *,
    window: int,
    dialect: SqlDialect,
    min_periods: int = 1,
    exclude_nan: bool = True,
) -> str:
    """滚动百分位 rank，对齐 ``pandas rolling.rank(pct=True, method='average')``。"""
    from backend.stat_valid import row_stat_invalid_sql, stat_valid_sql

    w = max(int(window), 1)
    mp = max(int(min_periods), 1)
    valid = stat_valid_sql("p._v", dialect=dialect, exclude_nan=exclude_nan)
    invalid_row = row_stat_invalid_sql("b._v", dialect=dialect, exclude_nan=exclude_nan)
    numbered = (
        f"SELECT ts, inst, _v, ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts) AS rn "
        f"FROM ({inner_sql}) t0"
    )
    if dialect == SqlDialect.CLICKHOUSE:
        valid_ch = valid
        rank_expr = (
            f"multiIf("
            f"{invalid_row}, NULL, "
            f"("
            f"SELECT multiIf("
            f"s.cnt < {mp}, NULL, "
            f"s.cnt = 0, NULL, "
            f"s.cnt <= 1, 1.0, "
            f"(s.cnt_le - (s.cnt_eq - 1) / 2.0) / s.cnt"
            f") "
            f"FROM ("
            f"SELECT "
            f"countIf({valid_ch}) AS cnt, "
            f"countIf({valid_ch} AND p._v <= b._v) AS cnt_le, "
            f"countIf({valid_ch} AND p._v = b._v) AS cnt_eq "
            f"FROM ({numbered}) p "
            f"WHERE p.inst = b.inst AND p.rn BETWEEN b.rn - {w - 1} AND b.rn"
            f") s"
            f")"
            f")"
        )
    else:
        rank_expr = (
            f"CASE WHEN {invalid_row} THEN NULL ELSE ("
            f"SELECT CASE "
            f"WHEN s.cnt < {mp} THEN NULL "
            f"WHEN s.cnt = 0 THEN NULL "
            f"WHEN s.cnt <= 1 THEN 1.0 "
            f"ELSE (s.cnt_le - (s.cnt_eq - 1) / 2.0) / s.cnt END "
            f"FROM ("
            f"SELECT "
            f"COUNT(*) FILTER (WHERE {valid}) AS cnt, "
            f"COUNT(*) FILTER (WHERE {valid} AND p._v <= b._v) AS cnt_le, "
            f"COUNT(*) FILTER (WHERE {valid} AND p._v = b._v) AS cnt_eq "
            f"FROM ({numbered}) p "
            f"WHERE p.inst = b.inst AND p.rn BETWEEN b.rn - {w - 1} AND b.rn"
            f") s"
            f") END"
        )
    return f"SELECT b.ts, b.inst, {rank_expr} AS _v FROM ({numbered}) b"


def _rolling_corr_pandas_compat_expr(
    *,
    corr_col: str,
    std_left_col: str,
    std_right_col: str,
    left_col: str,
    right_col: str,
    window_count_col: str,
    window: int,
    dialect: SqlDialect,
) -> str:
    """滚动相关；满窗零方差退化时对齐 pandas ``rolling.corr``（→ ``inf``）。"""
    w = max(int(window), 1)
    full = f"{window_count_col} >= {w}"
    if dialect == SqlDialect.CLICKHOUSE:
        return (
            f"multiIf("
            f"isNull({left_col}) OR isNull({right_col}), NULL, "
            f"isFinite({corr_col}), {corr_col}, "
            f"{full} AND (isNull({std_left_col}) OR {std_left_col} = 0) AND {std_right_col} > 0, inf, "
            f"{full} AND (isNull({std_right_col}) OR {std_right_col} = 0) AND {std_left_col} > 0, inf, "
            f"NULL)"
        )
    return (
        f"CASE "
        f"WHEN {left_col} IS NULL OR {right_col} IS NULL THEN NULL "
        f"WHEN isfinite({corr_col}) THEN {corr_col} "
        f"WHEN {full} AND ({std_left_col} IS NULL OR {std_left_col} = 0) AND {std_right_col} > 0 THEN 'Infinity'::DOUBLE "
        f"WHEN {full} AND ({std_right_col} IS NULL OR {std_right_col} = 0) AND {std_left_col} > 0 THEN 'Infinity'::DOUBLE "
        f"ELSE NULL END"
    )


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
    """截面 OLS y ~ x + const：pairwise-valid 样本，返回 (beta, alpha, n_valid)。"""
    if dialect == SqlDialect.DUCKDB:
        cov_fn = "covar_samp"
        var_fn = "var_samp"
    else:
        cov_fn = "covarSamp"
        var_fn = "varSamp"
    nf = _dialect_fn(dialect, "nullif")
    pair_mask = f"{y_col} IS NOT NULL AND {x_col} IS NOT NULL"
    pair_y = f"CASE WHEN {pair_mask} THEN {y_col} END"
    pair_x = f"CASE WHEN {pair_mask} THEN {x_col} END"
    mean_y = f"AVG({pair_y}) OVER ({partition})"
    mean_x = f"AVG({pair_x}) OVER ({partition})"
    cov_xy = f"{cov_fn}({pair_y}, {pair_x}) OVER ({partition})"
    var_x = f"{var_fn}({pair_x}) OVER ({partition})"
    beta = f"({cov_xy}) / {nf}({var_x}, 0)"
    alpha = f"({mean_y}) - ({beta}) * ({mean_x})"
    n_valid = f"COUNT(CASE WHEN {pair_mask} THEN 1 END) OVER ({partition})"
    return beta, alpha, n_valid


def _zscore_window_expr(
    *,
    value_col: str,
    partition: str,
    dialect: SqlDialect,
    std_fn_name: str = "stddev",
    min_periods: int = 1,
) -> str:
    """zscore / ts_zscore：std=0 → 0，NULL 保持缺失。"""
    std_fn = _dialect_fn(dialect, std_fn_name)
    avg = f"AVG({value_col}) OVER ({partition})"
    stdv = f"{std_fn}({value_col}) OVER ({partition})"
    core = (
        f"CASE WHEN {value_col} IS NULL THEN NULL "
        f"WHEN {stdv} IS NULL THEN NULL "
        f"WHEN {stdv} = 0 THEN 0 "
        f"ELSE ({value_col} - {avg}) / {stdv} END"
    )
    if min_periods <= 1:
        return core
    cnt = f"COUNT({value_col}) OVER ({partition})"
    return f"CASE WHEN {cnt} < {min_periods} THEN NULL ELSE {core} END"


def _normalize_window_expr(*, value_col: str, partition: str) -> str:
    """normalize / group_normalize：常数截面 → 0.5；单有效值 → NULL。"""
    lo = f"MIN({value_col}) OVER ({partition})"
    hi = f"MAX({value_col}) OVER ({partition})"
    span = f"({hi} - {lo})"
    cnt = f"COUNT({value_col}) OVER ({partition})"
    return (
        f"CASE WHEN {value_col} IS NULL THEN NULL "
        f"WHEN {cnt} <= 1 THEN NULL "
        f"WHEN {span} IS NULL OR {span} = 0 THEN 0.5 "
        f"ELSE ({value_col} - {lo}) / {span} END"
    )


def _scale_window_expr(*, value_col: str, partition: str, to_val: float, dialect: SqlDialect) -> str:
    """scale：sum(abs)=0 → 0，否则 x * to / sum(abs)。"""
    abs_fn = _dialect_fn(dialect, "abs")
    s = f"SUM({abs_fn}({value_col})) OVER ({partition})"
    return (
        f"CASE WHEN {value_col} IS NULL THEN NULL "
        f"WHEN {s} IS NULL OR {s} = 0 THEN 0 "
        f"ELSE {value_col} * ({to_val} / {s}) END"
    )


def _rolling_min_periods(window: int, *, default_mp: int | None = None) -> int:
    w = max(int(window), 1)
    return max(2, w // 2) if default_mp is None else max(int(default_mp), 1)


def _rolling_std_min_periods_sql(
    *,
    value_col: str,
    over: str,
    window: int,
    scale_expr: str,
    dialect: SqlDialect,
    min_periods: int | None = None,
    std_fn_name: str = "stddev",
) -> str:
    """带 min_periods 的 rolling std 表达式。"""
    mp = _rolling_min_periods(window, default_mp=min_periods)
    std_fn = _dialect_fn(dialect, std_fn_name)
    cnt = f"COUNT({value_col}) OVER ({over})"
    stdv = f"{std_fn}({value_col}) OVER ({over})"
    return (
        f"CASE WHEN {cnt} < {mp} THEN NULL ELSE {stdv} * {scale_expr} END"
    )


def _int_attr(node: PlanNode, *keys: str, input_index: int | None = None, default: int = 0) -> int:
    """从 attrs 或 positional literal 读取整数参数。"""
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
    """从计划节点 attrs 或子 literal 提取标量浮点值。"""
    for key in ("value", "fill_value", "const", "c"):
        if key in node.attrs and node.attrs[key] is not None:
            return float(node.attrs[key])
    if len(node.inputs) >= 2 and node.inputs[1].op == "literal":
        val = node.inputs[1].attrs.get("value")
        if val is not None:
            return float(val)
    return default


def _compare_binary_sql(op: str, left_sql: str, right_sql: str, *, dialect: SqlDialect) -> str:
    """二元比较算子 SQL（NULL/NaN → NULL）。"""
    from backend.elementwise_semantics import compare_sql
    from backend.long_alignment import anchor_join_sql

    expr = compare_sql(op, "l._v", "r._v", dialect_is_clickhouse=dialect == SqlDialect.CLICKHOUSE)
    return f"SELECT l.ts, l.inst, {expr} AS _v {anchor_join_sql(left_sql, right_sql)}"


def _rolling_ols_partition(w: int, *, prefix: str = "l") -> str:
    """滚动 OLS 窗口的 PARTITION/ORDER BY 子句。"""
    return (
        f"PARTITION BY {prefix}.inst ORDER BY {prefix}.ts "
        f"ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
    )


def _inst_cum_sum_sql(inner_sql: str) -> str:
    """pandas 对齐 cumsum：NULL 行输出 NULL，累计和跳过 NULL 输入。"""
    return _inst_cum_agg("SUM", inner_sql, sum_skip_null=True)


def _inst_cum_agg(
    agg: str,
    inner_sql: str,
    *,
    sum_skip_null: bool = False,
    null_guard: bool = True,
) -> str:
    """按 instrument 的累积窗口聚合 SQL；默认当前行 NULL → 输出 NULL。"""
    over = "PARTITION BY inst ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
    agg_u = agg.upper()
    if sum_skip_null or agg_u == "SUM":
        rolled = f"SUM(CASE WHEN _v IS NULL THEN 0 ELSE _v END) OVER ({over})"
    elif agg_u == "COUNT":
        rolled = f"SUM(CASE WHEN _v IS NOT NULL THEN 1 ELSE 0 END) OVER ({over})"
        null_guard = False
    else:
        rolled = f"{agg}(_v) OVER ({over})"
    if null_guard:
        body = f"CASE WHEN _v IS NULL THEN NULL ELSE {rolled} END"
    else:
        body = rolled
    return (
        f"SELECT ts, inst, {body} AS _v "
        f"FROM ({inner_sql}) t"
    )


def _cs_broadcast_agg(agg: str, inner_sql: str, *, all_null_null: bool = False) -> str:
    """截面广播聚合：按 ts 分区将聚合值广播到各行。"""
    rolled = f"{agg}(_v) OVER (PARTITION BY ts)"
    if not all_null_null:
        return (
            f"SELECT ts, inst, {rolled} AS _v "
            f"FROM ({inner_sql}) t"
        )
    valid = "COUNT(_v) OVER (PARTITION BY ts)"
    body = f"CASE WHEN {valid} = 0 THEN NULL ELSE {rolled} END"
    return (
        f"SELECT ts, inst, {body} AS _v "
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
    """累积差分：当前值减去窗口内首个非空值。"""
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
    """滚动极值距当前 bar 的距离（0=当前；并列取最近）。"""
    over = "PARTITION BY inst ORDER BY ts"
    win = f"{over} ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
    ext_fn = "MAX" if pick == "max" else "MIN"
    cases: list[str] = []
    for lag in range(w):
        val = "_v" if lag == 0 else f"LAG(_v, {lag}) OVER ({over})"
        cases.append(
            f"WHEN {val} IS NOT NULL AND w_ext IS NOT NULL AND ABS({val} - w_ext) < 1e-9 "
            f"THEN {float(lag)}"
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


def _duckdb_valid(value: str) -> str:
    return f"({value} IS NOT NULL AND NOT isnan({value}) AND NOT isinf({value}))"


def _raw_literal(node: PlanNode, input_index: int, default: Any = None) -> Any:
    if input_index >= len(node.inputs):
        return default
    child = node.inputs[input_index]
    if child.op != "literal":
        return default
    return child.attrs.get("value", default)


def _duckdb_join_layers(layers: list[_Layer], aliases: list[str]) -> str:
    sql = f"FROM ({layers[0].sql}) {aliases[0]}"
    for layer, alias in zip(layers[1:], aliases[1:], strict=True):
        sql += f" LEFT JOIN ({layer.sql}) {alias} USING (ts, inst)"
    return sql


def _duckdb_conditional_rolling_sql(
    value_sql: str | None,
    condition_sql: str,
    *,
    window: int,
    min_periods: int,
    op: str,
    ddof: int = 1,
) -> str:
    if value_sql is None:
        joined = f"FROM ({condition_sql}) c"
        over = (
            f"PARTITION BY c.inst ORDER BY c.ts "
            f"ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        )
        valid = _duckdb_valid("c._v")
        true = f"({valid} AND c._v <> 0)"
        count_valid = f"SUM(CASE WHEN {valid} THEN 1 ELSE 0 END) OVER ({over})"
        value = f"SUM(CASE WHEN {true} THEN 1 ELSE 0 END) OVER ({over})"
        return (
            f"SELECT c.ts, c.inst, CASE WHEN {count_valid} < {min_periods} "
            f"THEN NULL ELSE CAST({value} AS DOUBLE) END AS _v {joined}"
        )

    joined = (
        f"FROM ({value_sql}) x LEFT JOIN ({condition_sql}) c USING (ts, inst)"
    )
    over = (
        f"PARTITION BY x.inst ORDER BY x.ts "
        f"ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
    )
    selected = (
        f"({_duckdb_valid('x._v')} AND {_duckdb_valid('c._v')} AND c._v <> 0)"
    )
    count = f"SUM(CASE WHEN {selected} THEN 1 ELSE 0 END) OVER ({over})"
    if op == "sum":
        aggregate = f"SUM(CASE WHEN {selected} THEN x._v ELSE NULL END) OVER ({over})"
        required = min_periods
    elif op == "mean":
        aggregate = f"AVG(CASE WHEN {selected} THEN x._v ELSE NULL END) OVER ({over})"
        required = min_periods
    elif op == "std":
        fn = "STDDEV_POP" if ddof == 0 else "STDDEV_SAMP"
        aggregate = f"{fn}(CASE WHEN {selected} THEN x._v ELSE NULL END) OVER ({over})"
        required = max(min_periods, ddof + 1)
    else:  # pragma: no cover
        raise ValueError(op)
    return (
        f"SELECT x.ts, x.inst, CASE WHEN {count} < {required} "
        f"THEN NULL ELSE {aggregate} END AS _v {joined}"
    )


def _duckdb_multi_resid_sql(
    layers: list[_Layer],
    *,
    add_intercept: bool,
    min_obs: int,
) -> str:
    aliases = ["y"] + [f"x{i}" for i in range(1, len(layers))]
    joined = _duckdb_join_layers(layers, aliases)
    valid_parts = [_duckdb_valid("y._v")]
    valid_parts.extend(_duckdb_valid(f"{alias}._v") for alias in aliases[1:])
    ok = " AND ".join(valid_parts)
    raw_cols = ", ".join(
        ["y._v AS _y"] + [f"{alias}._v AS _x{i}" for i, alias in enumerate(aliases[1:], 1)]
    )
    base = (
        f"SELECT y.ts, y.inst, {raw_cols}, CASE WHEN {ok} THEN 1 ELSE 0 END AS _ok "
        f"{joined}"
    )
    n = "SUM(_ok) OVER (PARTITION BY ts)"
    if add_intercept:
        mean_y = "AVG(CASE WHEN _ok = 1 THEN _y END) OVER (PARTITION BY ts)"
        centered = [f"_x{i} - AVG(CASE WHEN _ok = 1 THEN _x{i} END) OVER (PARTITION BY ts)" for i in range(1, len(layers))]
        y_centered = f"_y - {mean_y}"
    else:
        mean_y = "0.0"
        centered = [f"_x{i}" for i in range(1, len(layers))]
        y_centered = "_y"
    stage = (
        f"SELECT *, {n} AS _n, {mean_y} AS _mean_y, "
        f"CASE WHEN _ok = 1 THEN {y_centered} END AS _yc, "
        + ", ".join(
            f"CASE WHEN _ok = 1 THEN {expr} END AS _v{i}"
            for i, expr in enumerate(centered, 1)
        )
        + f" FROM ({base}) b"
    )
    q_names: list[str] = []
    for i in range(1, len(layers)):
        residual = f"_v{i}"
        for q in q_names:
            coeff = (
                f"SUM(_v{i} * {q}) OVER (PARTITION BY ts) / "
                f"NULLIF(SUM({q} * {q}) OVER (PARTITION BY ts), 0)"
            )
            residual += f" - ({coeff}) * {q}"
        q = f"_q{i}"
        stage = f"SELECT *, ({residual}) AS {q} FROM ({stage}) qstage{i}"
        q_names.append(q)
    singular = " OR ".join(
        f"SUM({q} * {q}) OVER (PARTITION BY ts) IS NULL OR "
        f"SUM({q} * {q}) OVER (PARTITION BY ts) <= 1e-24"
        for q in q_names
    )
    fitted_terms = [
        (
            f"(SUM(_yc * {q}) OVER (PARTITION BY ts) / "
            f"NULLIF(SUM({q} * {q}) OVER (PARTITION BY ts), 0)) * {q}"
        )
        for q in q_names
    ]
    fitted = "_mean_y" + "".join(f" + {term}" for term in fitted_terms)
    return (
        f"SELECT ts, inst, CASE WHEN _ok = 0 OR _n < {min_obs} "
        f"OR ({singular}) THEN NULL ELSE _y - ({fitted}) END AS _v "
        f"FROM ({stage}) scored"
    )


def _duckdb_rolling_tstat_sql(
    y_sql: str,
    x_sql: str,
    *,
    window: int,
    min_periods: int,
    add_intercept: bool,
) -> str:
    over = (
        f"PARTITION BY y.inst ORDER BY y.ts "
        f"ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
    )
    valid = f"({_duckdb_valid('y._v')} AND {_duckdb_valid('x._v')})"
    yv = f"CASE WHEN {valid} THEN y._v END"
    xv = f"CASE WHEN {valid} THEN x._v END"
    n = f"COUNT({yv}) OVER ({over})"
    sx = f"SUM({xv}) OVER ({over})"
    sy = f"SUM({yv}) OVER ({over})"
    sxx = f"SUM(({xv}) * ({xv})) OVER ({over})"
    sxy = f"SUM(({xv}) * ({yv})) OVER ({over})"
    syy = f"SUM(({yv}) * ({yv})) OVER ({over})"
    if add_intercept:
        xx = f"({sxx} - ({sx}) * ({sx}) / NULLIF({n}, 0))"
        xy = f"({sxy} - ({sx}) * ({sy}) / NULLIF({n}, 0))"
        yy = f"({syy} - ({sy}) * ({sy}) / NULLIF({n}, 0))"
        dof = f"({n} - 2)"
    else:
        xx, xy, yy = sxx, sxy, syy
        dof = f"({n} - 1)"
    beta = f"({xy}) / NULLIF(({xx}), 0)"
    sse = f"GREATEST(0.0, ({yy}) - ({beta}) * ({xy}))"
    se = f"SQRT(({sse}) / NULLIF({dof}, 0) / NULLIF(({xx}), 0))"
    return (
        f"SELECT y.ts, y.inst, CASE WHEN {n} < {min_periods} OR {dof} <= 0 "
        f"OR ({xx}) <= 0 OR ({se}) <= 0 THEN NULL ELSE ({beta}) / ({se}) END AS _v "
        f"FROM ({y_sql}) y LEFT JOIN ({x_sql}) x USING (ts, inst)"
    )


def _compile_layer_impl(node: PlanNode, *, dialect: SqlDialect) -> _Layer | None:
    """递归将 PlanNode 编译为 ``_Layer`` 子查询；不支持的算子返回 ``None``。"""
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
            f"FROM ({left.sql}) l LEFT JOIN ({right.sql}) r USING (ts, inst)",
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
        from backend.elementwise_semantics import protected_div_sql
        from backend.numeric_semantics import protected_div_default, protected_epsilon_default

        eps = _float_attr(node, "epsilon", default=protected_epsilon_default())
        default = _float_attr(node, "default", default=protected_div_default())
        abs_fn = _dialect_fn(dialect, "abs")
        expr = protected_div_sql("l._v", "r._v", eps=eps, default=default, abs_fn=abs_fn)
        return _Layer(
            f"SELECT l.ts, l.inst, {expr} AS _v "
            f"FROM ({left.sql}) l LEFT JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=left.has_inst_window or right.has_inst_window,
            has_ts_partition=left.has_ts_partition or right.has_ts_partition,
        )

    if op in {"safe_div_null", "safe_div"}:
        if len(node.inputs) != 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        from backend.numeric_semantics import protected_epsilon_default

        eps = _float_attr(node, "epsilon", default=protected_epsilon_default())
        abs_fn = _dialect_fn(dialect, "abs")
        return _Layer(
            f"SELECT l.ts, l.inst, "
            f"CASE WHEN l._v IS NULL OR r._v IS NULL THEN NULL "
            f"WHEN {abs_fn}(r._v) <= {eps} THEN NULL "
            f"ELSE l._v / r._v END AS _v "
            f"FROM ({left.sql}) l LEFT JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=left.has_inst_window or right.has_inst_window,
            has_ts_partition=left.has_ts_partition or right.has_ts_partition,
        )

    if op == "div_or_default":
        if len(node.inputs) != 2:
            return None
        left = _compile_layer(node.inputs[0], dialect=dialect)
        right = _compile_layer(node.inputs[1], dialect=dialect)
        if left is None or right is None:
            return None
        from backend.elementwise_semantics import div_or_default_sql
        from backend.numeric_semantics import protected_div_default, protected_epsilon_default

        eps = _float_attr(node, "epsilon", default=protected_epsilon_default())
        default = _float_attr(node, "default", default=protected_div_default())
        abs_fn = _dialect_fn(dialect, "abs")
        expr = div_or_default_sql("l._v", "r._v", eps=eps, default=default, abs_fn=abs_fn)
        return _Layer(
            f"SELECT l.ts, l.inst, {expr} AS _v "
            f"FROM ({left.sql}) l LEFT JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=left.has_inst_window or right.has_inst_window,
            has_ts_partition=left.has_ts_partition or right.has_ts_partition,
        )

    if op == "protected_log":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        from backend.elementwise_semantics import protected_log_sql

        eps = _float_attr(node, "epsilon", default=1e-12)
        expr = protected_log_sql("_v", eps=eps, ln_fn=ln)
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "log_fill_invalid":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        from backend.elementwise_semantics import log_fill_invalid_sql

        eps = _float_attr(node, "epsilon", default=1e-12)
        expr = log_fill_invalid_sql("_v", eps=eps, ln_fn=ln)
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v "
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
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL OR _v < 0 THEN NULL "
            f"WHEN _v = 0 THEN -1.0/0.0 ELSE {ln}(_v) END AS _v "
            f"FROM ({inner.sql}) t",
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
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL OR _v < 0 THEN NULL ELSE {fn}(_v) END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op in {"clip", "cap"}:
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

    if op == "is_null":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        from backend.logical_semantics import is_null_sql

        return _Layer(
            f"SELECT ts, inst, {is_null_sql('_v')} AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "is_not_null":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        from backend.logical_semantics import is_null_sql

        return _Layer(
            f"SELECT ts, inst, CASE WHEN {is_null_sql('_v')} = 1.0 THEN 0.0 ELSE 1.0 END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "is_nan":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        from backend.logical_semantics import is_nan_sql

        expr = is_nan_sql("_v", dialect_is_clickhouse=dialect == SqlDialect.CLICKHOUSE)
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "is_finite":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        finite_fn = "isFinite" if dialect == SqlDialect.CLICKHOUSE else "isfinite"
        isnan_fn = "isNaN" if dialect == SqlDialect.CLICKHOUSE else "isnan"
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL OR {isnan_fn}(_v) THEN 0.0 "
            f"WHEN {finite_fn}(_v) THEN 1.0 ELSE 0.0 END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "is_infinite":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        from backend.elementwise_semantics import is_infinite_sql

        expr = is_infinite_sql("_v", dialect_is_clickhouse=dialect == SqlDialect.CLICKHOUSE)
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op in {"ts_count_if", "ts_sum_if", "ts_mean_if", "ts_std_if"}:
        if dialect != SqlDialect.DUCKDB:
            return None
        if op == "ts_count_if":
            condition = _compile_layer(node.inputs[0], dialect=dialect)
            if condition is None:
                return None
            window = int(node.attrs.get("window", _raw_literal(node, 1, 3)))
            min_periods = int(node.attrs.get("min_periods", _raw_literal(node, 2, 1)))
            sql = _duckdb_conditional_rolling_sql(
                None,
                condition.sql,
                window=window,
                min_periods=min_periods,
                op="count",
            )
        else:
            if len(node.inputs) < 2:
                return None
            value = _compile_layer(node.inputs[0], dialect=dialect)
            condition = _compile_layer(node.inputs[1], dialect=dialect)
            if value is None or condition is None:
                return None
            window = int(node.attrs.get("window", _raw_literal(node, 2, 3)))
            default_mp = 2 if op == "ts_std_if" else 1
            min_periods = int(
                node.attrs.get("min_periods", _raw_literal(node, 3, default_mp))
            )
            ddof = int(node.attrs.get("ddof", _raw_literal(node, 4, 1)))
            sql = _duckdb_conditional_rolling_sql(
                value.sql,
                condition.sql,
                window=window,
                min_periods=min_periods,
                op=op.removeprefix("ts_").removesuffix("_if"),
                ddof=ddof,
            )
        return _Layer(sql, has_inst_window=True)

    if op == "ts_last_if":
        if dialect != SqlDialect.DUCKDB or len(node.inputs) < 2:
            return None
        value = _compile_layer(node.inputs[0], dialect=dialect)
        condition = _compile_layer(node.inputs[1], dialect=dialect)
        if value is None or condition is None:
            return None
        window = int(node.attrs.get("window", _raw_literal(node, 2, 3)))
        over = (
            f"PARTITION BY x.inst ORDER BY x.ts "
            f"ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        )
        selected = (
            f"({_duckdb_valid('x._v')} AND {_duckdb_valid('c._v')} AND c._v <> 0)"
        )
        return _Layer(
            f"SELECT x.ts, x.inst, "
            f"arg_max(x._v, x.ts) FILTER (WHERE {selected}) OVER ({over}) AS _v "
            f"FROM ({value.sql}) x LEFT JOIN ({condition.sql}) c USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "ts_days_since":
        if dialect != SqlDialect.DUCKDB:
            return None
        condition = _compile_layer(node.inputs[0], dialect=dialect)
        if condition is None:
            return None
        raw_limit = node.attrs.get("max_lookback", _raw_literal(node, 1, None))
        limit = None if raw_limit is None else int(raw_limit)
        true = f"({_duckdb_valid('_v')} AND _v <> 0)"
        distance = "(_rn - _last_true)"
        limit_guard = "" if limit is None else f" OR {distance} >= {limit}"
        return _Layer(
            f"SELECT ts, inst, CASE WHEN _last_true IS NULL{limit_guard} "
            f"THEN NULL ELSE CAST({distance} AS DOUBLE) END AS _v FROM ("
            f"SELECT *, MAX(CASE WHEN {true} THEN _rn END) OVER ("
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
            f") AS _last_true FROM ("
            f"SELECT ts, inst, _v, ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts) AS _rn "
            f"FROM ({condition.sql}) c0"
            f") c1"
            f") c2",
            has_inst_window=True,
        )

    if op == "ts_true_streak":
        if dialect != SqlDialect.DUCKDB:
            return None
        condition = _compile_layer(node.inputs[0], dialect=dialect)
        if condition is None:
            return None
        false = f"(NOT {_duckdb_valid('_v')} OR _v = 0)"
        return _Layer(
            f"SELECT ts, inst, CAST(_rn - COALESCE(_last_false, 0) AS DOUBLE) AS _v FROM ("
            f"SELECT *, MAX(CASE WHEN {false} THEN _rn END) OVER ("
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
            f") AS _last_false FROM ("
            f"SELECT ts, inst, _v, ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts) AS _rn "
            f"FROM ({condition.sql}) s0"
            f") s1"
            f") s2",
            has_inst_window=True,
        )

    if op == "cs_bucket":
        if dialect != SqlDialect.DUCKDB:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        buckets = int(node.attrs.get("buckets", _raw_literal(node, 1, 10)))
        ascending = bool(node.attrs.get("ascending", _raw_literal(node, 2, True)))
        direction = "ASC" if ascending else "DESC"
        return _Layer(
            f"SELECT ts, inst, CASE WHEN _v IS NULL THEN NULL ELSE "
            f"LEAST({buckets}.0, FLOOR(((_rank + (_ties - 1) / 2.0) / _n) * {buckets}) + 1.0) "
            f"END AS _v FROM ("
            f"SELECT ts, inst, _v, "
            f"RANK() OVER (PARTITION BY ts ORDER BY _v {direction} NULLS LAST) AS _rank, "
            f"COUNT(*) OVER (PARTITION BY ts, _v) AS _ties, "
            f"COUNT(_v) OVER (PARTITION BY ts) AS _n "
            f"FROM ({inner.sql}) b0"
            f") b1",
            has_ts_partition=True,
        )

    if op == "cs_multi_resid":
        if dialect != SqlDialect.DUCKDB or len(node.inputs) < 2:
            return None
        layers: list[_Layer] = []
        for child in node.inputs:
            if child.op == "literal":
                return None
            layer = _compile_layer(child, dialect=dialect)
            if layer is None:
                return None
            layers.append(layer)
        add_intercept = bool(node.attrs.get("add_intercept", True))
        default_min = len(layers) + 1
        min_obs = int(node.attrs.get("min_obs") or default_min)
        return _Layer(
            _duckdb_multi_resid_sql(
                layers,
                add_intercept=add_intercept,
                min_obs=min_obs,
            ),
            has_ts_partition=True,
        )

    if op == "cs_wls_resid":
        if dialect != SqlDialect.DUCKDB or len(node.inputs) < 3:
            return None
        y = _compile_layer(node.inputs[0], dialect=dialect)
        x = _compile_layer(node.inputs[1], dialect=dialect)
        weight = _compile_layer(node.inputs[2], dialect=dialect)
        if y is None or x is None or weight is None:
            return None
        add_intercept = bool(node.attrs.get("add_intercept", True))
        min_obs = int(node.attrs.get("min_obs", 5))
        valid = (
            f"({_duckdb_valid('y._v')} AND {_duckdb_valid('x._v')} "
            f"AND {_duckdb_valid('w._v')} AND w._v > 0)"
        )
        joined = (
            f"SELECT y.ts, y.inst, y._v AS _y, x._v AS _x, w._v AS _w, "
            f"CASE WHEN {valid} THEN 1 ELSE 0 END AS _ok "
            f"FROM ({y.sql}) y LEFT JOIN ({x.sql}) x USING (ts, inst) "
            f"LEFT JOIN ({weight.sql}) w USING (ts, inst)"
        )
        n_expr = "SUM(_ok) OVER (PARTITION BY ts)"
        sum_w_expr = "SUM(CASE WHEN _ok = 1 THEN _w END) OVER (PARTITION BY ts)"
        if add_intercept:
            mx_expr = f"SUM(CASE WHEN _ok = 1 THEN _w * _x END) OVER (PARTITION BY ts) / NULLIF({sum_w_expr}, 0)"
            my_expr = f"SUM(CASE WHEN _ok = 1 THEN _w * _y END) OVER (PARTITION BY ts) / NULLIF({sum_w_expr}, 0)"
        else:
            mx_expr = my_expr = "0.0"
        centered = (
            f"SELECT *, {n_expr} AS _n, {mx_expr} AS _mx, {my_expr} AS _my "
            f"FROM ({joined}) w0"
        )
        dx, dy = "(_x - _mx)", "(_y - _my)"
        cov = f"SUM(CASE WHEN _ok = 1 THEN _w * ({dx}) * ({dy}) END) OVER (PARTITION BY ts)"
        var = f"SUM(CASE WHEN _ok = 1 THEN _w * ({dx}) * ({dx}) END) OVER (PARTITION BY ts)"
        moments = (
            f"SELECT *, {cov} AS _cov, {var} AS _var FROM ({centered}) w1"
        )
        beta = "_cov / NULLIF(_var, 0)"
        fitted = f"_my + ({beta}) * ({dx})"
        return _Layer(
            f"SELECT ts, inst, CASE WHEN _ok = 0 OR _n < {min_obs} "
            f"OR _var <= 1e-24 THEN NULL ELSE _y - ({fitted}) END AS _v "
            f"FROM ({moments}) wr",
            has_ts_partition=True,
        )

    if op == "period_lag":
        if dialect != SqlDialect.DUCKDB or len(node.inputs) < 2:
            return None
        value = _compile_layer(node.inputs[0], dialect=dialect)
        period = _compile_layer(node.inputs[1], dialect=dialect)
        if value is None or period is None:
            return None
        periods = int(node.attrs.get("periods", _raw_literal(node, 2, 1)))
        joined = (
            f"SELECT x.ts, x.inst, x._v AS xv, p._v AS pid "
            f"FROM ({value.sql}) x LEFT JOIN ({period.sql}) p USING (ts, inst)"
        )
        return _Layer(
            f"SELECT r.ts, r.inst, CASE WHEN r.pid IS NULL THEN NULL ELSE ("
            f"SELECT arg_max(h.xv, h.ts) FROM ({joined}) h "
            f"WHERE h.inst = r.inst AND h.ts <= r.ts AND h.pid = ("
            f"SELECT target.pid FROM ("
            f"SELECT q.pid, MIN(q.ts) AS first_seen FROM ({joined}) q "
            f"WHERE q.inst = r.inst AND q.ts <= r.ts AND q.pid IS NOT NULL "
            f"GROUP BY q.pid ORDER BY first_seen DESC LIMIT 1 OFFSET {periods}"
            f") target"
            f")) END AS _v FROM ({joined}) r",
            has_inst_window=True,
        )

    if op in {"period_change", "period_cagr", "yoy_by_period"}:
        if dialect != SqlDialect.DUCKDB or len(node.inputs) < 2:
            return None
        current = _compile_layer(node.inputs[0], dialect=dialect)
        periods = int(node.attrs.get("periods", _raw_literal(node, 2, 1 if op != "yoy_by_period" else 4)))
        previous = _compile_layer(
            PlanNode(op="period_lag", inputs=[node.inputs[0], node.inputs[1]], attrs={"periods": periods}),
            dialect=dialect,
        )
        if current is None or previous is None:
            return None
        joined = f"SELECT x.ts, x.inst, x._v AS xv, p._v AS pv FROM ({current.sql}) x LEFT JOIN ({previous.sql}) p USING (ts, inst)"
        valid = f"({_duckdb_valid('xv')} AND {_duckdb_valid('pv')})"
        if op == "period_change":
            mode = str(node.attrs.get("mode", _raw_literal(node, 3, "absolute"))).lower()
            expr = "xv - pv" if mode == "absolute" else "xv / NULLIF(pv, 0) - 1.0" if mode == "ratio" else "LN(xv / NULLIF(pv, 0))"
            if mode not in {"absolute", "ratio", "log"}:
                return None
            if mode == "log":
                valid += " AND xv / NULLIF(pv, 0) > 0"
        elif op == "period_cagr":
            ppy = int(node.attrs.get("periods_per_year", _raw_literal(node, 3, 4)))
            policy = str(node.attrs.get("sign_policy", _raw_literal(node, 4, "strict"))).lower()
            if policy == "strict":
                valid += " AND xv > 0 AND pv > 0"
                ratio = "xv / pv"
            elif policy == "absolute":
                valid += " AND ABS(pv) > 1e-12"
                ratio = "ABS(xv) / ABS(pv)"
            else:
                return None
            expr = f"POW({ratio}, {float(ppy) / float(periods)}) - 1.0"
        else:
            mode = str(node.attrs.get("denominator", _raw_literal(node, 3, "signed"))).lower()
            if mode not in {"signed", "absolute"}:
                return None
            valid += " AND ABS(pv) > 1e-12"
            denom = "pv" if mode == "signed" else "ABS(pv)"
            expr = f"(xv - pv) / {denom}"
        return _Layer(f"SELECT ts, inst, CASE WHEN {valid} THEN {expr} ELSE NULL END AS _v FROM ({joined}) fp", has_inst_window=True)

    if op in {"period_average", "ttm_from_quarterly"}:
        if dialect != SqlDialect.DUCKDB or len(node.inputs) < 2:
            return None
        count = int(node.attrs.get("periods", _raw_literal(node, 2, 2 if op == "period_average" else 4)))
        if count < 1:
            return None
        layers = [_compile_layer(node.inputs[0], dialect=dialect)]
        layers.extend(
            _compile_layer(PlanNode(op="period_lag", inputs=[node.inputs[0], node.inputs[1]], attrs={"periods": lag}), dialect=dialect)
            for lag in range(1, count)
        )
        if any(layer is None for layer in layers):
            return None
        aliases = [f"v{i}" for i in range(count)]
        sql = f"SELECT b.ts, b.inst, b._v AS {aliases[0]} FROM ({layers[0].sql}) b"
        for i in range(1, count):
            sql = f"SELECT j.*, p._v AS {aliases[i]} FROM ({sql}) j LEFT JOIN ({layers[i].sql}) p USING (ts, inst)"
        valid = " AND ".join(_duckdb_valid(alias) for alias in aliases)
        total = " + ".join(aliases)
        expr = f"({total}) / {float(count)}" if op == "period_average" else f"({total})"
        return _Layer(f"SELECT ts, inst, CASE WHEN {valid} THEN {expr} ELSE NULL END AS _v FROM ({sql}) fs", has_inst_window=True)

    if op == "quarter_from_cumulative":
        if dialect != SqlDialect.DUCKDB or len(node.inputs) < 3:
            return None
        current = _compile_layer(node.inputs[0], dialect=dialect)
        quarter = _compile_layer(node.inputs[2], dialect=dialect)
        previous = _compile_layer(PlanNode(op="period_lag", inputs=[node.inputs[0], node.inputs[1]], attrs={"periods": 1}), dialect=dialect)
        previous_q = _compile_layer(PlanNode(op="period_lag", inputs=[node.inputs[2], node.inputs[1]], attrs={"periods": 1}), dialect=dialect)
        if any(layer is None for layer in (current, quarter, previous, previous_q)):
            return None
        joined = (
            f"SELECT x.ts, x.inst, x._v AS xv, q._v AS qv, p._v AS pv, pq._v AS pqv "
            f"FROM ({current.sql}) x LEFT JOIN ({quarter.sql}) q USING (ts, inst) "
            f"LEFT JOIN ({previous.sql}) p USING (ts, inst) LEFT JOIN ({previous_q.sql}) pq USING (ts, inst)"
        )
        consecutive = "((pqv = 4 AND qv = 1) OR qv = pqv + 1)"
        expr = f"CASE WHEN qv = 1 THEN xv WHEN {consecutive} THEN xv - pv ELSE NULL END"
        return _Layer(f"SELECT ts, inst, {expr} AS _v FROM ({joined}) fq", has_inst_window=True)

    if op == "ttm_from_cumulative":
        if dialect != SqlDialect.DUCKDB or len(node.inputs) < 3:
            return None
        quarterly = PlanNode(op="quarter_from_cumulative", inputs=list(node.inputs[:3]), attrs={})
        return _compile_layer(
            PlanNode(op="ttm_from_quarterly", inputs=[quarterly, node.inputs[1]], attrs={"periods": 4}),
            dialect=dialect,
        )

    if op == "ts_regression_tstat":
        if dialect != SqlDialect.DUCKDB or len(node.inputs) < 2:
            return None
        y = _compile_layer(node.inputs[0], dialect=dialect)
        x = _compile_layer(node.inputs[1], dialect=dialect)
        if y is None or x is None:
            return None
        window = int(node.attrs.get("window", _raw_literal(node, 2, 3)))
        raw_mp = node.attrs.get("min_periods", _raw_literal(node, 3, None))
        min_periods = window if raw_mp is None else int(raw_mp)
        add_intercept = bool(node.attrs.get("add_intercept", _raw_literal(node, 4, True)))
        return _Layer(
            _duckdb_rolling_tstat_sql(
                y.sql,
                x.sql,
                window=window,
                min_periods=min_periods,
                add_intercept=add_intercept,
            ),
            has_inst_window=True,
        )

    if op == "ts_trend_tstat":
        if dialect != SqlDialect.DUCKDB:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        window = int(node.attrs.get("window", _raw_literal(node, 1, 3)))
        raw_mp = node.attrs.get("min_periods", _raw_literal(node, 2, None))
        min_periods = window if raw_mp is None else int(raw_mp)
        time_sql = (
            f"SELECT ts, inst, _v, "
            f"CAST(ROW_NUMBER() OVER (PARTITION BY inst ORDER BY ts) AS DOUBLE) AS _time "
            f"FROM ({inner.sql}) tr0"
        )
        y_sql = f"SELECT ts, inst, _v FROM ({time_sql}) tr1"
        x_sql = f"SELECT ts, inst, _time AS _v FROM ({time_sql}) tr2"
        return _Layer(
            _duckdb_rolling_tstat_sql(
                y_sql,
                x_sql,
                window=window,
                min_periods=min_periods,
                add_intercept=True,
            ),
            has_inst_window=True,
        )

    if op == "ts_max_drawdown":
        if dialect != SqlDialect.DUCKDB:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        window = int(node.attrs.get("window", _raw_literal(node, 1, 3)))
        min_periods = int(node.attrs.get("min_periods", _raw_literal(node, 2, 2)))
        over = "PARTITION BY inst ORDER BY ts"
        lags = ["_v"] + [f"LAG(_v, {i}) OVER ({over})" for i in range(1, window)]
        values = f"list_value({', '.join(lags)})"
        valid_list = f"list_filter({values}, v -> v IS NOT NULL AND NOT isnan(v) AND NOT isinf(v))"
        drawdowns = ["0.0"]
        for i, value_i in enumerate(lags):
            earlier = f"list_max(list_value({', '.join(lags[i:])}))"
            drawdowns.append(
                f"CASE WHEN {value_i} IS NULL OR {earlier} IS NULL THEN NULL "
                f"ELSE {value_i} / {earlier} - 1.0 END"
            )
        return _Layer(
            f"SELECT ts, inst, CASE WHEN list_count({valid_list}) < {min_periods} "
            f"OR list_min({valid_list}) <= 0 THEN NULL "
            f"ELSE LEAST({', '.join(drawdowns)}) END AS _v FROM ({inner.sql}) md",
            has_inst_window=True,
        )

    if op == "ts_partial_corr":
        if dialect != SqlDialect.DUCKDB or len(node.inputs) < 3:
            return None
        x = _compile_layer(node.inputs[0], dialect=dialect)
        y = _compile_layer(node.inputs[1], dialect=dialect)
        z = _compile_layer(node.inputs[2], dialect=dialect)
        if x is None or y is None or z is None:
            return None
        window = int(node.attrs.get("window", _raw_literal(node, 3, 3)))
        raw_mp = node.attrs.get("min_periods", _raw_literal(node, 4, None))
        min_periods = window if raw_mp is None else int(raw_mp)
        over = (
            f"PARTITION BY x.inst ORDER BY x.ts "
            f"ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        )
        valid = (
            f"({_duckdb_valid('x._v')} AND {_duckdb_valid('y._v')} "
            f"AND {_duckdb_valid('z._v')})"
        )
        xv = f"CASE WHEN {valid} THEN x._v END"
        yv = f"CASE WHEN {valid} THEN y._v END"
        zv = f"CASE WHEN {valid} THEN z._v END"
        n = f"COUNT({xv}) OVER ({over})"
        rxy = f"corr({xv}, {yv}) OVER ({over})"
        rxz = f"corr({xv}, {zv}) OVER ({over})"
        ryz = f"corr({yv}, {zv}) OVER ({over})"
        denom = f"SQRT(GREATEST(0.0, 1 - ({rxz}) * ({rxz})) * GREATEST(0.0, 1 - ({ryz}) * ({ryz})))"
        return _Layer(
            f"SELECT x.ts, x.inst, CASE WHEN {n} < {min_periods} OR ({denom}) <= 0 "
            f"THEN NULL ELSE (({rxy}) - ({rxz}) * ({ryz})) / ({denom}) END AS _v "
            f"FROM ({x.sql}) x LEFT JOIN ({y.sql}) y USING (ts, inst) "
            f"LEFT JOIN ({z.sql}) z USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "ts_nth_value":
        if dialect != SqlDialect.DUCKDB:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        window = int(node.attrs.get("window", _raw_literal(node, 1, 3)))
        nth = int(node.attrs.get("n", _raw_literal(node, 2, 1)))
        order = str(node.attrs.get("order", _raw_literal(node, 3, "largest"))).lower()
        raw_mp = node.attrs.get("min_periods", _raw_literal(node, 4, None))
        min_periods = nth if raw_mp is None else int(raw_mp)
        if order not in {"largest", "smallest"}:
            return None
        over = "PARTITION BY inst ORDER BY ts"
        lags = ["_v"] + [f"LAG(_v, {i}) OVER ({over})" for i in range(1, window)]
        values = f"list_filter(list_value({', '.join(lags)}), v -> v IS NOT NULL AND NOT isnan(v) AND NOT isinf(v))"
        sorted_values = (
            f"list_reverse_sort({values})"
            if order == "largest"
            else f"list_sort({values})"
        )
        required = max(min_periods, nth)
        return _Layer(
            f"SELECT ts, inst, CASE WHEN list_count({values}) < {required} THEN NULL "
            f"ELSE list_extract({sorted_values}, {nth}) END AS _v FROM ({inner.sql}) nth0",
            has_inst_window=True,
        )

    if op == "ts_mean":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        return _Layer(
            _inst_window(dialect, spec.size, "AVG", inner.sql, min_periods=spec.min_periods),
            has_inst_window=True,
        )

    if op == "ts_std":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        over = (
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        )
        std_key = "stddev_pop" if spec.ddof == 0 else "stddev"
        expr = _rolling_std_min_periods_sql(
            value_col="_v",
            over=over,
            window=spec.size,
            scale_expr="1",
            dialect=dialect,
            min_periods=spec.min_periods,
            std_fn_name=std_key,
        )
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_sum":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        return _Layer(
            _inst_window(dialect, spec.size, "SUM", inner.sql, min_periods=spec.min_periods),
            has_inst_window=True,
        )

    if op == "ts_max":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        return _Layer(
            _inst_window(dialect, spec.size, "MAX", inner.sql, min_periods=spec.min_periods),
            has_inst_window=True,
        )

    if op == "ts_min":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        return _Layer(
            _inst_window(dialect, spec.size, "MIN", inner.sql, min_periods=spec.min_periods),
            has_inst_window=True,
        )

    if op == "ts_zscore":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        over = (
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        )
        std_key = "stddev_pop" if spec.ddof == 0 else "stddev"
        expr = _zscore_window_expr(
            value_col="_v",
            partition=over,
            dialect=dialect,
            std_fn_name=std_key,
            min_periods=spec.min_periods,
        )
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "ts_sharpe":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node)
        ann = _float_attr(node, "ann_factor", default=252.0)
        sqrt_af = ann**0.5
        mp = max(2, w // 3)
        over = (
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        cnt = f"COUNT(_v) OVER ({over})"
        std = _dialect_fn(dialect, "stddev")
        nf = _dialect_fn(dialect, "nullif")
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE "
            f"WHEN {cnt} < {mp} THEN NULL "
            f"WHEN {std}(_v) OVER ({over}) = 0 THEN NULL "
            f"ELSE AVG(_v) OVER ({over}) / {nf}({std}(_v) OVER ({over}), 0) "
            f"END * {sqrt_af} AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op in {"ts_delay", "delay"}:
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
        pos_p = _literal_positional(node, 0)
        if pos_p is not None:
            p = pos_p
        qexpr = _quantile_over(dialect, "_v", p, "PARTITION BY ts")
        return _Layer(
            f"SELECT ts, inst, {qexpr} AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op in {"log_returns", "ts_log_return"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        d = _window_int(node, default=1)
        lag = f"LAG(_v, {d}) OVER (PARTITION BY inst ORDER BY ts)"
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL OR {lag} IS NULL OR _v <= 0 OR {lag} <= 0 THEN NULL "
            f"ELSE {ln}(_v / {lag}) END AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "volatility":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        w = _window_int(node, default=20)
        mp = _rolling_min_periods(w)
        over = f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        sqrt_fn = _dialect_fn(dialect, "sqrt")
        scale = f"{sqrt_fn}(252)"
        expr = _rolling_std_min_periods_sql(
            value_col="_v",
            over=over,
            window=w,
            scale_expr=scale,
            dialect=dialect,
            min_periods=mp,
        )
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "vwap":
        if len(node.inputs) < 2:
            return None
        price = _compile_layer(node.inputs[0], dialect=dialect)
        vol = _compile_layer(node.inputs[1], dialect=dialect)
        if price is None or vol is None:
            return None
        from backend.pairwise_rolling import sql_pairwise_vwap_expr

        spec = _window_spec(node, default=20)
        over = (
            f"PARTITION BY p.inst ORDER BY p.ts ROWS BETWEEN {spec.size - 1} PRECEDING AND CURRENT ROW"
        )
        vwap_expr = sql_pairwise_vwap_expr(
            price_col="p._v",
            volume_col="v._v",
            over=over,
            min_periods=spec.min_periods,
        )
        return _Layer(
            f"SELECT p.ts, p.inst, {vwap_expr} AS _v "
            f"FROM ({price.sql}) p LEFT JOIN ({vol.sql}) v USING (ts, inst)",
            has_inst_window=True,
        )

    if op in {"zscore", "standardize"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        expr = _zscore_window_expr(value_col="_v", partition="PARTITION BY ts", dialect=dialect)
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=True,
        )

    if op == "normalize":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        expr = _normalize_window_expr(value_col="_v", partition="PARTITION BY ts")
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({inner.sql}) t",
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

    cs_aggregates = {
        "c_mean": "AVG", "cs_mean": "AVG",
        "c_sum": "SUM", "cs_sum": "SUM",
        "c_count": "COUNT", "cs_count": "COUNT",
        "c_std": _dialect_fn(dialect, "stddev"),
        "cs_std": _dialect_fn(dialect, "stddev"),
    }
    if op in cs_aggregates:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        from backend.stat_valid import row_stat_invalid_sql

        invalid = row_stat_invalid_sql("_v", dialect=dialect, exclude_nan=True)
        cleaned = (
            f"SELECT ts, inst, CASE WHEN {invalid} THEN NULL ELSE _v END AS _v "
            f"FROM ({inner.sql}) t_clean"
        )
        agg = cs_aggregates[op]
        return _Layer(
            _cs_broadcast_agg(agg, cleaned, all_null_null=agg != "COUNT"),
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
        from backend.plan_params import int_mode_from_plan_node

        mode = 0 if op == "cs_resid" else int_mode_from_plan_node(node, input_index=2, default=0)
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
            f"FROM ({y_layer.sql}) y LEFT JOIN ({x_layer.sql}) x USING (ts, inst)",
            has_inst_window=y_layer.has_inst_window or x_layer.has_inst_window,
            has_ts_partition=True,
        )

    if op == "size_neutralize":
        # size_neutralize(x, market_cap) == cs_resid(x, ln(greatest(market_cap, 1)))
        if len(node.inputs) < 2:
            return None
        y_layer = _compile_layer(node.inputs[0], dialect=dialect)
        cap_layer = _compile_layer(node.inputs[1], dialect=dialect)
        if y_layer is None or cap_layer is None:
            return None
        ln = _dialect_fn(dialect, "ln")
        g = _dialect_fn(dialect, "greatest")
        ln_cap = f"{ln}({g}(c._v, 1))"
        part = "PARTITION BY y.ts"
        beta, alpha, n_valid = _cs_ols_components(
            dialect, y_col="y._v", x_col=ln_cap, partition=part
        )
        fit = f"({alpha}) + ({beta}) * ({ln_cap})"
        expr = (
            f"CASE WHEN y._v IS NULL OR c._v IS NULL THEN NULL "
            f"WHEN {n_valid} < 3 THEN NULL "
            f"ELSE y._v - ({fit}) END"
        )
        return _Layer(
            f"SELECT y.ts, y.inst, {expr} AS _v "
            f"FROM ({y_layer.sql}) y LEFT JOIN ({cap_layer.sql}) c USING (ts, inst)",
            has_inst_window=y_layer.has_inst_window or cap_layer.has_inst_window,
            has_ts_partition=True,
        )

    if op == "industry_size_neutralize":
        # Sequential dual: industry demean then size residual.
        if len(node.inputs) < 3:
            return None
        y_layer = _compile_layer(node.inputs[0], dialect=dialect)
        ind_layer = _compile_layer(node.inputs[1], dialect=dialect)
        cap_layer = _compile_layer(node.inputs[2], dialect=dialect)
        if y_layer is None or ind_layer is None or cap_layer is None:
            return None
        ln = _dialect_fn(dialect, "ln")
        g = _dialect_fn(dialect, "greatest")
        ln_cap = f"{ln}({g}(d.cap, 1))"
        demeaned_sql = (
            f"SELECT x.ts AS ts, x.inst AS inst, "
            f"(x._v - AVG(x._v) OVER (PARTITION BY x.ts, g._v)) AS dm, "
            f"c._v AS cap "
            f"FROM ({y_layer.sql}) x "
            f"LEFT JOIN ({ind_layer.sql}) g USING (ts, inst) "
            f"LEFT JOIN ({cap_layer.sql}) c USING (ts, inst)"
        )
        part = "PARTITION BY d.ts"
        beta, alpha, n_valid = _cs_ols_components(
            dialect, y_col="d.dm", x_col=ln_cap, partition=part
        )
        fit = f"({alpha}) + ({beta}) * ({ln_cap})"
        expr = (
            f"CASE WHEN d.dm IS NULL OR d.cap IS NULL THEN NULL "
            f"WHEN {n_valid} < 3 THEN NULL "
            f"ELSE d.dm - ({fit}) END"
        )
        return _Layer(
            f"SELECT d.ts, d.inst, {expr} AS _v FROM ({demeaned_sql}) d",
            has_inst_window=(
                y_layer.has_inst_window
                or ind_layer.has_inst_window
                or cap_layer.has_inst_window
            ),
            has_ts_partition=True,
        )

    if op == "scale":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        to_val = _float_attr(node, "to", default=1.0)
        if "to" not in (node.attrs or {}):
            for idx in range(1, len(node.inputs)):
                child = node.inputs[idx]
                if child.op == "literal":
                    val = child.attrs.get("value")
                    if val is not None and isinstance(val, (int, float)) and not isinstance(val, bool):
                        to_val = float(val)
                        break
        expr = _scale_window_expr(
            value_col="_v", partition="PARTITION BY ts", to_val=to_val, dialect=dialect
        )
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({inner.sql}) t",
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
                f"LEFT JOIN ({grp.sql}) g USING (ts, inst)",
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
            f"SELECT l.ts, l.inst, "
            f"CASE WHEN l._v IS NULL OR r._v IS NULL THEN NULL "
            f"ELSE {corr_fn}(l._v, r._v) OVER ({over}) END AS _v "
            f"FROM ({left.sql}) l LEFT JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "ts_autocorr":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        lag_pos = _literal_positional(node, 1, default=1.0)
        lag = max(int(lag_pos or 1), 1)
        w = max(lag + 2, _window_int(node))
        over = (
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        corr_expr = _rolling_corr_pandas_compat_expr(
            corr_col="_corr",
            std_left_col="_std_l",
            std_right_col="_std_r",
            left_col="curr_v",
            right_col="lagged_v",
            window_count_col="_win_cnt",
            window=w,
            dialect=dialect,
        )
        return _Layer(
            f"SELECT ts, inst, {corr_expr} AS _v "
            f"FROM ("
            f"SELECT ts, inst, curr_v, lagged_v, "
            f"COUNT(*) OVER ({over}) AS _win_cnt, "
            f"{'corr' if dialect == SqlDialect.DUCKDB else 'corrStable'}(curr_v, lagged_v) OVER ({over}) AS _corr, "
            f"{'stddev_samp' if dialect == SqlDialect.DUCKDB else 'stddevSamp'}(curr_v) OVER ({over}) AS _std_l, "
            f"{'stddev_samp' if dialect == SqlDialect.DUCKDB else 'stddevSamp'}(lagged_v) OVER ({over}) AS _std_r "
            f"FROM ("
            f"SELECT ts, inst, _v AS curr_v, "
            f"LAG(_v, {lag}) OVER (PARTITION BY inst ORDER BY ts) AS lagged_v "
            f"FROM ({inner.sql}) inner0"
            f") aligned"
            f") scored",
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

    if op in {
        "group_rank",
        "group_mean",
        "group_sum",
        "group_min",
        "group_max",
        "group_count",
        "group_zscore",
        "group_normalize",
        "group_std",
    }:
        if not node.inputs:
            return None
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        grp_layer = None
        if len(node.inputs) >= 2:
            grp_layer = _compile_layer(node.inputs[1], dialect=dialect)
            if grp_layer is None:
                return None
            part = "PARTITION BY x.ts, g._v"
            join = f"FROM ({inner.sql}) x LEFT JOIN ({grp_layer.sql}) g USING (ts, inst)"
        else:
            part = "PARTITION BY x.ts"
            join = f"FROM ({inner.sql}) x"
        if op == "group_rank":
            wrapped, keys = _group_partition_wrap(
                inner.sql,
                grp_layer.sql if grp_layer is not None else None,
            )
            return _Layer(
                _cs_average_rank_pct_sql(wrapped, partition_keys=keys, dialect=dialect),
                has_inst_window=inner.has_inst_window,
                has_ts_partition=True,
            )
        if op == "group_mean":
            expr = f"AVG(x._v) OVER ({part})"
        elif op == "group_sum":
            expr = f"CASE WHEN x._v IS NULL THEN NULL ELSE SUM(x._v) OVER ({part}) END"
        elif op == "group_min":
            expr = f"MIN(x._v) OVER ({part})"
        elif op == "group_max":
            expr = f"MAX(x._v) OVER ({part})"
        elif op == "group_count":
            expr = (
                f"CASE WHEN x._v IS NULL THEN NULL "
                f"ELSE CAST(COUNT(x._v) OVER ({part}) AS DOUBLE) END"
            )
        elif op == "group_std":
            std_fn = _dialect_fn(dialect, "stddev")
            cnt = f"COUNT(x._v) OVER ({part})"
            std_val = f"{std_fn}(x._v) OVER ({part})"
            expr = (
                f"CASE WHEN x._v IS NULL THEN NULL "
                f"WHEN {cnt} < 2 THEN 0.0 "
                f"WHEN {std_val} IS NULL THEN 0.0 "
                f"ELSE {std_val} END"
            )
        elif op == "group_zscore":
            expr = _group_zscore_expr(value_col="x._v", partition=part, dialect=dialect)
        elif op == "group_normalize":
            expr = _group_minmax_expr(value_col="x._v", partition=part, dialect=dialect)
        else:
            expr = "NULL"
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
        if len(node.inputs) < 2:
            return None
        grp_layer = _compile_layer(node.inputs[1], dialect=dialect)
        if grp_layer is None:
            return None
        side = str(node.attrs.get("side", "top")).lower()
        if side not in {"top", "bottom"}:
            return None
        wrapped, keys = _group_partition_wrap(
            inner.sql,
            grp_layer.sql if grp_layer is not None else None,
            value_alias="_oval",
        )
        keys_csv = ", ".join(keys)
        numbered = (
            f"SELECT ts, inst, _oval, {keys_csv}, _oval AS _v "
            f"FROM ({wrapped}) t0"
        )
        rank_frac = _average_rank_frac_correlated(
            row_value_col="b._v",
            partition_keys=keys,
            numbered_sql=numbered,
            row_alias="b",
            dialect=dialect,
            descending=side == "top",
        )
        expr = (
            f"CASE WHEN b._oval IS NULL THEN NULL "
            f"WHEN ({rank_frac}) <= {p} THEN 1.0 ELSE 0.0 END"
        )
        return _Layer(
            f"SELECT ts, inst, {expr} AS _v FROM ({numbered}) b",
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
            join = f"FROM ({inner.sql}) x LEFT JOIN ({grp.sql}) g USING (ts, inst)"
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
        spec = _window_spec(node)
        med = "median" if dialect == SqlDialect.DUCKDB else "median"
        return _Layer(
            _inst_window(dialect, spec.size, med, inner.sql, min_periods=spec.min_periods),
            has_inst_window=True,
        )

    if op == "ts_var":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        if spec.ddof == 0:
            var_fn = "VAR_POP" if dialect == SqlDialect.DUCKDB else "varPop"
        else:
            var_fn = "VAR_SAMP" if dialect == SqlDialect.DUCKDB else "varSamp"
        return _Layer(
            _inst_window(dialect, spec.size, var_fn, inner.sql, min_periods=spec.min_periods),
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
            join = f"FROM ({inner.sql}) x LEFT JOIN ({grp.sql}) g USING (ts, inst)"
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
        from backend.pair_window_spec import PairWindowSpec
        from backend.pairwise_rolling import sql_pairwise_beta_expr

        pspec = PairWindowSpec.from_plan_node(node)
        w = pspec.size
        over = (
            f"PARTITION BY l.inst ORDER BY l.ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        beta_expr = sql_pairwise_beta_expr(
            left_col="l._v",
            right_col="r._v",
            over=over,
            dialect_is_duckdb=dialect == SqlDialect.DUCKDB,
            min_periods=pspec.min_periods,
        )
        return _Layer(
            f"SELECT l.ts, l.inst, "
            f"CASE WHEN l._v IS NULL OR r._v IS NULL THEN NULL ELSE ({beta_expr}) END AS _v "
            f"FROM ({left.sql}) l LEFT JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "ts_mad":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        w = spec.size
        mp = spec.min_periods if node.attrs.get("min_periods") is not None else w
        scale = _float_attr(node, "scale", default=1.0)
        over = (
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        if dialect == SqlDialect.DUCKDB:
            vals = (
                f"list(_v) FILTER (WHERE _v IS NOT NULL AND isfinite(_v)) "
                f"OVER ({over})"
            )
            inf_count = (
                f"COUNT(*) FILTER (WHERE _v IS NOT NULL AND isinf(_v)) "
                f"OVER ({over})"
            )
            return _Layer(
                f"SELECT ts, inst, "
                f"CASE WHEN _inf_count > 0 OR length(_vals) < {mp} THEN NULL "
                f"ELSE {scale} * list_median(list_transform("
                f"_vals, x -> abs(x - list_median(_vals)))) END AS _v "
                f"FROM (SELECT ts, inst, {vals} AS _vals, "
                f"{inf_count} AS _inf_count FROM ({inner.sql}) t0) t",
                has_inst_window=True,
            )
        vals = f"groupArrayIf(_v, isFinite(_v)) OVER ({over})"
        inf_count = f"countIf(isInfinite(_v)) OVER ({over})"
        return _Layer(
            f"SELECT ts, inst, "
            f"if(_inf_count > 0 OR length(_vals) < {mp}, NULL, {scale} * "
            f"arrayReduce('medianExact', arrayMap("
            f"x -> abs(x - arrayReduce('medianExact', _vals)), _vals))) AS _v "
            f"FROM (SELECT ts, inst, {vals} AS _vals, "
            f"{inf_count} AS _inf_count FROM ({inner.sql}) t0) t",
            has_inst_window=True,
        )

    if op in {"ts_ema", "ema", "ewm_mean"}:
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

    if op == "ts_rank":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        spec = _window_spec(node)
        return _Layer(
            _ts_pct_rank_sql(
                inner.sql,
                window=spec.size,
                dialect=dialect,
                min_periods=spec.min_periods,
            ),
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
        return None

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
        if op == "nan_to_num":
            isnan_fn = "isNaN" if dialect == SqlDialect.CLICKHOUSE else "isnan"
            isinf_fn = "isInfinite" if dialect == SqlDialect.CLICKHOUSE else "isinf"
            val_expr = (
                f"CASE WHEN _v IS NULL OR {isnan_fn}(_v) OR {isinf_fn}(_v) THEN {lit} ELSE _v END"
            )
        else:
            val_expr = f"{coalesce}(_v, {lit})"
        return _Layer(
            f"SELECT ts, inst, {val_expr} AS _v FROM ({inner.sql}) t",
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

    if op in {"ts_decay_linear", "decay_linear"}:
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
        pow_fn = "pow" if dialect == SqlDialect.CLICKHOUSE else "POW"
        return _Layer(
            f"SELECT l.ts, l.inst, "
            f"CASE "
            f"WHEN l._v IS NULL OR r._v IS NULL THEN NULL "
            f"WHEN l._v < 0 AND r._v <> FLOOR(r._v) THEN NULL "
            f"WHEN l._v = 0 AND r._v < 0 THEN NULL "
            f"ELSE {pow_fn}(l._v, r._v) END AS _v "
            f"FROM ({left.sql}) l LEFT JOIN ({right.sql}) r USING (ts, inst)",
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
            f"CASE WHEN {_truthy_sql('l._v')} AND {_truthy_sql('r._v')} THEN 1.0 ELSE 0.0 END AS _v "
            f"FROM ({left.sql}) l LEFT JOIN ({right.sql}) r USING (ts, inst)",
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
            f"CASE WHEN {_truthy_sql('l._v')} OR {_truthy_sql('r._v')} THEN 1.0 ELSE 0.0 END AS _v "
            f"FROM ({left.sql}) l LEFT JOIN ({right.sql}) r USING (ts, inst)",
            has_inst_window=left.has_inst_window or right.has_inst_window,
            has_ts_partition=left.has_ts_partition or right.has_ts_partition,
        )

    if op == "not_":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN _v IS NULL THEN 1.0 "
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
            f"SELECT l.ts, l.inst, "
            f"CASE WHEN l._v IS NULL OR r._v IS NULL THEN NULL "
            f"ELSE {cov_fn}(l._v, r._v) OVER ({over}) END AS _v "
            f"FROM ({left.sql}) l LEFT JOIN ({right.sql}) r USING (ts, inst)",
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
        spec = _window_spec(node)
        w = spec.size
        mp = spec.min_periods if node.attrs.get("min_periods") is not None else w
        over = (
            f"PARTITION BY inst ORDER BY ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"
        )
        if dialect == SqlDialect.DUCKDB:
            valid = "_v IS NOT NULL AND isfinite(_v)"
            count_valid = f"COUNT(*) FILTER (WHERE {valid}) OVER ({over})"
            count_zero = f"COUNT(*) FILTER (WHERE {valid} AND _v = 0) OVER ({over})"
            count_neg = f"COUNT(*) FILTER (WHERE {valid} AND _v < 0) OVER ({over})"
            count_inf = (
                f"COUNT(*) FILTER (WHERE _v IS NOT NULL AND isinf(_v)) OVER ({over})"
            )
        else:
            valid = "isFinite(_v)"
            count_valid = f"countIf({valid}) OVER ({over})"
            count_zero = f"countIf({valid} AND _v = 0) OVER ({over})"
            count_neg = f"countIf({valid} AND _v < 0) OVER ({over})"
            count_inf = f"countIf(isInfinite(_v)) OVER ({over})"
        log_sum = (
            f"SUM(CASE WHEN {valid} AND _v != 0 THEN {ln}(ABS(_v)) ELSE NULL END) "
            f"OVER ({over})"
        )
        return _Layer(
            f"SELECT ts, inst, "
            f"CASE WHEN {count_inf} > 0 OR {count_valid} < {mp} THEN NULL "
            f"WHEN {count_zero} > 0 THEN 0.0 "
            f"WHEN {log_sum} > 709.782712893384 THEN NULL "
            f"ELSE CASE WHEN MOD({count_neg}, 2) = 1 THEN -1.0 ELSE 1.0 END "
            f"* EXP({log_sum}) END AS _v "
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

    if op == "ts_regression_slope":
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
            f"FROM ({y_layer.sql}) y LEFT JOIN ({x_layer.sql}) x USING (ts, inst)",
            has_inst_window=True,
        )

    if op == "cum_sum":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _inst_cum_sum_sql(inner.sql),
            has_inst_window=True,
        )

    if op == "cum_max":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _inst_cum_agg("MAX", inner.sql),
            has_inst_window=True,
        )

    if op == "cum_min":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _inst_cum_agg("MIN", inner.sql),
            has_inst_window=True,
        )

    if op == "cum_prod":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        prod_fn = "product"
        return _Layer(
            _inst_cum_agg(prod_fn, inner.sql),
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
        over = "PARTITION BY inst ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
        run = f"SUM(CASE WHEN _v IS NULL THEN 0 ELSE _v END) OVER ({over})"
        cnt = f"SUM(CASE WHEN _v IS NOT NULL THEN 1 ELSE 0 END) OVER ({over})"
        body = f"CASE WHEN _v IS NULL THEN NULL WHEN {cnt} <= 0 THEN NULL ELSE {run} / {cnt} END"
        return _Layer(
            f"SELECT ts, inst, {body} AS _v FROM ({inner.sql}) t",
            has_inst_window=True,
        )

    if op == "expanding_sum":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _inst_cum_agg("SUM", inner.sql, sum_skip_null=True),
            has_inst_window=True,
        )

    if op in {"cum_std", "expanding_std"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        std_fn = _dialect_fn(dialect, "stddev")
        return _Layer(
            _inst_cum_agg(std_fn, inner.sql),
            has_inst_window=True,
        )

    if op == "count":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            _inst_cum_agg("COUNT", inner.sql),
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
            f"CASE WHEN _v IS NULL THEN NULL "
            f"WHEN {abs_fn}(_v) = 0 THEN NULL "
            f"ELSE {ln_fn}({abs_fn}(_v)) END AS _v "
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

    if op == "tanh":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        # Both supported dialects expose tanh. NULL remains NULL and IEEE
        # infinities saturate to +/-1 in DuckDB, matching NumPy/Polars.
        return _Layer(
            f"SELECT ts, inst, tanh(_v) AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "cbrt":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        return _Layer(
            "SELECT ts, inst, sign(_v) * power(abs(_v), 1.0 / 3.0) AS _v "
            f"FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op == "truncate":
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        decimals = int(node.attrs.get("decimals", node.attrs.get("k", node.attrs.get("d", 0))))
        scale = 10.0 ** decimals
        return _Layer(
            f"SELECT ts, inst, trunc(_v * {scale!r}) / {scale!r} AS _v FROM ({inner.sql}) t",
            has_inst_window=inner.has_inst_window,
            has_ts_partition=inner.has_ts_partition,
        )

    if op in {"ewm_std", "ts_ewm_std"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        span = _window_int(node, default=20)
        return _Layer(
            _ewm_weighted_moment_sql(inner.sql, span, dialect=dialect, sqrt=True),
            has_inst_window=True,
        )

    if op in {"ewm_var", "ts_ewm_var"}:
        inner = _compile_layer(node.inputs[0], dialect=dialect)
        if inner is None:
            return None
        span = _window_int(node, default=20)
        return _Layer(
            _ewm_weighted_moment_sql(inner.sql, span, dialect=dialect, sqrt=False),
            has_inst_window=True,
        )

    if op in {"Slope", "ts_time_slope"}:
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

    if op in {"ewm_cov", "ewm_corr", "ts_ewm_cov", "ts_ewm_corr"}:
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
                corr=(op in {"ewm_corr", "ts_ewm_corr"}),
            ),
            has_inst_window=True,
        )

    return None


def _collect_columns(node: PlanNode, out: set[str]) -> None:
    """递归收集计划树引用的列名到 ``out`` 集合。"""
    if node.op == "column":
        name = node.attrs.get("name")
        if name:
            out.add(str(name))
    for child in node.inputs:
        _collect_columns(child, out)


def plan_is_sql_capable(plan: PlanNode) -> bool:
    """判断计划树是否全部由 SQL backend 支持（委托 ``sql_registry.is_sql_capable``）。"""
    from backend.sql_pushdown.sql_registry import is_sql_capable

    return is_sql_capable(plan)


def _build_filter_clause(
    filt: SqlPushdownFilter | None,
    *,
    dialect: SqlDialect,
) -> str:
    """根据 ``SqlPushdownFilter`` 生成 base CTE 的 WHERE 子句。"""
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
    """构造 base CTE：标准化 ``ts``/``inst`` 轴列并应用过滤。"""
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
    """将单因子逻辑计划编译为可执行 SQL。

    DuckDB 使用 registry 数据集名（``{{dataset}}`` 占位符），ClickHouse 使用物理表名。
    计划不可 SQL 化或编译失败时返回 ``None``。
    """
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
    """将多个 SQL 可编译子树合并为一条 WITH 查询（共享 base CTE）。

    返回 ``BatchCompiledSql``，含各 ``sid`` 对应的 SQL 列别名；任一子树
    不可编译时返回 ``None``。
    """
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
