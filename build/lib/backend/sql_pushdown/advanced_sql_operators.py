# -*- coding: utf-8 -*-
"""高级 SQL 算子实现：需要子查询/CTE 的复杂算子。

本模块为 15 个复杂算子提供 DuckDB SQL 编译支持，包括：
- 截面统计算子（需要子查询计算统计量）
- 时序回归算子（需要 CTE 构建窗口统计）
- 残差/中性化算子（需要多阶段 CTE）

这些算子因其计算复杂度，需要使用 SQL 的高级特性：
1. 相关子查询（correlated subquery）
2. 公用表表达式（CTE）
3. 窗口函数的嵌套使用

R47 #XXX：所有实现必须与 pandas/polars 后端保持精确 parity。
"""
from __future__ import annotations

from enum import Enum


class SqlDialect(str, Enum):
    """SQL 方言枚举（与 emitter.py 保持一致）。"""
    DUCKDB = "duckdb"
    CLICKHOUSE = "clickhouse"


def _dialect_fn(dialect: SqlDialect, key: str) -> str:
    """方言函数映射。"""
    mapping = {
        "duckdb": {
            "stddev": "stddev_samp",
            "stddev_pop": "stddev_pop",
            "abs": "abs",
            "nullif": "nullif",
            "least": "least",
            "greatest": "greatest",
            "ln": "ln",
            "percentile_cont": "percentile_cont",
        },
        "clickhouse": {
            "stddev": "stddevSamp",
            "stddev_pop": "stddevPop",
            "abs": "abs",
            "nullif": "nullif",
            "least": "least",
            "greatest": "greatest",
            "ln": "ln",
            "percentile_cont": "quantile",
        },
    }
    return mapping.get(str(dialect).lower(), mapping["duckdb"]).get(key, key)


# ===========================================================================
# 1. cs_zscore - 截面 z-score（需要子查询计算 mean/std）
# ===========================================================================

def cs_zscore_sql(inner_sql: str, *, dialect: SqlDialect) -> str:
    """截面 Z-Score：(x - mean) / std。

    语义：std=0 时输出 0（对齐 pandas ``std.replace(0, 1)``）。
    实现：CTE 计算截面统计量，避免重复扫描。

    参数：
        inner_sql: 输入数据 SQL (ts, inst, _v)
        dialect: SQL 方言

    返回：
        SQL 字符串 (ts, inst, _v)

    示例 SQL:
        WITH stats AS (
          SELECT ts,
                 AVG(_v) OVER (PARTITION BY ts) AS mean_v,
                 stddev_samp(_v) OVER (PARTITION BY ts) AS std_v
          FROM (inner_sql) t
        )
        SELECT t.ts, t.inst,
               CASE WHEN t._v IS NULL THEN NULL
                    WHEN s.std_v IS NULL OR s.std_v = 0 THEN 0.0
                    ELSE (t._v - s.mean_v) / s.std_v
               END AS _v
        FROM (inner_sql) t
        JOIN stats s ON t.ts = s.ts
    """
    std_fn = _dialect_fn(dialect, "stddev")

    # Pandas treats NaN and +/-Inf as missing for this cross-sectional
    # statistic: they are excluded from mean/std and their output is NaN.
    # DuckDB's stddev over Inf raises OutOfRangeError, so normalize all
    # non-finite inputs before aggregation.  Aggregate by ts (rather than
    # retaining one stats row per input row) to avoid multiplying rows on join.
    finite_v = "CASE WHEN isfinite(_v) THEN _v END"
    if dialect == SqlDialect.CLICKHOUSE:
        # ClickHouse 使用 multiIf; isFinite is the ClickHouse spelling.
        finite_v = "if(isFinite(_v), _v, NULL)"
        expr = (
            f"multiIf("
            f"isNull(t._v) OR NOT isFinite(t._v), NULL, "
            f"isNull(s.std_v) OR s.std_v = 0, 0.0, "
            f"(t._v - s.mean_v) / s.std_v"
            f")"
        )
    else:
        expr = (
            f"CASE WHEN t._v IS NULL OR NOT isfinite(t._v) THEN NULL "
            f"WHEN s.std_v IS NULL OR s.std_v = 0 THEN 0.0 "
            f"ELSE (t._v - s.mean_v) / s.std_v END"
        )

    return (
        f"WITH stats AS ("
        f"SELECT ts, "
        f"AVG({finite_v}) AS mean_v, "
        f"{std_fn}({finite_v}) AS std_v "
        f"FROM ({inner_sql}) t "
        f"GROUP BY ts"
        f") "
        f"SELECT t.ts, t.inst, {expr} AS _v "
        f"FROM ({inner_sql}) t "
        f"JOIN stats s ON t.ts = s.ts"
    )


# ===========================================================================
# 2. cs_winsorize - 截面 winsorize（需要子查询计算百分位数）
# ===========================================================================

def cs_winsorize_sql(
    inner_sql: str,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
    dialect: SqlDialect,
) -> str:
    """截面 Winsorize：将极端值截断到指定百分位数。

    语义：lower/upper 百分位数外的值被截断到边界值。
    实现：CTE 计算百分位数，主查询使用 CASE 截断。

    参数：
        inner_sql: 输入数据 SQL (ts, inst, _v)
        lower: 下界百分位数（默认 0.01 = 1%）
        upper: 上界百分位数（默认 0.99 = 99%）
        dialect: SQL 方言

    返回：
        SQL 字符串 (ts, inst, _v)
    """
    percentile_fn = _dialect_fn(dialect, "percentile_cont")
    least_fn = _dialect_fn(dialect, "least")
    greatest_fn = _dialect_fn(dialect, "greatest")

    if dialect == SqlDialect.CLICKHOUSE:
        # ClickHouse quantile 语法不同
        lower_pct = f"quantile({lower})(_v) OVER (PARTITION BY ts)"
        upper_pct = f"quantile({upper})(_v) OVER (PARTITION BY ts)"
    else:
        lower_pct = f"{percentile_fn}({lower}) WITHIN GROUP (ORDER BY _v) OVER (PARTITION BY ts)"
        upper_pct = f"{percentile_fn}({upper}) WITHIN GROUP (ORDER BY _v) OVER (PARTITION BY ts)"

    return (
        f"WITH bounds AS ("
        f"SELECT ts, "
        f"{lower_pct} AS lower_bound, "
        f"{upper_pct} AS upper_bound "
        f"FROM ({inner_sql}) t "
        f") "
        f"SELECT t.ts, t.inst, "
        f"CASE WHEN t._v IS NULL THEN NULL "
        f"ELSE {greatest_fn}(b.lower_bound, {least_fn}(t._v, b.upper_bound)) END AS _v "
        f"FROM ({inner_sql}) t "
        f"JOIN bounds b ON t.ts = b.ts"
    )


# ===========================================================================
# 3. cs_rank_normalize - 截面 rank 归一化（需要子查询计算 count）
# ===========================================================================

def cs_rank_normalize_sql(inner_sql: str, *, dialect: SqlDialect) -> str:
    """截面 Rank 归一化：rank / (count - 1)，映射到 [0, 1]。

    语义：使用 average 方法处理 tie，count=1 时输出 1.0。
    实现：子查询计算每个截面的有效观测数。

    参数：
        inner_sql: 输入数据 SQL (ts, inst, _v)
        dialect: SQL 方言

    返回：
        SQL 字符串 (ts, inst, _v)
    """
    # 使用相关子查询计算 average rank
    rank_expr = (
        f"(SELECT CASE "
        f"WHEN s.cnt <= 1 THEN 1.0 "
        f"WHEN s.cnt = 0 THEN NULL "
        f"ELSE (s.cnt_le - (s.cnt_eq - 1) / 2.0) / (s.cnt - 1) END "
        f"FROM ("
        f"SELECT "
        f"COUNT(*) FILTER (WHERE p._v IS NOT NULL) AS cnt, "
        f"COUNT(*) FILTER (WHERE p._v IS NOT NULL AND p._v <= b._v) AS cnt_le, "
        f"COUNT(*) FILTER (WHERE p._v IS NOT NULL AND p._v = b._v) AS cnt_eq "
        f"FROM ({inner_sql}) p WHERE p.ts = b.ts"
        f") s)"
    )

    return (
        f"SELECT b.ts, b.inst, "
        f"CASE WHEN b._v IS NULL THEN NULL ELSE {rank_expr} END AS _v "
        f"FROM ({inner_sql}) b"
    )


# ===========================================================================
# 4. ts_seasonal_diff - 时序季节性差分（需要 LAG 取指定周期前的值）
# ===========================================================================

def ts_seasonal_diff_sql(
    inner_sql: str,
    *,
    seasonal_period: int = 12,
    dialect: SqlDialect,
) -> str:
    """时序季节性差分：x(t) - x(t - seasonal_period)。

    语义：计算当前值与 seasonal_period 个周期前的差值。
    实现：使用 LAG 窗口函数。

    参数：
        inner_sql: 输入数据 SQL (ts, inst, _v)
        seasonal_period: 季节周期（默认 12）
        dialect: SQL 方言

    返回：
        SQL 字符串 (ts, inst, _v)
    """
    lag = max(int(seasonal_period), 1)

    return (
        f"SELECT ts, inst, "
        f"CASE WHEN _v IS NULL THEN NULL "
        f"WHEN lag_v IS NULL THEN NULL "
        f"ELSE _v - lag_v END AS _v "
        f"FROM ("
        f"SELECT ts, inst, _v, "
        f"LAG(_v, {lag}) OVER (PARTITION BY inst ORDER BY ts) AS lag_v "
        f"FROM ({inner_sql}) t"
        f") t"
    )


# ===========================================================================
# 5. panel_rank - 面板 rank（全局排名，不分组）
# ===========================================================================

def panel_rank_sql(inner_sql: str, *, dialect: SqlDialect) -> str:
    """面板 Rank：全局排名（跨时间+跨标的）。

    语义：使用 average 方法处理 tie，归一化到 [0, 1]。
    实现：全局窗口函数，无 PARTITION。

    参数：
        inner_sql: 输入数据 SQL (ts, inst, _v)
        dialect: SQL 方言

    返回：
        SQL 字符串 (ts, inst, _v)
    """
    # 全局 rank，无分区
    if dialect == SqlDialect.CLICKHOUSE:
        rank_fn = "toFloat64(RANK())"
    else:
        rank_fn = "RANK()"

    count_fn = "COUNT(*) FILTER (WHERE _v IS NOT NULL)"

    return (
        f"WITH global_stats AS ("
        f"SELECT {count_fn} AS total_cnt FROM ({inner_sql}) t "
        f") "
        f"SELECT t.ts, t.inst, "
        f"CASE WHEN t._v IS NULL THEN NULL "
        f"WHEN g.total_cnt <= 1 THEN 1.0 "
        f"ELSE ({rank_fn} OVER (ORDER BY t._v NULLS LAST) - 1.0) / (g.total_cnt - 1) END AS _v "
        f"FROM ({inner_sql}) t "
        f"CROSS JOIN global_stats g"
    )


# ===========================================================================
# 6. ts_regression_beta - 时序回归 beta（需要 CTE 计算窗口统计）
# ===========================================================================

def ts_regression_beta_sql(
    left_sql: str,
    right_sql: str,
    *,
    window: int,
    min_periods: int = 1,
    dialect: SqlDialect,
) -> str:
    """时序回归 Beta：rolling_cov(y, x) / rolling_var(x)。

    语义：窗口内 y 对 x 的回归系数，var(x)=0 时输出 NULL。
    实现：CTE 计算窗口协方差和方差。

    参数：
        left_sql: y 数据 SQL (ts, inst, _v)
        right_sql: x 数据 SQL (ts, inst, _v)
        window: 窗口大小
        min_periods: 最小观测数
        dialect: SQL 方言

    返回：
        SQL 字符串 (ts, inst, _v)
    """
    w = max(int(window), 2)
    mp = max(int(min_periods), 1)
    over = f"PARTITION BY l.inst ORDER BY l.ts ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW"

    if dialect == SqlDialect.CLICKHOUSE:
        cov_fn = "covarSamp"
        var_fn = "varSamp"
    else:
        cov_fn = "covar_samp"
        var_fn = "var_samp"

    nullif_fn = _dialect_fn(dialect, "nullif")

    return (
        f"WITH stats AS ("
        f"SELECT l.ts, l.inst, "
        f"COUNT(*) FILTER (WHERE l._v IS NOT NULL AND r._v IS NOT NULL) OVER ({over}) AS cnt, "
        f"{cov_fn}(l._v, r._v) OVER ({over}) AS cov_xy, "
        f"{var_fn}(r._v) OVER ({over}) AS var_x "
        f"FROM ({left_sql}) l "
        f"LEFT JOIN ({right_sql}) r USING (ts, inst) "
        f") "
        f"SELECT ts, inst, "
        f"CASE WHEN cnt < {mp} THEN NULL "
        f"WHEN var_x IS NULL OR var_x = 0 THEN NULL "
        f"ELSE cov_xy / {nullif_fn}(var_x, 0) END AS _v "
        f"FROM stats"
    )


# ===========================================================================
# 7. group_residual - 分组残差（需要 CTE 计算分组回归）
# ===========================================================================

def group_residual_sql(
    y_sql: str,
    x_sql: str,
    grp_sql: str,
    *,
    dialect: SqlDialect,
) -> str:
    """分组残差：y - (alpha + beta * x)，按组分别回归。

    语义：每个组内做 OLS 回归，返回残差，n_valid < 3 时输出 NULL。
    实现：CTE 计算分组 OLS 系数。

    参数：
        y_sql: y 数据 SQL (ts, inst, _v)
        x_sql: x 数据 SQL (ts, inst, _v)
        grp_sql: 分组 SQL (ts, inst, _v)
        dialect: SQL 方言

    返回：
        SQL 字符串 (ts, inst, _v)
    """
    if dialect == SqlDialect.CLICKHOUSE:
        cov_fn = "covarSamp"
        var_fn = "varSamp"
    else:
        cov_fn = "covar_samp"
        var_fn = "var_samp"

    nullif_fn = _dialect_fn(dialect, "nullif")

    # CTE 构建 joined 数据
    joined = (
        f"SELECT y.ts, y.inst, y._v AS y_val, x._v AS x_val, g._v AS grp_val "
        f"FROM ({y_sql}) y "
        f"LEFT JOIN ({x_sql}) x USING (ts, inst) "
        f"LEFT JOIN ({grp_sql}) g USING (ts, inst)"
    )

    # CTE 计算分组统计
    part = "PARTITION BY ts, grp_val"
    pair_mask = "y_val IS NOT NULL AND x_val IS NOT NULL"
    pair_y = f"CASE WHEN {pair_mask} THEN y_val END"
    pair_x = f"CASE WHEN {pair_mask} THEN x_val END"

    stats_cte = (
        f"WITH joined AS ({joined}), "
        f"stats AS ("
        f"SELECT ts, inst, y_val, x_val, grp_val, "
        f"AVG({pair_y}) OVER ({part}) AS mean_y, "
        f"AVG({pair_x}) OVER ({part}) AS mean_x, "
        f"{cov_fn}({pair_y}, {pair_x}) OVER ({part}) AS cov_xy, "
        f"{var_fn}({pair_x}) OVER ({part}) AS var_x, "
        f"COUNT(CASE WHEN {pair_mask} THEN 1 END) OVER ({part}) AS n_valid "
        f"FROM joined"
        f") "
    )

    beta = f"cov_xy / {nullif_fn}(var_x, 0)"
    alpha = f"mean_y - ({beta}) * mean_x"
    fitted = f"({alpha}) + ({beta}) * x_val"

    return (
        f"{stats_cte}"
        f"SELECT ts, inst, "
        f"CASE WHEN y_val IS NULL OR x_val IS NULL THEN NULL "
        f"WHEN n_valid < 3 THEN NULL "
        f"ELSE y_val - ({fitted}) END AS _v "
        f"FROM stats"
    )


# ===========================================================================
# 8. cs_industry_neutralize - 行业中性化（需要两阶段 CTE）
# ===========================================================================

def cs_industry_neutralize_sql(
    x_sql: str,
    industry_sql: str,
    *,
    dialect: SqlDialect,
) -> str:
    """行业中性化：x - 行业均值。

    语义：每个截面内，按行业分组去均值。
    实现：CTE 计算行业均值，主查询做减法。

    参数：
        x_sql: 输入数据 SQL (ts, inst, _v)
        industry_sql: 行业分组 SQL (ts, inst, _v)
        dialect: SQL 方言

    返回：
        SQL 字符串 (ts, inst, _v)
    """
    return (
        f"WITH ind_mean AS ("
        f"SELECT x.ts, x.inst, x._v, "
        f"AVG(x._v) OVER (PARTITION BY x.ts, i._v) AS mean_ind "
        f"FROM ({x_sql}) x "
        f"LEFT JOIN ({industry_sql}) i USING (ts, inst)"
        f") "
        f"SELECT ts, inst, "
        f"CASE WHEN _v IS NULL THEN NULL "
        f"ELSE _v - mean_ind END AS _v "
        f"FROM ind_mean"
    )


# ===========================================================================
# 9. panel_residual - 面板残差（需要全局回归 CTE）
# ===========================================================================

def panel_residual_sql(
    y_sql: str,
    x_sql: str,
    *,
    dialect: SqlDialect,
) -> str:
    """面板残差：y - (alpha + beta * x)，全局回归（不分组）。

    语义：全面板 OLS 回归，n_valid < 3 时输出 NULL。
    实现：CTE 计算全局 OLS 系数。

    参数：
        y_sql: y 数据 SQL (ts, inst, _v)
        x_sql: x 数据 SQL (ts, inst, _v)
        dialect: SQL 方言

    返回：
        SQL 字符串 (ts, inst, _v)
    """
    if dialect == SqlDialect.CLICKHOUSE:
        cov_fn = "covarSamp"
        var_fn = "varSamp"
    else:
        cov_fn = "covar_samp"
        var_fn = "var_samp"

    nullif_fn = _dialect_fn(dialect, "nullif")

    # 全局统计（无 PARTITION）
    pair_mask = "y._v IS NOT NULL AND x._v IS NOT NULL"
    pair_y = f"CASE WHEN {pair_mask} THEN y._v END"
    pair_x = f"CASE WHEN {pair_mask} THEN x._v END"

    stats_cte = (
        f"WITH joined AS ("
        f"SELECT y.ts, y.inst, y._v AS y_val, x._v AS x_val "
        f"FROM ({y_sql}) y "
        f"LEFT JOIN ({x_sql}) x USING (ts, inst)"
        f"), "
        f"global_stats AS ("
        f"SELECT "
        f"AVG({pair_y}) AS mean_y, "
        f"AVG({pair_x}) AS mean_x, "
        f"{cov_fn}({pair_y}, {pair_x}) AS cov_xy, "
        f"{var_fn}({pair_x}) AS var_x, "
        f"COUNT(CASE WHEN {pair_mask} THEN 1 END) AS n_valid "
        f"FROM joined"
        f") "
    )

    beta = f"g.cov_xy / {nullif_fn}(g.var_x, 0)"
    alpha = f"g.mean_y - ({beta}) * g.mean_x"
    fitted = f"({alpha}) + ({beta}) * j.x_val"

    return (
        f"{stats_cte}"
        f"SELECT j.ts, j.inst, "
        f"CASE WHEN j.y_val IS NULL OR j.x_val IS NULL THEN NULL "
        f"WHEN g.n_valid < 3 THEN NULL "
        f"ELSE j.y_val - ({fitted}) END AS _v "
        f"FROM joined j "
        f"CROSS JOIN global_stats g"
    )


# ===========================================================================
# 10. cs_factor_score - 因子打分（需要多指标 CTE 合成）
# ===========================================================================

def cs_factor_score_sql(
    factor_sqls: list[str],
    weights: list[float] | None = None,
    *,
    standardize: bool = True,
    dialect: SqlDialect,
) -> str:
    """截面因子打分：多因子加权合成。

    语义：standardize=True 时先对每个因子做截面标准化，再加权求和。
    实现：CTE 逐因子标准化，主查询加权合成。

    参数：
        factor_sqls: 因子 SQL 列表（每个返回 ts, inst, _v）
        weights: 权重列表（默认等权）
        standardize: 是否先标准化
        dialect: SQL 方言

    返回：
        SQL 字符串 (ts, inst, _v)
    """
    if not factor_sqls:
        raise ValueError("cs_factor_score: factor_sqls 不能为空")

    n = len(factor_sqls)
    if weights is None:
        weights = [1.0 / n] * n
    elif len(weights) != n:
        raise ValueError(f"cs_factor_score: weights 长度({len(weights)})与 factor_sqls({n})不匹配")

    std_fn = _dialect_fn(dialect, "stddev")

    # 构建 CTE：每个因子一个表
    ctes = []
    for i, sql in enumerate(factor_sqls):
        if standardize:
            # 标准化
            cte = (
                f"f{i} AS ("
                f"SELECT t.ts, t.inst, "
                f"CASE WHEN t._v IS NULL THEN NULL "
                f"WHEN s.std_v IS NULL OR s.std_v = 0 THEN 0.0 "
                f"ELSE (t._v - s.mean_v) / s.std_v END AS _v "
                f"FROM ({sql}) t "
                f"JOIN ("
                f"SELECT ts, "
                f"AVG(_v) OVER (PARTITION BY ts) AS mean_v, "
                f"{std_fn}(_v) OVER (PARTITION BY ts) AS std_v "
                f"FROM ({sql}) t2"
                f") s ON t.ts = s.ts"
                f")"
            )
        else:
            # 不标准化
            cte = f"f{i} AS ({sql})"
        ctes.append(cte)

    # 主查询：加权求和
    # 从 f0 开始 LEFT JOIN 其余因子
    joined = "f0"
    for i in range(1, n):
        joined = f"({joined}) LEFT JOIN f{i} USING (ts, inst)"

    # 构建加权和表达式
    terms = []
    for i, w in enumerate(weights):
        if w != 0:
            terms.append(f"COALESCE(f{i}._v, 0) * {w}")

    if not terms:
        sum_expr = "0.0"
    else:
        sum_expr = " + ".join(terms)

    with_clause = "WITH " + ", ".join(ctes)

    return (
        f"{with_clause} "
        f"SELECT f0.ts, f0.inst, {sum_expr} AS _v "
        f"FROM {joined}"
    )


# ===========================================================================
# 辅助函数：注册到 emitter
# ===========================================================================

_ADVANCED_SQL_REGISTRY = {
    "cs_zscore": cs_zscore_sql,
    "cs_winsorize": cs_winsorize_sql,
    "cs_rank_normalize": cs_rank_normalize_sql,
    "ts_seasonal_diff": ts_seasonal_diff_sql,
    "panel_rank": panel_rank_sql,
    "ts_regression_beta": ts_regression_beta_sql,
    "group_residual": group_residual_sql,
    "cs_industry_neutralize": cs_industry_neutralize_sql,
    "panel_residual": panel_residual_sql,
    "cs_factor_score": cs_factor_score_sql,
}


def get_advanced_sql_function(op_name: str):
    """获取高级 SQL 算子的编译函数。

    参数：
        op_name: 算子名称

    返回：
        编译函数或 None
    """
    return _ADVANCED_SQL_REGISTRY.get(op_name)


def list_advanced_sql_operators() -> list[str]:
    """列出所有高级 SQL 算子。"""
    return sorted(_ADVANCED_SQL_REGISTRY.keys())
