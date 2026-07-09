# -*- coding: utf-8 -*-
"""SQL 下推 emitter 单元测试。"""
from __future__ import annotations

from backend.sql_pushdown.emitter import compile_plan_to_sql, plan_is_sql_capable
from planner.logical_plan import PlanNode


def _col(name: str) -> PlanNode:
    return PlanNode(op="column", attrs={"name": name})


def _lit(v) -> PlanNode:
    return PlanNode(op="literal", attrs={"value": v})


def test_ts_mean_column_sql():
    plan = PlanNode(
        op="ts_mean",
        inputs=[_col("close")],
        attrs={"d": 3},
    )
    assert plan_is_sql_capable(plan)
    compiled = compile_plan_to_sql(
        plan,
        dataset="cn_stock_daily",
        time_column="trade_date",
        instrument_column="ticker",
    )
    assert compiled is not None
    assert "cn_stock_daily" in compiled.query
    assert "AVG" in compiled.query
    assert "ROWS BETWEEN 2 PRECEDING" in compiled.query
    assert compiled.referenced_columns == frozenset({"close"})


def test_rank_ts_mean_sql():
    plan = PlanNode(
        op="rank",
        inputs=[
            PlanNode(
                op="ts_mean",
                inputs=[_col("close")],
                attrs={"d": 2},
            )
        ],
    )
    assert plan_is_sql_capable(plan)
    compiled = compile_plan_to_sql(
        plan,
        dataset="panel",
        time_column="ts",
        instrument_column="inst",
    )
    assert compiled is not None
    assert "RANK()" in compiled.query
    assert "AVG" in compiled.query


def test_macd_not_sql_capable():
    plan = PlanNode(op="MACD", inputs=[_col("close")], attrs={"window": 12})
    assert not plan_is_sql_capable(plan)


def test_ts_rank_sql():
    plan = PlanNode(op="ts_rank", inputs=[_col("close")], attrs={"d": 5})
    assert plan_is_sql_capable(plan)
    compiled = compile_plan_to_sql(
        plan,
        dataset="panel",
        time_column="ts",
        instrument_column="inst",
    )
    assert compiled is not None
    assert "RANK()" in compiled.query
    assert "ROWS BETWEEN 4 PRECEDING" in compiled.query


def test_ewm_mean_sql():
    plan = PlanNode(op="ewm_mean", inputs=[_col("close")], attrs={"span": 10})
    assert plan_is_sql_capable(plan)
    compiled = compile_plan_to_sql(
        plan,
        dataset="panel",
        time_column="ts",
        instrument_column="inst",
    )
    assert compiled is not None
    assert "POW" in compiled.query or "exponentialMovingAverage" in compiled.query


def test_zscore_sql():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    plan = PlanNode(op="zscore", inputs=[_col("close")])
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    assert "STDDEV" in compiled.query or "stddev" in compiled.query.lower()


def test_ts_corr_sql():
    from backend.sql_pushdown.emitter import compile_plan_to_sql

    plan = PlanNode(
        op="ts_corr",
        inputs=[_col("close"), _col("volume")],
        attrs={"d": 3},
    )
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
    )
    assert compiled is not None
    assert "corr" in compiled.query.lower()
    assert '"close"' in compiled.query and '"volume"' in compiled.query


def test_ts_corr_sql_with_literal_window_input():
    """窗口参数在 literal 子节点时也应可编译（DSL 常见形态）。"""
    from backend.sql_pushdown.emitter import compile_plan_to_sql

    plan = PlanNode(
        op="ts_corr",
        inputs=[_col("close"), _col("volume"), PlanNode(op="literal", attrs={"value": 3}, inputs=[])],
        attrs={},
    )
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
    )
    assert compiled is not None
    assert "corr" in compiled.query.lower()


def test_group_neutralize_sql():
    from backend.sql_pushdown.emitter import compile_plan_to_sql

    plan = PlanNode(
        op="group_neutralize",
        inputs=[_col("close"), _col("industry")],
    )
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
    )
    assert compiled is not None
    assert "PARTITION BY x.ts, g._v" in compiled.query


def test_where_sql():
    from backend.sql_pushdown.emitter import compile_plan_to_sql

    plan = PlanNode(
        op="where",
        inputs=[_col("flag"), _col("a"), _col("b")],
    )
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
    )
    assert compiled is not None
    assert "CASE WHEN" in compiled.query


def test_ffill_sql():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    plan = PlanNode(op="ffill", inputs=[_col("close")])
    assert plan_is_sql_capable(plan)
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    assert "LAST_VALUE" in compiled.query
    assert "IGNORE NULLS" in compiled.query


def test_fillna_const_sql():
    from backend.sql_pushdown.emitter import compile_plan_to_sql

    plan = PlanNode(
        op="fillna_const",
        inputs=[_col("close"), _lit(0.0)],
    )
    assert plan_is_sql_capable(plan)
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
    )
    assert compiled is not None
    assert "COALESCE" in compiled.query.upper()


def test_ts_decay_linear_sql():
    from backend.sql_pushdown.emitter import compile_plan_to_sql

    plan = PlanNode(
        op="ts_decay_linear",
        inputs=[_col("close")],
        attrs={"d": 3},
    )
    assert plan_is_sql_capable(plan)
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
    )
    assert compiled is not None
    assert "LAG(_v, 1)" in compiled.query
    assert "NULLIF" in compiled.query.upper()


def test_decay_linear_alias_sql_capable():
    plan = PlanNode(
        op="decay_linear",
        inputs=[_col("close")],
        attrs={"window": 5},
    )
    assert plan_is_sql_capable(plan)


def test_bfill_sql_causal_passthrough():
    """bfill 为因果算子：SQL 透传 inner，不引用未来值。"""
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    plan = PlanNode(op="bfill", inputs=[_col("close")])
    assert plan_is_sql_capable(plan)
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    assert "FIRST_VALUE" not in compiled.query
    assert "ORDER BY ts DESC" not in compiled.query


def test_rank_skips_null_in_sql():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    plan = PlanNode(op="rank", inputs=[_col("close")])
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    assert "WHEN _v IS NULL THEN NULL" in compiled.query
    assert "COUNT(_v)" in compiled.query


def test_group_rank_skips_null_in_sql():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    plan = PlanNode(
        op="group_rank",
        inputs=[_col("close"), _col("industry")],
    )
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    assert "WHEN x._v IS NULL THEN NULL" in compiled.query
    assert "COUNT(x._v)" in compiled.query


def test_group_zscore_zero_std_sql():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    plan = PlanNode(
        op="group_zscore",
        inputs=[_col("close"), _col("industry")],
    )
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    assert "WHEN x._v IS NULL THEN NULL" in compiled.query
    assert "THEN 0" in compiled.query


def test_clickhouse_tier2_emitter():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    for op, attrs in (
        ("ffill", {}),
        ("bfill", {}),
        ("ts_decay_linear", {"d": 3}),
    ):
        plan = PlanNode(op=op, inputs=[_col("close")], attrs=attrs)
        compiled = compile_plan_to_sql(
            plan,
            table="panel",
            time_column="ts",
            instrument_column="inst",
            dialect=SqlDialect.CLICKHOUSE,
        )
        assert compiled is not None, op
        assert "panel" in compiled.query

    rank_plan = PlanNode(op="rank", inputs=[_col("close")])
    rank_sql = compile_plan_to_sql(
        rank_plan,
        table="panel",
        time_column="ts",
        instrument_column="inst",
        dialect=SqlDialect.CLICKHOUSE,
    )
    assert rank_sql is not None
    assert "isNull(_v)" in rank_sql.query
    assert "nullIf(COUNT(_v)" in rank_sql.query


def test_coalesce_sql():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    plan = PlanNode(
        op="coalesce",
        inputs=[_col("close"), _col("open"), _col("volume")],
    )
    assert plan_is_sql_capable(plan)
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    assert "COALESCE" in compiled.query
    assert "t0" in compiled.query and "t1" in compiled.query and "t2" in compiled.query
    assert compiled.referenced_columns == frozenset({"close", "open", "volume"})


def test_coalesce_clickhouse_sql():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    plan = PlanNode(op="coalesce", inputs=[_col("close"), _col("open")])
    compiled = compile_plan_to_sql(
        plan,
        table="panel",
        time_column="ts",
        instrument_column="inst",
        dialect=SqlDialect.CLICKHOUSE,
    )
    assert compiled is not None
    assert "coalesce(t0._v, t1._v)" in compiled.query


def test_protected_div_sql():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    plan = PlanNode(
        op="protected_div",
        inputs=[_col("close"), _col("volume")],
    )
    assert plan_is_sql_capable(plan)
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    assert "COALESCE" in compiled.query
    assert "CASE WHEN l._v IS NULL" in compiled.query


def test_winsorize_preserves_null_sql():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    plan = PlanNode(op="winsorize", inputs=[_col("close")], attrs={"a": 0.05})
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    assert "WHEN _v IS NULL THEN NULL" in compiled.query


def test_clip_positional_bounds_sql():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    plan = PlanNode(
        op="clip",
        inputs=[_col("close"), _lit(-2.0), _lit(2.0)],
    )
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    assert "-2" in compiled.query and "2" in compiled.query
    assert "WHEN _v IS NULL THEN NULL" in compiled.query


def test_protected_log_sql():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    plan = PlanNode(op="protected_log", inputs=[_col("close")])
    assert plan_is_sql_capable(plan)
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    assert "WHEN _v IS NULL" in compiled.query


def test_protected_sqrt_sql():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    plan = PlanNode(op="protected_sqrt", inputs=[_col("close")])
    assert plan_is_sql_capable(plan)
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    assert "WHEN _v IS NULL THEN NULL" in compiled.query


def test_nan_to_num_sql():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    plan = PlanNode(op="nan_to_num", inputs=[_col("close"), _lit(-1.0)])
    assert plan_is_sql_capable(plan)
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    assert "COALESCE(_v, -1" in compiled.query


def test_is_nan_sql():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    plan = PlanNode(op="is_nan", inputs=[_col("close")])
    assert plan_is_sql_capable(plan)
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    assert "isnan(_v)" in compiled.query


def test_is_finite_sql():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    plan = PlanNode(op="is_finite", inputs=[_col("close")])
    assert plan_is_sql_capable(plan)
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    assert "isfinite(_v)" in compiled.query


def test_fillna_zero_sql():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    plan = PlanNode(
        op="fillna",
        inputs=[_col("close"), _lit("zero")],
    )
    assert plan_is_sql_capable(plan)
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    assert "COALESCE(_v, 0" in compiled.query


def test_where_numeric_condition_sql():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    plan = PlanNode(
        op="where",
        inputs=[
            PlanNode(op="is_finite", inputs=[_col("close")]),
            _col("close"),
            _lit(0.0),
        ],
    )
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    assert "c._v IS NOT NULL AND c._v <> 0" in compiled.query


def test_normalize_sql():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    plan = PlanNode(op="normalize", inputs=[_col("close")])
    assert plan_is_sql_capable(plan)
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    assert "MIN(_v)" in compiled.query and "MAX(_v)" in compiled.query
    assert "WHEN _v IS NULL THEN NULL" in compiled.query


def test_standardize_resolves_to_zscore_sql():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    plan = PlanNode(op="standardize", inputs=[_col("close")])
    assert plan_is_sql_capable(plan)
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    assert "STDDEV" in compiled.query or "stddev" in compiled.query.lower()


def test_group_normalize_sql():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    plan = PlanNode(
        op="group_normalize",
        inputs=[_col("close"), _col("industry")],
    )
    assert plan_is_sql_capable(plan)
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    assert "MIN(x._v)" in compiled.query and "MAX(x._v)" in compiled.query
    assert "THEN 0.5" in compiled.query


def test_group_percentile_sql():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    plan = PlanNode(
        op="group_percentile",
        inputs=[_col("close"), _col("industry"), _lit(0.5)],
    )
    assert plan_is_sql_capable(plan)
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    assert "WHEN x._v IS NULL THEN 0.0" in compiled.query
    assert "<= 0.5" in compiled.query


def test_group_decay_linear_sql():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    plan = PlanNode(
        op="group_decay_linear",
        inputs=[_col("close"), _col("industry"), _lit(5)],
    )
    assert plan_is_sql_capable(plan)
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    assert "RANK()" in compiled.query
    assert "WHEN x._v IS NULL THEN NULL" in compiled.query


def test_cs_resid_sql():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    plan = PlanNode(op="cs_resid", inputs=[_col("close"), _col("open")])
    assert plan_is_sql_capable(plan)
    compiled = compile_plan_to_sql(
        plan,
        dataset="d",
        time_column="t",
        instrument_column="i",
        dialect=SqlDialect.DUCKDB,
    )
    assert compiled is not None
    assert "covar_samp" in compiled.query
    assert "var_samp" in compiled.query
    assert "< 3" in compiled.query
    assert "y._v -" in compiled.query


def test_cs_regression_sql_modes():
    from backend.sql_pushdown.emitter import SqlDialect, compile_plan_to_sql

    for mode, needle in ((0, "y._v -"), (1, "covar_samp"), (2, "+")):
        plan = PlanNode(
            op="cs_regression",
            inputs=[_col("close"), _col("open"), _lit(mode)],
        )
        assert plan_is_sql_capable(plan)
        compiled = compile_plan_to_sql(
            plan,
            dataset="d",
            time_column="t",
            instrument_column="i",
            dialect=SqlDialect.DUCKDB,
        )
        assert compiled is not None
        assert needle in compiled.query

