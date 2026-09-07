import numpy as np
import pandas as pd
import pytest

from factor_engine.planner.logical_plan import PlanNode


def _plan(**attrs):
    return PlanNode("ts_zscore", inputs=(PlanNode("column", attrs={"name": "x"}),), attrs={"window": 3, "min_periods": 2, **attrs})


def _expected(values, **attrs):
    x = pd.DataFrame({"x": values})
    finite = x.replace([np.inf, -np.inf], np.nan)
    stats = finite if attrs.get("includes_current_bar", True) else finite.shift(1)
    mean = stats.rolling(3, min_periods=2).mean()
    std = stats.rolling(3, min_periods=2).std(ddof=attrs.get("ddof", 1))
    out = (finite - mean) / std
    out = out.mask(std.eq(0), 0.0 if attrs.get("zero_std_policy", "zero") == "zero" else np.nan)
    if attrs.get("nan_policy", "propagate") == "propagate":
        out = out.mask(stats.isna().astype(int).rolling(3, min_periods=1).sum() > 0)
    return out.where(finite.notna())["x"].to_numpy()


@pytest.mark.parametrize("attrs", [
    {"ddof": 0, "nan_policy": "ignore"},
    {"ddof": 1, "includes_current_bar": False, "nan_policy": "ignore"},
    {"nan_policy": "propagate"},
    {"zero_std_policy": "nan", "nan_policy": "ignore"},
])
def test_public_polars_long_compile_matches_reference(attrs):
    pl = pytest.importorskip("polars")
    from factor_engine.backend.polars_expr_emitter import compile_plan_to_polars
    values = [1.0, np.nan, 3.0, 3.0, 3.0]
    base = pl.DataFrame({"ts": range(5), "inst": ["A"] * 5, "x": values}).lazy()
    compiled = compile_plan_to_polars(_plan(**attrs), base)
    assert compiled is not None
    got = compiled.frame.collect()[compiled.value_col].to_numpy()
    np.testing.assert_allclose(got, _expected(values, **attrs), equal_nan=True)


@pytest.mark.parametrize("attrs", [
    {"ddof": 0, "nan_policy": "ignore"},
    {"ddof": 1, "includes_current_bar": False, "nan_policy": "ignore"},
    {"nan_policy": "propagate"},
    {"zero_std_policy": "nan", "nan_policy": "ignore"},
])
def test_public_duckdb_compile_matches_reference(attrs):
    duckdb = pytest.importorskip("duckdb")
    from factor_engine.backend.sql_pushdown.emitter import compile_plan_to_sql
    values = [1.0, np.nan, 3.0, 3.0, 3.0]
    data = pd.DataFrame({"trade_date": range(5), "ticker": ["A"] * 5, "x": values})
    compiled = compile_plan_to_sql(_plan(**attrs), dataset="input_data", time_column="trade_date", instrument_column="ticker")
    assert compiled is not None
    con = duckdb.connect()
    con.register("input_data", data)
    got = con.execute(compiled.query.replace("{{input_data}}", "input_data")).fetchdf().sort_values("ts")["value"].to_numpy()
    np.testing.assert_allclose(got, _expected(values, **attrs), equal_nan=True)


def test_fastpath_rewrite_preserves_constant_window_nan_semantics():
    from factor_engine.planner.rewrite_fastpath import rewrite_plan_for_fastpath
    col = PlanNode("column", attrs={"name": "x"})
    mean = PlanNode("ts_mean", inputs=(col,), attrs={"window": 3, "min_periods": 2})
    std = PlanNode("ts_std", inputs=(col,), attrs={"window": 3, "min_periods": 2})
    rewritten = rewrite_plan_for_fastpath(PlanNode("divide", inputs=(PlanNode("subtract", inputs=(col, mean)), std)))
    assert rewritten.op == "ts_zscore"
    assert rewritten.attrs["zero_std_policy"] == "nan"
