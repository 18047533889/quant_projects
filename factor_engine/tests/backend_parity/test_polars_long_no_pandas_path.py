# -*- coding: utf-8
"""PolarsLong native scan path：禁止 load_column / prefetch（真实 production 路径）。"""
from __future__ import annotations

import pytest

pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from cleaned_operators import load_all
from runtime.engine import FactorEngine


class NoPandasSource:
    """仅 ``scan_polars_long``；load/prefetch 一调用即失败。"""

    def __init__(self, lazy_base) -> None:
        self._lazy = lazy_base

    def load_column(self, name: str):
        raise AssertionError(f"load_column must not be called: {name!r}")

    def load_columns(self, names: list[str]):
        raise AssertionError(f"load_columns must not be called: {names!r}")

    def prefetch_columns(self, names: list[str]):
        raise AssertionError(f"prefetch_columns must not be called: {names!r}")

    def scan_polars_long(self, columns: list[str]):
        import polars as pl

        cols = ["ts", "inst"] + [c for c in sorted(columns) if c not in {"ts", "inst"}]
        return self._lazy.select([pl.col(c) for c in cols if c in self._lazy.collect_schema().names()])

    def scan_index_long(self):
        return self._lazy.select(["ts", "inst"]).unique()


@pytest.fixture(scope="module")
def source():
    import polars as pl

    load_all()
    df = pl.DataFrame(
        {
            "ts": pl.Series(
                ["2024-01-01", "2024-01-02", "2024-01-03"] * 2,
                dtype=pl.Date,
            ),
            "inst": ["A", "A", "A", "B", "B", "B"],
            "close": [10.0, 11.0, 10.5, 20.0, 21.0, 20.5],
            "open": [9.5, 10.5, 10.0, 19.5, 20.5, 20.0],
            "volume": [100.0, 110.0, 105.0, 200.0, 210.0, 205.0],
            "ret": [0.01, 0.02, -0.01, 0.03, 0.04, 0.01],
            "group_id": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            "flag": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            "neg": [1.0, -1.0, 4.0, 2.0, 3.0, -2.0],
        }
    )
    return NoPandasSource(df.lazy())


F = make_cleaned_call_factory

NO_PANDAS_CASES = [
    ("abs", lambda: F("abs")(col("close"))),
    ("neg", lambda: F("neg")(col("close"))),
    ("sign", lambda: F("sign")(col("close"))),
    ("add", lambda: F("add")(col("close"), col("open"))),
    ("subtract", lambda: F("subtract")(col("close"), col("open"))),
    ("multiply", lambda: F("multiply")(col("close"), col("open"))),
    ("divide", lambda: F("divide")(col("close"), col("open"))),
    ("maximum", lambda: F("maximum")(col("close"), col("open"))),
    ("minimum", lambda: F("minimum")(col("close"), col("open"))),
    ("gt", lambda: F("gt")(col("close"), col("open"))),
    ("lt", lambda: F("lt")(col("close"), col("open"))),
    ("ge", lambda: F("ge")(col("close"), col("open"))),
    ("le", lambda: F("le")(col("close"), col("open"))),
    ("eq", lambda: F("eq")(col("close"), col("open"))),
    ("ne", lambda: F("ne")(col("close"), col("open"))),
    ("and_", lambda: F("and_")(col("flag"), col("volume"))),
    ("or_", lambda: F("or_")(col("flag"), col("volume"))),
    ("not_", lambda: F("not_")(col("flag"))),
    ("coalesce", lambda: F("coalesce")(col("close"), col("open"))),
    ("fillna_const", lambda: F("fillna_const")(col("close"), 0.0)),
    ("ffill", lambda: F("ffill")(col("close"))),
    ("clip", lambda: F("clip")(col("close"), 0.0, 100.0)),
    ("floor", lambda: F("floor")(col("close"))),
    ("ceil", lambda: F("ceil")(col("close"))),
    ("exp", lambda: F("exp")(col("close"))),
    ("log", lambda: F("log")(col("neg"))),
    ("sqrt", lambda: F("sqrt")(col("neg"))),
    ("power", lambda: F("power")(col("close"), col("volume"))),
    ("inverse", lambda: F("inverse")(col("close"))),
    ("protected_sqrt", lambda: F("protected_sqrt")(col("neg"))),
    ("protected_div", lambda: F("protected_div")(col("close"), col("volume"))),
    ("protected_log", lambda: F("protected_log")(col("neg"))),
    ("safe_div_null", lambda: F("safe_div_null")(col("close"), col("volume"))),
    ("nan_to_num", lambda: F("nan_to_num")(col("close"))),
    ("is_finite", lambda: F("is_finite")(col("close"))),
    ("is_null", lambda: F("is_null")(col("close"))),
    ("is_nan", lambda: F("is_nan")(col("close"))),
    ("rank", lambda: F("rank")(col("close"))),
    ("rank_pct", lambda: F("rank_pct")(col("close"))),
    ("zscore", lambda: F("zscore")(col("close"))),
    ("scale", lambda: F("scale")(col("close"))),
    ("normalize", lambda: F("normalize")(col("close"))),
    ("winsorize", lambda: F("winsorize")(col("close"))),
    ("log_returns", lambda: F("log_returns")(col("close"))),
    ("cum_sum", lambda: F("cum_sum")(col("close"))),
    ("cum_max", lambda: F("cum_max")(col("close"))),
    ("cum_min", lambda: F("cum_min")(col("close"))),
    ("cum_prod", lambda: F("cum_prod")(col("close"))),
    ("cum_delta", lambda: F("cum_delta")(col("close"))),
    ("expanding_sum", lambda: F("expanding_sum")(col("close"))),
    ("expanding_mean", lambda: F("expanding_mean")(col("close"))),
    ("count", lambda: F("count")(col("close"))),
    ("is_not_null", lambda: F("is_not_null")(col("close"))),
    ("is_infinite", lambda: F("is_infinite")(col("close"))),
    ("cs_mean", lambda: F("cs_mean")(col("close"))),
    ("cs_std", lambda: F("cs_std")(col("close"))),
    ("cs_sum", lambda: F("cs_sum")(col("close"))),
    ("cs_count", lambda: F("cs_count")(col("close"))),
    ("cs_demean", lambda: F("cs_demean")(col("close"))),
    ("cs_pct_rank", lambda: F("cs_pct_rank")(col("close"))),
    ("cs_mad", lambda: F("cs_mad")(col("close"))),
    ("cs_mad_zscore", lambda: F("cs_mad_zscore")(col("close"))),
    ("group_mean", lambda: F("group_mean")(col("close"), col("group_id"))),
    ("group_zscore", lambda: F("group_zscore")(col("close"), col("group_id"))),
    ("group_rank", lambda: F("group_rank")(col("close"), col("group_id"))),
    ("group_neutralize", lambda: F("group_neutralize")(col("close"), col("group_id"))),
    ("group_normalize", lambda: F("group_normalize")(col("close"), col("group_id"))),
    ("group_percentile", lambda: F("group_percentile")(col("close"), col("group_id"), 0.5)),
    ("group_std", lambda: F("group_std")(col("close"), col("group_id"))),
    ("group_winsorize", lambda: F("group_winsorize")(col("close"), col("group_id"))),
    ("ts_mean", lambda: F("ts_mean")(col("close"), 3)),
    ("ts_std", lambda: F("ts_std")(col("close"), 3)),
    ("ts_var", lambda: F("ts_var")(col("close"), 2)),
    ("ts_sum", lambda: F("ts_sum")(col("close"), 2)),
    ("ts_max", lambda: F("ts_max")(col("close"), 2)),
    ("ts_min", lambda: F("ts_min")(col("close"), 2)),
    ("ts_median", lambda: F("ts_median")(col("close"), 2)),
    ("ts_delay", lambda: F("ts_delay")(col("close"), 1)),
    ("ts_delta", lambda: F("ts_delta")(col("close"), 1)),
    ("ts_pct", lambda: F("ts_pct")(col("close"), 1)),
    ("ts_zscore", lambda: F("ts_zscore")(col("close"), 2)),
    ("ts_rank", lambda: F("ts_rank")(col("close"), 2)),
    ("ts_sharpe", lambda: F("ts_sharpe")(col("close"), 2)),
    ("ts_autocorr", lambda: F("ts_autocorr")(col("close"), 3, 1)),
    ("ts_corr", lambda: F("ts_corr")(col("close"), col("open"), 2)),
    ("ts_cov", lambda: F("ts_cov")(col("ret"), col("close"), 2)),
    ("ts_beta", lambda: F("ts_beta")(col("ret"), col("close"), 2, min_periods=2)),
    ("volatility", lambda: F("volatility")(col("close"), 2)),
    ("vwap", lambda: F("vwap")(col("close"), col("volume"), 3)),
    ("div_or_default", lambda: F("div_or_default")(col("close"), col("volume"))),
    ("log_fill_invalid", lambda: F("log_fill_invalid")(col("close"))),
    ("log_abs", lambda: F("log_abs")(col("close"))),
    ("signed_log", lambda: F("signed_log")(col("close"))),
    ("signed_sqrt", lambda: F("signed_sqrt")(col("close"))),
    ("group_sum", lambda: F("group_sum")(col("close"), col("group_id"))),
    ("group_min", lambda: F("group_min")(col("close"), col("group_id"))),
    ("group_max", lambda: F("group_max")(col("close"), col("group_id"))),
    ("group_count", lambda: F("group_count")(col("close"), col("group_id"))),
    ("tanh", lambda: F("tanh")(col("close"))),
    (
        "where",
        lambda: F("where")(
            F("gt")(col("close"), 0),
            col("close"),
            col("volume"),
        ),
    ),
]


@pytest.mark.parametrize("factory_name,expr_builder", NO_PANDAS_CASES)
def test_polars_long_no_pandas_path(source, factory_name, expr_builder):
    from cleaned_operators.operator_surface import DAILY_CANONICALS
    from cleaned_operators.registry import OperatorRegistry
    if OperatorRegistry._aliases.get(factory_name, factory_name) not in DAILY_CANONICALS:
        pytest.skip("not on the production daily surface")
    eng = FactorEngine(backend=build_backend("polars_long"), data_source=source)
    out = eng.run(Factor(name="t", expr=expr_builder()))
    assert out.get("used_polars_long_path") is True
    assert not out.get("polars_long_fallback_reason")
    bps = out.get("backend_path_summary") or {}
    assert bps.get("fastpath_route") in {"fast", "mixed", "slow"}
    assert len(out["result"]) == 6


def test_run_many_cse_no_pandas_path(source):
    sub = make_cleaned_call_factory("ts_mean")(col("close"), 2)
    f1 = Factor(name="a", expr=sub)
    f2 = Factor(name="b", expr=make_cleaned_call_factory("rank")(sub))
    eng = FactorEngine(backend=build_backend("polars_long"), data_source=source)
    out = eng.run_many([f1, f2], enable_cse=True)
    assert "backend_paths" in out
    assert out["backend_paths"]["a"]["primary_route"] == "polars_long_native"
    assert out["backend_paths"]["b"]["primary_route"] == "polars_long_native"
