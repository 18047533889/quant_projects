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
            "grp": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        }
    )
    return NoPandasSource(df.lazy())


NO_PANDAS_CASES = [
    ("ts_mean", lambda: make_cleaned_call_factory("ts_mean")(col("close"), 3)),
    ("ts_std", lambda: make_cleaned_call_factory("ts_std")(col("close"), 3)),
    ("rank", lambda: make_cleaned_call_factory("rank")(col("close"))),
    ("zscore", lambda: make_cleaned_call_factory("zscore")(col("close"))),
    ("group_mean", lambda: make_cleaned_call_factory("group_mean")(col("close"), col("grp"))),
    ("group_zscore", lambda: make_cleaned_call_factory("group_zscore")(col("close"), col("grp"))),
    ("group_winsorize", lambda: make_cleaned_call_factory("group_winsorize")(col("close"), col("grp"))),
    ("vwap", lambda: make_cleaned_call_factory("vwap")(col("close"), col("volume"), 3)),
    (
        "protected_div",
        lambda: make_cleaned_call_factory("protected_div")(col("close"), col("volume")),
    ),
    ("protected_log", lambda: make_cleaned_call_factory("protected_log")(col("close"))),
    (
        "where",
        lambda: make_cleaned_call_factory("where")(
            make_cleaned_call_factory("gt")(col("close"), 0),
            col("close"),
            col("volume"),
        ),
    ),
    ("ts_corr", lambda: make_cleaned_call_factory("ts_corr")(col("close"), col("open"), 2)),
    ("ts_cov", lambda: make_cleaned_call_factory("ts_cov")(col("ret"), col("close"), 2)),
    ("ts_beta", lambda: make_cleaned_call_factory("ts_beta")(col("ret"), col("close"), 2)),
    ("ts_delay", lambda: make_cleaned_call_factory("ts_delay")(col("close"), 1)),
    ("log_returns", lambda: make_cleaned_call_factory("log_returns")(col("close"))),
    ("cs_mad", lambda: make_cleaned_call_factory("cs_mad")(col("close"))),
    ("cs_mad_zscore", lambda: make_cleaned_call_factory("cs_mad_zscore")(col("close"))),
]


@pytest.mark.parametrize("factory_name,expr_builder", NO_PANDAS_CASES)
def test_polars_long_no_pandas_path(source, factory_name, expr_builder):
    eng = FactorEngine(backend=build_backend("polars_long"), data_source=source)
    out = eng.run(Factor(name="t", expr=expr_builder()))
    assert out.get("used_polars_long_path") is True
    assert not out.get("polars_long_fallback_reason")
    bps = out.get("backend_path_summary") or {}
    assert bps.get("fastpath_route") in {"fast", "mixed"}
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
