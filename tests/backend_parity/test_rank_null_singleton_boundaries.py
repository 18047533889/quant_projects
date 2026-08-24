# -*- coding: utf-8
"""rank NULL/singleton 边界：全 NULL、单有效值+NULL/NaN。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.backend.rank_spec import polars_cs_rank_expr
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.common.cs_broadcast import cs_rank_01
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.factory import build_data_source
from tests.backend_parity.duckdb_ieee_fixture import seed_duckdb_ieee_panel, write_duckdb_registry
from tests.helpers import InMemorySeriesSource


def _wide_rank_panel() -> pd.DataFrame:
    """三列宽表：单有效值 + NULL/NaN 边界。"""
    return pd.DataFrame(
        {
            "A": [42.0, np.nan, np.nan, 1.0, 2.0],
            "B": [np.nan, np.nan, np.nan, 3.0, 4.0],
            "C": [np.nan, 42.0, np.nan, 5.0, 6.0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"]),
    )


@pytest.fixture(scope="module")
def rank_edge_source():
    load_all()
    wide = _wide_rank_panel()
    rows: list[tuple] = []
    for ts, row in wide.iterrows():
        for inst in ("A", "B", "C"):
            val = row[inst]
            rows.append((ts, inst, val))
    idx = pd.MultiIndex.from_tuples(
        [(ts, inst) for ts, inst, _ in rows],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([v for _, _, v in rows], index=idx)
    return InMemorySeriesSource(data={"close": close})


def _run(source, expr, backend: str) -> pd.Series:
    out = FactorEngine(backend=build_backend(backend), data_source=source).run(
        Factor(name="t", expr=expr)
    )
    return out["result"].sort_index()


def test_cs_rank_01_pandas_null_singleton_boundaries():
    x = pd.DataFrame({"A": [42.0], "B": [np.nan], "C": [np.nan]})
    out = cs_rank_01(x)
    assert out.loc[0, "A"] == pytest.approx(0.5)
    assert pd.isna(out.loc[0, "B"])
    assert pd.isna(out.loc[0, "C"])


def test_cs_rank_01_pandas_all_null_cross_section():
    x = pd.DataFrame({"A": [np.nan], "B": [np.nan], "C": [np.nan]})
    out = cs_rank_01(x)
    assert out.isna().all().all()


def test_cs_rank_01_pandas_one_valid_one_nan():
    x = pd.DataFrame({"A": [1.0, 5.0], "B": [np.nan, np.nan]})
    out = cs_rank_01(x)
    assert out.loc[0, "A"] == pytest.approx(0.5)
    assert pd.isna(out.loc[0, "B"])
    assert out.loc[1, "A"] == pytest.approx(0.5)
    assert pd.isna(out.loc[1, "B"])


def test_rank_polars_long_matches_pandas_on_boundaries(rank_edge_source):
    expr = make_cleaned_call_factory("rank")(col("close"))
    pd_out = _run(rank_edge_source, expr, "pandas")
    pl_out = _run(rank_edge_source, expr, "polars_long")
    pd.testing.assert_series_equal(pd_out, pl_out, check_names=False, rtol=1e-6, atol=1e-6)


@pytest.fixture
def duckdb_ieee_source(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    data_root = tmp_path / "data"
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(write_duckdb_registry(tmp_path, data_root)))
    seed_duckdb_ieee_panel(data_root)
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass
    return build_data_source({"type": "data_access", "dataset": "test_daily"})


def _duck_col(name: str):
    mapping = {
        "close": "Close",
        "open": "Open",
        "ret": "Ret",
        "numer": "Numer",
        "denom": "Denom",
        "neg": "Neg",
        "flag": "Flag",
        "grp": "Grp",
        "volume": "Volume",
    }
    return col(mapping.get(name, name))


def test_rank_duckdb_singleton_null_not_filled(duckdb_ieee_source):
    """2024-01-02：A=42 有效，B=NaN，C=NULL → 仅 A=0.5。"""
    load_all()
    expr = make_cleaned_call_factory("rank")(_duck_col("close"))
    out = _run(duckdb_ieee_source, expr, "duckdb_sql")
    ts = pd.Timestamp("2024-01-02")
    assert out.loc[(ts, "A")] == pytest.approx(0.5)
    assert pd.isna(out.loc[(ts, "B")])
    assert pd.isna(out.loc[(ts, "C")])


def test_rank_duckdb_all_null_cross_section(duckdb_ieee_source):
    load_all()
    expr = make_cleaned_call_factory("rank")(_duck_col("close"))
    out = _run(duckdb_ieee_source, expr, "duckdb_sql")
    ts = pd.Timestamp("2024-01-03")
    for sym in ("A", "B", "C"):
        assert pd.isna(out.loc[(ts, sym)])


def test_is_nan_duckdb_detects_ieee_nan(duckdb_ieee_source):
    """DuckDB 原生 NaN fixture：is_nan 须识别 IEEE NaN（非 SQL NULL）。"""
    from factor_engine.cleaned_operators.operator_surface import DAILY_CANONICALS
    if "is_nan" not in DAILY_CANONICALS:
        pytest.skip("is_nan is not production-certified")
    load_all()
    expr = make_cleaned_call_factory("is_nan")(_duck_col("close"))
    out = _run(duckdb_ieee_source, expr, "duckdb_sql")
    ts = pd.Timestamp("2024-01-02")
    assert out.loc[(ts, "A")] == pytest.approx(0.0)
    assert out.loc[(ts, "B")] == pytest.approx(1.0)
    assert out.loc[(ts, "C")] == pytest.approx(0.0)


def test_polars_cs_rank_expr_direct_null_order():
    import polars as pl

    df = pl.DataFrame(
        {
            "ts": ["2024-01-02"] * 3,
            "inst": ["A", "B", "C"],
            "v": [42.0, None, float("nan")],
        }
    )
    expr = polars_cs_rank_expr("v", partition_cols=("ts",), order_by="inst", canon="rank")
    got = df.select(expr.alias("r"))["r"].to_list()
    assert got[0] == pytest.approx(0.5)
    assert got[1] is None
    assert got[2] is None or (isinstance(got[2], float) and math.isnan(got[2]))
