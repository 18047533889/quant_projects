# -*- coding: utf-8
"""P0 算子 edge-case 三后端 parity（NaN / 除零 / tie / std=0 / 单值截面）。"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from api.cleaned_ops import make_cleaned_call_factory
from api.columns import col
from api.factor import Factor
from backend.factory import build_backend
from cleaned_operators import load_all
from runtime.engine import FactorEngine
from storage.factory import build_data_source
from tests.helpers import InMemorySeriesSource


@pytest.fixture(scope="module")
def edge_source():
    load_all()
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-04"), "A"),
            (pd.Timestamp("2024-01-05"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-03"), "B"),
            (pd.Timestamp("2024-01-04"), "B"),
            (pd.Timestamp("2024-01-05"), "B"),
        ],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([10.0, 11.0, np.nan, 12.0, 20.0, 20.0, 21.0, 22.0], index=idx)
    numer = pd.Series([1.0, 2.0, 3.0, 4.0, 0.0, 1e-15, -1e-15, np.nan], index=idx)
    denom = pd.Series([0.0, 1.0, 1e-15, 2.0, 1.0, 0.0, -1.0, 1.0], index=idx)
    neg = pd.Series([0.0, -1.0, -4.0, 1e-15, 2.0, np.nan, 3.0, -2.0], index=idx)
    flag = pd.Series([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], index=idx)
    grp = pd.Series([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0], index=idx)
    volume = pd.Series([100.0, 0.0, 110.0, 120.0, 200.0, 210.0, 0.0, 220.0], index=idx)
    return InMemorySeriesSource(
        data={
            "close": close,
            "numer": numer,
            "denom": denom,
            "neg": neg,
            "flag": flag,
            "grp": grp,
            "volume": volume,
        }
    )


def _write_duckdb_registry(tmp_path: Path, root: Path) -> Path:
    content = f"""
test_daily:
  kind: static
  access_mode: published
  layout: plain
  hive_partitioning: false
  union_by_name: true
  root: {root}
  glob: "**/*.parquet"
  time_column: TradeDate
  instrument_column: Symbol
  schema:
    TradeDate: date
    Symbol: string
    Close: double
    Numer: double
    Denom: double
    Neg: double
    Flag: double
    Grp: int64
    Volume: double
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _seed_duckdb(root: Path, mem: InMemorySeriesSource) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    mapping = {
        "close": "Close",
        "numer": "Numer",
        "denom": "Denom",
        "neg": "Neg",
        "flag": "Flag",
        "grp": "Grp",
        "volume": "Volume",
    }
    for (ts, sym), close in mem.data["close"].items():
        row = {
            "TradeDate": ts.date(),
            "Symbol": sym,
            "Close": float(close) if pd.notna(close) else None,
        }
        for src, dst in mapping.items():
            if src == "close":
                continue
            val = mem.data[src].loc[(ts, sym)]
            row[dst] = float(val) if pd.notna(val) else None
        rows.append(row)
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")


@pytest.fixture
def duckdb_source(tmp_path, monkeypatch, edge_source):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("DATA_ACCESS_CONFIG", str(_write_duckdb_registry(tmp_path, tmp_path / "data")))
    _seed_duckdb(tmp_path / "data", edge_source)
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass
    return build_data_source({"type": "data_access", "dataset": "test_daily"})


def _run(source, expr, backend: str):
    return FactorEngine(backend=build_backend(backend), data_source=source).run(
        Factor(name="t", expr=expr)
    )


def _result_series(run_out) -> pd.Series:
    return run_out["result"].sort_index()


def _col(name: str):
    return col(
        {
            "close": "Close",
            "numer": "Numer",
            "denom": "Denom",
            "neg": "Neg",
            "flag": "Flag",
            "grp": "Grp",
            "volume": "Volume",
        }.get(name, name)
    )


EDGE_CASES = [
    ("protected_div", lambda: make_cleaned_call_factory("protected_div")(col("numer"), col("denom"))),
    ("protected_log", lambda: make_cleaned_call_factory("protected_log")(col("neg"))),
    ("protected_sqrt", lambda: make_cleaned_call_factory("protected_sqrt")(col("neg"))),
    ("divide", lambda: make_cleaned_call_factory("divide")(col("numer"), col("denom"))),
    ("sqrt", lambda: make_cleaned_call_factory("sqrt")(col("neg"))),
    ("log", lambda: make_cleaned_call_factory("log")(col("neg"))),
    ("where", lambda: make_cleaned_call_factory("where")(col("flag"), col("close"), col("numer"))),
    ("rank", lambda: make_cleaned_call_factory("rank")(col("close"))),
    ("zscore", lambda: make_cleaned_call_factory("zscore")(col("close"))),
    ("ts_zscore", lambda: make_cleaned_call_factory("ts_zscore")(col("close"), 3)),
    ("ts_std", lambda: make_cleaned_call_factory("ts_std")(col("close"), 3)),
    ("group_zscore", lambda: make_cleaned_call_factory("group_zscore")(col("close"), col("grp"))),
    ("winsorize", lambda: make_cleaned_call_factory("winsorize")(col("close"))),
    ("cs_mad", lambda: make_cleaned_call_factory("cs_mad")(col("close"))),
    ("cs_mad_zscore", lambda: make_cleaned_call_factory("cs_mad_zscore")(col("close"))),
    ("vwap", lambda: make_cleaned_call_factory("vwap")(col("close"), col("volume"), 2)),
    ("power", lambda: make_cleaned_call_factory("power")(col("close"), col("numer"))),
]


@pytest.mark.parametrize("name,expr_builder", EDGE_CASES)
def test_p0_edge_polars_long_matches_pandas(edge_source, name, expr_builder):
    expr = expr_builder()
    pd_out = _result_series(_run(edge_source, expr, "pandas"))
    long_out = _result_series(_run(edge_source, expr, "polars_long"))
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False, rtol=1e-6, atol=1e-6)


DUCKDB_EDGE_CASES = [
    ("protected_div", lambda: make_cleaned_call_factory("protected_div")(_col("numer"), _col("denom"))),
    ("protected_log", lambda: make_cleaned_call_factory("protected_log")(_col("neg"))),
    ("protected_sqrt", lambda: make_cleaned_call_factory("protected_sqrt")(_col("neg"))),
    ("divide", lambda: make_cleaned_call_factory("divide")(_col("numer"), _col("denom"))),
    ("sqrt", lambda: make_cleaned_call_factory("sqrt")(_col("neg"))),
    ("log", lambda: make_cleaned_call_factory("log")(_col("neg"))),
    ("where", lambda: make_cleaned_call_factory("where")(_col("flag"), _col("close"), _col("numer"))),
    ("rank", lambda: make_cleaned_call_factory("rank")(_col("close"))),
    ("zscore", lambda: make_cleaned_call_factory("zscore")(_col("close"))),
    ("ts_zscore", lambda: make_cleaned_call_factory("ts_zscore")(_col("close"), 3)),
    ("ts_std", lambda: make_cleaned_call_factory("ts_std")(_col("close"), 3)),
    ("ts_mean", lambda: make_cleaned_call_factory("ts_mean")(_col("close"), 3)),
    ("ts_pct", lambda: make_cleaned_call_factory("ts_pct")(_col("close"), 1)),
    ("group_zscore", lambda: make_cleaned_call_factory("group_zscore")(_col("close"), _col("grp"))),
    ("winsorize", lambda: make_cleaned_call_factory("winsorize")(_col("close"))),
    ("cs_mad", lambda: make_cleaned_call_factory("cs_mad")(_col("close"))),
    ("cs_mad_zscore", lambda: make_cleaned_call_factory("cs_mad_zscore")(_col("close"))),
    ("vwap", lambda: make_cleaned_call_factory("vwap")(_col("close"), _col("volume"), 2)),
    ("power", lambda: make_cleaned_call_factory("power")(_col("close"), _col("numer"))),
    ("scale", lambda: make_cleaned_call_factory("scale")(_col("close"))),
    ("normalize", lambda: make_cleaned_call_factory("normalize")(_col("close"))),
    ("volatility", lambda: make_cleaned_call_factory("volatility")(_col("close"), 3)),
]


@pytest.mark.parametrize("name,expr_builder", DUCKDB_EDGE_CASES)
def test_p0_edge_duckdb_matches_pandas(duckdb_source, name, expr_builder):
    from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution

    expr = expr_builder()
    pd_out = _result_series(_run(duckdb_source, expr, "pandas"))
    sql_run = _run(duckdb_source, expr, "duckdb_sql")
    assert_duckdb_real_sql_execution(sql_run)
    sql_out = _result_series(sql_run)
    pd.testing.assert_series_equal(pd_out, sql_out, check_names=False, rtol=1e-6, atol=1e-6)


def test_rank_single_valid_value_is_half(edge_source):
    idx = edge_source.data["close"].index
    single_ts = idx[0][0]
    mask = idx.get_level_values("timestamp") == single_ts
    sub_idx = idx[mask][:1]
    one = pd.Series([42.0], index=sub_idx)
    src = InMemorySeriesSource(data={"close": one})
    expr = make_cleaned_call_factory("rank")(col("close"))
    pd_out = _result_series(_run(src, expr, "pandas")).iloc[0]
    long_out = _result_series(_run(src, expr, "polars_long")).iloc[0]
    assert pd_out == pytest.approx(0.5)
    assert long_out == pytest.approx(0.5)


def test_normalize_single_valid_value_is_null(edge_source):
    from backend.numeric_semantics import normalize_single_valid_is_null

    assert normalize_single_valid_is_null()
    idx = edge_source.data["close"].index
    single_ts = idx[0][0]
    mask = idx.get_level_values("timestamp") == single_ts
    sub_idx = idx[mask][:1]
    one = pd.Series([42.0], index=sub_idx)
    src = InMemorySeriesSource(data={"close": one})
    expr = make_cleaned_call_factory("normalize")(col("close"))
    pd_out = _result_series(_run(src, expr, "pandas")).iloc[0]
    long_out = _result_series(_run(src, expr, "polars_long")).iloc[0]
    assert math.isnan(pd_out)
    assert math.isnan(long_out)


def test_ts_pct_zero_prev_is_null(edge_source):
    from backend.numeric_semantics import ts_pct_zero_prev_is_null

    assert ts_pct_zero_prev_is_null()
    idx = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-04"), "A"),
        ],
        names=["timestamp", "instrument"],
    )
    close = pd.Series([0.0, 5.0, 10.0], index=idx)
    src = InMemorySeriesSource(data={"close": close})
    expr = make_cleaned_call_factory("ts_pct")(col("close"), 1)
    pd_out = _result_series(_run(src, expr, "pandas"))
    long_out = _result_series(_run(src, expr, "polars_long"))
    assert math.isnan(pd_out.iloc[1])
    assert math.isnan(long_out.iloc[1])
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False, rtol=1e-6, atol=1e-6)


def test_vwap_zero_volume_polars_native(edge_source):
    expr = make_cleaned_call_factory("vwap")(col("close"), col("volume"), 2)
    out = FactorEngine(backend=build_backend("polars_long"), data_source=edge_source).run(
        Factor(name="t", expr=expr)
    )
    assert out.get("used_polars_long_native") is True
    pd_out = _result_series(_run(edge_source, expr, "pandas"))
    long_out = out["result"].sort_index()
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False, rtol=1e-6, atol=1e-6)
