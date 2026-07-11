# -*- coding: utf-8
"""Production-safe 算子 bulk parity：补齐 MEMORY/EDGE 未覆盖的 P0/P1 执行 case。"""
from __future__ import annotations

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


def _build_source():
    load_all()
    dates = pd.date_range("2024-01-02", periods=10, freq="D")
    insts = ["A", "B", "C", "D"]
    idx = pd.MultiIndex.from_product([dates, insts], names=["timestamp", "instrument"])
    n = len(idx)
    rng = np.random.default_rng(42)
    close = pd.Series(10.0 + rng.normal(0, 1, n).cumsum() * 0.1, index=idx)
    open_ = close - 0.3
    volume = pd.Series(rng.integers(50, 300, n).astype(float), index=idx)
    volume.iloc[3] = 0.0
    ret = close.pct_change(fill_method=None).fillna(0.0)
    grp = pd.Series([1, 1, 2, 2] * len(dates), index=idx, dtype=float)
    flag = pd.Series((close > 0).astype(float), index=idx)
    return InMemorySeriesSource(
        data={
            "close": close,
            "open": open_,
            "volume": volume,
            "ret": ret,
            "grp": grp,
            "flag": flag,
        }
    )


@pytest.fixture(scope="module")
def bulk_source():
    return _build_source()


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
    Open: double
    Volume: double
    Ret: double
    Grp: int64
    Flag: double
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _seed_duckdb(root: Path, mem: InMemorySeriesSource) -> None:
    root.mkdir(parents=True, exist_ok=True)
    mapping = {
        "close": "Close",
        "open": "Open",
        "volume": "Volume",
        "ret": "Ret",
        "grp": "Grp",
        "flag": "Flag",
    }
    rows = []
    for (ts, sym) in mem.data["close"].index:
        row = {"TradeDate": ts.date(), "Symbol": sym}
        for src, dst in mapping.items():
            val = mem.data[src].loc[(ts, sym)]
            row[dst] = float(val) if pd.notna(val) else None
        rows.append(row)
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")


@pytest.fixture(scope="module")
def duckdb_bulk_source(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("duckdb_bulk")
    import os

    os.environ["DATA_ACCESS_SKIP_COS_MIRROR"] = "1"
    mem = _build_source()
    reg = _write_duckdb_registry(tmp, tmp / "data")
    os.environ["DATA_ACCESS_CONFIG"] = str(reg)
    _seed_duckdb(tmp / "data", mem)
    try:
        from data_access import reset_store

        reset_store()
    except ImportError:
        pass
    return build_data_source({"type": "data_access", "dataset": "test_daily"})


def _col(name: str):
    return col(
        {
            "close": "Close",
            "open": "Open",
            "volume": "Volume",
            "ret": "Ret",
            "grp": "Grp",
            "flag": "Flag",
        }.get(name, name)
    )


def _run(source, expr, backend: str):
    return FactorEngine(backend=build_backend(backend), data_source=source).run(
        Factor(name="t", expr=expr)
    )


def _result_series(run_out) -> pd.Series:
    return run_out["result"].sort_index()


F = make_cleaned_call_factory

POLARS_BULK_CASES = [
    ("abs", lambda: F("abs")(col("close"))),
    ("neg", lambda: F("neg")(col("close"))),
    ("sign", lambda: F("sign")(col("close"))),
    ("multiply", lambda: F("multiply")(col("close"), col("open"))),
    ("subtract", lambda: F("subtract")(col("close"), col("open"))),
    ("maximum", lambda: F("maximum")(col("close"), col("open"))),
    ("minimum", lambda: F("minimum")(col("close"), col("open"))),
    ("clip", lambda: F("clip")(col("close"), 0.0, 100.0)),
    ("floor", lambda: F("floor")(col("close"))),
    ("ceil", lambda: F("ceil")(col("close"))),
    ("inverse", lambda: F("inverse")(col("close"))),
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
    ("fillna", lambda: F("fillna")(col("close"), 0.0)),
    ("nan_to_num", lambda: F("nan_to_num")(col("close"))),
    ("ffill", lambda: F("ffill")(col("close"))),
    ("is_finite", lambda: F("is_finite")(col("close"))),
    ("is_null", lambda: F("is_null")(col("close"))),
    ("is_nan", lambda: F("is_nan")(col("close"))),
    ("is_not_null", lambda: F("is_not_null")(col("close"))),
    ("is_infinite", lambda: F("is_infinite")(col("close"))),
    ("ts_delay", lambda: F("ts_delay")(col("close"), 2)),
    ("ts_delta", lambda: F("ts_delta")(col("close"), 1)),
    ("ts_pct", lambda: F("ts_pct")(col("close"), 1)),
    ("ts_max", lambda: F("ts_max")(col("close"), 3)),
    ("ts_min", lambda: F("ts_min")(col("close"), 3)),
    ("ts_sum", lambda: F("ts_sum")(col("close"), 3)),
    ("ts_var", lambda: F("ts_var")(col("close"), 3)),
    ("cs_demean", lambda: F("cs_demean")(col("close"))),
    ("scale", lambda: F("scale")(col("close"))),
    ("normalize", lambda: F("normalize")(col("close"))),
    ("group_neutralize", lambda: F("group_neutralize")(col("close"), col("grp"))),
    ("group_normalize", lambda: F("group_normalize")(col("close"), col("grp"))),
    ("group_std", lambda: F("group_std")(col("close"), col("grp"))),
    ("c_std", lambda: F("c_std")(col("close"))),
    ("c_sum", lambda: F("c_sum")(col("close"))),
    ("c_count", lambda: F("c_count")(col("close"))),
    ("cum_sum", lambda: F("cum_sum")(col("close"))),
    ("cum_max", lambda: F("cum_max")(col("close"))),
    ("cum_min", lambda: F("cum_min")(col("close"))),
    ("cum_prod", lambda: F("cum_prod")(col("close"))),
    ("expanding_sum", lambda: F("expanding_sum")(col("close"))),
    ("count", lambda: F("count")(col("close"))),
    ("volatility", lambda: F("volatility")(col("close"), 3)),
    ("log", lambda: F("log")(col("close"))),
    ("sqrt", lambda: F("sqrt")(col("close"))),
    ("ts_rank", lambda: F("ts_rank")(col("close"), 3)),
    ("ts_autocorr", lambda: F("ts_autocorr")(col("close"), 4, 1)),
    ("is_nan", lambda: F("is_nan")(col("close"))),
    ("is_not_null", lambda: F("is_not_null")(col("close"))),
    ("is_infinite", lambda: F("is_infinite")(col("close"))),
    ("cum_delta", lambda: F("cum_delta")(col("close"))),
    ("expanding_mean", lambda: F("expanding_mean")(col("close"))),
    ("divide", lambda: F("divide")(col("close"), col("open"))),
    ("exp", lambda: F("exp")(col("close"))),
    ("winsorize", lambda: F("winsorize")(col("close"))),
    ("protected_sqrt", lambda: F("protected_sqrt")(col("close"))),
    ("safe_div_null", lambda: F("safe_div_null")(col("close"), col("volume"))),
    ("div_or_default", lambda: F("div_or_default")(col("close"), col("volume"))),
    ("log_fill_invalid", lambda: F("log_fill_invalid")(col("close"))),
    ("log_abs", lambda: F("log_abs")(col("close"))),
    ("signed_log", lambda: F("signed_log")(col("close"))),
    ("signed_sqrt", lambda: F("signed_sqrt")(col("close"))),
    ("group_sum", lambda: F("group_sum")(col("close"), col("grp"))),
    ("group_min", lambda: F("group_min")(col("close"), col("grp"))),
    ("group_max", lambda: F("group_max")(col("close"), col("grp"))),
    ("group_count", lambda: F("group_count")(col("close"), col("grp"))),
    ("ts_sharpe", lambda: F("ts_sharpe")(col("close"), 3)),
]

DUCKDB_BULK_CASES = [
    ("abs", lambda: F("abs")(_col("close"))),
    ("neg", lambda: F("neg")(_col("close"))),
    ("sign", lambda: F("sign")(_col("close"))),
    ("add", lambda: F("add")(_col("close"), _col("open"))),
    ("multiply", lambda: F("multiply")(_col("close"), _col("open"))),
    ("subtract", lambda: F("subtract")(_col("close"), _col("open"))),
    ("divide", lambda: F("divide")(_col("close"), _col("open"))),
    ("maximum", lambda: F("maximum")(_col("close"), _col("open"))),
    ("minimum", lambda: F("minimum")(_col("close"), _col("open"))),
    ("clip", lambda: F("clip")(_col("close"), 0.0, 100.0)),
    ("floor", lambda: F("floor")(_col("close"))),
    ("ceil", lambda: F("ceil")(_col("close"))),
    ("inverse", lambda: F("inverse")(_col("close"))),
    ("exp", lambda: F("exp")(_col("close"))),
    ("log", lambda: F("log")(_col("close"))),
    ("sqrt", lambda: F("sqrt")(_col("close"))),
    ("gt", lambda: F("gt")(_col("close"), _col("open"))),
    ("lt", lambda: F("lt")(_col("close"), _col("open"))),
    ("ge", lambda: F("ge")(_col("close"), _col("open"))),
    ("le", lambda: F("le")(_col("close"), _col("open"))),
    ("eq", lambda: F("eq")(_col("close"), _col("open"))),
    ("ne", lambda: F("ne")(_col("close"), _col("open"))),
    ("and_", lambda: F("and_")(_col("flag"), _col("volume"))),
    ("or_", lambda: F("or_")(_col("flag"), _col("volume"))),
    ("not_", lambda: F("not_")(_col("flag"))),
    ("coalesce", lambda: F("coalesce")(_col("close"), _col("open"))),
    ("fillna_const", lambda: F("fillna_const")(_col("close"), 0.0)),
    ("fillna", lambda: F("fillna")(_col("close"), 0.0)),
    ("nan_to_num", lambda: F("nan_to_num")(_col("close"))),
    ("ffill", lambda: F("ffill")(_col("close"))),
    ("is_finite", lambda: F("is_finite")(_col("close"))),
    ("is_null", lambda: F("is_null")(_col("close"))),
    ("ts_delay", lambda: F("ts_delay")(_col("close"), 2)),
    ("ts_delta", lambda: F("ts_delta")(_col("close"), 1)),
    ("ts_pct", lambda: F("ts_pct")(_col("close"), 1)),
    ("ts_max", lambda: F("ts_max")(_col("close"), 3)),
    ("ts_min", lambda: F("ts_min")(_col("close"), 3)),
    ("ts_sum", lambda: F("ts_sum")(_col("close"), 3)),
    ("ts_var", lambda: F("ts_var")(_col("close"), 3)),
    ("ts_zscore", lambda: F("ts_zscore")(_col("close"), 3)),
    ("ts_median", lambda: F("ts_median")(_col("close"), 3)),
    ("log_returns", lambda: F("log_returns")(_col("close"))),
    ("volatility", lambda: F("volatility")(_col("close"), 3)),
    ("ts_rank", lambda: F("ts_rank")(_col("close"), 3)),
    ("ts_sharpe", lambda: F("ts_sharpe")(_col("close"), 3)),
    ("ts_autocorr", lambda: F("ts_autocorr")(_col("close"), 4, 1)),
    ("group_normalize", lambda: F("group_normalize")(_col("close"), _col("grp"))),
    ("group_percentile", lambda: F("group_percentile")(_col("close"), _col("grp"), 0.5)),
    ("group_std", lambda: F("group_std")(_col("close"), _col("grp"))),
    ("is_nan", lambda: F("is_nan")(_col("close"))),
    ("is_not_null", lambda: F("is_not_null")(_col("close"))),
    ("is_infinite", lambda: F("is_infinite")(_col("close"))),
    ("cs_demean", lambda: F("cs_demean")(_col("close"))),
    ("scale", lambda: F("scale")(_col("close"))),
    ("normalize", lambda: F("normalize")(_col("close"))),
    ("cs_pct_rank", lambda: F("cs_pct_rank")(_col("close"))),
    ("rank_pct", lambda: F("rank_pct")(_col("close"))),
    ("group_neutralize", lambda: F("group_neutralize")(_col("close"), _col("grp"))),
    ("group_normalize", lambda: F("group_normalize")(_col("close"), _col("grp"))),
    ("group_std", lambda: F("group_std")(_col("close"), _col("grp"))),
    ("group_rank", lambda: F("group_rank")(_col("close"), _col("grp"))),
    ("group_percentile", lambda: F("group_percentile")(_col("close"), _col("grp"), 0.5)),
    ("c_mean", lambda: F("c_mean")(_col("close"))),
    ("c_std", lambda: F("c_std")(_col("close"))),
    ("c_sum", lambda: F("c_sum")(_col("close"))),
    ("c_count", lambda: F("c_count")(_col("close"))),
    ("cum_sum", lambda: F("cum_sum")(_col("close"))),
    ("cum_max", lambda: F("cum_max")(_col("close"))),
    ("cum_min", lambda: F("cum_min")(_col("close"))),
    ("cum_prod", lambda: F("cum_prod")(_col("close"))),
    ("cum_delta", lambda: F("cum_delta")(_col("close"))),
    ("expanding_mean", lambda: F("expanding_mean")(_col("close"))),
    ("expanding_sum", lambda: F("expanding_sum")(_col("close"))),
    ("count", lambda: F("count")(_col("close"))),
    ("power", lambda: F("power")(_col("close"), _col("volume"))),
    ("protected_sqrt", lambda: F("protected_sqrt")(_col("close"))),
    ("safe_div_null", lambda: F("safe_div_null")(_col("close"), _col("volume"))),
    ("div_or_default", lambda: F("div_or_default")(_col("close"), _col("volume"))),
    ("log_fill_invalid", lambda: F("log_fill_invalid")(_col("close"))),
    ("log_abs", lambda: F("log_abs")(_col("close"))),
    ("signed_log", lambda: F("signed_log")(_col("close"))),
    ("signed_sqrt", lambda: F("signed_sqrt")(_col("close"))),
    ("group_sum", lambda: F("group_sum")(_col("close"), _col("grp"))),
    ("group_min", lambda: F("group_min")(_col("close"), _col("grp"))),
    ("group_max", lambda: F("group_max")(_col("close"), _col("grp"))),
    ("group_count", lambda: F("group_count")(_col("close"), _col("grp"))),
    ("where", lambda: F("where")(_col("flag"), _col("close"), _col("open"))),
    ("rolling_beta", lambda: F("rolling_beta")(_col("ret"), _col("close"), 3)),
    ("ts_corr", lambda: F("ts_corr")(_col("close"), _col("open"), 3)),
    ("ts_cov", lambda: F("ts_cov")(_col("ret"), _col("close"), 3)),
    ("ts_beta", lambda: F("ts_beta")(_col("ret"), _col("close"), 3)),
    ("group_mean", lambda: F("group_mean")(_col("close"), _col("grp"))),
    ("group_zscore", lambda: F("group_zscore")(_col("close"), _col("grp"))),
    ("group_winsorize", lambda: F("group_winsorize")(_col("close"), _col("grp"))),
    ("winsorize", lambda: F("winsorize")(_col("close"))),
    ("vwap", lambda: F("vwap")(_col("close"), _col("volume"), 3)),
    ("cs_mad", lambda: F("cs_mad")(_col("close"))),
    ("cs_mad_zscore", lambda: F("cs_mad_zscore")(_col("close"))),
]


def _assert_series(pd_out, other_out):
    left = pd_out.astype(float)
    right = other_out.astype(float)
    pd.testing.assert_series_equal(left, right, check_names=False, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name,expr_builder", POLARS_BULK_CASES)
def test_bulk_polars_long_matches_pandas(bulk_source, name, expr_builder):
    expr = expr_builder()
    pd_out = _result_series(_run(bulk_source, expr, "pandas"))
    long_out = _result_series(_run(bulk_source, expr, "polars_long"))
    _assert_series(pd_out, long_out)


@pytest.mark.parametrize("name,expr_builder", DUCKDB_BULK_CASES)
def test_bulk_duckdb_matches_pandas(duckdb_bulk_source, name, expr_builder):
    from tests.backend_parity.duckdb_parity_helpers import assert_duckdb_real_sql_execution

    expr = expr_builder()
    pd_out = _result_series(_run(duckdb_bulk_source, expr, "pandas"))
    sql_run = _run(duckdb_bulk_source, expr, "duckdb_sql")
    assert_duckdb_real_sql_execution(sql_run)
    sql_out = _result_series(sql_run)
    _assert_series(pd_out, sql_out)
