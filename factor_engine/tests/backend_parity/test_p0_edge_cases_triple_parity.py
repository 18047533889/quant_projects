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
    open_ = close - 0.5
    ret = pd.Series([0.01, 0.02, -0.01, 0.03, 0.04, 0.01, 0.02, 0.05], index=idx)
    numer = pd.Series([1.0, 2.0, 3.0, 4.0, 0.0, 1e-15, -1e-15, np.nan], index=idx)
    denom = pd.Series([0.0, 1.0, 1e-15, 2.0, 1.0, 0.0, -1.0, 1.0], index=idx)
    neg = pd.Series([0.0, -1.0, -4.0, 1e-15, 2.0, np.nan, 3.0, -2.0], index=idx)
    flag = pd.Series([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], index=idx)
    grp = pd.Series([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0], index=idx)
    volume = pd.Series([100.0, 0.0, 110.0, 120.0, 200.0, 210.0, 0.0, 220.0], index=idx)
    return InMemorySeriesSource(
        data={
            "close": close,
            "open": open_,
            "ret": ret,
            "numer": numer,
            "denom": denom,
            "neg": neg,
            "flag": flag,
            "group_id": grp,
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
    Open: double
    Ret: double
    Numer: double
    Denom: double
    Neg: double
    Flag: double
    group_id: int64
    Volume: double
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _seed_duckdb(root: Path, mem: InMemorySeriesSource) -> None:
    root.mkdir(parents=True, exist_ok=True)
    mapping = {
        "close": "Close",
        "open": "Open",
        "ret": "Ret",
        "numer": "Numer",
        "denom": "Denom",
        "neg": "Neg",
        "flag": "Flag",
        "group_id": "group_id",
        "volume": "Volume",
    }
    rows = []
    for (ts, sym) in mem.data["close"].index:
        row = {"TradeDate": ts.date(), "Symbol": sym}
        for src, dst in mapping.items():
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
            "open": "Open",
            "ret": "Ret",
            "numer": "Numer",
            "denom": "Denom",
            "neg": "Neg",
            "flag": "Flag",
            "volume": "Volume",
        }.get(name, name)
    )


EDGE_CASES = [
    ("protected_div", lambda: make_cleaned_call_factory("protected_div")(col("numer"), col("denom"))),
    ("safe_div_null", lambda: make_cleaned_call_factory("safe_div_null")(col("numer"), col("denom"))),
    ("protected_log", lambda: make_cleaned_call_factory("protected_log")(col("neg"))),
    ("protected_sqrt", lambda: make_cleaned_call_factory("protected_sqrt")(col("neg"))),
    ("divide", lambda: make_cleaned_call_factory("divide")(col("numer"), col("denom"))),
    ("add", lambda: make_cleaned_call_factory("add")(col("close"), col("open"))),
    ("sqrt", lambda: make_cleaned_call_factory("sqrt")(col("neg"))),
    ("log", lambda: make_cleaned_call_factory("log")(col("neg"))),
    ("where", lambda: make_cleaned_call_factory("where")(col("flag"), col("close"), col("numer"))),
    ("rank", lambda: make_cleaned_call_factory("rank")(col("close"))),
    ("zscore", lambda: make_cleaned_call_factory("zscore")(col("close"))),
    ("ts_zscore", lambda: make_cleaned_call_factory("ts_zscore")(col("close"), 3)),
    ("ts_std", lambda: make_cleaned_call_factory("ts_std")(col("close"), 3)),
    ("ts_mean", lambda: make_cleaned_call_factory("ts_mean")(col("close"), 3)),
    ("ts_delay", lambda: make_cleaned_call_factory("ts_delay")(col("close"), 1)),
    ("group_mean", lambda: make_cleaned_call_factory("group_mean")(col("close"), col("group_id"))),
    ("group_winsorize", lambda: make_cleaned_call_factory("group_winsorize")(col("close"), col("group_id"))),
    ("ts_corr", lambda: make_cleaned_call_factory("ts_corr")(col("close"), col("open"), 3)),
    ("ts_cov", lambda: make_cleaned_call_factory("ts_cov")(col("ret"), col("close"), 3)),
    ("ts_beta", lambda: make_cleaned_call_factory("ts_beta")(col("ret"), col("close"), 3)),
    ("group_zscore", lambda: make_cleaned_call_factory("group_zscore")(col("close"), col("group_id"))),
    ("winsorize", lambda: make_cleaned_call_factory("winsorize")(col("close"))),
    ("cs_mad", lambda: make_cleaned_call_factory("cs_mad")(col("close"))),
    ("cs_mad_zscore", lambda: make_cleaned_call_factory("cs_mad_zscore")(col("close"))),
    ("vwap", lambda: make_cleaned_call_factory("vwap")(col("close"), col("volume"), 2)),
    ("power", lambda: make_cleaned_call_factory("power")(col("close"), col("numer"))),
    ("ts_delta", lambda: make_cleaned_call_factory("ts_delta")(col("close"), 1)),
    ("ts_pct", lambda: make_cleaned_call_factory("ts_pct")(col("close"), 1)),
    ("multiply", lambda: make_cleaned_call_factory("multiply")(col("close"), col("open"))),
    ("subtract", lambda: make_cleaned_call_factory("subtract")(col("close"), col("open"))),
    ("ts_max", lambda: make_cleaned_call_factory("ts_max")(col("close"), 3)),
    ("ts_min", lambda: make_cleaned_call_factory("ts_min")(col("close"), 3)),
    ("safe_div_null", lambda: make_cleaned_call_factory("safe_div_null")(col("numer"), col("denom"))),
    ("sign", lambda: make_cleaned_call_factory("sign")(col("close"))),
    ("fillna_const", lambda: make_cleaned_call_factory("fillna_const")(col("close"), 0.0)),
    ("cum_sum", lambda: make_cleaned_call_factory("cum_sum")(col("close"))),
    ("abs", lambda: make_cleaned_call_factory("abs")(col("close"))),
    ("neg", lambda: make_cleaned_call_factory("neg")(col("close"))),
    ("exp", lambda: make_cleaned_call_factory("exp")(col("ret"))),
    ("floor", lambda: make_cleaned_call_factory("floor")(col("close"))),
    ("ceil", lambda: make_cleaned_call_factory("ceil")(col("close"))),
    ("maximum", lambda: make_cleaned_call_factory("maximum")(col("close"), col("open"))),
    ("minimum", lambda: make_cleaned_call_factory("minimum")(col("close"), col("open"))),
    ("gt", lambda: make_cleaned_call_factory("gt")(col("close"), col("open"))),
    ("lt", lambda: make_cleaned_call_factory("lt")(col("close"), col("open"))),
    ("eq", lambda: make_cleaned_call_factory("eq")(col("close"), col("open"))),
    ("ge", lambda: make_cleaned_call_factory("ge")(col("close"), col("open"))),
    ("le", lambda: make_cleaned_call_factory("le")(col("close"), col("open"))),
    ("ne", lambda: make_cleaned_call_factory("ne")(col("close"), col("open"))),
    ("and_", lambda: make_cleaned_call_factory("and_")(col("flag"), col("volume"))),
    ("or_", lambda: make_cleaned_call_factory("or_")(col("flag"), col("volume"))),
    ("not_", lambda: make_cleaned_call_factory("not_")(col("flag"))),
    ("coalesce", lambda: make_cleaned_call_factory("coalesce")(col("close"), col("open"))),
    ("ts_sum", lambda: make_cleaned_call_factory("ts_sum")(col("close"), 3)),
    ("ts_var", lambda: make_cleaned_call_factory("ts_var")(col("close"), 3)),
    ("ts_median", lambda: make_cleaned_call_factory("ts_median")(col("close"), 3)),
    ("rank_pct", lambda: make_cleaned_call_factory("rank_pct")(col("close"))),
    ("cs_pct_rank", lambda: make_cleaned_call_factory("cs_pct_rank")(col("close"))),
    ("cs_demean", lambda: make_cleaned_call_factory("cs_demean")(col("close"))),
    ("cs_mean", lambda: make_cleaned_call_factory("cs_mean")(col("close"))),
    ("cs_std", lambda: make_cleaned_call_factory("cs_std")(col("close"))),
    ("cs_sum", lambda: make_cleaned_call_factory("cs_sum")(col("close"))),
    ("cs_count", lambda: make_cleaned_call_factory("cs_count")(col("close"))),
    ("log_returns", lambda: make_cleaned_call_factory("log_returns")(col("close"))),
    ("volatility", lambda: make_cleaned_call_factory("volatility")(col("ret"), 3)),
    ("count", lambda: make_cleaned_call_factory("count")(col("close"))),
    ("cum_max", lambda: make_cleaned_call_factory("cum_max")(col("close"))),
    ("cum_min", lambda: make_cleaned_call_factory("cum_min")(col("close"))),
    ("cum_prod", lambda: make_cleaned_call_factory("cum_prod")(col("close"))),
    ("cum_delta", lambda: make_cleaned_call_factory("cum_delta")(col("close"))),
    ("expanding_sum", lambda: make_cleaned_call_factory("expanding_sum")(col("close"))),
    ("expanding_mean", lambda: make_cleaned_call_factory("expanding_mean")(col("close"))),
    ("is_null", lambda: make_cleaned_call_factory("is_null")(col("close"))),
    ("is_not_null", lambda: make_cleaned_call_factory("is_not_null")(col("close"))),
    ("is_infinite", lambda: make_cleaned_call_factory("is_infinite")(col("close"))),
    ("ts_autocorr", lambda: make_cleaned_call_factory("ts_autocorr")(col("close"), 4, 1)),
    ("ts_rank", lambda: make_cleaned_call_factory("ts_rank")(col("close"), 3)),
    ("group_rank", lambda: make_cleaned_call_factory("group_rank")(col("close"), col("group_id"))),
    ("group_neutralize", lambda: make_cleaned_call_factory("group_neutralize")(col("close"), col("group_id"))),
    ("clip", lambda: make_cleaned_call_factory("clip")(col("close"), 0.0, 100.0)),
    ("ffill", lambda: make_cleaned_call_factory("ffill")(col("close"))),
    ("group_normalize", lambda: make_cleaned_call_factory("group_normalize")(col("close"), col("group_id"))),
    ("group_percentile", lambda: make_cleaned_call_factory("group_percentile")(col("close"), col("group_id"), 0.5)),
    ("group_std", lambda: make_cleaned_call_factory("group_std")(col("close"), col("group_id"))),
    ("inverse", lambda: make_cleaned_call_factory("inverse")(col("close"))),
    ("is_finite", lambda: make_cleaned_call_factory("is_finite")(col("close"))),
    ("nan_to_num", lambda: make_cleaned_call_factory("nan_to_num")(col("close"))),
    ("scale", lambda: make_cleaned_call_factory("scale")(col("close"))),
    ("normalize", lambda: make_cleaned_call_factory("normalize")(col("close"))),
    ("is_nan", lambda: make_cleaned_call_factory("is_nan")(col("close"))),
    ("ts_sharpe", lambda: make_cleaned_call_factory("ts_sharpe")(col("close"), 3)),
    ("div_or_default", lambda: make_cleaned_call_factory("div_or_default")(col("numer"), col("denom"))),
    ("log_fill_invalid", lambda: make_cleaned_call_factory("log_fill_invalid")(col("neg"))),
    ("log_abs", lambda: make_cleaned_call_factory("log_abs")(col("neg"))),
    ("signed_log", lambda: make_cleaned_call_factory("signed_log")(col("neg"))),
    ("signed_sqrt", lambda: make_cleaned_call_factory("signed_sqrt")(col("neg"))),
    ("group_sum", lambda: make_cleaned_call_factory("group_sum")(col("close"), col("group_id"))),
    ("group_min", lambda: make_cleaned_call_factory("group_min")(col("close"), col("group_id"))),
    ("group_max", lambda: make_cleaned_call_factory("group_max")(col("close"), col("group_id"))),
    ("group_count", lambda: make_cleaned_call_factory("group_count")(col("close"), col("group_id"))),
    ("tanh", lambda: make_cleaned_call_factory("tanh")(col("neg"))),
]


@pytest.mark.parametrize("name,expr_builder", EDGE_CASES)
def test_p0_edge_polars_long_matches_pandas(edge_source, name, expr_builder):
    from cleaned_operators.operator_surface import DAILY_CANONICALS
    from cleaned_operators.registry import OperatorRegistry
    if OperatorRegistry._aliases.get(name, name) not in DAILY_CANONICALS:
        pytest.skip("not on the production daily surface")
    expr = expr_builder()
    pd_out = _result_series(_run(edge_source, expr, "pandas"))
    long_out = _result_series(_run(edge_source, expr, "polars_long"))
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False, rtol=1e-6, atol=1e-6)


DUCKDB_EDGE_CASES = [
    ("protected_div", lambda: make_cleaned_call_factory("protected_div")(_col("numer"), _col("denom"))),
    ("safe_div_null", lambda: make_cleaned_call_factory("safe_div_null")(_col("numer"), _col("denom"))),
    ("protected_log", lambda: make_cleaned_call_factory("protected_log")(_col("neg"))),
    ("protected_sqrt", lambda: make_cleaned_call_factory("protected_sqrt")(_col("neg"))),
    ("divide", lambda: make_cleaned_call_factory("divide")(_col("numer"), _col("denom"))),
    ("add", lambda: make_cleaned_call_factory("add")(_col("close"), _col("open"))),
    ("sqrt", lambda: make_cleaned_call_factory("sqrt")(_col("neg"))),
    ("log", lambda: make_cleaned_call_factory("log")(_col("neg"))),
    ("where", lambda: make_cleaned_call_factory("where")(_col("flag"), _col("close"), _col("numer"))),
    ("rank", lambda: make_cleaned_call_factory("rank")(_col("close"))),
    ("zscore", lambda: make_cleaned_call_factory("zscore")(_col("close"))),
    ("ts_zscore", lambda: make_cleaned_call_factory("ts_zscore")(_col("close"), 3)),
    ("ts_std", lambda: make_cleaned_call_factory("ts_std")(_col("close"), 3)),
    ("ts_mean", lambda: make_cleaned_call_factory("ts_mean")(_col("close"), 3)),
    ("ts_delay", lambda: make_cleaned_call_factory("ts_delay")(_col("close"), 1)),
    ("ts_pct", lambda: make_cleaned_call_factory("ts_pct")(_col("close"), 1)),
    ("group_mean", lambda: make_cleaned_call_factory("group_mean")(_col("close"), _col("group_id"))),
    ("group_winsorize", lambda: make_cleaned_call_factory("group_winsorize")(_col("close"), _col("group_id"))),
    ("ts_corr", lambda: make_cleaned_call_factory("ts_corr")(_col("close"), _col("open"), 3)),
    ("ts_cov", lambda: make_cleaned_call_factory("ts_cov")(_col("ret"), _col("close"), 3)),
    ("ts_beta", lambda: make_cleaned_call_factory("ts_beta")(_col("ret"), _col("close"), 3)),
    ("group_zscore", lambda: make_cleaned_call_factory("group_zscore")(_col("close"), _col("group_id"))),
    ("winsorize", lambda: make_cleaned_call_factory("winsorize")(_col("close"))),
    ("cs_mad", lambda: make_cleaned_call_factory("cs_mad")(_col("close"))),
    ("cs_mad_zscore", lambda: make_cleaned_call_factory("cs_mad_zscore")(_col("close"))),
    ("vwap", lambda: make_cleaned_call_factory("vwap")(_col("close"), _col("volume"), 2)),
    ("power", lambda: make_cleaned_call_factory("power")(_col("close"), _col("numer"))),
    ("scale", lambda: make_cleaned_call_factory("scale")(_col("close"))),
    ("normalize", lambda: make_cleaned_call_factory("normalize")(_col("close"))),
    ("volatility", lambda: make_cleaned_call_factory("volatility")(_col("ret"), 3)),
    ("count", lambda: make_cleaned_call_factory("count")(_col("close"))),
    ("cum_max", lambda: make_cleaned_call_factory("cum_max")(_col("close"))),
    ("cum_min", lambda: make_cleaned_call_factory("cum_min")(_col("close"))),
    ("cum_prod", lambda: make_cleaned_call_factory("cum_prod")(_col("close"))),
    ("cum_delta", lambda: make_cleaned_call_factory("cum_delta")(_col("close"))),
    ("expanding_sum", lambda: make_cleaned_call_factory("expanding_sum")(_col("close"))),
    ("expanding_mean", lambda: make_cleaned_call_factory("expanding_mean")(_col("close"))),
    ("is_null", lambda: make_cleaned_call_factory("is_null")(_col("close"))),
    ("is_not_null", lambda: make_cleaned_call_factory("is_not_null")(_col("close"))),
    ("is_infinite", lambda: make_cleaned_call_factory("is_infinite")(_col("close"))),
    ("ts_autocorr", lambda: make_cleaned_call_factory("ts_autocorr")(_col("close"), 4, 1)),
    ("ts_rank", lambda: make_cleaned_call_factory("ts_rank")(_col("close"), 3)),
    ("ts_delta", lambda: make_cleaned_call_factory("ts_delta")(_col("close"), 1)),
    ("ts_pct", lambda: make_cleaned_call_factory("ts_pct")(_col("close"), 1)),
    ("multiply", lambda: make_cleaned_call_factory("multiply")(_col("close"), _col("open"))),
    ("subtract", lambda: make_cleaned_call_factory("subtract")(_col("close"), _col("open"))),
    ("ts_max", lambda: make_cleaned_call_factory("ts_max")(_col("close"), 3)),
    ("ts_min", lambda: make_cleaned_call_factory("ts_min")(_col("close"), 3)),
    ("safe_div_null", lambda: make_cleaned_call_factory("safe_div_null")(_col("numer"), _col("denom"))),
    ("sign", lambda: make_cleaned_call_factory("sign")(_col("close"))),
    ("fillna_const", lambda: make_cleaned_call_factory("fillna_const")(_col("close"), 0.0)),
    ("cum_sum", lambda: make_cleaned_call_factory("cum_sum")(_col("close"))),
    ("abs", lambda: make_cleaned_call_factory("abs")(_col("close"))),
    ("neg", lambda: make_cleaned_call_factory("neg")(_col("close"))),
    ("exp", lambda: make_cleaned_call_factory("exp")(_col("ret"))),
    ("floor", lambda: make_cleaned_call_factory("floor")(_col("close"))),
    ("ceil", lambda: make_cleaned_call_factory("ceil")(_col("close"))),
    ("maximum", lambda: make_cleaned_call_factory("maximum")(_col("close"), _col("open"))),
    ("minimum", lambda: make_cleaned_call_factory("minimum")(_col("close"), _col("open"))),
    ("gt", lambda: make_cleaned_call_factory("gt")(_col("close"), _col("open"))),
    ("lt", lambda: make_cleaned_call_factory("lt")(_col("close"), _col("open"))),
    ("eq", lambda: make_cleaned_call_factory("eq")(_col("close"), _col("open"))),
    ("ge", lambda: make_cleaned_call_factory("ge")(_col("close"), _col("open"))),
    ("le", lambda: make_cleaned_call_factory("le")(_col("close"), _col("open"))),
    ("ne", lambda: make_cleaned_call_factory("ne")(_col("close"), _col("open"))),
    ("and_", lambda: make_cleaned_call_factory("and_")(_col("flag"), _col("volume"))),
    ("or_", lambda: make_cleaned_call_factory("or_")(_col("flag"), _col("volume"))),
    ("not_", lambda: make_cleaned_call_factory("not_")(_col("flag"))),
    ("coalesce", lambda: make_cleaned_call_factory("coalesce")(_col("close"), _col("open"))),
    ("ts_sum", lambda: make_cleaned_call_factory("ts_sum")(_col("close"), 3)),
    ("ts_var", lambda: make_cleaned_call_factory("ts_var")(_col("close"), 3)),
    ("ts_median", lambda: make_cleaned_call_factory("ts_median")(_col("close"), 3)),
    ("rank_pct", lambda: make_cleaned_call_factory("rank_pct")(_col("close"))),
    ("cs_pct_rank", lambda: make_cleaned_call_factory("cs_pct_rank")(_col("close"))),
    ("cs_demean", lambda: make_cleaned_call_factory("cs_demean")(_col("close"))),
    ("cs_mean", lambda: make_cleaned_call_factory("cs_mean")(_col("close"))),
    ("cs_std", lambda: make_cleaned_call_factory("cs_std")(_col("close"))),
    ("cs_sum", lambda: make_cleaned_call_factory("cs_sum")(_col("close"))),
    ("cs_count", lambda: make_cleaned_call_factory("cs_count")(_col("close"))),
    ("log_returns", lambda: make_cleaned_call_factory("log_returns")(_col("close"))),
    ("group_rank", lambda: make_cleaned_call_factory("group_rank")(_col("close"), _col("group_id"))),
    ("group_neutralize", lambda: make_cleaned_call_factory("group_neutralize")(_col("close"), _col("group_id"))),
    ("clip", lambda: make_cleaned_call_factory("clip")(_col("close"), 0.0, 100.0)),
    ("ffill", lambda: make_cleaned_call_factory("ffill")(_col("close"))),
    ("group_normalize", lambda: make_cleaned_call_factory("group_normalize")(_col("close"), _col("group_id"))),
    ("group_percentile", lambda: make_cleaned_call_factory("group_percentile")(_col("close"), _col("group_id"), 0.5)),
    ("group_std", lambda: make_cleaned_call_factory("group_std")(_col("close"), _col("group_id"))),
    ("inverse", lambda: make_cleaned_call_factory("inverse")(_col("close"))),
    ("is_finite", lambda: make_cleaned_call_factory("is_finite")(_col("close"))),
    ("nan_to_num", lambda: make_cleaned_call_factory("nan_to_num")(_col("close"))),
    ("ts_sharpe", lambda: make_cleaned_call_factory("ts_sharpe")(_col("close"), 3)),
    ("div_or_default", lambda: make_cleaned_call_factory("div_or_default")(_col("numer"), _col("denom"))),
    ("log_fill_invalid", lambda: make_cleaned_call_factory("log_fill_invalid")(_col("neg"))),
    ("log_abs", lambda: make_cleaned_call_factory("log_abs")(_col("neg"))),
    ("signed_log", lambda: make_cleaned_call_factory("signed_log")(_col("neg"))),
    ("signed_sqrt", lambda: make_cleaned_call_factory("signed_sqrt")(_col("neg"))),
    ("group_sum", lambda: make_cleaned_call_factory("group_sum")(_col("close"), _col("group_id"))),
    ("group_min", lambda: make_cleaned_call_factory("group_min")(_col("close"), _col("group_id"))),
    ("group_max", lambda: make_cleaned_call_factory("group_max")(_col("close"), _col("group_id"))),
    ("group_count", lambda: make_cleaned_call_factory("group_count")(_col("close"), _col("group_id"))),
    ("tanh", lambda: make_cleaned_call_factory("tanh")(_col("neg"))),
]


@pytest.mark.parametrize("name,expr_builder", DUCKDB_EDGE_CASES)
def test_p0_edge_duckdb_matches_pandas(duckdb_source, name, expr_builder):
    from cleaned_operators.operator_surface import DAILY_CANONICALS
    from cleaned_operators.registry import OperatorRegistry
    if OperatorRegistry._aliases.get(name, name) not in DAILY_CANONICALS:
        pytest.skip("not on the production daily surface")
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
    from cleaned_operators.operator_surface import DAILY_CANONICALS
    if "vwap" not in DAILY_CANONICALS:
        pytest.skip("vwap is a field/recipe, not a production primitive")
    expr = make_cleaned_call_factory("vwap")(col("close"), col("volume"), 2)
    out = FactorEngine(backend=build_backend("polars_long"), data_source=edge_source).run(
        Factor(name="t", expr=expr)
    )
    assert out.get("used_polars_long_native") is True
    pd_out = _result_series(_run(edge_source, expr, "pandas"))
    long_out = out["result"].sort_index()
    pd.testing.assert_series_equal(pd_out, long_out, check_names=False, rtol=1e-6, atol=1e-6)
