# -*- coding: utf-8
"""Adversarial three-backend (pandas / polars_long / duckdb_sql) probe.

Runs a broad operator set across ALL THREE backends on hostile data (NaN
holes, an inf, uneven group sizes, a single-instrument panel, tiny windows,
negative values) and asserts NaN-aware alignment.  Covers operators with a
real polars/duckdb fastpath AND operators that must hybrid-fallback to the
pandas reference — both are required to agree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("polars")

from factor_engine.api.cleaned_ops import make_cleaned_call_factory
from factor_engine.api.columns import col
from factor_engine.api.factor import Factor
from factor_engine.backend.factory import build_backend
from factor_engine.cleaned_operators import load_all
from factor_engine.runtime.engine import FactorEngine
from factor_engine.storage.factory import build_data_source
from tests.helpers import InMemorySeriesSource

F = make_cleaned_call_factory


def _memory_source() -> InMemorySeriesSource:
    load_all()
    dates = pd.date_range("2024-01-02", periods=45, freq="B")
    insts = ["A", "B", "C", "D"]
    idx = pd.MultiIndex.from_product([dates, insts], names=["timestamp", "instrument"])
    rng = np.random.default_rng(20260809)
    close = pd.Series(rng.normal(100, 10, len(idx)), index=idx)
    close[(rng.random(len(idx)) < 0.12)] = np.nan  # NaN holes
    # an inf to exercise inf semantics
    close.iloc[7] = np.inf
    open_ = close.shift(1, fill_value=close.iloc[0]) + rng.normal(0, 0.5, len(idx))
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 1, len(idx)))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 1, len(idx)))
    volume = pd.Series(rng.integers(100, 2000, len(idx)).astype(float), index=idx)
    volume[(rng.random(len(idx)) < 0.05)] = np.nan
    ret = close / close.shift(1, fill_value=close.iloc[0]) - 1.0
    grp = pd.Series(
        [1.0 if (i % 4) < 2 else 2.0 for i in range(len(idx))], index=idx
    )
    pre_close = close.shift(1, fill_value=close.iloc[0])
    amount = volume * close
    return InMemorySeriesSource(
        data={
            "close": close,
            "open": open_,
            "high": high,
            "low": low,
            "volume": volume,
            "ret": ret,
            "group_id": grp,
            "pre_close": pre_close,
            "amount": amount,
        }
    )


def _write_duckdb_registry(tmp_path, root) -> str:
    content = f"""
probe_daily:
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
    Open: double
    High: double
    Low: double
    Close: double
    Volume: double
    Ret: double
    group_id: int64
    pre_close: double
    amount: double
"""
    path = tmp_path / "datasets.yaml"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return str(path)


def _seed_duckdb_panel(root, mem: InMemorySeriesSource) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for (ts, sym), close in mem.data["close"].items():
        rows.append(
            {
                "TradeDate": ts.date(),
                "Symbol": sym,
                "Open": _v(mem.data["open"].loc[(ts, sym)]),
                "High": _v(mem.data["high"].loc[(ts, sym)]),
                "Low": _v(mem.data["low"].loc[(ts, sym)]),
                "Close": _v(close),
                "Volume": _v(mem.data["volume"].loc[(ts, sym)]),
                "Ret": _v(mem.data["ret"].loc[(ts, sym)]),
                "group_id": int(mem.data["group_id"].loc[(ts, sym)]),
                "pre_close": _v(mem.data["pre_close"].loc[(ts, sym)]),
                "amount": _v(mem.data["amount"].loc[(ts, sym)]),
            }
        )
    pd.DataFrame(rows).to_parquet(root / "panel.parquet")


def _v(x):
    # Preserve inf exactly (parquet float can hold it); only NaN becomes SQL NULL.
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return float(x)


def _col(name: str):
    mapping = {
        "close": "Close",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "volume": "Volume",
        "ret": "Ret",
        "group_id": "group_id",
        "pre_close": "pre_close",
        "amount": "amount",
    }
    return col(mapping.get(name, name))


@pytest.fixture(scope="module")
def probe_source():
    return _memory_source()


@pytest.fixture
def duckdb_probe_source(tmp_path, monkeypatch, probe_source):
    monkeypatch.setenv("DATA_ACCESS_SKIP_COS_MIRROR", "1")
    monkeypatch.setenv("DATA_ACCESS_CONFIG", _write_duckdb_registry(tmp_path, tmp_path / "data"))
    _seed_duckdb_panel(tmp_path / "data", probe_source)
    try:
        from data_access import reset_store
        reset_store()
    except ImportError:
        pass
    return build_data_source({"type": "data_access", "dataset": "probe_daily"})


def _run(source, expr, backend_name: str):
    return FactorEngine(
        backend=build_backend(backend_name), data_source=source, run_mode="research"
    ).run(Factor(name="t", expr=expr))


def _compare(a: pd.Series, b: pd.Series, name: str, backend: str) -> None:
    both = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="outer").sort_index()
    # NaN must align to NaN; inf must align to inf (same sign).
    na_a = both["a"].isna()
    na_b = both["b"].isna()
    assert na_a.equals(na_b), f"{name} [{backend}]: NaN pattern differs:\n{both[na_a != na_b].head(10)}"
    va = both["a"].to_numpy(dtype=float)
    vb = both["b"].to_numpy(dtype=float)
    a_inf = np.isposinf(va) if va.dtype == np.float64 else np.zeros(len(va), bool)
    b_inf = np.isposinf(vb) if vb.dtype == np.float64 else np.zeros(len(vb), bool)
    a_ninf = np.isneginf(va) if va.dtype == np.float64 else np.zeros(len(va), bool)
    b_ninf = np.isneginf(vb) if vb.dtype == np.float64 else np.zeros(len(vb), bool)
    inf_ok = (a_inf & b_inf) | (a_ninf & b_ninf)
    inf_bad = ((a_inf | a_ninf) | (b_inf | b_ninf)) & ~inf_ok
    both = both.assign(_inf_bad=inf_bad)
    finite = both["a"].notna() & both["b"].notna() & ~(a_inf | a_ninf | b_inf | b_ninf)
    diff = (both["a"] - both["b"]).abs()
    denom = both["a"].abs()
    bad = (finite & ~(diff <= 1e-5 + 1e-5 * denom)) | both["_inf_bad"]
    n_bad = int(bad.sum())
    assert n_bad == 0, (
        f"{name} [{backend}]: {n_bad} mismatched cells; max_abs_diff="
        f"{float(diff[finite].max()) if finite.any() else 0}\n"
        f"{both[bad].head(6)}"
    )


# Operators runnable on plain bars (no extras).
_BAR_CASES = [
    ("ts_mean", lambda: F("ts_mean")(col("close"), 3)),
    ("ts_mean_k2", lambda: F("ts_mean")(col("close"), 2)),
    ("ts_std", lambda: F("ts_std")(col("close"), 5)),
    ("ts_var", lambda: F("ts_var")(col("close"), 5)),
    ("ts_skew", lambda: F("ts_skew")(col("close"), 6)),
    ("ts_kurt", lambda: F("ts_kurt")(col("close"), 6)),
    ("ts_zscore", lambda: F("ts_zscore")(col("close"), 5)),
    ("ts_rank", lambda: F("ts_rank")(col("close"), 5)),
    ("ts_median", lambda: F("ts_median")(col("close"), 5)),
    ("ts_sum", lambda: F("ts_sum")(col("close"), 5)),
    ("ts_min", lambda: F("ts_min")(col("close"), 5)),
    ("ts_max", lambda: F("ts_max")(col("close"), 5)),
    ("ts_delay", lambda: F("ts_delay")(col("close"), 2)),
    ("ts_delta", lambda: F("ts_delta")(col("close"), 3)),
    ("ts_pct", lambda: F("ts_pct")(col("close"), 3)),
    ("ts_sharpe", lambda: F("ts_sharpe")(col("close"), 5)),
    ("ts_autocorr", lambda: F("ts_autocorr")(col("close"), 5, 1)),
    ("ts_corr", lambda: F("ts_corr")(col("close"), col("open"), 5)),
    ("ts_cov", lambda: F("ts_cov")(col("close"), col("open"), 5)),
    ("ts_beta", lambda: F("ts_beta")(col("ret"), col("close"), 5)),
    ("ts_decay_linear", lambda: F("ts_decay_linear")(col("close"), 5)),
    ("expanding_mean", lambda: F("expanding_mean")(col("close"))),
    ("expanding_std", lambda: F("expanding_std")(col("close"))),
    ("cum_delta", lambda: F("cum_delta")(col("close"))),
    ("cumsum", lambda: F("cumsum")(col("close"))),
    ("rank", lambda: F("rank")(col("close"))),
    ("rank_pct", lambda: F("rank_pct")(col("close"))),
    ("cs_mean", lambda: F("cs_mean")(col("close"))),
    ("cs_std", lambda: F("cs_std")(col("close"))),
    ("cs_zscore", lambda: F("cs_zscore")(col("close"))),
    ("cs_rank", lambda: F("cs_rank")(col("close"))),
    ("cs_pct_rank", lambda: F("cs_pct_rank")(col("close"))),
    ("cs_mad", lambda: F("cs_mad")(col("close"))),
    ("cs_mad_zscore", lambda: F("cs_mad_zscore")(col("close"))),
    ("cs_quantile", lambda: F("cs_quantile")(col("close"), 0.5)),
    ("cs_min", lambda: F("cs_min")(col("close"))),
    ("cs_max", lambda: F("cs_max")(col("close"))),
    ("cs_count", lambda: F("cs_count")(col("close"))),
    ("cs_sum", lambda: F("cs_sum")(col("close"))),
    ("group_mean", lambda: F("group_mean")(col("close"), col("group_id"))),
    ("group_std", lambda: F("group_std")(col("close"), col("group_id"))),
    ("group_zscore", lambda: F("group_zscore")(col("close"), col("group_id"))),
    ("group_rank", lambda: F("group_rank")(col("close"), col("group_id"))),
    ("group_percentile", lambda: F("group_percentile")(col("close"), col("group_id"), 0.5)),
    ("group_winsorize", lambda: F("group_winsorize")(col("close"), col("group_id"))),
    ("group_median", lambda: F("group_median")(col("close"), col("group_id"))),
    ("winsorize", lambda: F("winsorize")(col("close"))),
    ("normalize", lambda: F("normalize")(col("close"))),
    ("zscore", lambda: F("zscore")(col("close"))),
    ("protected_div", lambda: F("protected_div")(col("close"), col("volume"))),
    ("protected_log", lambda: F("protected_log")(col("close"))),
    ("log_returns", lambda: F("log_returns")(col("close"))),
    ("returns", lambda: F("returns")(col("close"))),
    ("add", lambda: F("add")(col("close"), col("open"))),
    ("subtract", lambda: F("subtract")(col("close"), col("open"))),
    ("multiply", lambda: F("multiply")(col("close"), col("open"))),
    ("divide", lambda: F("divide")(col("close"), col("open"))),
    ("power", lambda: F("power")(col("close"), col("volume"))),
    ("gt", lambda: F("gt")(col("close"), col("open"))),
    ("where", lambda: F("where")(col("group_id"), col("close"), col("open"))),
    ("ts_returns", lambda: F("ts_returns")(col("close"), 5)),
    ("ts_ewm_mean", lambda: F("ts_ewm_mean")(col("close"), 5)),
    ("ts_ewm_std", lambda: F("ts_ewm_std")(col("close"), 5)),
    ("SMA", lambda: F("SMA")(col("close"), 5)),
    ("EMA", lambda: F("EMA")(col("close"), 5)),
    ("WMA", lambda: F("WMA")(col("close"), 5)),
    ("RSI", lambda: F("RSI")(col("close"), 5)),
    ("bollinger_mid", lambda: F("bollinger_mid")(col("close"), 5)),
    ("bollinger_upper", lambda: F("bollinger_upper")(col("close"), 5)),
    ("bollinger_lower", lambda: F("bollinger_lower")(col("close"), 5)),
    ("MACD_line", lambda: F("MACD_line")(col("close"), 5, 10)),
    ("MACD_signal", lambda: F("MACD_signal")(col("close"), 5, 10, 3)),
    ("MACD_hist", lambda: F("MACD_hist")(col("close"), 5, 10, 3)),
    ("vwap", lambda: F("vwap")(col("close"), col("volume"), 5)),
    ("VWAP", lambda: F("VWAP")(col("close"), col("volume"))),
    ("returns_daily", lambda: F("returns")(col("close"))),
    ("true_range", lambda: F("true_range")(col("high"), col("low"), col("close"))),
    ("atr", lambda: F("atr")(col("high"), col("low"), col("close"), 5)),
    ("pct_rank_ts", lambda: F("ts_pct_rank")(col("close"), 5)),
    ("avg_true_range_pct", lambda: F("avg_true_range_pct")(col("high"), col("low"), col("close"), 5)),
    ("typical_price", lambda: F("typical_price")(col("high"), col("low"), col("close"))),
    ("median_price", lambda: F("median_price")(col("high"), col("low"))),
    ("high_low_range", lambda: F("high_low_range")(col("high"), col("low"))),
    # Synthetic pre_close is derived from the same continuous close series.
    ("overnight_return", lambda: F("overnight_return")(col("open"), col("pre_close"), price_basis="CONTINUOUS")),
    ("open_close_return", lambda: F("open_close_return")(col("open"), col("close"))),
    ("candle_body", lambda: F("candle_body")(col("open"), col("close"))),
    ("candle_upper_shadow", lambda: F("candle_upper_shadow")(col("open"), col("high"), col("close"))),
    ("candle_lower_shadow", lambda: F("candle_lower_shadow")(col("open"), col("low"), col("close"))),
    ("candle_real_body", lambda: F("candle_real_body")(col("open"), col("close"))),
    ("hilo_pct", lambda: F("hilo_pct")(col("high"), col("low"), col("close"))),
    ("ts_quantile", lambda: F("ts_quantile")(col("close"), 5, 0.5)),
    ("ts_max_drawdown", lambda: F("ts_max_drawdown")(col("close"), 10)),
    ("ts_drawdown", lambda: F("ts_drawdown")(col("close"))),
    ("volume_ratio", lambda: F("volume_ratio")(col("volume"), 5)),
    ("dollar_volume", lambda: F("dollar_volume")(col("close"), col("volume"))),
    ("amount_weighted_price", lambda: F("amount_weighted_price")(col("close"), col("volume"))),
    ("price_pct_chg", lambda: F("price_pct_chg")(col("close"), 3)),
    ("ts_pct_rank", lambda: F("ts_pct_rank")(col("close"), 5)),
    ("ts_rank_mean", lambda: F("ts_rank_mean")(col("close"), 5)),
    ("ts_rank_std", lambda: F("ts_rank_std")(col("close"), 5)),
    ("ts_linear_reg_slope", lambda: F("ts_linear_reg_slope")(col("close"), 5)),
    ("ts_linear_reg_residual", lambda: F("ts_linear_reg_residual")(col("close"), 5)),
    ("ts_time_slope", lambda: F("ts_time_slope")(col("close"), 5)),
    ("cs_skew", lambda: F("cs_skew")(col("close"))),
    ("cs_kurt", lambda: F("cs_kurt")(col("close"))),
    ("cs_range", lambda: F("cs_range")(col("close"))),
    ("cs_demean", lambda: F("cs_demean")(col("close"))),
    ("cs_robust_resid", lambda: F("cs_robust_resid")(col("close"), col("open"))),
]

# These legacy names are deliberately absent from the active canonical surface.
# Keep testing their fail-closed contract, but do not pretend they have three
# executable backends and then count the expected KeyError as a parity failure.
_REMOVED_CASE_NAMES = {
    "expanding_mean", "expanding_std", "cum_delta", "cumsum", "cs_min",
    "cs_max", "group_median", "protected_log", "ts_returns", "RSI",
    "bollinger_mid", "bollinger_upper", "bollinger_lower", "vwap", "VWAP",
    "atr", "pct_rank_ts", "avg_true_range_pct", "typical_price",
    "median_price", "high_low_range", "candle_real_body", "hilo_pct",
    "ts_drawdown", "volume_ratio", "amount_weighted_price", "price_pct_chg",
    "ts_pct_rank", "ts_rank_mean", "ts_rank_std", "ts_linear_reg_slope",
    "ts_linear_reg_residual", "cs_skew", "cs_kurt", "cs_range",
}
_REMOVED_CASES = [case for case in _BAR_CASES if case[0] in _REMOVED_CASE_NAMES]
_BAR_CASES = [case for case in _BAR_CASES if case[0] not in _REMOVED_CASE_NAMES]


@pytest.mark.parametrize("name,expr_builder", _REMOVED_CASES)
def test_removed_probe_names_fail_closed(name, expr_builder, probe_source):
    with pytest.raises(KeyError, match="unknown operator canonical"):
        _run(probe_source, expr_builder(), "pandas")


@pytest.mark.parametrize("name,expr_builder", _BAR_CASES)
def test_triple_bar_parity(probe_source, duckdb_probe_source, name, expr_builder):
    expr = expr_builder()
    pd_out = _run(probe_source, expr, "pandas")["result"].sort_index()
    long_out = _run(probe_source, expr, "polars_long")["result"].sort_index()
    _compare(pd_out, long_out, name, "polars_long")
    # DuckDB needs a mapping column name for every input used.
    _compare(pd_out, duckdb_out := _run(duckdb_probe_source, expr, "duckdb_sql")["result"].sort_index(), name, "duckdb_sql")


# Single-instrument panel — polars group-by / duckdb cross-section edge cases.
def test_triple_single_instrument(probe_source, duckdb_probe_source):
    single_values = pd.Series(
        probe_source.data["close"].xs("A", level="instrument"),
        name="close",
    )
    single_values.index = single_values.index.rename("timestamp")
    single = pd.concat({"A": single_values}, names=["instrument"]).swaplevel()
    single.index = single.index.set_names(["timestamp", "instrument"])
    single = single.sort_index()
    from tests.helpers import InMemorySeriesSource as _IMS
    src = _IMS(data={"close": single, "open": single - 0.5, "volume": single * 2, "group_id": pd.Series(1.0, index=single.index)})
    for name, expr_builder in [
        ("rank", lambda: F("rank")(col("close"))),
        ("zscore", lambda: F("zscore")(col("close"))),
        ("ts_mean", lambda: F("ts_mean")(col("close"), 3)),
        ("group_mean", lambda: F("group_mean")(col("close"), col("group_id"))),
        ("cs_mean", lambda: F("cs_mean")(col("close"))),
    ]:
        pd_out = _run(src, expr_builder(), "pandas")["result"].sort_index()
        long_out = _run(src, expr_builder(), "polars_long")["result"].sort_index()
        _compare(pd_out, long_out, f"single:{name}", "polars_long")


def test_true_range_polars_native_null_matches_pandas():
    """A native Polars NULL in high/low must not fall through a NULL predicate."""
    import polars as pl

    from factor_engine.cleaned_operators.registry import OperatorRegistry

    high_pd = pd.DataFrame({"A": [3.0, np.nan, 5.0]})
    low_pd = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    close_pd = pd.DataFrame({"A": [2.0, 2.0, 4.0]})
    high_pl = pl.DataFrame({"A": [3.0, None, 5.0]})
    low_pl = pl.DataFrame({"A": [1.0, 2.0, 3.0]})
    close_pl = pl.DataFrame({"A": [2.0, 2.0, 4.0]})

    pandas_op = OperatorRegistry.get("true_range", "pandas_numpy")
    polars_op = OperatorRegistry.get("true_range", "polars")
    expected = pandas_op.calculate(high_pd, low_pd, close_pd)["A"]
    actual = pd.Series(polars_op.calculate(high_pl, low_pl, close_pl)["A"].to_list())
    pd.testing.assert_series_equal(expected, actual, check_names=False)
