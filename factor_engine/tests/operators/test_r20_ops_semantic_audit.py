import numpy as np
import pandas as pd

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators._numpy_kernels import downside_beta_
from factor_engine.cleaned_operators.composite_fastpath import pd_vp_weighted_price
from factor_engine.cleaned_operators.microstructure import ops as legacy_micro
from factor_engine.cleaned_operators.microstructure.session import pct_change_by_session, rolling_by_session
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.tools.catalog_r20_ops_params_recipes import migrate_formula


def _panel(values, index=None):
    if index is None:
        index = pd.date_range("2025-01-01", periods=len(values))
    return pd.DataFrame({"A": np.asarray(values, dtype=float)}, index=index)


def test_vp_weighted_price_window_is_used_and_matches_shared_kernel():
    load_all()
    n = 35
    close = _panel(10 + np.linspace(0, 3, n) + 0.2 * np.sin(np.arange(n)))
    volume = _panel(np.linspace(100, 900, n))
    open_ = close * (1 + 0.002 * np.cos(np.arange(n))[:, None])
    high = pd.DataFrame(np.maximum(open_, close) + 0.15, index=close.index, columns=close.columns)
    low = pd.DataFrame(np.minimum(open_, close) - 0.12, index=close.index, columns=close.columns)
    op = OperatorRegistry.get("recipe_vp_weighted_price", mode="any")
    actual = op.calculate(close, volume, open_, high, low, 20)
    expected = pd_vp_weighted_price(close, volume, open_, high, low, window=20, min_periods=5)
    pd.testing.assert_frame_equal(actual, expected)
    short = op.calculate(close, volume, open_, high, low, 5)
    assert not np.allclose(actual.iloc[-8:], short.iloc[-8:], equal_nan=True)
    prefix = op.calculate(close.iloc[:24], volume.iloc[:24], open_.iloc[:24], high.iloc[:24], low.iloc[:24], 20)
    pd.testing.assert_frame_equal(actual.iloc[:24], prefix)


def test_reviewed_micro_adapters_equal_session_aware_legacy_kernels_and_are_prefix_causal():
    load_all()
    day1 = pd.date_range("2025-01-02 09:31", periods=18, freq="min")
    day2 = pd.date_range("2025-01-03 09:31", periods=18, freq="min")
    index = day1.append(day2)
    close = _panel(10 + 0.03 * np.arange(36) + 0.1 * np.sin(np.arange(36)), index)
    volume = _panel(100 + (np.arange(36) % 7) * 15, index)
    high, low = close + 0.05, close - 0.04
    cases = [
        ("recipe_micro_spread", legacy_micro.MicroSpreadOp(), (high, low, close)),
        ("recipe_micro_trade_imbalance", legacy_micro.MicroTradeImbalanceOp(), (close, volume, 20, 2)),
        ("recipe_micro_vpin", legacy_micro.MicroVpinOp(), (close, volume, 20, 2)),
        ("recipe_micro_amihud_hf", legacy_micro.MicroAmihudHfOp(), (close, volume)),
    ]
    for canonical, legacy, args in cases:
        reviewed = OperatorRegistry.get(canonical, mode="any")
        actual = reviewed.calculate(*args)
        if canonical == "recipe_micro_spread":
            expected = (high - low) / close
        else:
            ret = pct_change_by_session(close["A"])
            if canonical == "recipe_micro_amihud_hf":
                series = ret.abs() / (close["A"] * volume["A"]).replace(0, np.nan)
            else:
                magnitude = ret.abs() if canonical == "recipe_micro_vpin" else np.sign(ret)
                series = rolling_by_session(magnitude * volume["A"], 20, "sum", min_periods=2) / rolling_by_session(volume["A"], 20, "sum", min_periods=2).replace(0, np.nan)
            expected = series.to_frame("A")
        pd.testing.assert_frame_equal(actual, expected)
        cut_args = tuple(x.iloc[:27] if isinstance(x, pd.DataFrame) else x for x in args)
        prefix = reviewed.calculate(*cut_args)
        pd.testing.assert_frame_equal(actual.iloc[:27], prefix)


def test_reviewed_downside_beta_matches_exact_kernel_missingness_and_prefix():
    load_all()
    rng = np.random.default_rng(7)
    market = rng.normal(0, 0.012, 90)
    stock = 1.7 * market + rng.normal(0, 0.002, 90)
    market[[15, 70]] = np.nan
    stock[22] = np.nan
    ret, benchmark = _panel(stock), _panel(market)
    op = OperatorRegistry.get("recipe_downside_beta", mode="any")
    actual = op.calculate(ret, benchmark, 60)
    expected = _panel(downside_beta_(stock, market, 60), ret.index)
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.iloc[:59].isna().all().all()
    prefix = op.calculate(ret.iloc[:73], benchmark.iloc[:73], 60)
    pd.testing.assert_frame_equal(actual.iloc[:73], prefix)


def test_catalog_rewrites_name_parameters_and_preserves_scalar_frequency_meaning():
    vp, _ = migrate_formula("vp_weighted_price(close, volume)")
    assert vp.endswith(", 20)") and "recipe_vp_weighted_price" in vp
    beta, _ = migrate_formula("downside_beta(ret)")
    assert "recipe_downside_beta" in beta and "BenchmarkIndexDailyBar" in beta
    micro, _ = migrate_formula("micro_vpin(MinuteOHLCVA)")
    assert "recipe_micro_vpin" in micro and "StockMinuteBar" in micro
    fft, _ = migrate_formula("fft(turnover_ratio)")
    assert fft == "ts_signal_spectral_entropy(turnover_ratio, 60)"
    wavelet, _ = migrate_formula("wavelet(turnover_ratio)")
    assert wavelet == "subtract(ts_mean(turnover_ratio, 5), ts_mean(turnover_ratio, 20))"
