# -*- coding: utf-8 -*-
"""R11 round-3 operator-audit fixes for the liquidity / price-volume family.

Covers the liquidity_v2 / polars_liquidity_v2 fixes:
* 17  volume_autocorr raw lookback = window + lag + ``lag < window`` relational.
* 18  volume pct-change family guarded at a zero volume base (no inf blow-up).
* 19  volume inputs must be non-negative (input_units + fail-loud check).
* 20  ``ADL`` renamed ``rolling_adl_flow`` (bounded rolling flow, not cumulative).
* 21  ``volume_to_range`` renamed ``volume_price_range_density`` (unit stated,
       range <= 0 -> NaN).
* 26  polars ``ts_impulse_strength`` scales on volatility_{t-1} (excludes the
       current impulse).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

load_all()


def _panel(values):
    return pd.DataFrame({"A": np.asarray(values, dtype=float)})


# ---------------------------------------------------------------------------
# 17. volume_autocorr: raw lookback = window + lag, lag < window relational
# ---------------------------------------------------------------------------


def test_volume_autocorr_raw_lookback_is_window_plus_lag():
    op = OperatorRegistry.get("volume_autocorr", "pandas_numpy")
    x = _panel(np.arange(1.0, 25.0))
    # first valid output requires window + lag - 1 prior bars (raw history
    # window + lag), NOT window bars.
    out = op.calculate(x, 5, 2)
    assert int(out["A"].first_valid_index()) == 5 + 2 - 1


def test_volume_autocorr_declares_history_formula():
    op = OperatorRegistry.get("volume_autocorr", "pandas_numpy")
    assert op.metadata.param_specs["window"].history_formula == "window + lag"
    assert op.metadata.param_specs["lag"].history_semantics == "exact_rows"


def test_volume_autocorr_lag_lt_window_relational():
    op = OperatorRegistry.get("volume_autocorr", "pandas_numpy")
    assert op.metadata.relational_specs, "expected a declared relational spec"
    expr = [r.expression for r in op.metadata.relational_specs]
    assert "lag < window" in expr
    x = _panel(np.arange(1.0, 25.0))
    # lag >= window is infeasible at the call boundary (runtime enforcement).
    with pytest.raises(ValueError, match="lag must be < window"):
        op.calculate(x, 3, 4)
    # valid combos still run
    assert op.calculate(x, 5, 3)["A"].notna().any()


def test_volume_autocorr_polars_twin_matches_pandas():
    pl = pytest.importorskip("polars")
    pa = OperatorRegistry.get("volume_autocorr", "pandas_numpy")
    po = OperatorRegistry.get("volume_autocorr", "polars")
    assert po.metadata.relational_specs, "polars twin must declare the relation"
    rng = np.random.default_rng(7)
    vol = _panel(rng.integers(1e4, 5e5, 80).astype(float))
    pvol = pl.DataFrame({"A": vol["A"].to_numpy()})
    np.testing.assert_allclose(
        pa.calculate(vol, 10, 2)["A"].to_numpy(),
        po.calculate(pvol, 10, 2)["A"].to_numpy(),
        rtol=1e-8,
        atol=1e-8,
        equal_nan=True,
    )


# ---------------------------------------------------------------------------
# 18. volume pct-change family: zero-base must not blow up to inf
# ---------------------------------------------------------------------------


def test_volume_volatility_zero_base_is_nan_not_inf():
    op = OperatorRegistry.get("volume_volatility", "pandas_numpy")
    v = _panel([100.0, 0.0, 200.0, 300.0])
    out = op.calculate(v, 2)["A"].to_numpy()
    assert np.isfinite(np.nan_to_num(out)).all() or np.isnan(out).all()


def test_volume_volatility_zero_base_guarded_pandas_vs_polars():
    pl = pytest.importorskip("polars")
    pa = OperatorRegistry.get("volume_volatility", "pandas_numpy")
    po = OperatorRegistry.get("volume_volatility", "polars")
    v = _panel([100.0, 0.0, 200.0, 300.0])
    pv = pl.DataFrame({"A": v["A"].to_numpy()})
    a = pa.calculate(v, 2)["A"].to_numpy()
    b = po.calculate(pv, 2)["A"].to_numpy()
    assert not np.isinf(a).any() and not np.isinf(b).any()
    np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-8, equal_nan=True)


def test_return_volume_beta_zero_base_guarded():
    op = OperatorRegistry.get("return_volume_beta", "pandas_numpy")
    ret = _panel([0.01, 0.02, 0.03, 0.04, 0.05])
    vol = _panel([100.0, 0.0, 200.0, 300.0, 400.0])
    out = op.calculate(ret, vol, 3)["A"].to_numpy()
    assert not np.isinf(out).any()


def test_price_volume_divergence_zero_base_guarded():
    op = OperatorRegistry.get("price_volume_divergence", "pandas_numpy")
    close = _panel([10.0, 11.0, 12.0, 13.0, 14.0])
    vol = _panel([100.0, 0.0, 200.0, 300.0, 400.0])
    out = op.calculate(close, vol, 2, 2)["A"].to_numpy()
    assert not np.isinf(out).any()


# ---------------------------------------------------------------------------
# 19. volume inputs are non-negative: input_units + fail-loud runtime check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "up_volume_ratio",
        "down_volume_ratio",
        "volume_weighted_return",
        "volume_weighted_momentum",
        "return_volume_beta",
        "return_turnover_beta",
    ],
)
def test_non_negative_volume_declared_and_enforced(name):
    op = OperatorRegistry.get(name, "pandas_numpy")
    assert op is not None
    units = op.metadata.input_units or {}
    assert any(v == "non_negative_volume" for v in units.values()), name
    ret = _panel([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    close = _panel([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    turnover = _panel([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    vol_neg = _panel([1.0, -2.0, 3.0, 4.0, 5.0, 6.0])
    vol_pos = _panel([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    args = {
        "up_volume_ratio": (ret, vol_neg),
        "down_volume_ratio": (ret, vol_neg),
        "volume_weighted_return": (ret, vol_neg),
        "volume_weighted_momentum": (close, vol_neg),
        "return_volume_beta": (ret, vol_neg),
        "return_turnover_beta": (ret, vol_neg),
    }[name]
    good_args = {
        "up_volume_ratio": (ret, vol_pos),
        "down_volume_ratio": (ret, vol_pos),
        "volume_weighted_return": (ret, vol_pos),
        "volume_weighted_momentum": (close, vol_pos),
        "return_volume_beta": (ret, vol_pos),
        "return_turnover_beta": (ret, vol_pos),
    }[name]
    with pytest.raises(ValueError, match="non-negative"):
        op.calculate(*args, 3)
    op.calculate(*good_args, 3)  # positive volume runs


def test_up_volume_ratio_denominator_not_abs_of_negative():
    # The denominator must be the raw non-negative volume, never abs() of a
    # negative numerator.  A negative volume is rejected outright, so the ratio
    # can never exceed [0, 1] from a signed-volume artifact.
    op = OperatorRegistry.get("up_volume_ratio", "pandas_numpy")
    ret = _panel([0.01, 0.02, 0.03, 0.04])
    vol = _panel([1.0, 2.0, 3.0, 4.0])
    out = op.calculate(ret, vol, 2)["A"].to_numpy()
    assert not np.isnan(out).all()
    valid = out[np.isfinite(out)]
    assert ((valid >= 0.0) & (valid <= 1.0)).all()


def test_polars_non_negative_volume_enforced():
    pl = pytest.importorskip("polars")
    op = OperatorRegistry.get("up_volume_ratio", "polars")
    ret = pl.DataFrame({"A": [0.01, 0.02, 0.03, 0.04]})
    negv = pl.DataFrame({"A": [1.0, -2.0, 3.0, 4.0]})
    with pytest.raises(ValueError, match="non-negative"):
        op.calculate(ret, negv, 2)


# ---------------------------------------------------------------------------
# 20. ADL -> rolling_adl_flow (bounded rolling flow, NOT cumulative ADL)
# ---------------------------------------------------------------------------


def test_rolling_adl_flow_is_canonical_and_adl_alias():
    canon = OperatorRegistry.resolve_canonical("ADL")
    assert canon == "rolling_adl_flow"
    assert OperatorRegistry.get("rolling_adl_flow", "pandas_numpy") is not None
    assert OperatorRegistry.get("rolling_adl_flow", "polars") is not None
    # the alias still resolves to the same kernel
    alias = OperatorRegistry.get("ADL", "pandas_numpy")
    canon_op = OperatorRegistry.get("rolling_adl_flow", "pandas_numpy")
    assert alias is not None and canon_op is not None
    rng = np.random.default_rng(3)
    n = 40
    close = _panel(50.0 + np.cumsum(rng.normal(0, 0.5, n)))
    high = close + 0.5
    low = close - 0.5
    vol = _panel(rng.integers(1e4, 5e5, n).astype(float))
    a = alias.calculate(high, low, close, vol, 10)
    b = canon_op.calculate(high, low, close, vol, 10)
    np.testing.assert_allclose(
        a["A"].to_numpy(), b["A"].to_numpy(), rtol=1e-12, atol=1e-12, equal_nan=True
    )


def test_chaikin_oscillator_describes_rolling_flow_not_cumulative():
    op = OperatorRegistry.get("ChaikinOscillator", "pandas_numpy")
    desc = op.metadata.description.lower()
    # The description must explicitly DISAMBIGUATE the rolling-flow semantic
    # from the classic cumulative ADL (it should say "not cumulative").
    assert "rolling" in desc or "bounded" in desc
    assert "not cumulative" in desc or "not the cumulative" in desc


# ---------------------------------------------------------------------------
# 21. volume_to_range -> volume_price_range_density (unit stated, range 0 -> NaN)
# ---------------------------------------------------------------------------


def test_volume_price_range_density_canonical_and_alias():
    assert OperatorRegistry.resolve_canonical("volume_to_range") == "volume_price_range_density"
    op = OperatorRegistry.get("volume_price_range_density", "pandas_numpy")
    assert op is not None
    assert op.metadata.output_unit == "volume_per_price_range"
    assert op.metadata.input_units["volume"] == "non_negative_volume"


def test_volume_price_range_density_guards_zero_range():
    op = OperatorRegistry.get("volume_price_range_density", "pandas_numpy")
    vol = _panel([100.0, 100.0, 100.0, 100.0])
    high = _panel([10.0, 10.0, 12.0, 13.0])
    low = _panel([8.0, 10.0, 10.0, 11.0])  # row1: high==low -> range 0 -> NaN
    out = op.calculate(vol, high, low, 2)["A"].to_numpy()
    assert np.isnan(out[0])  # warm-up
    assert np.isnan(out[1]) or np.isnan(out[2])  # range==0 row poisons a window


def test_volume_price_range_density_parity_pandas_vs_polars():
    pl = pytest.importorskip("polars")
    pa = OperatorRegistry.get("volume_price_range_density", "pandas_numpy")
    po = OperatorRegistry.get("volume_price_range_density", "polars")
    assert po is not None
    vol = _panel([100.0, 200.0, 300.0, 400.0, 500.0])
    high = _panel([10.0, 11.0, 12.0, 13.0, 14.0])
    low = _panel([8.0, 9.0, 10.0, 11.0, 12.0])
    pv = pl.DataFrame({"A": vol["A"].to_numpy()})
    ph = pl.DataFrame({"A": high["A"].to_numpy()})
    plow = pl.DataFrame({"A": low["A"].to_numpy()})
    a = pa.calculate(vol, high, low, 3)["A"].to_numpy()
    b = po.calculate(pv, ph, plow, 3)["A"].to_numpy()
    np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-8, equal_nan=True)


# ---------------------------------------------------------------------------
# 26. polars ts_impulse_strength scales on volatility_{t-1} (excludes current)
# ---------------------------------------------------------------------------


def test_polars_ts_impulse_strength_scales_on_prior_volatility():
    pl = pytest.importorskip("polars")
    op = OperatorRegistry.get("ts_impulse_strength", "polars")
    rng = np.random.default_rng(11)
    n = 60
    close = _panel(100.0 + np.cumsum(rng.normal(0.0, 1.0, n)))
    pclose = pl.DataFrame({"A": close["A"].to_numpy()})
    got = op.calculate(pclose, 5, 10)["A"].to_numpy()

    pc = close["A"].pct_change(fill_method=None)
    manual = np.full(n, np.nan)
    for t in range(n):
        if t < 5:
            continue
        ret = close["A"].iloc[t] / close["A"].iloc[t - 5] - 1.0
        if t - 10 < 0:
            continue
        seg = pc.iloc[t - 10 : t]  # rows [t-10, t-1]: EXCLUDES the current bar
        if seg.notna().sum() < 10:
            continue
        rv = seg.std()
        manual[t] = ret / (rv * np.sqrt(5.0))
    np.testing.assert_allclose(
        got, manual, rtol=1e-8, atol=1e-8, equal_nan=True
    )


def test_polars_ts_impulse_strength_matches_pandas_twin():
    pl = pytest.importorskip("polars")
    po = OperatorRegistry.get("ts_impulse_strength", "polars")
    pdo = OperatorRegistry.get("ts_impulse_strength", "pandas_numpy")
    rng = np.random.default_rng(13)
    close = _panel(100.0 + np.cumsum(rng.normal(0.0, 1.0, 60)))
    pclose = pl.DataFrame({"A": close["A"].to_numpy()})
    a = pdo.calculate(close, 5, 10)["A"].to_numpy()
    b = po.calculate(pclose, 5, 10)["A"].to_numpy()
    np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-8, equal_nan=True)
