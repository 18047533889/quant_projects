# -*- coding: utf-8 -*-
"""Independent oracle test for the KAMA Efficiency Ratio (ER) fix.

The module's ``kama()`` computes the Kaufman Efficiency Ratio with the
standard formula:

    direction  = |x[t] - x[t - period_er]|          (net change over lookback)
    volatility = sum_{i=t-period_er+1}^{t} |x[i] - x[i-1]|   (total path)
    ER = direction / volatility

This file re-implements KAMA with a slow, obviously-correct textbook loop on
the *same lagged inputs* (``shift(1)``, matching the module's causal
contract) and asserts an exact match. It covers:

  * a random-walk price series,
  * a constant-price series (flat -> ER -> 0 -> KAMA holds its level),
  * a trending series.

The oracle deliberately mirrors the module's warmup/seed semantics so the
comparison isolates the ER computation.
"""
import numpy as np
import pandas as pd

from factor_preprocess.transforms.smoothing import kama


def _df(values, asset="A"):
    n = len(values)
    return pd.DataFrame(
        {
            "asset_id": [asset] * n,
            "date": list(range(n)),
            "value": values,
        }
    )


def _oracle_kama(values, period_fast, period_slow, period_er):
    """Slow, textbook KAMA on the lagged series (shift(1)), causal contract."""
    fast_sc = 2.0 / (period_fast + 1.0)
    slow_sc = 2.0 / (period_slow + 1.0)

    lagged = np.full(len(values), np.nan)
    lagged[1:] = values[:-1]  # shift(1): output at t uses only <= t-1

    n = len(lagged)
    out = np.full(n, np.nan)
    kama_val = np.nan

    for i in range(n):
        if i < period_er:
            out[i] = np.nan
            continue

        # --- Standard Kaufman Efficiency Ratio over [t-period_er, t] ---
        # direction = |x[t] - x[t-period_er]|
        a = lagged[i - period_er]
        b = lagged[i]
        if np.isnan(a) or np.isnan(b):
            er = 0.0  # undefined -> treat as flat (hold level)
        else:
            direction = abs(b - a)
            # volatility = sum_{i=t-period_er+1}^{t} |x[i] - x[i-1]|
            volatility = 0.0
            for j in range(i - period_er + 1, i + 1):
                volatility += abs(lagged[j] - lagged[j - 1])
            er = direction / volatility if volatility > 0 else 0.0

        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        # --- Recursive KAMA update (same seed semantics as the module) ---
        if not np.isfinite(kama_val):
            if not np.isnan(lagged[i]):
                kama_val = lagged[i]
            out[i] = np.nan
            continue
        kama_val = kama_val + sc * (lagged[i] - kama_val)
        out[i] = kama_val
    return out


def _assert_match(series, period_fast=2, period_slow=30, period_er=10):
    df = _df(series)
    actual = kama(df, period_fast, period_slow, period_er).to_numpy()
    expected = _oracle_kama(series, period_fast, period_slow, period_er)
    assert len(actual) == len(expected)
    mask = ~np.isnan(expected)
    assert mask.sum() > 0, "oracle produced no finite outputs"
    np.testing.assert_allclose(
        actual[mask], expected[mask], rtol=1e-9, atol=1e-9,
        err_msg="kama() deviates from the textbook Kaufman oracle",
    )


def test_kama_oracle_random_walk():
    rng = np.random.RandomState(42)
    steps = rng.normal(0, 1.0, 200)
    price = 100.0 + np.cumsum(steps)
    _assert_match(price)


def test_kama_oracle_constant_price_holds_level():
    # Flat price: ER -> 0 -> slowest sc -> KAMA holds its seeded level.
    price = np.full(80, 50.0)
    _assert_match(price)


def test_kama_oracle_trending_series():
    t = np.arange(150)
    price = 10.0 + 0.5 * t
    _assert_match(price)


def test_kama_oracle_use_current_skips_shift():
    # use_current=True must equal the oracle computed on the unshifted series.
    rng = np.random.RandomState(7)
    price = 100.0 + np.cumsum(rng.normal(0, 1.0, 120))
    df = _df(price)
    actual = kama(df, use_current=True).to_numpy()
    expected = _oracle_kama(price, 2, 30, 10)
    # Oracle uses shift(1); for use_current we compare against a variant that
    # does NOT shift. Recompute oracle on the unshifted series.
    lagged = np.asarray(price, dtype=float)
    n = len(lagged)
    out = np.full(n, np.nan)
    fast_sc = 2.0 / 3.0
    slow_sc = 2.0 / 31.0
    kama_val = np.nan
    for i in range(n):
        if i < 10:
            out[i] = np.nan
            continue
        a = lagged[i - 10]
        b = lagged[i]
        if np.isnan(a) or np.isnan(b):
            er = 0.0
        else:
            direction = abs(b - a)
            volatility = sum(abs(lagged[j] - lagged[j - 1])
                             for j in range(i - 9, i + 1))
            er = direction / volatility if volatility > 0 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        if not np.isfinite(kama_val):
            if not np.isnan(lagged[i]):
                kama_val = lagged[i]
            out[i] = np.nan
            continue
        kama_val = kama_val + sc * (lagged[i] - kama_val)
        out[i] = kama_val
    mask = ~np.isnan(out)
    np.testing.assert_allclose(
        actual[mask], out[mask], rtol=1e-9, atol=1e-9,
        err_msg="use_current=True does not match unshifted textbook KAMA",
    )
