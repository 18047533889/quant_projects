# -*- coding: utf-8 -*-
"""review #5 R5-09/10/11: full-history == checkpoint-incremental == pandas.

The checkpoint segment kernels must be the *same math* as the pandas
full-history reference (``ewm(alpha, adjust=False, ignore_na=False)`` with
min_periods gating and output carry-over at missing bars), not a parallel
hand-rolled recurrence with different seeds.

* ``test_kernel_matches_pandas_reference`` — each ``recursive_kernel`` segment
  run over a whole series (with NaN gaps / consecutive NaN / leading NaN /
  Inf) reproduces the production pandas implementation bit-for-bit.
* ``test_stitch_at_every_split_matches_full`` — for every split point in
  ``[1, N)`` the checkpoint-resume stitching of the two halves equals the
  full-series output, so checkpoint-incremental and full-history agree on a
  segment grid that never existed in the full run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recursive_kernel import (
    adx_segment,
    atr_wilder_segment,
    ema_segment,
    macd_segment,
    rsi_wilder_segment,
)

from cleaned_operators.technical.signal import (
    _compute_atr_wilder,
    _compute_dmi_adx,
    _compute_rsi_wilder,
)


@pytest.fixture(scope="module")
def noisy_close():
    rng = np.random.default_rng(20260808)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 260))
    # NaN gaps: leading, single, consecutive run, Inf (poisons one more bar of
    # diff/TR), and a trailing run.
    close[0] = np.nan
    close[40] = np.nan
    close[80:84] = np.nan
    close[160] = np.inf
    close[220:226] = np.nan
    return close


def _frames(close):
    return pd.DataFrame({"c": close})


# --------------------------------------------------------------------------
# full-series parity with the pandas production reference
# --------------------------------------------------------------------------
def test_ema_matches_pandas(noisy_close):
    ref = _frames(noisy_close)["c"].ewm(span=10, adjust=False).mean().to_numpy()
    values, _ = ema_segment(noisy_close, {}, 10)
    np.testing.assert_allclose(values, ref, equal_nan=True, rtol=1e-12, atol=1e-12)


def test_macd_family_matches_pandas(noisy_close):
    close = _frames(noisy_close)["c"]
    line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = line.ewm(span=9, adjust=False).mean()
    for output, ref in (
        ("line", line),
        ("signal", signal),
        ("hist", line - signal),
    ):
        values, _ = macd_segment(noisy_close, {}, 12, 26, 9, output)
        np.testing.assert_allclose(
            values, ref.to_numpy(), equal_nan=True, rtol=1e-12, atol=1e-12,
            err_msg=f"MACD_{output}",
        )


def test_rsi_wilder_matches_pandas(noisy_close):
    ref = _compute_rsi_wilder(_frames(noisy_close), 14)["c"].to_numpy()
    values, _ = rsi_wilder_segment(noisy_close, {}, 14)
    np.testing.assert_allclose(values, ref, equal_nan=True, rtol=1e-12, atol=1e-12)


def test_atr_wilder_matches_pandas(noisy_close):
    high = noisy_close + 1.2
    low = noisy_close - 1.2
    ref = _compute_atr_wilder(_frames(high), _frames(low), _frames(noisy_close), 14)["c"].to_numpy()
    values, _ = atr_wilder_segment(high, low, noisy_close, {}, 14)
    np.testing.assert_allclose(values, ref, equal_nan=True, rtol=1e-12, atol=1e-12)


def test_adx_matches_pandas(noisy_close):
    high = noisy_close + 1.2
    low = noisy_close - 1.2
    ref = _compute_dmi_adx(pd.Series(high), pd.Series(low), pd.Series(noisy_close), 14).to_numpy()
    values, _ = adx_segment(high, low, noisy_close, {}, 14)
    np.testing.assert_allclose(values, ref, equal_nan=True, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------
# checkpoint stitching == full history at every split point
# --------------------------------------------------------------------------
def _stitch_ok(kernel, args, kwargs, n):
    full, _ = kernel(*args, {}, **kwargs)
    for split in range(1, n):
        first, state = kernel(*[a[:split] for a in args], {}, **kwargs)
        second, _ = kernel(*[a[split:] for a in args], state, **kwargs)
        stitched = np.concatenate([first, second])
        if not np.allclose(stitched, full, equal_nan=True, rtol=1e-12, atol=1e-12):
            return False
    return True


@pytest.mark.parametrize(
    "name, kernel, params",
    [
        ("ts_ema", ema_segment, {"span": 10}),
        ("RSI_WILDER", rsi_wilder_segment, {"window": 14}),
        ("MACD_line", lambda x, s, **k: macd_segment(x, s, 12, 26, 9, "line"), {}),
        ("MACD_signal", lambda x, s, **k: macd_segment(x, s, 12, 26, 9, "signal"), {}),
        ("MACD_hist", lambda x, s, **k: macd_segment(x, s, 12, 26, 9, "hist"), {}),
    ],
)
def test_stitch_every_split_single_input(name, kernel, params, noisy_close):
    assert _stitch_ok(kernel, (noisy_close,), params, len(noisy_close)), name


def test_stitch_every_split_multi_input(noisy_close):
    high = noisy_close + 1.2
    low = noisy_close - 1.2
    assert _stitch_ok(atr_wilder_segment, (high, low, noisy_close), {"window": 14}, len(noisy_close))
    assert _stitch_ok(adx_segment, (high, low, noisy_close), {"window": 14}, len(noisy_close))
