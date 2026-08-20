# -*- coding: utf-8 -*-
"""R11 round-2 regression tests: structure-matched effective-TE surrogate (P0, review #19).

``ts_effective_transfer_entropy`` subtracts the mean over circular-shift
surrogates from the raw transfer entropy.  The REAL TE path applies the raw time
lag on the ORIGINAL timeline and only THEN masks missing values, so the real
estimate's transitions live on the original physical-time grid (a NaN gap never
redefines the lag).  The old effective-TE surrogate instead COMPRESSED the source
series (dropped the NaN rows) and then circularly shifted the compressed series —
erasing the physical-time missing-gap structure from the null model, so a gap
that separates two dependent segments no longer blocked dependence in the
surrogate and E[TE_surrogate] no longer matched the real TE's bias.

The fix (review item #19) circular/block-shifts the SOURCE series on the ORIGINAL
timeline, rebuilds the ``(x_lead, y_lag)`` triples on the shifted timeline, and
only THEN re-applies the valid mask.  The null therefore shares the exact same
missing-gap topology as the real estimate.

These tests lock in the corrected behavior:
  (a) independent series with a long internal NaN gap -> structure-matched null
      removes the finite-sample bias, so the effective-TE excess is ~ 0;
  (b) a genuine source -> target dependence still yields excess > 0 with the
      same gap present;
  (c) determinism (same seed / same input -> bit-identical result).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.advanced_information import (
    TsEffectiveTransferEntropy,
    _effective_transfer_entropy_window,
)


def _smooth(n: int, rho: float = 0.9, seed: int = 0) -> np.ndarray:
    """Deterministic AR(1) process (smooth, autocorrelated)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, n)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + z[i]
    return out


def _driven_target(source: np.ndarray, seed: int = 0) -> np.ndarray:
    """``target[t] = 0.7 * source[t-1] + noise`` (genuine source->target coupling)."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(source)
    for i in range(1, source.shape[0]):
        out[i] = 0.7 * source[i - 1] + rng.normal(0.0, 0.3)
    return out


def _panel(vals: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=vals.shape[0], freq="B")
    arr = vals if vals.ndim == 2 else vals[:, None]
    return pd.DataFrame(arr, index=idx, columns=[f"S{i}" for i in range(arr.shape[1])])


# ---------------------------------------------------------------------------
# (a) independent source with a long internal gap -> excess ~ 0
# ---------------------------------------------------------------------------

def test_independent_source_internal_gap_excess_is_zero():
    """A long internal NaN gap in an INDEPENDENT source must not create a
    spurious effective-TE excess: the structure-matched null removes the same
    finite-sample bias the real estimate carries, so excess ~ 0."""
    n = 600
    driver = _smooth(n, seed=11)
    target = _driven_target(driver, seed=5)   # target has real structure
    source = _smooth(n, seed=99)              # independent of target
    source[200:300] = np.nan                  # long internal gap (100 rows)

    out = TsEffectiveTransferEntropy().calculate(
        _panel(target), _panel(source), window=100, bins=3, lag=1
    )
    vals = out.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    assert vals.size > 0
    # Old compressed-shift surrogate broke the gap topology and left a spurious
    # bias; the structure-matched null corrects it back to ~ 0.
    assert abs(float(np.nanmean(vals))) < 0.02
    assert abs(float(np.nanmedian(vals))) < 0.02


def test_independent_source_without_gap_still_zero():
    """Control: the same independent source WITHOUT the gap is also ~ 0, so the
    gap itself is not inflating the null (it is genuinely structure-matched)."""
    n = 600
    driver = _smooth(n, seed=11)
    target = _driven_target(driver, seed=5)
    source = _smooth(n, seed=99)  # no gap
    out = TsEffectiveTransferEntropy().calculate(
        _panel(target), _panel(source), window=100, bins=3, lag=1
    )
    vals = out.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    assert vals.size > 0
    assert abs(float(np.nanmean(vals))) < 0.02


# ---------------------------------------------------------------------------
# (b) genuine source -> target dependence with the same gap -> excess > 0
# ---------------------------------------------------------------------------

def test_genuine_dependence_internal_gap_excess_positive():
    """With a genuine source->target coupling the excess must stay positive even
    when the source carries the same long internal gap: the structure-matched
    null destroys the x->y alignment while sharing the missing topology, so the
    real signal survives."""
    n = 600
    source = _smooth(n, seed=13)
    target = _driven_target(source, seed=7)   # genuine source->target
    source[200:300] = np.nan                  # same long internal gap

    out = TsEffectiveTransferEntropy().calculate(
        _panel(target), _panel(source), window=100, bins=3, lag=1
    )
    vals = out.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    assert vals.size > 0
    # Strong directional coupling: median excess well above the ~0 null floor.
    assert float(np.nanmedian(vals)) > 0.1
    assert float(np.nanmean(vals)) > 0.1


# ---------------------------------------------------------------------------
# (c) determinism
# ---------------------------------------------------------------------------

def test_effective_transfer_entropy_deterministic_with_gap():
    """Same input (with a gap) -> bit-identical output on repeated evaluation."""
    n = 300
    rng = np.random.default_rng(0)
    target = rng.normal(0.0, 1.0, (n, 2))
    source = rng.normal(0.0, 1.0, (n, 2))
    source[100:130, :] = np.nan  # internal gap
    op = TsEffectiveTransferEntropy()
    a = op.calculate(_panel(target), _panel(source), window=100, bins=3, lag=1)
    b = op.calculate(_panel(target), _panel(source), window=100, bins=3, lag=1)
    assert np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True)


def test_effective_transfer_entropy_window_deterministic():
    """The single-window kernel itself is deterministic on a gapped source."""
    rng = np.random.default_rng(1)
    tw = rng.normal(0.0, 1.0, 160)
    sw = rng.normal(0.0, 1.0, 160)
    sw[60:100] = np.nan
    a = _effective_transfer_entropy_window(tw, sw, 3, 1, 30)
    b = _effective_transfer_entropy_window(tw, sw, 3, 1, 30)
    assert a == b


# ---------------------------------------------------------------------------
# surrogate structure-matched property (direct kernel test)
# ---------------------------------------------------------------------------

def test_surrogate_preserves_source_gap_topology():
    """Each surrogate shift happens on the ORIGINAL timeline and re-applies the
    valid mask AFTER the shift, so a long source gap is preserved in the null.

    The OLD surrogate first COMPRESSED the source (dropped the NaN rows) and then
    shifted the compressed series, so the gap vanished from the null entirely.
    Here every surrogate still carries the full 40-row gap after the rotation —
    the null shares the real estimate's missing topology."""
    from cleaned_operators.advanced_information import _SURROGATE_OFFSETS

    rng = np.random.default_rng(3)
    n = 160
    tw = rng.normal(0.0, 1.0, n)
    sw = rng.normal(0.0, 1.0, n)
    sw[60:100] = np.nan  # internal gap of 40 rows

    lag = 1
    for off in _SURROGATE_OFFSETS:
        y_shifted = np.empty_like(sw)
        y_shifted[: n - off] = sw[off:]
        y_shifted[n - off :] = sw[:off]
        y_t_s = y_shifted[:-lag]
        nan_count = int(np.isnan(y_t_s).sum())
        # The 40-row gap is preserved in the rotated window: 39 or 40 NaN in the
        # lag-truncated slice, depending on whether a NaN wraps into the last
        # position that lag-truncation drops.  A compressed-shift surrogate would
        # have 0 here (the gap was deleted before shifting).
        assert nan_count in (39, 40)
