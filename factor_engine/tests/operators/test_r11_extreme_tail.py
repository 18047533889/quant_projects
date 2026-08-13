# -*- coding: utf-8 -*-
"""R11 fixes for cleaned_operators/extreme_tail.py (operator-semantic review).

Covers the four R11 findings:

* ISSUE 1 — ``_hill_series`` is now a TRUE Hill estimator with a SINGLE tail
  selection: one threshold at ``tail_fraction`` and ALL exceedances beyond it
  (the pre-R11 code POT-thresholded at ``frac`` and then applied
  ``k = floor(mags.size * frac)`` a SECOND time, which with
  window=120/frac=0.2/min_tail_count=10 produced ``k=4 < 10`` and long NaN
  stretches under default parameters).
  * Pareto golden test: a synthetic Pareto with known tail index ξ is recovered
    within a loose tolerance.
  * Default-parameter finite-rate test: 400 rows at window=120 keep the output
    finite far above 0 (the double-tail NaN bug is gone).
  * Upper/lower mirror test: negating a series mirrors its tails (with the R11
    ``u > 0`` threshold guard, the raw ``lower`` side is NaN for a centred
    series, so the mirror is asserted through the negated series).
* ISSUE 2 — ``_mean_excess_slope_series`` computes the lower tail on the
  mirrored series ``y = -x`` and reuses the exact upper-tail computation, so
  ``ME_lower(x) == ME_upper(-x)`` and the slope sign is interpreted identically
  on both sides (heavy tail -> positive slope on both sides).
* ISSUE 3 — ``_extremal_index_series`` NaN gaps BREAK clusters and an explicit
  ``run_length`` parameter controls whether a gap of non-exceedances keeps one
  cluster (default 1).
* ISSUE 4 — the numeric price-level heuristic is demoted to a data-quality
  WARNING (never a hard ValueError); the estimator still runs.  The typed
  FieldSpec ``input_units`` declaration is the authoritative contract.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from cleaned_operators.extreme_tail import (
    _extremal_index_series,
    _hill_series,
    _mean_excess_slope_series,
    TsExtremalIndex,
    TsHillTailIndex,
)


@pytest.fixture(autouse=True)
def _clear_price_warning_dedup():
    """ISSUE 4: reset the module-level price-level warning dedup flag so each
    test observes the warning independently (it persists across calls/cases)."""
    import cleaned_operators.extreme_tail as _et

    _et._WARNED_PRICE_LEVEL.clear()
    yield


def _col(data: list[float]) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(data, dtype=float).reshape(-1, 1), columns=["c"])


# ---------------------------------------------------------------------------
# ISSUE 1 — TRUE single-selection Hill estimator
# ---------------------------------------------------------------------------

def test_hill_pareto_golden_recovers_true_tail_index():
    """A synthetic Pareto with a known tail index ξ is recovered by the Hill
    estimator within a loose tolerance (single tail selection, no double frac).

    ``X = U ** (-ξ)`` with ``U ~ U(0,1)`` has survival ``P(X > x) = x ** (-1/ξ)``
    (a Pareto(scale=1)), i.e. GPD tail index exactly ``ξ``.
    """
    rng = np.random.default_rng(11)
    for xi in (0.3, 0.5):
        u = rng.uniform(0.0, 1.0, size=4000)
        pareto = u ** (-xi)  # tail index = xi
        out = _hill_series(
            pareto, window=pareto.size, side="upper", tail_fraction=0.2, min_tail_count=10
        )
        est = float(out[-1])
        assert np.isfinite(est)
        assert abs(est - xi) < 0.25


def test_hill_default_parameters_finite_rate():
    """window=120 / tail_fraction=0.2 / min_tail_count=10 must NOT produce the
    long NaN stretches of the pre-R11 double-tail-selection bug: with a single
    selection the exceedance count is ~0.2*window = 24 >= 10, so the trailing
    output stays finite well above 0."""
    rng = np.random.default_rng(3)
    x = np.abs(rng.standard_t(df=3, size=400)) + 0.01
    out = _hill_series(x, window=120, side="upper", tail_fraction=0.2, min_tail_count=10)
    trailing = out[200:]
    finite_ratio = float(np.isfinite(trailing).mean())
    assert finite_ratio > 0.3


def test_hill_upper_lower_negation_mirror():
    """Negating a centred heavy-tailed series mirrors its tails: the upper tail
    of x must be close to the upper tail of -x (== the lower tail of x).  The
    lower tail is ONE kernel on the mirrored series ``z = -x``, so
    ``HillLower(x) == HillUpper(-x)`` directly (no separate lower code path, no
    ``u<=0`` guard that NaN's the lower side of a centred signed series)."""
    rng = np.random.default_rng(55)
    x = rng.standard_t(df=4, size=4000)
    up_x = _hill_series(x, window=x.size, side="upper", tail_fraction=0.2, min_tail_count=10)[-1]
    up_nx = _hill_series(-x, window=x.size, side="upper", tail_fraction=0.2, min_tail_count=10)[-1]
    lo_x = _hill_series(x, window=x.size, side="lower", tail_fraction=0.2, min_tail_count=10)[-1]
    assert np.isfinite(up_x)
    assert np.isfinite(up_nx)
    assert abs(float(up_x) - float(up_nx) < 0.1
    # The direct invariant the Master-Audit item requires: HillLower(x) equals
    # HillUpper(-x) exactly (finite for a centred signed series — the raw
    # side='lower' call is no longer NaN'd by a negative quantile threshold).
    assert np.isfinite(lo_x)
    assert abs(float(lo_x) - float(up_nx) < 1e-9
    # R26-060..062: a strictly-POSITIVE magnitude / level has a left tail BOUNDED
    # below by 0 — classic Hill (the ``z = -x`` mirror) does not apply to it and
    # would produce a misleading negative ξ.  The positive-level lower tail is
    # explicitly unsupported -> NaN; the UPPER tail stays finite.
    mag = np.abs(rng.standard_t(df=3, size=4000)) + 0.05
    lo_mag = _hill_series(mag, window=mag.size, side="lower", tail_fraction=0.2, min_tail_count=10)[-1]
    up_mag = _hill_series(mag, window=mag.size, side="upper", tail_fraction=0.2, min_tail_count=10)[-1]
    assert np.isnan(lo_mag), "strictly-positive level lower tail must be unsupported (R26-061)"
    assert np.isfinite(up_mag)


# ---------------------------------------------------------------------------
# ISSUE 2 — mean-excess lower tail is the exact mirror
# ---------------------------------------------------------------------------

def test_mean_excess_lower_mirror_equals_upper_on_negated():
    """ME_lower(x) == ME_upper(-x) exactly: the lower tail reuses the upper-tail
    computation on the mirrored series ``y = -x`` (no sign-reversal bug)."""
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 1.0, 600) + 0.5 * rng.standard_t(df=3, size=600)
    lo = _mean_excess_slope_series(x, window=600, side="lower", min_tail_count=20)[-1]
    up_neg = _mean_excess_slope_series(-x, window=600, side="upper", min_tail_count=20)[-1]
    assert np.isfinite(lo)
    assert np.isfinite(up_neg)
    assert abs(float(lo) - float(up_neg) < 1e-9


def test_mean_excess_slope_positive_on_heavy_tail_both_sides():
    """Heavy tail -> positive mean-excess slope on BOTH sides.  A Student-t
    series shifted so every value is positive yet the left tail stays unbounded
    gives a heavy lower tail too, so both slopes must be positive (the pre-R11
    lower-tail sign bug regressed on the x-space threshold and reversed the
    slope)."""
    rng = np.random.default_rng(1)
    x = rng.standard_t(df=2.5, size=800) + 5.0
    up = _mean_excess_slope_series(x, window=800, side="upper", min_tail_count=20)[-1]
    lo = _mean_excess_slope_series(x, window=800, side="lower", min_tail_count=20)[-1]
    assert np.isfinite(up)
    assert np.isfinite(lo)
    assert float(up) > 0.0
    assert float(lo) > 0.0


# ---------------------------------------------------------------------------
# ISSUE 3 — extremal index cluster semantics (NaN breaks, run_length)
# ---------------------------------------------------------------------------

def test_extremal_index_nan_gap_breaks_cluster():
    """A NaN gap separates two exceedance blocks -> MORE clusters than without
    the gap (θ with the NaN gap is larger).  Pre-R11 carried ``prev`` across NaN
    and miscounted ``1,1,NaN,1,1`` as ONE cluster."""
    no_gap = np.array([0.0, 0, 1, 1, 1, 1, 0, 0])
    gap = np.array([0.0, 0, 1, 1, np.nan, 1, 1, 0, 0])
    th_no = _extremal_index_series(no_gap, window=30, side="upper", q=0.5, min_exceed=2)[-1]
    th_gap = _extremal_index_series(gap, window=30, side="upper", q=0.5, min_exceed=2)[-1]
    assert th_no == pytest.approx(0.25)  # one cluster of 4
    assert th_gap == pytest.approx(0.5)  # NaN splits into two clusters of 2
    assert th_gap > th_no


def test_extremal_index_run_length_effect():
    """run_length=2 with a single-bar gap between exceedances keeps ONE cluster,
    while run_length=1 gives two."""
    x = np.array([0.0, 0, 1, 1, 0, 1, 1, 0, 0])
    rl1 = _extremal_index_series(x, window=30, side="upper", q=0.5, min_exceed=2, run_length=1)[-1]
    rl2 = _extremal_index_series(x, window=30, side="upper", q=0.5, min_exceed=2, run_length=2)[-1]
    assert rl1 == pytest.approx(0.5)  # the single zero-gap breaks the run
    assert rl2 == pytest.approx(0.25)  # run_length=2 bridges the single-bar gap
    assert rl2 < rl1


def test_extremal_index_nan_gap_via_operator():
    """The operator path forwards run_length and honours the NaN-breaks rule."""
    op = TsExtremalIndex()
    df = _col([0.0, 0, 1, 1, np.nan, 1, 1, 0, 0])
    out = op._calculate_series(df, window=30, side="upper", q=0.5, min_exceed=2)
    assert out.iloc[-1, 0] == pytest.approx(0.5)


def test_extremal_index_run_length_validation():
    with pytest.raises(ValueError, match="run_length"):
        _extremal_index_series(
            np.array([0.0, 1.0, 0.0, 1.0, 0.0]), window=10, side="upper", q=0.5, min_exceed=2, run_length=0
        )


# ---------------------------------------------------------------------------
# ISSUE 4 — price-level heuristic demoted to a DQ warning (never a raise)
# ---------------------------------------------------------------------------

def test_price_level_warns_but_estimator_runs():
    """A raw trending price level (positive, monotonically increasing) no longer
    raises ValueError; a data-quality warning is emitted; the estimator still
    returns values."""
    price = np.linspace(100.0, 300.0, 250)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = _hill_series(price, window=200, side="upper", tail_fraction=0.2, min_tail_count=10)
    assert any("price level" in str(w.message) for w in rec)
    assert np.isfinite(out).any()


def test_price_level_operator_does_not_raise():
    """End-to-end through TsHillTailIndex: no exception for a raw price level,
    a warning is emitted, and the output column contains finite values."""
    op = TsHillTailIndex()
    rng = np.random.default_rng(4)  # distinct from the seed-0 rw in the shared tree
    rw = np.cumsum(rng.normal(0.0, 1.0, 300)) + 100.0
    df = _col(rw.tolist())
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = op._calculate_series(df, window=150, side="upper", tail_fraction=0.2, min_tail_count=10)
    assert any("price level" in str(w.message) for w in rec)
    assert np.isfinite(out.to_numpy(dtype=float)).any()


def test_signed_returns_do_not_warn():
    """A signed return series (mixed signs) is not flagged by the price-level
    heuristic — no warning is emitted."""
    rng = np.random.default_rng(2)
    ret = rng.normal(0.0, 0.02, 500)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _hill_series(ret, window=300, side="upper", tail_fraction=0.2, min_tail_count=10)
    assert not any("price level" in str(w.message) for w in rec)
