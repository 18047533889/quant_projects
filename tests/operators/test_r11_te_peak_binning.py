# -*- coding: utf-8 -*-
"""R11 transfer-entropy peak binning + surrogate-excess regression tests.

Covers two operator-semantic review fixes in ``advanced_information.py``:

* ISSUE 1 — shared reference binning across lags: ``_te_peak_window`` used to let
  every lag re-derive its OWN quantile edges from a different shifted sample, so
  raw TE values were not comparable across lags.  Now one reference binning is
  built once from the unshifted parent window (``_shared_bins``) and passed into
  every lag's ``_te_from_transitions``.
* ISSUE 2 — max-selection bias: ``peak_strength`` is ``max_l TE_l``, which is
  positively biased even for independent inputs.  The new
  ``ts_transfer_entropy_peak_excess`` subtracts the mean surrogate max
  (deterministic, seeded, shared-binned) from the real max.

Modules are imported directly (their registration is import-triggered) instead of
``load_all()`` so this file stays self-contained while the concurrent R10/R11
session is mid-edit on ``runtime/`` / ``storage/`` / ``polars_dynamics.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factor_engine.cleaned_operators.advanced_information as ai  # noqa: F401
from factor_engine.cleaned_operators.advanced_information import (
    _quantile_edges,
    _shared_bins,
    _te_from_transitions,
    _te_peak_window,
    _te_peak_window_excess,
)
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _frame(v: np.ndarray, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        np.asarray(v, dtype=float).reshape(-1, 1),
        index=pd.date_range("2024-01-01", periods=n, freq="B"),
        columns=["A"],
    )


def _coupled(target_t: np.ndarray, source_t: np.ndarray, lag: int) -> None:
    """In place: ``target_t[i] = source_t[i-lag] + noise`` (driving source)."""
    n = target_t.shape[0]
    rng = np.random.default_rng(12345 + lag)
    eps = rng.normal(size=n)
    for i in range(lag, n):
        target_t[i] = source_t[i - lag] + 0.3 * eps[i]


def _op(name: str):
    op = OperatorRegistry.get(name, "pandas_numpy")
    assert op is not None, name
    return op


# ---------------------------------------------------------------------------
# ISSUE 1 (a) — shared reference binning, identical edges across lags
# ---------------------------------------------------------------------------
def test_shared_bins_are_parent_window_edges():
    rng = np.random.default_rng(0)
    tw = rng.normal(size=80)
    sw = rng.normal(size=80)
    xe, ye = _shared_bins(tw, sw, 3)
    # The reference binning is built ONCE from the unshifted parent window.
    assert np.array_equal(xe, _quantile_edges(tw, 3))
    assert np.array_equal(ye, _quantile_edges(sw, 3))


def test_peak_window_passes_identical_shared_edges_to_every_lag(monkeypatch):
    rng = np.random.default_rng(1)
    tw = rng.normal(size=80)
    sw = rng.normal(size=80)
    recorded: list[tuple[np.ndarray | None, np.ndarray | None]] = []

    original = _te_from_transitions

    def spy(xs, ys, xn, bins, min_cells_ratio=1.0, bins_x=None, bins_y=None):
        recorded.append((bins_x, bins_y))
        return original(
            xs, ys, xn, bins, min_cells_ratio=min_cells_ratio,
            bins_x=bins_x, bins_y=bins_y,
        )

    monkeypatch.setattr(ai, "_te_from_transitions", spy)
    peak, lag_norm = _te_peak_window(tw, sw, 3, 30, 1.0)
    assert np.isfinite(peak) or np.isnan(peak)
    assert len(recorded) >= 2, "the lag grid must evaluate at least two lags"
    # Every lag is digitized against the SAME reference edges (not re-derived).
    first_x, first_y = recorded[0]
    assert first_x is not None and first_y is not None
    for bx, by in recorded[1:]:
        assert bx is first_x or np.array_equal(bx, first_x)
        assert by is first_y or np.array_equal(by, first_y)


def test_shared_bins_differ_from_per_lag_self_derived_edges():
    """The fix is observable: per-lag self-derived edges differ from the shared
    reference, which is exactly why the old ``peak_lag`` mixed dependence with
    discretization changes."""
    rng = np.random.default_rng(2)
    tw = rng.normal(size=60)
    sw = rng.normal(size=60)
    xe, ye = _shared_bins(tw, sw, 3)
    # lag=1 masked transition sample
    x_t = tw[:-1]
    y_t = sw[:-1]
    mask = np.isfinite(x_t) & np.isfinite(y_t) & np.isfinite(tw[1:])
    xs1 = x_t[mask]
    ys1 = y_t[mask]
    per_lag_x = _quantile_edges(xs1, 3)
    per_lag_y = _quantile_edges(ys1, 3)
    # On this sample the per-lag edges genuinely differ from the shared ones
    # (otherwise the old re-discretization would have been harmless).
    assert not np.array_equal(xe, per_lag_x) or not np.array_equal(ye, per_lag_y)


# ---------------------------------------------------------------------------
# ISSUE 1 (b) — peak_lag still identifies a genuine coupling lag after the fix
# ---------------------------------------------------------------------------
def test_peak_lag_identifies_genuine_lag_two_small_window():
    rng = np.random.default_rng(3)
    n = 120
    x = rng.normal(size=n)
    y = np.zeros(n)
    _coupled(y, x, lag=2)  # y[i] = x[i-2] + 0.3 * noise
    tw, sw = y[:70], x[:70]
    _, lag_norm = _te_peak_window(tw, sw, 3, 30, 1.0)
    assert np.isfinite(lag_norm)
    assert round(lag_norm * 10) == 2  # lag 2 wins under shared binning


# ---------------------------------------------------------------------------
# ISSUE 2 (a) — independence: raw peak positive, excess ~ 0
# ---------------------------------------------------------------------------
def test_independence_excess_near_zero_while_raw_peak_positive():
    rng = np.random.default_rng(4)
    raw_peaks: list[float] = []
    excesses: list[float] = []
    for _ in range(20):
        tw = rng.normal(size=60)
        sw = rng.normal(size=60)
        p, _ = _te_peak_window(tw, sw, 3, 30, 1.0)
        e, _ = _te_peak_window_excess(tw, sw, 3, 30, 1.0, n_surrogates=20, seed=0)
        if np.isfinite(p) and np.isfinite(e):
            raw_peaks.append(p)
            excesses.append(e)
    assert len(raw_peaks) >= 15
    # raw max-over-lags is positively biased even under independence ...
    assert float(np.mean(raw_peaks)) > 0.01
    # ... but the surrogate-standardized excess is ~ 0.
    assert abs(float(np.mean(excesses))) < 0.1
    assert float(np.max(np.abs(np.asarray(excesses)))) < 0.1


# ---------------------------------------------------------------------------
# ISSUE 2 (b) — genuine coupling: excess > 0 and peak lag is 2
# ---------------------------------------------------------------------------
def test_dependence_excess_positive_and_peak_lag_two():
    rng = np.random.default_rng(5)
    n = 200
    x = rng.normal(size=n)
    y = np.zeros(n)
    _coupled(y, x, lag=2)
    tw, sw = y[:60], x[:60]
    _, lag_norm = _te_peak_window(tw, sw, 3, 30, 1.0)
    excess, elag_norm = _te_peak_window_excess(tw, sw, 3, 30, 1.0, n_surrogates=20, seed=0)
    assert np.isfinite(excess)
    assert round(lag_norm * 10) == 2
    assert round(elag_norm * 10) == 2
    assert excess > 0.05  # clearly above the independent null


def test_excess_operator_positive_under_coupling():
    rng = np.random.default_rng(6)
    n = 200
    x = rng.normal(size=n)
    y = np.zeros(n)
    _coupled(y, x, lag=2)
    out = _op("ts_transfer_entropy_peak_excess").calculate(
        _frame(y, n), _frame(x, n), window=60, bins=3, n_surrogates=20, seed=0
    ).to_numpy(dtype=float)
    vals = out[np.isfinite(out)]
    assert vals.size > 0
    assert float(np.mean(vals)) > 0.05


# ---------------------------------------------------------------------------
# ISSUE 2 (c) — determinism: same inputs + same seed -> identical output
# ---------------------------------------------------------------------------
def test_excess_operator_deterministic_same_seed():
    rng = np.random.default_rng(7)
    n = 120
    tw = _frame(rng.normal(size=n), n)
    sw = _frame(rng.normal(size=n), n)
    op = _op("ts_transfer_entropy_peak_excess")
    a = op.calculate(tw, sw, window=60, bins=3, n_surrogates=20, seed=0).to_numpy(dtype=float)
    b = op.calculate(tw, sw, window=60, bins=3, n_surrogates=20, seed=0).to_numpy(dtype=float)
    assert np.array_equal(a, b, equal_nan=True)


def test_excess_operator_declares_non_searchable_surrogate_params():
    op = _op("ts_transfer_entropy_peak_excess")
    specs = op.metadata.param_specs or {}
    ns = specs.get("n_surrogates")
    sd = specs.get("seed")
    assert ns is not None and ns.searchable is False
    assert sd is not None and sd.searchable is False
    # The reference (strength) operators are untouched.
    assert _op("ts_transfer_entropy_peak_strength").metadata.param_specs.get("n_surrogates") is None
