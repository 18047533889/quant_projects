# -*- coding: utf-8 -*-
"""R11 round-2 Directional-Change contract + initial-extrema audits.

Covers the round-11 P0 batch for the directional-change family
(``ts_dc_overshoot_ratio`` / ``ts_dc_event_rate`` /
``ts_dc_duration_asymmetry`` / ``ts_dc_overshoot_asymmetry``):

* TASK 1 (P0)  — PositivePrice contract: the DC family measures
  ``p/origin - 1`` and ``origin*(1±theta)``, so the input panel is declared
  ``input_units={"x": "price"}`` and the operator FAILS CLOSED on any
  non-positive price (``<= 0``) instead of silently computing garbage from a
  signed numeric field (returns / spreads).
* TASK 2       — initial-extrema seeding: before the first confirmation the
  undecided state tracks the pre-confirmation running HIGH and LOW separately,
  so the first event fires from the pre-confirmation running extremum (the
  max/min accumulated before the first confirmation) — never from the
  arbitrary first bar.
* TASK 3       — degenerate-input audit: ``threshold <= 0`` is rejected, and
  insufficient price history fails closed to NaN (no phantom confirmation).

Also re-verifies the existing R11 guarantees still hold on positive prices:
a missing DC scale still BREAKS the episode (never straddles a gap), a NaN
price bar still censors the clock, ``fixed_absolute`` stays prefix-invariant,
and the polars backend remains at exact parity with the pandas reference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all

load_all()

from factor_engine.cleaned_operators.directional_change import _dc_column
from factor_engine.cleaned_operators.registry import OperatorRegistry

_DC_NAMES = (
    "ts_dc_overshoot_ratio",
    "ts_dc_event_rate",
    "ts_dc_duration_asymmetry",
    "ts_dc_overshoot_asymmetry",
)


def _get(name: str):
    return OperatorRegistry.get(name, "pandas_numpy") or OperatorRegistry.get(name)


def _col(data: list[float]) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(data, dtype=float).reshape(-1, 1), columns=["c"])


def _ones(n: int) -> pd.DataFrame:
    return pd.DataFrame(np.ones((n, 1), dtype=float), columns=["c"])


def _frame(values: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=values.shape[0], freq="B")
    cols = [f"S{i}" for i in range(values.shape[1])]
    return pd.DataFrame(values, index=idx, columns=cols)


# ---------------------------------------------------------------------------
# TASK 1 (P0) — PositivePrice typed contract
# ---------------------------------------------------------------------------
def test_metadata_declares_price_input_contract():
    for name in _DC_NAMES:
        op = _get(name)
        assert op.metadata.input_units.get("x") == "price", name
        assert "price" in op.metadata.compatible_units.get("x", ()), name


@pytest.mark.parametrize("name", _DC_NAMES)
def test_non_positive_price_raises_typed_contract_violation(name):
    op = _get(name)
    # zero price
    with pytest.raises(ValueError, match="positive price"):
        op.calculate(_col([0.0, 2.0, 1.0, 3.0]), _ones(4), threshold=1.0, window=10)
    # negative price
    with pytest.raises(ValueError, match="positive price"):
        op.calculate(_col([100.0, 99.0, -1.0, 101.0]), _ones(4), threshold=1.0, window=10)
    # NaN is fine (it censors the clock) but a non-positive FINITE value raises
    with pytest.raises(ValueError, match="positive price"):
        op.calculate(_col([100.0, np.nan, 0.0, 101.0]), _ones(4), threshold=1.0, window=10)


def test_dc_kernel_raises_on_non_positive_price():
    # The raw kernel fails closed too (the runtime gate is the kernel's own).
    with pytest.raises(ValueError, match="positive price"):
        _dc_column(np.asarray([100.0, 0.0], dtype=float), np.ones(2), 1.0, 10, "adaptive")
    with pytest.raises(ValueError, match="positive price"):
        _dc_column(np.asarray([100.0, 99.0, -5.0], dtype=float), np.ones(3), 1.0, 10, "adaptive")


def test_all_positive_prices_never_raise():
    op = _get("ts_dc_event_rate")
    out = op.calculate(_col([100.0, 102.0, 101.0, 104.0, 103.0]), _ones(5), threshold=1.0, window=10)
    assert out.shape == (5, 1)
    assert np.isfinite(out.iloc[-1, 0])


# ---------------------------------------------------------------------------
# TASK 2 — initial-extrema seeding (pre-confirmation running high/low)
# ---------------------------------------------------------------------------
def test_first_down_event_uses_pre_confirmation_running_high_not_first_bar():
    # Path: rises to 101 at bar 1, then falls.  The pre-confirmation running
    # HIGH is 101 (bar 1) — NOT the first bar's 100.  The first event is a DOWN
    # that must fire at bar 3 when price reaches 101 - 2 = 99.  If the first
    # extreme were pinned to the first bar (100), the down event would fire one
    # bar earlier at bar 2 (100 - 2 = 98) — or, with the old shared-extreme
    # bug, never at all (the running low drags the extreme and the peak is lost).
    x = np.asarray([100.0, 101.0, 100.0, 99.0, 98.0, 99.0, 100.0, 101.0, 102.0])
    out = _dc_column(x, np.ones_like(x), 2.0, 10, "adaptive")
    evr = out["event_rate"]
    # No event yet at bars 0..2: the down event needs the running high (101) to
    # be exceeded by the threshold, i.e. price 99 at bar 3.
    assert np.isnan(evr[0])
    assert evr[1] == pytest.approx(0.0)
    assert evr[2] == pytest.approx(0.0)
    assert evr[3] == pytest.approx(0.25)  # 1 event / 4 finite bars
    # The first down leg (start 3) is completed at bar 6 by an up event; its
    # overshoot = 99 - min(99, 98) = 1, threshold 2 -> ratio 0.5.
    assert out["overshoot_ratio"][6] == pytest.approx(0.5)


def test_first_up_event_uses_pre_confirmation_running_low_not_first_bar():
    # Path: dips to 99 at bar 1, then rises.  The pre-confirmation running LOW
    # is 99 (bar 1) — NOT the first bar's 100.  The first event is an UP that
    # fires at bar 2 when price reaches 99 + 2 = 101.  If the first extreme were
    # pinned to the first bar (100), 101 - 100 = 1 < 2 so no event would fire.
    x = np.asarray([100.0, 99.0, 101.0])
    out = _dc_column(x, np.ones_like(x), 2.0, 10, "adaptive")
    evr = out["event_rate"]
    assert np.isnan(evr[0])
    assert evr[1] == pytest.approx(0.0)
    assert evr[2] == pytest.approx(0.3333333333333333)  # 1 event / 3 finite bars


def test_monotone_decline_still_confirms_first_down_from_initial_peak():
    # Regression: with a single shared ``extreme`` a monotone decline never
    # confirms (the running low keeps up with price and the peak 100 is lost).
    # The dual-track seeding measures the down event from the pre-confirmation
    # running high (100): price falls 100 - 98 = 2 at bar 2 -> down event.
    x = np.asarray([100.0, 99.0, 98.0, 97.0, 96.0])
    out = _dc_column(x, np.ones_like(x), 2.0, 10, "adaptive")
    evr = out["event_rate"]
    assert evr[1] == pytest.approx(0.0)
    assert evr[2] == pytest.approx(0.3333333333333333)  # first down event at bar 2
    # Monotone decline never REVERSES, so no leg is completed -> overshoot NaN.
    assert np.isnan(out["overshoot_ratio"]).all()


def test_monotone_rise_first_up_then_no_completed_leg_overshoot_nan():
    # A monotone rise confirms a first UP event at bar 1 (price 2 - 1 = 1 >= 1)
    # but never reverses, so overshoot_ratio stays NaN (no completed leg).
    x = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    out = _dc_column(x, np.ones_like(x), 1.0, 10, "adaptive")
    evr = out["event_rate"]
    assert evr[1] == pytest.approx(0.5)
    assert np.isnan(out["overshoot_ratio"]).all()


# ---------------------------------------------------------------------------
# TASK 3 — degenerate-input audit
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0.0, -1.0, -1e-9])
def test_threshold_non_positive_rejected(bad):
    op = _get("ts_dc_event_rate")
    x = _col([100.0, 102.0, 101.0])
    with pytest.raises(ValueError, match="threshold > 0"):
        op.calculate(x, _ones(3), threshold=bad, window=10)


def test_insufficient_history_fails_closed_nan_not_phantom():
    op_osr = _get("ts_dc_overshoot_ratio")
    op_evr = _get("ts_dc_event_rate")
    # A single finite bar: seed only, no confirmation -> all NaN, no phantom.
    one = op_osr.calculate(_col([100.0]), _ones(1), threshold=1.0, window=10)
    assert np.isnan(one.iloc[0, 0])
    evr_one = op_evr.calculate(_col([100.0]), _ones(1), threshold=1.0, window=10)
    assert np.isnan(evr_one.iloc[0, 0])
    # Two bars that confirm a first event but never complete a leg -> the
    # overshoot ratio (which needs a COMPLETED leg) fails closed to NaN even
    # though the first confirmation is real (event_rate = 1/2).
    two = op_osr.calculate(_col([100.0, 102.0]), _ones(2), threshold=1.0, window=10)
    assert np.isnan(two.iloc[-1, 0])
    evr_two = op_evr.calculate(_col([100.0, 102.0]), _ones(2), threshold=1.0, window=10)
    assert evr_two.iloc[-1, 0] == pytest.approx(0.5)
    # All-NaN panel: every output row is NaN.
    nan_panel = pd.DataFrame(np.full((5, 1), np.nan), columns=["c"])
    out = op_evr.calculate(nan_panel, _ones(5), threshold=1.0, window=10)
    assert np.isnan(out.to_numpy(dtype=float)).all()


# ---------------------------------------------------------------------------
# (d) — a normal positive-price series keeps its expected oscillation count
# ---------------------------------------------------------------------------
def test_positive_price_oscillation_event_rate_golden():
    # Sawtooth on strictly positive prices, thr=1, scale=1 (adaptive):
    # events at bars 1 (up), 2 (down), 4 (up), 5 (down) -> 4 events / 6 bars.
    x = _col([1.0, 3.0, 2.0, 4.0, 5.0, 3.0])
    out = _get("ts_dc_event_rate").calculate(x, _ones(6), threshold=1.0, window=10)
    assert out.iloc[-1, 0] == pytest.approx(4.0 / 6.0)
    # The last completed leg is the up leg @3..5 (max 5, start 4): overshoot
    # ratio = (5 - 4) / 1 = 1.0.
    osr = _get("ts_dc_overshoot_ratio").calculate(x, _ones(6), threshold=1.0, window=10)
    assert osr.iloc[-1, 0] == pytest.approx(1.0)


def test_asymmetry_positive_prices_expected():
    # Hand-traced legs: up@1 (dur 1, os 0), down@2 (dur 1, os 0),
    # up@3..4 (dur 2, os 2), down@5..6 (dur 2, os 1).
    # duration_asym = (median(1,2) - median(1,2)) / sum = 0.0;
    # overshoot_asym = (median(0,2) - median(0,1)) / sum = (1 - 0.5)/1.5 = 1/3.
    x = _col([1.0, 3.0, 2.0, 4.0, 6.0, 5.0, 4.0, 6.0])
    d = _get("ts_dc_duration_asymmetry").calculate(x, _ones(8), threshold=1.0, window=12)
    o = _get("ts_dc_overshoot_asymmetry").calculate(x, _ones(8), threshold=1.0, window=12)
    assert d.iloc[-1, 0] == pytest.approx(0.0)
    assert o.iloc[-1, 0] == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# Re-verified existing guarantees on positive prices
# ---------------------------------------------------------------------------
def test_scale_gap_still_breaks_episode_positive_prices():
    x = _col([1.0, 3.0, 2.0, 4.0, 5.0, 3.0])
    s_gap = _col([1.0, 1.0, np.nan, 1.0, 1.0, 1.0])
    s_full = _ones(6)
    rate_full = _get("ts_dc_event_rate").calculate(x, s_full, threshold=1.0, window=10)
    rate_gap = _get("ts_dc_event_rate").calculate(x, s_gap, threshold=1.0, window=10)
    # Without the gap there are events at bars 1,2,4,5 (4/6); with a NaN scale
    # at bar 2 the episode breaks and only the post-gap events survive — the
    # gap never connects the two sides into one episode.  R26-116/117: the
    # event-rate denominator is CLOCK-OBSERVABLE bars (price valid AND scale
    # valid), so the broken-clock bar 2 is excluded -> 3 events / 5 observable
    # bars.
    assert rate_full.iloc[-1, 0] == pytest.approx(4.0 / 6.0)
    assert rate_gap.iloc[-1, 0] == pytest.approx(3.0 / 5.0)


def test_nan_price_censors_clock_positive_prices():
    x = _col([1.0, 2.0, np.nan, 3.0, 4.0, 2.0])
    out = _get("ts_dc_event_rate").calculate(x, _ones(6), threshold=1.0, window=10)
    # The NaN bar (and the reseed bar after it) have no event_rate; the post-gap
    # events (bars 4, 5) plus the pre-gap event (bar 1) total 3 events / 5 finite.
    assert np.isnan(out.iloc[2, 0])
    assert np.isnan(out.iloc[3, 0])
    assert out.iloc[-1, 0] == pytest.approx(3.0 / 5.0)


def test_fixed_absolute_prefix_invariance_positive_prices():
    rng = np.random.default_rng(11)
    n = 200
    xw = np.cumsum(rng.normal(0.0, 0.1, n)) + 100.0
    sw = np.abs(rng.normal(1.0, 0.3, n)) + 0.5
    op = _get("ts_dc_overshoot_ratio")
    full = op.calculate(
        _frame(xw[:, None]), _frame(sw[:, None]), threshold=1.0, window=60,
        threshold_mode="fixed_absolute",
    ).to_numpy(dtype=float)
    prefix = op.calculate(
        _frame(xw[:160, None]), _frame(sw[:160, None]), threshold=1.0, window=60,
        threshold_mode="fixed_absolute",
    ).to_numpy(dtype=float)
    np.testing.assert_allclose(full[:160], prefix, equal_nan=True, rtol=1e-9, atol=1e-12)


def test_polars_backend_parity_positive_prices():
    pl = pytest.importorskip("polars")
    x_pd = _col([1.0, 3.0, 2.0, 4.0, 5.0, 3.0])
    np_out = _get("ts_dc_event_rate").calculate(x_pd, _ones(6), threshold=1.0, window=10)
    pl_op = OperatorRegistry.get("ts_dc_event_rate", "polars")
    if pl_op is None:
        pytest.skip("polars backend unavailable")
    pl_x = pl.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=6, freq="B"), "c": [1.0, 3.0, 2.0, 4.0, 5.0, 3.0]}
    )
    pl_s = pl.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=6, freq="B"), "c": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}
    )
    pl_out = pl_op.calculate(pl_x, pl_s, threshold=1.0, window=10)
    np.testing.assert_allclose(
        np.asarray(pl_out["c"].to_list(), dtype=float),
        np_out["c"].to_numpy(dtype=float),
        equal_nan=True,
    )
