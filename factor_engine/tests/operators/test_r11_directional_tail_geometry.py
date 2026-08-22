# -*- coding: utf-8 -*-
"""R11 long-tail audit findings #29-#43 — geometry / DC / tail / marked-event.

Covers:
* #29 robust correlation is genuinely robust (biweight ≠ Pearson on outliers).
* #30 ts_beta_break_score reports the true beta Cov(y,x)/Var(x) (not corr).
* #32 fixed_absolute DC threshold is prefix-invariant (no rolling-boundary drift).
* #34 a missing DC scale BREAKS the episode (no event straddles the gap).
* #35 group_tail_lead baseline uses the common observation cohort.
* #36 tail lead requires min conditioning events and outputs effective_event_n.
* #37 relation diffusion complete graph == closed-form group mean + deviation.
* #39 ts_hill_tail_index rejects a raw nonstationary price level.
* #41 extreme-run missing gap censors (never connects two clusters).
* #42 marked_event NaN event observation censors the spanning interval.
* #43 MarkMissingPolicy distinguishes the three event/mark states.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all

load_all()

from cleaned_operators.registry import OperatorRegistry


def _col(data: list[float]) -> pd.DataFrame:
    return pd.DataFrame(np.asarray(data, dtype=float).reshape(-1, 1), columns=["c"])


def _frame(values: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=values.shape[0], freq="B")
    cols = [f"S{i}" for i in range(values.shape[1])]
    return pd.DataFrame(values, index=idx, columns=cols)


def _get(name: str):
    return OperatorRegistry.get(name, "pandas_numpy") or OperatorRegistry.get(name)


# ---------------------------------------------------------------------------
# #29  robust correlation must genuinely differ from Pearson
# ---------------------------------------------------------------------------
def test_robust_correlation_differs_from_pearson_on_outlier():
    from cleaned_operators.feature_geometry import _biweight_midcorr

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    y = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 0.01])
    bicor = _biweight_midcorr(x, y)
    pearson = float(np.corrcoef(x, y)[0, 1])
    # Pearson is dragged negative by the single outlier; biweight downweights it.
    assert abs(bicor - pearson) > 0.5
    assert bicor > 0.0 and pearson < 0.0


def test_feature_mode_share_robust_band_differs_from_pearson_band():
    # Two genuinely-correlated fields plus one outlying point: the biweight
    # midcorrelation gives a different dominant-mode share than Pearson.
    rng = np.random.default_rng(3)
    base = rng.normal(0.0, 1.0, 60)
    f1 = base + 0.1 * rng.normal(0.0, 1.0, 60)
    f2 = base + 0.1 * rng.normal(0.0, 1.0, 60)
    f3 = rng.normal(0.0, 1.0, 60)
    # Inject a single huge outlier into f1 (row 30).
    f1[30] = 100.0
    panel = _frame(np.column_stack([f1, f2, f3]))
    out = _get("ts_feature_mode_share").calculate(panel, panel, panel, window=60)
    robust_share = float(out.iloc[-1, 0])

    # Independent Pearson-based reference.
    z = np.column_stack([f1, f2, f3])
    corr = np.corrcoef(z, rowvar=False)
    w = np.maximum(np.linalg.eigvalsh(corr), 0.0)
    pearson_share = float(w[-1] / w.sum())

    assert np.isfinite(robust_share)
    assert abs(robust_share - pearson_share) > 1e-3


# ---------------------------------------------------------------------------
# #30  beta break is the true beta, not a correlation
# ---------------------------------------------------------------------------
def test_beta_break_uses_true_cov_var_slope():
    from cleaned_operators.feature_geometry import _true_beta

    # y = 3x + noise-free: true beta = 3, correlation = 1.
    x = np.arange(1.0, 61.0)
    y = 3.0 * x + 5.0
    beta = _true_beta(y, x, 5)
    assert beta == pytest.approx(3.0, rel=1e-9)
    assert abs(float(np.corrcoef(x, y)[0, 1]) - 1.0) < 1e-9  # but corr != 3


# ---------------------------------------------------------------------------
# #32  fixed_absolute threshold is prefix-invariant (no boundary drift)
# ---------------------------------------------------------------------------
def test_dc_fixed_absolute_prefix_invariance_and_anchor_stability():
    rng = np.random.default_rng(11)
    n = 200
    # Strictly positive price path (the concurrent R13/R14 directional-change
    # contract requires input_units='price' to be > 0; a bare random walk can go
    # negative and is no longer a valid price fixture).
    x = pd.DataFrame(
        np.exp(np.cumsum(rng.normal(0.0, 0.1, n))), index=pd.date_range("2024-01-01", periods=n, freq="B")
    )
    s = pd.DataFrame(
        np.abs(rng.normal(1.0, 0.3, n)) + 0.5, index=pd.date_range("2024-01-01", periods=n, freq="B")
    )
    op = _get("ts_dc_overshoot_ratio")
    full = op.calculate(x, s, threshold=1.0, window=60, threshold_mode="fixed_absolute").to_numpy(
        dtype=float
    )
    prefix = op.calculate(
        x.iloc[:160], s.iloc[:160], threshold=1.0, window=60, threshold_mode="fixed_absolute"
    ).to_numpy(dtype=float)
    # The value at row r is identical whether the series is 160 or 200 bars long:
    # the anchor was pinned once at stream init and never re-derived from the
    # sliding window's first scale.
    np.testing.assert_allclose(full[:160], prefix, equal_nan=True, rtol=1e-9, atol=1e-12)

    # Also verify the anchor does not drift across different total lengths.
    short = op.calculate(
        x.iloc[:60], s.iloc[:60], threshold=1.0, window=60, threshold_mode="fixed_absolute"
    ).to_numpy(dtype=float)
    assert np.isclose(full[59, 0], short[-1, 0], equal_nan=True)


# ---------------------------------------------------------------------------
# #34  missing DC scale BREAKS the episode (never skips-and-continues)
# ---------------------------------------------------------------------------
def test_dc_scale_missing_breaks_episode():
    # The NaN scale at bar 2 breaks the episode: fewer events survive than the
    # full-scale run (rate 4/6 -> 3/5 under round-2 §24 pre-confirmation
    # running-extrema seeding; prices strictly positive by contract).
    # R26-116/117: the event-rate denominator is CLOCK-OBSERVABLE bars (price
    # valid AND scale valid), so the broken-clock bar 2 is excluded.
    # Prices must be strictly positive (concurrent R13/R14 price>0 contract);
    # adding a constant preserves the DC path shape and event positions.
    x = _col([1.0, 3.0, 2.0, 4.0, 5.0, 3.0])
    s_gap = _col([1.0, 1.0, np.nan, 1.0, 1.0, 1.0])
    s_full = _col([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    rate_gap = _get("ts_dc_event_rate").calculate(x, s_gap, threshold=1.0, window=10)
    rate_full = _get("ts_dc_event_rate").calculate(x, s_full, threshold=1.0, window=10)
    assert rate_full.iloc[-1, 0] == pytest.approx(4.0 / 6.0)
    assert rate_gap.iloc[-1, 0] == pytest.approx(3.0 / 5.0)


# ---------------------------------------------------------------------------
# #35  group_tail_lead baseline uses the common observation cohort
# ---------------------------------------------------------------------------
def test_group_tail_lead_common_cohort_reduces_counts_for_others():
    from cleaned_operators.tail_systemic import _tail_lead_effective_n_series

    rng = np.random.default_rng(5)
    rows, cols = 40, 3
    x = rng.normal(0.0, 1.0, (rows, cols))
    # Member 1 is extreme on days 25..34 (deeply negative) — inside the last
    # trailing window [25,39].
    x[25:35, 1] = -5.0
    g = np.tile(np.array(["A", "A", "A"]), (rows, 1))

    x_all = x.copy()
    x_missing = x.copy()
    # Member 0 suspended on days 25..26 only.  (A 2-day block keeps the
    # post-suspension warm-up short: member 0's extreme indicator is NaN on
    # 25..26 and recovers by row 27, so the common cohort excludes exactly the
    # two suspended days.)
    x_missing[25:27, 0] = np.nan

    eff_all = _tail_lead_effective_n_series(x_all, g, 15, 0.2, "lower", 1, 10, True, 5)
    eff_missing = _tail_lead_effective_n_series(x_missing, g, 15, 0.2, "lower", 1, 10, True, 5)

    # Member 1 loses the two suspended days (25..26) from its conditioning count
    # because the common cohort requires EVERY member to be observable on day d.
    assert eff_all[-1, 1] >= 8
    assert eff_missing[-1, 1] < eff_all[-1, 1]
    assert eff_all[-1, 1] - eff_missing[-1, 1] <= 5


# ---------------------------------------------------------------------------
# #36  tail lead requires a minimum conditioning-event count + effective_event_n
# ---------------------------------------------------------------------------
def test_group_tail_lead_min_conditioning_events_and_effective_n():
    from cleaned_operators.tail_systemic import _tail_lead_effective_n_series, _tail_lead_series

    rng = np.random.default_rng(6)
    rows, cols = 60, 2
    g = np.tile(np.array(["A", "A"]), (rows, 1))
    # Member 0: extreme on the last window (days 40..59).  Member 1: a strictly
    # increasing series, so it is NEVER extreme on its own except for the single
    # planted day 50 (x=row index is far above the prior-window lower quantile).
    x = rng.normal(0.0, 1.0, (rows, cols))
    x[:, 1] = np.arange(rows, dtype=float)
    x[50, 1] = -5.0
    x[40:60, 0] = -5.0

    score = _tail_lead_series(x, g, 20, 0.2, "lower", 1, 10, True, min_conditioning_events=5)
    eff = _tail_lead_effective_n_series(x, g, 20, 0.2, "lower", 1, 10, True, min_conditioning_events=5)

    fin = np.isfinite(score)
    # Whenever a score is emitted, the effective conditioning count reached 5.
    assert bool((eff[fin] >= 5).all())
    # Member 1 has a single own conditioning event (day 50) -> its score stays NaN.
    assert eff[-1, 1] == pytest.approx(1.0)
    assert np.isnan(score[-1, 1])
    # Member 0 has enough conditioning events -> finite score.
    assert eff[-1, 0] >= 5
    assert np.isfinite(score[-1, 0])


# ---------------------------------------------------------------------------
# #37  relation diffusion complete graph == closed-form group mean + deviation
# ---------------------------------------------------------------------------
def test_relation_diffusion_closed_form_equals_group_mean_plus_deviation():
    rng = np.random.default_rng(9)
    rows, cols = 10, 4
    x = rng.normal(0.0, 1.0, (rows, cols))
    g = np.tile(np.array(["A", "A", "A", "B"]), (rows, 1))
    alpha, steps = 0.5, 3
    out = _get("relation_diffusion_score").calculate(
        _frame(x), _frame(g.astype(object)), alpha=alpha, steps=steps
    ).to_numpy(dtype=float)

    # Closed form: P=(J-I)/(m-1); P^k x = mean + c^k (x-mean); cascade
    # coefficient for the deviation is (1-alpha) c geo_sum + (alpha c)^K.
    for r in range(rows):
        for key in set(g[r]):
            idx = [c for c in range(cols) if g[r, c] == key]
            if len(idx) < 2:
                continue
            m = len(idx)
            vals = x[r, idx]
            mean = float(vals.mean())
            c = -1.0 / (m - 1)
            d = alpha * c
            geo_sum = (1.0 - d**steps) / (1.0 - d)
            coeff = (1.0 - alpha) * c * geo_sum + d**steps
            expect = mean + coeff * (vals - mean)
            for pos, i in enumerate(idx):
                assert out[r, i] == pytest.approx(expect[pos], rel=1e-9, abs=1e-9)


def test_relation_diffusion_two_member_constant_preserved():
    # The closed form still preserves constant graph signals (round-3 P0-05).
    x = np.tile(np.array([3.0, 3.0, 3.0]), (20, 1))
    g = np.tile(np.array(["A", "A", "A"]), (20, 1))
    out = _get("relation_diffusion_score").calculate(
        _frame(x), _frame(g.astype(object)), alpha=0.5, steps=3
    ).to_numpy(dtype=float)
    assert np.all(np.isfinite(out[-1]))
    assert np.allclose(out[-1], 3.0, atol=1e-9)


# ---------------------------------------------------------------------------
# #39  ts_hill_tail_index flags a raw nonstationary price level
# ---------------------------------------------------------------------------
def test_hill_flags_raw_price_level():
    op = _get("ts_hill_tail_index")
    # A raw nonstationary price level must be detected.  The R11 operator model
    # treats the numeric heuristic as a data-quality warning (the authoritative
    # input-unit contract lives in the typed FieldSpec ``input_units``), so the
    # operator emits a RuntimeWarning and still runs — it does not silently
    # accept a price level as if it were a return series.
    with pytest.warns(RuntimeWarning, match="price level"):
        op.calculate(_col(np.linspace(100.0, 300.0, 200).tolist()), 100, "upper", 0.2, 10)
    with pytest.warns(RuntimeWarning, match="price level"):
        rng = np.random.default_rng(0)
        rw = np.cumsum(rng.normal(0.0, 1.0, 300)) + 100.0
        op.calculate(_col(rw.tolist()), 150, "upper", 0.2, 10)


def test_hill_accepts_return_and_positive_magnitude():
    rng = np.random.default_rng(1)
    op = _get("ts_hill_tail_index")
    ret = _col((rng.normal(0.0, 0.02, 500)).tolist())
    mag = _col(np.abs(rng.normal(0.0, 0.02, 500)).tolist())
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # returns/magnitudes must NOT be flagged
        assert np.isfinite(op.calculate(ret, 300, "upper", 0.2, 10).to_numpy(dtype=float)[-1, 0])
        assert np.isfinite(op.calculate(mag, 300, "upper", 0.2, 10).to_numpy(dtype=float)[-1, 0])


# ---------------------------------------------------------------------------
# #41  extreme-run missing gap censors (never connects two clusters)
# ---------------------------------------------------------------------------
def test_extremal_index_gap_does_not_connect_clusters():
    n = 30
    x = np.zeros(n)
    x[5] = 1.0
    x[7] = np.nan  # unknown bar between the two extreme bars
    x[8] = 1.0
    out = _get("ts_extremal_index").calculate(
        _col(x.tolist()), window=30, side="upper", q=0.9, min_exceed=2
    )
    # The two extremes on either side of the gap are SEPARATE clusters (the gap
    # must not connect them) -> theta = clusters / count = 2 / 2 = 1.
    assert out.iloc[-1, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# #42  marked_event NaN event observation censors the spanning interval
# ---------------------------------------------------------------------------
def test_marked_event_nan_observation_censors_interval():
    op = _get("event_interval_mark_coupling")
    # Events at rows 0,2,5,9,14.  In ev_nan a NaN event observation at row 3
    # lies strictly between the events at rows 2 and 5, so the interval 2->5
    # spans an unknown bar and must be censored (never treated as a no-event 0).
    # After censoring, only 3 valid intervals remain -> fail closed to NaN.
    ev_zero = _col([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    ev_nan = _col([1.0, 0.0, 1.0, np.nan, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    mk = _col([1.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 16.0])

    out_zero = op.calculate(ev_zero, mk, window=15, mark_missing_policy="censor").iloc[-1, 0]
    out_nan = op.calculate(ev_nan, mk, window=15, mark_missing_policy="censor").iloc[-1, 0]
    # The uncensored version has intervals [2,3,4,5] vs marks [2,4,8,16] (> 0.9);
    # the NaN-spanning interval is censored -> only 3 valid -> NaN.
    assert out_zero > 0.9
    assert np.isnan(out_nan)


# ---------------------------------------------------------------------------
# #43  MarkMissingPolicy distinguishes the three event/mark states
# ---------------------------------------------------------------------------
def test_mark_missing_policy_censor_vs_drop():
    # 6 events, marks [1,2,NaN,4,5,6]: the mark at event index 2 is unavailable
    # ("event occurred, mark unavailable").  censor breaks the sequence (NaN),
    # drop re-indexes the survivors (finite).
    ev = _col([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    mk = _col([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    op = _get("event_mark_autocorr")
    censor = op.calculate(ev, mk, history_window=6, event_lag=1, mark_missing_policy="censor")
    drop = op.calculate(ev, mk, history_window=6, event_lag=1, mark_missing_policy="drop")
    assert np.isnan(censor.iloc[-1, 0])
    assert np.isfinite(drop.iloc[-1, 0])


def test_event_states_three_way_classification():
    from cleaned_operators.marked_event import _event_states

    ev = np.array([1.0, 0.0, 1.0, np.nan, 1.0])
    mk = np.array([1.5, 0.0, np.nan, 0.0, 3.0])
    idx, marks, has_mark = _event_states(ev, mk)
    # Events: bars 0 (mark available), 2 (mark unavailable), 4 (mark available).
    assert idx.tolist() == [0, 2, 4]
    assert marks[0] == 1.5
    assert np.isnan(marks[1])
    assert marks[2] == 3.0
    assert has_mark.tolist() == [True, False, True]
