# -*- coding: utf-8 -*-
"""Round-11 #3/#4/#9/#10/#145/#146/#147/#148 regression tests.

Covers the statefulness + missing-semantics long-tail fixes for the event /
sequential operator family:

* #3    survival trio (``ts_state_age_percentile`` / ``ts_state_exit_hazard`` /
        ``ts_state_residual_life``) are EPISODE-stateful — each active-row
        output depends on the *completed-run distribution* accumulated since the
        dataset origin plus the current run's age, so a chunk computed without
        that state diverges from the full run.  The execution contract reports
        ``is_stateful`` / full_history.
* #4    ``ts_cusum_pressure`` is a recursive two-sided CUSUM (S⁺/S⁻ accumulate
        across rows) — declared recursive / required_full_history.
* #9    ``event_decay_asof`` is a recursive exponential-decay accumulator —
        declared recursive / required_full_history.
* #10   ``ts_time_since_change`` needs ``previous_state`` + ``last_change_time``
        at chunk start even with ``max_lookback`` — declared recursive /
        required_full_history.
* #145  event spacing must NOT span a NaN gap: an interval crossing an UNKNOWN
        row is censored (window → NaN), never a precise interval.
* #146  ``ts_time_since_change`` splits ``since_transition`` (NaN until the first
        observed transition) vs ``state_age`` (clock starts at the first valid
        state).
* #147  ``max_lookback`` is an EXCLUSIVE upper bound: a distance exactly equal
        to the limit emits NaN (boundary pinned by the tests).
* #148  ``event_decay_asof`` distinguishes EventBool (``event_kind="bool"``, each
        occurrence contributes exactly 1) from MarkedEvent (``event_kind="marked"``,
        the output unit inherits the magnitude).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry
from runtime.execution_contract import execution_contract, history_requirement

ensure_cleaned_loaded()

SURVIVAL_TRIO = (
    "ts_state_age_percentile",
    "ts_state_exit_hazard",
    "ts_state_residual_life",
)
RECURSIVE_OPS = ("ts_cusum_pressure", "event_decay_asof", "ts_time_since_change")


def _bdate(n: int, start: str = "2024-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


# ---------------------------------------------------------------------------
# #3 / #4 / #9 / #10 — execution contracts are stateful / full-history
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("canonical", SURVIVAL_TRIO)
def test_survival_trio_declared_episode_full_history(canonical):
    contract = execution_contract(canonical)
    assert contract.is_stateful
    assert contract.state_model == "episode"
    assert contract.chunking == "required_full_history"
    req = history_requirement(canonical)
    assert req.is_full_history
    assert req.kind == "full_history"


@pytest.mark.parametrize("canonical", RECURSIVE_OPS)
def test_recursive_ops_declared_stateful(canonical):
    contract = execution_contract(canonical)
    assert contract.is_stateful
    assert contract.state_model == "recursive"
    assert contract.chunking == "required_full_history"
    req = history_requirement(canonical)
    assert req.is_full_history
    assert req.kind == "full_history"


def test_survival_trio_full_differs_from_chunk_without_state():
    """#3: the completed-run distribution is carried across the whole history, so
    a chunk computed without that state diverges from the full run."""
    n = 60
    state = pd.DataFrame(np.zeros((n, 1)), index=_bdate(n), columns=["A"])
    state.iloc[5:9, 0] = 1.0    # completed run length 4
    state.iloc[15:19, 0] = 1.0  # completed run length 4
    state.iloc[40:52, 0] = 1.0  # current run (ends after row 51)
    op = OperatorRegistry.get("ts_state_age_percentile", "pandas_numpy")
    full = op.calculate(state, history_window=60, min_completed_runs=2)
    # chunk starting mid-history without the earlier completed runs
    chunk = op.calculate(state.iloc[30:], history_window=60, min_completed_runs=2)
    # At the last active row (global 51 / chunk 21) the full run has the two
    # completed runs {4, 4} in memory (ECDF = 1.0); the chunk has none (NaN).
    full_last = full["A"].iloc[51]
    chunk_last = chunk["A"].iloc[21]
    assert np.isfinite(full_last)
    assert np.isnan(chunk_last)
    assert not np.allclose(full_last, chunk_last, equal_nan=True)


def test_event_decay_asof_full_replay_invariance():
    """#9: the exponential-decay accumulator is reproducible from a from-scratch
    full replay (recursive state, not a finite-window statistic)."""
    rng = np.random.default_rng(3)
    ev = pd.DataFrame(
        rng.choice([0.0, 1.0], size=(40, 2)).astype(float),
        columns=["A", "B"],
        index=_bdate(40),
    )
    op = OperatorRegistry.get("event_decay_asof", "pandas_numpy")
    out = op.calculate(ev, half_life=12.0)
    weight = 0.5 ** (1.0 / 12.0)
    for col in out.columns:
        acc = 0.0
        seen = False
        for t in range(len(ev)):
            v = ev[col].iloc[t]
            if np.isfinite(v):
                seen = True
                acc = acc * weight + float(v)
            elif not seen:
                assert np.isnan(out[col].iloc[t])
                continue
            else:  # carry policy: missing decays the accumulator
                acc = acc * weight
            assert out[col].iloc[t] == pytest.approx(acc)


def test_ts_cusum_pressure_full_replay_invariance():
    """#4: the S⁺/S⁻ accumulation matches a manual from-scratch full replay."""
    rng = np.random.default_rng(5)
    x = pd.DataFrame(
        rng.normal(size=(50, 2)),
        columns=["A", "B"],
        index=_bdate(50),
    )
    op = OperatorRegistry.get("ts_cusum_pressure", "pandas_numpy")
    out = op.calculate(x, reference_window=10, drift=0.5, min_periods=3)
    w = 10
    k = 0.5
    eps = 1e-12
    for col in out.columns:
        xv = x[col].to_numpy(dtype=float)
        sp = 0.0
        sm = 0.0
        for t in range(len(xv)):
            xt = xv[t]
            if not np.isfinite(xt):
                assert np.isnan(out[col].iloc[t])
                sp = 0.0
                sm = 0.0
                continue
            lo = max(0, t - w)
            seg = xv[lo:t]
            valid = seg[np.isfinite(seg)]
            if valid.size < 3:
                assert np.isnan(out[col].iloc[t])
                sp = 0.0
                sm = 0.0
                continue
            med = float(np.median(valid))
            mad = float(np.median(np.abs(valid - med)))
            if mad <= eps:
                assert np.isnan(out[col].iloc[t])
                sp = 0.0
                sm = 0.0
                continue
            scale = 1.4826 * mad
            z = (float(xt) - med) / scale
            sp = max(0.0, sp + z - k)
            sm = min(0.0, sm + z + k)
            assert out[col].iloc[t] == pytest.approx(sp + sm)


# ---------------------------------------------------------------------------
# #145 — event spacing does NOT span a NaN gap
# ---------------------------------------------------------------------------


def test_event_spacing_mean_censored_across_nan_gap():
    dates = _bdate(7)
    clean = pd.DataFrame([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0], index=dates, columns=["A"])
    naned = pd.DataFrame([1.0, 0.0, np.nan, 1.0, 0.0, 1.0, 0.0], index=dates, columns=["A"])
    op = OperatorRegistry.get("ts_event_spacing_mean", "pandas_numpy")
    sm_clean = op.calculate(clean, window=7, min_events=2)
    sm_nan = op.calculate(naned, window=7, min_events=2)
    # identical event positions; the NaN between event 0 and 3 censors the window
    assert sm_clean["A"].iloc[4] == pytest.approx(3.0)
    assert np.isnan(sm_nan["A"].iloc[4])
    assert np.isnan(sm_nan["A"].iloc[6])


def test_event_spacing_cv_censored_across_nan_gap():
    dates = _bdate(8)
    clean = pd.DataFrame(
        [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], index=dates, columns=["A"]
    )
    naned = pd.DataFrame(
        [1.0, 0.0, np.nan, 1.0, 0.0, 1.0, 0.0, 1.0], index=dates, columns=["A"]
    )
    op = OperatorRegistry.get("ts_event_spacing_cv", "pandas_numpy")
    cv_clean = op.calculate(clean, window=8, min_events=3)
    cv_nan = op.calculate(naned, window=8, min_events=3)
    # 3 gaps with no NaN -> finite CV; the 0->3 gap crossing NaN censors -> NaN
    assert np.isfinite(cv_clean["A"].iloc[7])
    assert np.isnan(cv_nan["A"].iloc[7])


# ---------------------------------------------------------------------------
# #146 — time_since_change: two initial-state semantics
# ---------------------------------------------------------------------------


def test_time_since_change_two_initial_semantics_differ():
    dates = _bdate(6)
    cond = pd.DataFrame([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], index=dates, columns=["A"])
    op = OperatorRegistry.get("ts_time_since_change", "pandas_numpy")
    st = op.calculate(cond, missing_policy="break", initial_semantics="since_transition")
    sa = op.calculate(cond, missing_policy="break", initial_semantics="state_age")
    # since_transition: NaN until the first observed transition at row 3
    assert np.isnan(st["A"].iloc[0]) and np.isnan(st["A"].iloc[2])
    assert st["A"].iloc[3] == 0.0 and st["A"].iloc[5] == 2.0
    # state_age: the clock starts at the first valid state (row 0)
    assert sa["A"].iloc[0] == 0.0 and sa["A"].iloc[2] == 2.0
    assert sa["A"].iloc[3] == 0.0 and sa["A"].iloc[5] == 2.0
    # the two definitions disagree on the rows before the first transition
    assert not np.allclose(st["A"].iloc[:3].to_numpy(), sa["A"].iloc[:3].to_numpy(), equal_nan=True)


def test_time_since_change_rejects_unknown_semantics():
    dates = _bdate(2)
    cond = pd.DataFrame([1.0, 1.0], index=dates, columns=["A"])
    op = OperatorRegistry.get("ts_time_since_change", "pandas_numpy")
    with pytest.raises(ValueError, match="initial_semantics"):
        op.calculate(cond, initial_semantics="bogus")


# ---------------------------------------------------------------------------
# #147 — max_lookback exact-boundary behaviour (exclusive upper bound)
# ---------------------------------------------------------------------------


def test_time_since_change_max_lookback_exclusive_boundary():
    dates = _bdate(6)
    cond = pd.DataFrame([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], index=dates, columns=["A"])
    op = OperatorRegistry.get("ts_time_since_change", "pandas_numpy")
    # change at row 3 -> distances 0, 1, 2 at rows 3, 4, 5.
    lim2 = op.calculate(cond, max_lookback=2, missing_policy="break")
    assert lim2["A"].iloc[3] == 0.0 and lim2["A"].iloc[4] == 1.0
    assert np.isnan(lim2["A"].iloc[5])  # distance 2 == limit 2 -> NOT emitted
    # one more row of headroom -> distance 2 IS emitted
    lim3 = op.calculate(cond, max_lookback=3, missing_policy="break")
    assert lim3["A"].iloc[5] == 2.0


# ---------------------------------------------------------------------------
# #148 — event_decay_asof: EventBool vs MarkedEvent
# ---------------------------------------------------------------------------


def test_event_decay_asof_event_vs_marked():
    dates = _bdate(4)
    ev = pd.DataFrame([0.0, 2.0, 0.0, 0.0], index=dates, columns=["A"])
    op = OperatorRegistry.get("event_decay_asof", "pandas_numpy")
    marked = op.calculate(ev, half_life=10.0, event_kind="marked")
    eb = op.calculate(ev, half_life=10.0, event_kind="bool")
    # MarkedEvent: magnitude 2.0 preserved and decays
    assert marked["A"].iloc[1] == pytest.approx(2.0)
    assert marked["A"].iloc[3] == pytest.approx(2.0 * (0.5 ** (2.0 / 10.0)))
    # EventBool: each occurrence contributes exactly 1
    assert eb["A"].iloc[1] == pytest.approx(1.0)
    assert eb["A"].iloc[3] == pytest.approx(0.5 ** (2.0 / 10.0))


def test_event_decay_asof_rejects_unknown_event_kind():
    dates = _bdate(2)
    ev = pd.DataFrame([0.0, 1.0], index=dates, columns=["A"])
    op = OperatorRegistry.get("event_decay_asof", "pandas_numpy")
    with pytest.raises(ValueError, match="event_kind"):
        op.calculate(ev, event_kind="bogus")
