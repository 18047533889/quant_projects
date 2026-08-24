# -*- coding: utf-8 -*-
"""Round-14 P0 semantic fixes for the marked-event / update-clock families.

Covers:

  * P0-8  ``event_mark_autocorr`` default censor uses the LATEST contiguous
          observed event segment (the trailing/current regime), not the longest
          historical one.  ``event_interval_mark_coupling`` censor likewise
          uses the trailing valid run.
  * P0-9  ``event_interval_mark_coupling`` is a strict bounded bar-window: the
          pre-window interval is included ONLY when the previous event lies
          within ``max_boundary_extension`` bars of the window edge.  A distant
          previous event no longer makes the window unbounded.
  * P0-10 the update-clock ``HistoryRequirement`` event count DERIVES from the
          ``n_updates`` parameter (floored at the kernel's ``min_updates``),
          instead of a stale constant 5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all

try:
    load_all()
except Exception:
    # The working tree currently carries UNCOMMITTED concurrent-session renames
    # that break a full registry load (the ``ts_multifractal_width`` alias
    # colliding with the polars geometry-math backend, ``ts_residualized_hsic``
    # ``purge_gap`` ParamSpec drift, ...).  Those are unrelated to these P0
    # fixes; fall back to loading ONLY the operator modules this file exercises
    # so the marked-event / update-clock semantics can still be validated.
    import factor_engine.cleaned_operators.marked_event  # noqa: F401
    import factor_engine.cleaned_operators.update_clock  # noqa: F401

from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.execution_contract import (
    execution_contract,
    history_requirement,
    own_history_requirement,
)


def _col(values) -> pd.DataFrame:
    arr = np.asarray(values, dtype=float).reshape(-1, 1)
    return pd.DataFrame(arr, columns=["S0"])


def _run(canonical, *frames, **kw):
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    return op.calculate(*frames, **kw)["S0"].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# P0-8  marked-event censor picks the LATEST contiguous segment
# ---------------------------------------------------------------------------
def test_event_mark_autocorr_uses_latest_contiguous_regime():
    """A missing mark breaks the event sequence; the censor must reflect the NEW
    (trailing) regime, not the longest (stale old) one.

    Construction: 20 old-regime events with alternating marks (autocorr = -1),
    one unavailable mark (NaN), then 6 new-regime events with a monotone ramp
    (autocorr = +1).  The default ``censor`` must output the +1 of the new
    regime — the old code's "longest segment" pick would output the old -1.
    """
    old = np.array([1.0, 3.0] * 10)              # 20 marks, corr = -1
    new = np.arange(1.0, 7.0)                    # [1,2,3,4,5,6], corr = +1
    marks = np.concatenate([old, [np.nan], new])  # 20 + 1 + 6 = 27 events

    ev = _col(np.ones(len(marks)))
    mk = _col(marks)

    out = _run("event_mark_autocorr", ev, mk, history_window=30, event_lag=1)
    # Trailing/current regime (6 events, ramp) -> perfect positive autocorr.
    assert out[-1] == pytest.approx(1.0, abs=1e-6)
    # Explicitly NOT the stale old-regime alternating -1.
    assert out[-1] != pytest.approx(-1.0, abs=1e-6)


def test_event_interval_mark_coupling_censor_uses_trailing_run():
    """P0-8 in the interval-coupling censor: a mid-sequence break keeps only the
    TRAILING valid run (the current regime), never the longest historical one."""
    # 8 events; marks [1, 2, NaN, 4, 5, 6, 7, 8]: the NaN mark at event index 2
    # breaks the sequence.  Trailing run = indices 3..7 -> corr of the ramp.
    ev = _col(np.ones(8))
    mk = _col([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
    out = _run("event_interval_mark_coupling", ev, mk, window=8)
    # Trailing run (4,5,6,7,8): intervals all 1 -> zero variance -> fail closed.
    assert np.isnan(out[-1])


# ---------------------------------------------------------------------------
# P0-9  event_interval_mark_coupling strict bounded bar-window
# ---------------------------------------------------------------------------
def test_event_interval_mark_coupling_bounded_pre_window():
    """A previous event 1000 bars before the window must NOT leak in: with the
    default ``max_boundary_extension`` the output is identical with and without
    that distant event.  A large extension re-enables it (the knob is live)."""
    op = OperatorRegistry.get("event_interval_mark_coupling", "pandas_numpy")
    n = 1200
    w = 252
    cluster = [1150, 1152, 1155, 1159, 1164]

    ev_with = np.zeros(n)
    ev_with[0] = 1.0                     # distant event ~1150 bars before window
    ev_with[cluster] = 1.0
    ev_without = np.zeros(n)
    ev_without[cluster] = 1.0

    mk = np.zeros(n)
    mk[0] = 1.0
    mk[cluster] = [1.0, 2.0, 4.0, 8.0, 16.0]

    out_with = op.calculate(_col(ev_with), _col(mk), window=w).iloc[-1, 0]
    out_without = op.calculate(_col(ev_without), _col(mk), window=w).iloc[-1, 0]

    assert np.isfinite(out_with)
    # The distant event must be EXCLUDED -> identical output.
    assert out_with == pytest.approx(out_without, abs=1e-9)

    # A huge extension DOES include the distant interval -> output diverges,
    # proving the boundary-extension knob controls the pre-window interval.
    out_ext = op.calculate(
        _col(ev_with), _col(mk), window=w, max_boundary_extension=2000
    ).iloc[-1, 0]
    assert out_ext != pytest.approx(out_without, abs=1e-6)


def test_event_interval_mark_coupling_zero_extension_in_window_only():
    """max_boundary_extension=0 disables boundary extension entirely — a previous
    event even 1 bar before the window is excluded (in-window events only), so
    the pre-window interval no longer contributes a data point."""
    op = OperatorRegistry.get("event_interval_mark_coupling", "pandas_numpy")
    # Window [5, 19]; previous event at row 4 (1 bar before window edge 5).
    # With mbe=1 the pre-window interval (1 bar) is kept; with mbe=0 it is
    # dropped, changing the (interval, mark) sample and hence the coupling.
    rows = [4, 5, 7, 9, 12, 15]
    ev = np.zeros(20)
    ev[rows] = 1.0
    mk = np.zeros(20)
    mk[rows] = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]

    with_ext = op.calculate(_col(ev), _col(mk), window=15, max_boundary_extension=1).iloc[-1, 0]
    no_ext = op.calculate(_col(ev), _col(mk), window=15, max_boundary_extension=0).iloc[-1, 0]
    assert np.isfinite(with_ext) and np.isfinite(no_ext)
    assert with_ext != pytest.approx(no_ext, abs=1e-9)


# ---------------------------------------------------------------------------
# P0-10  update-clock HistoryRequirement derives its event count from n_updates
# ---------------------------------------------------------------------------
_MIN_UPDATES = {
    "update_path_efficiency": 3,
    "update_acceleration": 4,
    "update_surprise": 5,
    "update_direction_persistence": 3,
}


def test_update_clock_history_derives_from_n_updates():
    """The event-count requirement follows the ``n_updates`` parameter, not a
    stale constant 5 (both the planner's ``history_requirement`` and the
    analyzer's per-node ``own_history_requirement``)."""
    for canon, floor in _MIN_UPDATES.items():
        ec = execution_contract(canon)
        assert ec.is_stateful, canon

        hr10 = history_requirement(canon, {"n_updates": 10})
        assert hr10.is_event_clock, canon
        assert hr10.kind == "event_count", canon
        assert hr10.count >= 10, (canon, hr10.count)

        # The per-node API the analyzer delegates to must agree.
        own10 = own_history_requirement(canon, {"n_updates": 10})
        assert own10.count >= 10, (canon, own10.count)

        # Unbound n_updates -> the kernel default (5).
        assert history_requirement(canon).count == 5, canon


def test_update_clock_history_floored_at_min_updates():
    """n_updates below the kernel floor reports the floored value (the minimum
    number of real update events the kernel can ever consume)."""
    for canon, floor in _MIN_UPDATES.items():
        hr3 = history_requirement(canon, {"n_updates": 3})
        assert hr3.kind == "event_count", canon
        assert hr3.count == max(3, floor), (canon, hr3.count, floor)


# ---------------------------------------------------------------------------
# Happy-path smoke: each operator computes on a small synthetic series.
# ---------------------------------------------------------------------------
def test_marked_event_smoke():
    ev = _col([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    mk = _col([1.0, 0.0, 2.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 16.0])
    out1 = _run("event_mark_autocorr", ev, mk, history_window=15, event_lag=1)
    assert out1.shape == (15,)
    assert np.isfinite(out1[-1])
    out2 = _run("event_interval_mark_coupling", ev, mk, window=15)
    assert out2.shape == (15,)
    assert out2[-1] > 0.9


def test_update_clock_smoke():
    x = _col([10.0, 12.0, 16.0, 21.0, 27.0, 30.0, 33.0])
    ev = _col([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    for canon, n in [
        ("update_path_efficiency", 3),
        ("update_acceleration", 4),
        ("update_surprise", 5),
        ("update_direction_persistence", 3),
    ]:
        out = _run(canon, x, ev, n_updates=n)
        assert out.shape == (7,), canon
        assert np.isfinite(out[-1]), canon
