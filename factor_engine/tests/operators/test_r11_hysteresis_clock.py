# -*- coding: utf-8 -*-
"""Round-11 hysteresis-family + update/report event-clock audit tests.

Covers findings #5/#6/#7/#8/#44/#45/#46/#47/#48/#49:

  #5   hysteresis family is genuinely stateful (shared HysteresisStateKernel).
  #6   ONE HysteresisMissingPolicy drives state/age/integral/entry consistently.
  #7   update-clock operators are event-history state (history_kind=event_count).
  #8   report-timing history is report_count, never trading bars.
  #44  update-clock NaN update_event censors (two update nodes are never joined
       across an unknown observation day).
  #45  update-acceleration baseline scale excludes the current delta.
  #46  report_timing NEVER infers a filing event from "field is finite".
  #47  a revision fires only when revision_id changes / an explicit event, never
       from ``prev_x`` being finite.
  #48  report window defaults/grid are report counts (4/8/12/20).
  #49  revision magnitude requires a typed same-period RevisionPair.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all

load_all()

from cleaned_operators.registry import OperatorRegistry
from runtime.execution_contract import execution_contract, history_requirement


def _panel(vals):
    arr = np.asarray(vals, dtype=float).reshape(-1, 1)
    return pd.DataFrame(arr, columns=["S0"])


def _str_panel(vals):
    return pd.DataFrame({"S0": list(vals)})


def _col(vals):
    return _panel(vals)


def _run(canonical, *frames, **kw):
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    return op.calculate(*frames, **kw)["S0"].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# #5 / #6 hysteresis family: shared kernel + shared missing policy
# ---------------------------------------------------------------------------

def test_hysteresis_family_declared_stateful_full_history():
    """The whole family reports a stateful, honest full-history contract."""
    for canon in (
        "ts_hysteresis_state",
        "ts_hysteresis_age",
        "ts_state_integral",
        "ts_state_entry_strength",
    ):
        ec = execution_contract(canon)
        assert ec.is_stateful, canon
        assert ec.chunking == "required_full_history", canon


def test_hysteresis_family_shared_missing_policy_carry_vs_break():
    """One policy drives state/age/integral/entry strength consistently (#6).

    A state is entered, then an unknown (NaN) day, then a mild value that keeps
    the state alive ONLY if the NaN carried the state machine.  CARRY keeps the
    whole family alive; BREAK resets the whole family.
    """
    z = _col([0.0, 1.2, 1.5, np.nan, 0.8])
    upper, lower = 1.0, 0.5

    st = _run("ts_hysteresis_state", z, upper=upper, lower=lower, missing_policy="CARRY_STATE_FREEZE_CLOCK")
    age = _run("ts_hysteresis_age", z, upper=upper, lower=lower, cap=10, missing_policy="CARRY_STATE_FREEZE_CLOCK")
    integ = _run("ts_state_integral", z, upper=upper, lower=lower, missing_policy="CARRY_STATE_FREEZE_CLOCK")
    ent = _run("ts_state_entry_strength", z, upper=upper, lower=lower, missing_policy="CARRY_STATE_FREEZE_CLOCK")
    # Carried through the NaN: state +1, age continued to 3 (0.3 with cap=10),
    # integral kept accruing (0.7 + 1.0 + 0.3 = 2.0), entry value (0.2) carried.
    assert st[-1] == 1.0
    assert age[-1] == pytest.approx(3.0 / 10.0)
    assert integ[-1] == pytest.approx(2.0)
    assert ent[-1] == pytest.approx(0.2)

    st_b = _run("ts_hysteresis_state", z, upper=upper, lower=lower, missing_policy="BREAK")
    age_b = _run("ts_hysteresis_age", z, upper=upper, lower=lower, cap=10, missing_policy="BREAK")
    integ_b = _run("ts_state_integral", z, upper=upper, lower=lower, missing_policy="BREAK")
    ent_b = _run("ts_state_entry_strength", z, upper=upper, lower=lower, missing_policy="BREAK")
    # NaN reset the machine: the mild 0.8 cannot re-enter the +1 band -> neutral.
    assert st_b[-1] == 0.0
    assert age_b[-1] == 0.0
    assert integ_b[-1] == 0.0
    assert ent_b[-1] == 0.0


def test_hysteresis_family_regression_goldens():
    assert np.allclose(
        _run("ts_hysteresis_state", _col([0.5, 1.2, 0.8, 0.3, -1.3, -0.5]), upper=1.0, lower=0.5),
        [0, 1, 1, 0, -1, -1],
    )
    assert np.allclose(
        _run("ts_state_entry_strength", _col([0.5, 1.2, 0.8, 0.3, -1.3]), upper=1.0, lower=0.5),
        [0.0, 0.2, 0.2, 0.0, -0.3],
        atol=1e-6,
    )


def test_hysteresis_invalid_missing_policy_fails_closed():
    with pytest.raises(ValueError):
        _run("ts_hysteresis_state", _col([1.0, 2.0]), upper=1.0, lower=0.5, missing_policy="UNKNOWN")
    with pytest.raises(ValueError):
        _run("ts_hysteresis_state", _col([1.0, 2.0]), upper=1.0, lower=0.5, missing_policy="BOGUS")


# ---------------------------------------------------------------------------
# #7 update-clock event-history state declarations
# ---------------------------------------------------------------------------

def test_update_clock_declared_event_count_history():
    for canon in (
        "update_path_efficiency",
        "update_acceleration",
        "update_surprise",
        "update_direction_persistence",
    ):
        ec = execution_contract(canon)
        assert ec.is_stateful, canon
        hr = history_requirement(canon)
        assert hr.is_event_clock, canon
        assert hr.kind == "event_count", canon
        assert hr.count == 5, canon  # the default ``n_updates``


# ---------------------------------------------------------------------------
# #44 update-clock NaN update_event censors the update chain
# ---------------------------------------------------------------------------

def test_update_clock_nan_event_censors_chain():
    # Real update nodes at rows 0 (10), 2 (12), 5 (18); row 4 is an UNKNOWN
    # observation day (NaN update_event).  With BREAK (default) the row-5 node
    # cannot look back past the unknown day, so the 3-update window would span a
    # censored boundary -> fail closed NaN.  With the explicit SKIP opt-in the
    # legacy transparent join is preserved.
    x = _col([10.0, 10.0, 12.0, 12.0, 15.0, 15.0, 18.0])
    ev = _col([1.0, 0.0, 1.0, 0.0, np.nan, 1.0, 0.0])

    brk = _run("update_path_efficiency", x, ev, n_updates=3)
    assert np.isnan(brk[5]) and np.isnan(brk[6])

    skip = _run("update_path_efficiency", x, ev, n_updates=3, missing_policy="SKIP")
    assert skip[6] == pytest.approx(1.0)


def test_update_clock_goldens():
    x1 = _col([10.0, 10.0, 10.0, 12.0, 12.0, 15.0, 15.0])
    ev1 = _col([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    assert _run("update_path_efficiency", x1, ev1, n_updates=3)[-1] == pytest.approx(1.0)

    x2 = _col([10.0, 12.0, 16.0, 21.0, 27.0, 27.0])
    ev2 = _col([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    assert _run("update_acceleration", x2, ev2, n_updates=5)[-1] == pytest.approx(1.0)
    assert _run("update_surprise", x2, ev2, n_updates=5)[-1] == pytest.approx(2.0 / 1.4826, rel=1e-3)
    assert _run("update_direction_persistence", x2, ev2, n_updates=5)[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# #45 update-acceleration baseline excludes the current delta
# ---------------------------------------------------------------------------

def test_update_acceleration_baseline_excludes_current_delta():
    # deltas [1, 2, 100].  The baseline MAD must use t-1 and earlier only:
    # MAD([1,2]) = 0.5 -> score (100-2)/0.5 = 196.  A self-inclusive scale
    # MAD([1,2,100]) = 1.0 would give 98 — the current delta would inflate its
    # own denominator.
    x = _col([10.0, 11.0, 13.0, 113.0])
    ev = _col([1.0, 1.0, 1.0, 1.0])
    out = _run("update_acceleration", x, ev, n_updates=4)
    assert out[-1] == pytest.approx(196.0)


# ---------------------------------------------------------------------------
# #8 / #46 report-filing event clock
# ---------------------------------------------------------------------------

def test_report_timing_declared_report_count_history():
    for canon in ("report_filing_delay_surprise", "report_revision_magnitude"):
        ec = execution_contract(canon)
        assert ec.is_stateful, canon
        hr = history_requirement(canon)
        assert hr.is_event_clock, canon
        assert hr.kind == "report_count", canon


def test_report_filing_delay_forward_filled_no_fabricated_event():
    """#46: a forward-filled panel has a finite value every day — the old code
    treated every finite day as a filing.  Now NO filing event exists, so the
    output is all NaN (fail closed), never a manufactured daily filing."""
    delay = _col([10.0] * 20)
    out = _run("report_filing_delay_surprise", delay)
    assert np.all(np.isnan(out))


def test_report_filing_delay_explicit_event_clock():
    """An explicit filing_event drives the report clock; output fires only on
    event rows and only once min_periods past events exist."""
    delay = _col([10.0, np.nan, 11.0, np.nan, 13.0, np.nan, 16.0])
    ev = _col([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    out = _run("report_filing_delay_surprise", delay, ev, window=8, min_periods=3)
    assert np.isnan(out[0])          # no past events
    assert np.isnan(out[1]) and np.isnan(out[2]) and np.isnan(out[3])
    assert np.isnan(out[4]) and np.isnan(out[5])
    assert np.isfinite(out[6])       # three past events (rows 0, 2, 4)


def test_report_filing_delay_sparse_vintage_fires():
    """#46 acceptable source: sparse vintage rows (finite after missing) are
    filing events even without an explicit event panel."""
    delay = _col([10.0, np.nan, 11.0, np.nan, 13.0, np.nan, 16.0])
    out = _run("report_filing_delay_surprise", delay, window=8, min_periods=3)
    assert np.isnan(out[0]) and np.isnan(out[1]) and np.isnan(out[2]) and np.isnan(out[3])
    assert np.isnan(out[4]) and np.isnan(out[5])
    assert np.isfinite(out[6])


# ---------------------------------------------------------------------------
# #47 / #49 revision event clock + same-period RevisionPair
# ---------------------------------------------------------------------------

def test_revision_fires_only_on_revision_event():
    """#47: ``prev_x`` finite is NOT a revision.  Only explicit revision_event
    rows fire, and only once min_periods past revision events exist."""
    x = _col([10.0, 10.0, 10.0, 12.0, 12.0, 15.0])
    prev_x = _col([9.0, 9.0, 9.0, 10.0, 10.0, 12.0])
    period = _str_panel(["2024Q1"] * 6)
    rev_ev = _col([0.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    out = _run("report_revision_magnitude", x, prev_x, period, period, rev_ev, window=8, min_periods=2)
    # Non-revision rows (0,1,3) stay NaN; rows 2/4 are revisions with too few
    # past events; row 5 has two past revisions and a same-period pair -> finite.
    assert np.isnan(out[0]) and np.isnan(out[1]) and np.isnan(out[2])
    assert np.isnan(out[3]) and np.isnan(out[4])
    assert np.isfinite(out[5])


def test_revision_same_period_pair_required():
    """#49: a revision whose period ids differ is NOT a valid RevisionPair and
    fails closed even when an explicit event fires."""
    x = _col([10.0, 10.0, 10.0, 12.0, 12.0, 15.0])
    prev_x = _col([9.0, 9.0, 9.0, 10.0, 10.0, 12.0])
    cur_period = _str_panel(["2024Q1"] * 6)
    # Row 5 is a revision event whose PREVIOUS vintage belongs to a different
    # ReportPeriod ("2024Q2") than the current one ("2024Q1") — not a valid
    # same-period RevisionPair -> fail closed NaN.
    prev_period = _str_panel(["2024Q1"] * 5 + ["2024Q2"])
    rev_ev = _col([0.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    out = _run("report_revision_magnitude", x, prev_x, cur_period, prev_period, rev_ev, window=8, min_periods=2)
    assert np.isnan(out[5])          # prev_period "2024Q2" != cur_period "2024Q1" at row 5


def test_revision_forward_filled_no_fabricated_revision():
    """#47/#49: x forward-filled (finite every day) with a finite prev_x — the
    only potential "revision" is the first vintage at row 0, which has no past
    events -> all NaN.  No revision is manufactured from finiteness."""
    x = _col([10.0] * 8)
    prev_x = _col([9.0] * 8)
    period = _str_panel(["2024Q1"] * 8)
    out = _run("report_revision_magnitude", x, prev_x, period, period, window=8, min_periods=2)
    assert np.all(np.isnan(out))
