from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.event_response import (
    EventHistoricalResponseMean,
    EventHistoricalResponseSignBalance,
    EventResponseDispersion,
    EventResponseEffectiveEvents,
    EventResponseOverlapRatio,
)
from factor_engine.runtime.execution_contract import history_requirement


def _frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"A": values})


def test_episode_window_anchor_public_calculate() -> None:
    event = np.zeros(13)
    event[[4, 5, 8, 11]] = 1.0
    response = np.zeros(13)
    response[[6, 9, 12]] = [100.0, 2.0, 4.0]
    ef, rf = _frame(event), _frame(response)

    effective = EventResponseEffectiveEvents().calculate(
        ef, history_window=7, horizon=1, refractory=3
    )
    overlap = EventResponseOverlapRatio().calculate(
        ef, history_window=7, horizon=1, refractory=3
    )
    mean = EventHistoricalResponseMean().calculate(
        rf, ef, history_window=7, horizon=1, mode="sum", min_events=1,
        refractory=3,
    )
    dispersion = EventResponseDispersion().calculate(
        rf, ef, history_window=7, horizon=1, min_events=2, refractory=3
    )
    sign_balance = EventHistoricalResponseSignBalance().calculate(
        rf, ef, history_window=7, horizon=1, min_events=1, refractory=3
    )

    # At t=12, lo=5. Event 5 continues the episode begun at 4 and must not be
    # reinvented merely because event 4 is just outside the rolling cohort.
    assert effective.iloc[12, 0] == 2.0
    assert overlap.iloc[12, 0] == pytest.approx(1.0 - 2.0 / 3.0)
    assert mean.iloc[12, 0] == 3.0
    assert sign_balance.iloc[12, 0] == 1.0
    assert dispersion.iloc[12, 0] == pytest.approx(
        np.std([2.0, 4.0], ddof=1) / 3.0
    )


def test_episode_boundary_control() -> None:
    event = np.zeros(10)
    event[[2, 5]] = 1.0
    ef = _frame(event)
    effective = EventResponseEffectiveEvents().calculate(
        ef, history_window=4, horizon=1, refractory=3
    )
    overlap = EventResponseOverlapRatio().calculate(
        ef, history_window=4, horizon=1, refractory=3
    )
    # A gap exactly equal to refractory begins a new episode.
    assert effective.iloc[9, 0] == 1.0
    assert overlap.iloc[9, 0] == 0.0


def test_long_episode_crossing_window_boundary() -> None:
    event = np.zeros(24)
    event[2:21] = 1.0
    ef = _frame(event)
    effective = EventResponseEffectiveEvents().calculate(
        ef, history_window=5, horizon=1, refractory=2
    )
    overlap = EventResponseOverlapRatio().calculate(
        ef, history_window=5, horizon=1, refractory=2
    )
    # The sole episode began long before lo=15; none of its continuation events
    # in [15, 19] may become a fresh anchor.
    assert effective.iloc[20, 0] == 0.0
    assert overlap.iloc[20, 0] == 1.0


def test_episode_no_refractory_control() -> None:
    event = np.zeros(10)
    event[[3, 4, 5]] = 1.0
    ef = _frame(event)
    effective = EventResponseEffectiveEvents().calculate(
        ef, history_window=6, horizon=1, refractory=0
    )
    overlap = EventResponseOverlapRatio().calculate(
        ef, history_window=6, horizon=1, refractory=0
    )
    assert effective.iloc[9, 0] == 3.0
    assert overlap.iloc[9, 0] == 0.0


def test_missing_predecessor_coverage_is_not_known_zero() -> None:
    known = np.zeros(10)
    known[5] = 1.0
    unknown = known.copy()
    unknown[4] = np.nan
    response = np.zeros(10)
    response[6] = 7.0

    known_eff = EventResponseEffectiveEvents().calculate(
        _frame(known), history_window=4, horizon=1, refractory=3
    )
    unknown_eff = EventResponseEffectiveEvents().calculate(
        _frame(unknown), history_window=4, horizon=1, refractory=3
    )
    known_mean = EventHistoricalResponseMean().calculate(
        _frame(response), _frame(known), history_window=4, horizon=1,
        min_events=1, refractory=3,
    )
    unknown_mean = EventHistoricalResponseMean().calculate(
        _frame(response), _frame(unknown), history_window=4, horizon=1,
        min_events=1, refractory=3,
    )

    assert known_eff.iloc[9, 0] == 1.0
    # The response kernel refuses to admit the unproven anchor; the diagnostic
    # cannot report a confident count when that anchor's status is unknown.
    assert np.isnan(unknown_eff.iloc[9, 0])
    assert known_mean.iloc[9, 0] == 7.0
    assert np.isnan(unknown_mean.iloc[9, 0])


@pytest.mark.parametrize("refractory", [0, 3])
def test_unknown_raw_cohort_keeps_diagnostics_unknown(refractory: int) -> None:
    known = np.zeros(10)
    known[5] = 1.0
    unknown = known.copy()
    unknown[7] = np.nan  # inside t=9 cohort [5, 8]
    for operator in (EventResponseEffectiveEvents, EventResponseOverlapRatio):
        known_out = operator().calculate(
            _frame(known), history_window=4, horizon=1,
            refractory=refractory,
        )
        unknown_out = operator().calculate(
            _frame(unknown), history_window=4, horizon=1,
            refractory=refractory,
        )
        assert np.isfinite(known_out.iloc[9, 0])
        assert np.isnan(unknown_out.iloc[9, 0])


def test_old_missing_coverage_does_not_poison_later_episode() -> None:
    event = np.zeros(16)
    event[2] = np.nan
    event[10] = 1.0
    effective = EventResponseEffectiveEvents().calculate(
        _frame(event), history_window=5, horizon=1, refractory=3
    )
    # Only the local predecessor span [8, 9] determines event 10's status.
    assert effective.iloc[15, 0] == 1.0


def test_episode_markers_are_prefix_causal() -> None:
    base = np.zeros(16)
    base[[2, 3, 7]] = 1.0
    changed_future = base.copy()
    changed_future[[12, 13]] = 1.0
    op = EventResponseEffectiveEvents()
    lhs = op.calculate(_frame(base), history_window=8, horizon=2, refractory=4)
    rhs = op.calculate(
        _frame(changed_future), history_window=8, horizon=2, refractory=4
    )
    pd.testing.assert_frame_equal(lhs.iloc[:12], rhs.iloc[:12])


def test_finite_history_prefix_reproduces_last_row_public_calculate() -> None:
    rng = np.random.default_rng(43)
    event = np.zeros(80)
    event[[65, 68, 74]] = 1.0
    response = rng.normal(size=80)
    history_window, horizon, refractory = 12, 3, 5
    params = {
        "history_window": history_window,
        "horizon": horizon,
        "refractory": refractory,
        "min_events": 1,
    }
    required = history_requirement("event_historical_response_mean", params)
    assert required.kind == "finite"
    assert required.rows == history_window - 1 + horizon + refractory - 1

    op = EventHistoricalResponseMean()
    full = op.calculate(_frame(response), _frame(event), **params)
    target = len(event) - 1
    start = target - required.rows
    chunk = op.calculate(_frame(response[start:]), _frame(event[start:]), **params)
    assert np.isfinite(full.iloc[-1, 0])
    assert np.isfinite(chunk.iloc[-1, 0])
    assert chunk.iloc[-1, 0] == pytest.approx(full.iloc[-1, 0], nan_ok=True)


def test_long_episode_finite_prefix_chunk_equivalence() -> None:
    event = np.zeros(60)
    event[5:56] = 1.0
    params = {"history_window": 10, "horizon": 2, "refractory": 3}
    required = history_requirement("event_response_overlap_ratio", params)
    target = 59
    start = target - required.rows
    for operator in (EventResponseEffectiveEvents, EventResponseOverlapRatio):
        full = operator().calculate(_frame(event), **params)
        chunk = operator().calculate(_frame(event[start:]), **params)
        assert np.isfinite(full.iloc[-1, 0])
        assert np.isfinite(chunk.iloc[-1, 0])
        assert chunk.iloc[-1, 0] == full.iloc[-1, 0]


@pytest.mark.parametrize(
    ("canonical", "expected_default_rows"),
    [
        ("event_historical_response_mean", 124),
        ("event_historical_response_sign_balance", 124),
        ("event_response_dispersion", 129),
        ("event_response_effective_events", 128),
        ("event_response_overlap_ratio", 128),
    ],
)
def test_history_authority_defaults_and_explicit_refractory(
    canonical: str, expected_default_rows: int
) -> None:
    default = history_requirement(canonical, {})
    assert default.kind == "finite"
    assert default.rows == expected_default_rows

    explicit = history_requirement(
        canonical, {"history_window": 7, "horizon": 2, "refractory": 3}
    )
    assert explicit.kind == "finite"
    assert explicit.rows == 10  # (7 - 1) + 2 + (3 - 1)


def test_diagnostic_none_and_zero_refractory_history() -> None:
    for canonical in (
        "event_response_effective_events", "event_response_overlap_ratio"
    ):
        implicit = history_requirement(
            canonical, {"history_window": 7, "horizon": 2}
        )
        explicit_none = history_requirement(
            canonical,
            {"history_window": 7, "horizon": 2, "refractory": None},
        )
        zero = history_requirement(
            canonical,
            {"history_window": 7, "horizon": 2, "refractory": 0},
        )
        assert implicit.rows == explicit_none.rows == 9
        assert zero.rows == 8
