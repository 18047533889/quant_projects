# -*- coding: utf-8 -*-
"""R11 P0 audit — streaming confirmed-pivot ledger (no retrospective rewrite).

The retrospective history-rewrite bug (findings #1/#2): the pivot chains in
``structural_levels._scan_confirmed_pivots`` and ``extrema_divergence._confirmed_extrema``
deleted a previously-confirmed pivot when a future more-extreme same-side
candidate appeared, and the factors then re-read that final (rewritten) list for
every historical row — so future data changed already-published past factor
values (future-function / PIT violation).

This suite verifies the shared ``StreamingConfirmedPivotLedger`` fix:

* prefix-invariance  ``factor(full_data)[:T] == factor(data[:T])`` for several T;
* a future higher-high / lower-low does not change past factor values;
* NaN inside a confirmation window blocks confirmation;
* no asymmetric end-window confirmation at the series start/end;
* ``PivotSuperseded`` effective_at semantics (active pivot set flips exactly at
  the new pivot's confirmation row).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import OperatorRegistry, load_all
from factor_engine.cleaned_operators.common._pivot_ledger import PEAK, confirmed_pivot_events


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module", autouse=True)
def _registry_loaded():
    load_all()
    yield


def _frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"A": np.asarray(values, dtype=float)})


def _price_series(n: int = 150, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    base = (
        100.0
        + 20.0 * np.sin(t / 6.0)
        + 5.0 * np.sin(t / 2.3)
        + 2.5 * np.sin(t / 11.0)
    )
    return np.maximum(base + rng.normal(0.0, 1.0, n), 1.0)


def _y_series(x: np.ndarray, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return x + rng.normal(0.0, 1.5, len(x))


def _calc(name: str, *frames: pd.DataFrame, **params: object) -> np.ndarray:
    op = OperatorRegistry.get(name, "pandas_numpy")
    assert op is not None, name
    out = op.calculate(*frames, **params)
    return np.asarray(out, dtype=float)


STRUCTURAL_CALLS: dict[str, dict[str, object]] = {
    "ts_structural_level_density": {
        "window": 40,
        "prominence": 0.02,
        "confirmation": 3,
        "bandwidth": 0.03,
    },
    "ts_nearest_structural_level_distance": {
        "window": 40,
        "prominence": 0.02,
        "confirmation": 3,
        "direction": "any",
    },
    "ts_structural_level_strength": {
        "window": 40,
        "prominence": 0.02,
        "confirmation": 3,
        "decay": 0.05,
    },
}

EXTREMA_CALLS: dict[str, dict[str, object]] = {
    "ts_extrema_divergence_strength": {
        "window": 30,
        "prominence": 0.02,
        "confirmation": 3,
        "match_lag": 4,
        "side": "peak",
    },
    "ts_extrema_confirmation_rate": {
        "window": 30,
        "prominence": 0.02,
        "confirmation": 3,
        "tolerance": 3,
        "side": "peak",
    },
}


# ---------------------------------------------------------------------------
# prefix invariance (property test)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [3, 11, 27])
def test_structural_levels_prefix_invariant_random_T(seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = _price_series(160, seed=seed)
    ts = sorted(int(v) for v in rng.integers(24, 150, size=7))
    ts = [25] + ts
    for name, params in STRUCTURAL_CALLS.items():
        out_full = _calc(name, _frame(x), **params)
        for t in ts:
            out_pref = _calc(name, _frame(x[:t]), **params)
            np.testing.assert_allclose(
                out_full[:t],
                out_pref,
                equal_nan=True,
                err_msg=f"{name} seed={seed} T={t}",
            )


@pytest.mark.parametrize("seed", [3, 11, 27])
def test_extrema_divergence_prefix_invariant_random_T(seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = _price_series(160, seed=seed)
    y = _y_series(x, seed=seed + 1)
    ts = sorted(int(v) for v in rng.integers(24, 150, size=7))
    ts = [25] + ts
    for name, params in EXTREMA_CALLS.items():
        out_full = _calc(name, _frame(x), _frame(y), **params)
        for t in ts:
            out_pref = _calc(name, _frame(x[:t]), _frame(y[:t]), **params)
            np.testing.assert_allclose(
                out_full[:t],
                out_pref,
                equal_nan=True,
                err_msg=f"{name} seed={seed} T={t}",
            )


# ---------------------------------------------------------------------------
# future higher-high / lower-low does not rewrite the past
# ---------------------------------------------------------------------------
def _higher_high_series() -> np.ndarray:
    """Peak P1 (bar 5, 110) confirmed at row 8; higher peak P2 (bar 12, 121)
    confirmed at row 15 with no intervening confirmed trough, so P2 supersedes
    P1 effective row 15."""
    return np.array(
        [
            100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 109.5, 109.0, 109.5,
            112.0, 115.0, 118.0, 121.0, 117.0, 113.0, 110.0, 108.0, 105.0,
            103.0, 101.0, 100.0,
        ],
        dtype=float,
    )


def _lower_low_series() -> np.ndarray:
    """Mirror of ``_higher_high_series``: trough T1 (bar 5, 110) superseded by
    deeper trough T2 (bar 12, 99) effective row 15."""
    return np.array(
        [
            120.0, 118.0, 116.0, 114.0, 112.0, 110.0, 110.5, 111.0, 110.5,
            108.0, 105.0, 102.0, 99.0, 103.0, 107.0, 110.0, 112.0, 115.0,
            117.0, 119.0, 120.0,
        ],
        dtype=float,
    )


@pytest.mark.parametrize(
    "series_factory, superseded_pivot, new_pivot",
    [
        (_higher_high_series, 5, 12),
        (_lower_low_series, 5, 12),
    ],
)
def test_future_more_extreme_does_not_change_past_factor_values(
    series_factory, superseded_pivot: int, new_pivot: int
) -> None:
    x = series_factory()
    conf = 3

    # ledger-level: exactly one supersession with correct effective_at
    res = confirmed_pivot_events(x, conf, 0.02, positive_only=True)
    assert len(res.superseded) == 1
    ss = res.superseded[0]
    assert ss.effective_at == 15
    assert ss.old_pivot.pivot_at == superseded_pivot
    assert ss.new_pivot.pivot_at == new_pivot
    # the superseded pivot stays active before effective_at and flips after
    before = res.active_at(ss.effective_at - 1)
    after = res.active_at(ss.effective_at)
    assert any(ev is ss.old_pivot for ev in before)
    assert not any(ev is ss.new_pivot for ev in before)
    assert any(ev is ss.new_pivot for ev in after)
    assert not any(ev is ss.old_pivot for ev in after)

    # factor-level: rows before effective_at are identical whether the future
    # more-extreme pivot exists or not.
    y = _y_series(x, seed=5)
    params_small = {**STRUCTURAL_CALLS["ts_structural_level_density"], "window": 20}
    for name, params in [
        ("ts_structural_level_density", params_small),
        ("ts_structural_level_strength", {**STRUCTURAL_CALLS["ts_structural_level_strength"], "window": 20}),
        ("ts_nearest_structural_level_distance", {**STRUCTURAL_CALLS["ts_nearest_structural_level_distance"], "window": 20}),
    ]:
        out_full = _calc(name, _frame(x), **params)
        out_cut = _calc(name, _frame(x[:15]), **params)
        np.testing.assert_allclose(
            out_full[:15],
            out_cut,
            equal_nan=True,
            err_msg=f"{name}: future more-extreme pivot rewrote rows < effective_at",
        )
    for name, params in EXTREMA_CALLS.items():
        small_params = {**params, "window": 20}
        out_full = _calc(name, _frame(x), _frame(y), **small_params)
        out_cut = _calc(name, _frame(x[:15]), _frame(y[:15]), **small_params)
        np.testing.assert_allclose(
            out_full[:15],
            out_cut,
            equal_nan=True,
            err_msg=f"{name}: future more-extreme pivot rewrote rows < effective_at",
        )


# ---------------------------------------------------------------------------
# confirmation contract
# ---------------------------------------------------------------------------
def test_nan_inside_confirmation_window_blocks_confirmation() -> None:
    conf = 2
    # bar 4 (10.0) is a strict local max over its finite neighbours, but the
    # full confirmation window [2, 6] contains NaN at bar 6 -> must NOT confirm.
    x = np.array([5.0, 6.0, 7.0, 8.0, 10.0, 9.0, np.nan, 8.0, 7.0])
    res = confirmed_pivot_events(x, conf, 0.0, positive_only=True)
    assert all(ev.pivot_at != 4 for ev in res.events)
    assert all(ev.pivot_at != 4 for ev in res.active_at(len(x) - 1))

    # the same shape without the NaN confirms bar 4 at row 6.
    x_clean = np.array([5.0, 6.0, 7.0, 8.0, 10.0, 9.0, 8.0, 8.0, 7.0])
    res_clean = confirmed_pivot_events(x_clean, conf, 0.0, positive_only=True)
    assert any(ev.pivot_at == 4 and ev.side == PEAK for ev in res_clean.events)


def test_nan_in_window_blocks_operator_output_change() -> None:
    # A NaN inside the confirmation window must prevent a pivot from being used
    # at any later row (fail-closed, not silently re-detected with a shrunken
    # window).
    x = np.array([5.0, 6.0, 7.0, 8.0, 10.0, 9.0, np.nan, 8.0, 7.0])
    res = confirmed_pivot_events(x, 2, 0.0, positive_only=True)
    assert len(res.events) == 0


def test_no_asymmetric_end_window_confirmation() -> None:
    conf = 3
    # Bar 0 (50) is a "peak" under an asymmetric left window and bar 7 (60) a
    # "peak" under an asymmetric right window, but the full ±confirmation
    # neighbourhood is unavailable for both, so NO pivot may be confirmed.
    x = np.array([50.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 60.0])
    res = confirmed_pivot_events(x, conf, 0.0, positive_only=True)
    assert len(res.events) == 0
    # … and through the operators (a start/end pivot must not leak into output).
    out = _calc("ts_structural_level_density", _frame(x), **STRUCTURAL_CALLS["ts_structural_level_density"])
    assert np.all(np.isnan(out))


def test_pivot_superseded_effective_at_semantics() -> None:
    x = _higher_high_series()
    res = confirmed_pivot_events(x, 3, 0.02, positive_only=True)
    assert len(res.events) == 2
    p1, p2 = res.events
    assert p1.pivot_at == 5 and p1.side == PEAK and p1.confirmed_at == 8
    assert p2.pivot_at == 12 and p2.side == PEAK and p2.confirmed_at == 15
    assert p2.supersedes is p1
    assert len(res.superseded) == 1
    ss = res.superseded[0]
    assert ss.old_pivot is p1
    assert ss.new_pivot is p2
    assert ss.effective_at == p2.confirmed_at
    # supersession only affects rows >= effective_at
    assert p1 in res.active_at(10)
    assert p1 not in res.active_at(15)
    assert p2 in res.active_at(15)


def test_ledger_prefix_invariance_direct() -> None:
    """The raw ledger itself is prefix-invariant: scanning data[:T] must give the
    same events/active sets that scanning the full series yields for rows < T."""
    x = _price_series(200, seed=5)
    full = confirmed_pivot_events(x, 3, 0.02, positive_only=True)
    for t in [30, 60, 100, 150, 190]:
        pref = confirmed_pivot_events(x[:t], 3, 0.02, positive_only=True)
        full_events = [(e.pivot_at, e.side, round(e.value, 10)) for e in full.active_at(t - 1)]
        pref_events = [(e.pivot_at, e.side, round(e.value, 10)) for e in pref.active_at(t - 1)]
        assert full_events == pref_events, f"active pivot sets differ at T={t}"
