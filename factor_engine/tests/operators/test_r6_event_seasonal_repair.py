"""Final-registry execution checks for the R6 event/seasonal repair batch."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry

load_all()

INTRA = (
    "intra_event_window_reduce intra_event_pre_post_contrast intra_impulse_event_detector "
    "intra_post_impulse_response intra_probe_outcome_score intra_supply_absorption_score "
    "intra_consolidation_quality intra_response_curve_features intra_liquidity_resilience_curve_fit"
).split()
ERD = (
    "erd_half_life_decay_count erd_recency_decay erd_event_rate_decay_slope erd_marked_event_decay "
    "erd_sign_consistent_decay erd_cross_events_spacing erd_post_event_hazard "
    "erd_event_window_return_gradient erd_event_response_amplitude erd_burst_duration"
).split()
SEASONAL = (
    "cs1_weekday_lag_ratio cs1_weekday_lag_zscore cs1_monthly_lag_ratio "
    "cs1_seasonal_relative_rank cs1_weekday_effect_strength cs1_weekday_anomaly "
    "cs1_week_cycle_phase cs1_weekly_harmonic_power cs1_seasonal_residual_smoothness"
).split()


def _data():
    idx = []
    for offset in range(2):
        day = pd.Timestamp("2024-01-03") + pd.Timedelta(days=offset)
        idx.extend(day + pd.Timedelta(minutes=m) for m in (*range(571, 691), *range(781, 901)))
    n = len(idx)
    r = 0.0004 * np.sin(np.arange(n) / 4.0)
    r[[50, 90, 290, 330]] = 0.04
    price = pd.DataFrame({"A": 100 * np.exp(np.cumsum(r))}, index=pd.DatetimeIndex(idx))
    volume = pd.DataFrame({"A": 20 + 5 * np.sin(np.arange(n) / 8.0) ** 2}, index=price.index)
    for e in (50, 90, 290, 330):
        volume.iloc[e, 0] = 200
        volume.iloc[e + 1:e + 31, 0] = 20 + 180 * np.exp(-np.arange(30) / 5)
    amount = price * volume
    mask = pd.DataFrame(0.0, index=price.index, columns=price.columns)
    mask.iloc[[50, 290], 0] = 1.0
    ix = pd.date_range("2024-01-01", periods=100)
    x = pd.DataFrame({"A": 100 + np.arange(100) * .1 + np.sin(np.arange(100) * 2 * np.pi / 5)}, index=ix)
    event = pd.DataFrame({"A": ((np.arange(100) % 7) < 2).astype(float)}, index=ix)
    mark = pd.DataFrame({"A": np.where(event.A > 0, np.sin(np.arange(100)) + 1, 0)}, index=ix)
    ret = x.pct_change().fillna(0)
    return price, volume, amount, mask, x, event, mark, ret


def _calls():
    price, volume, amount, mask, x, event, mark, ret = _data()
    return {
        "intra_event_window_reduce": (price, mask, 5, 15, "mean", "first", 3),
        "intra_event_pre_post_contrast": (price, mask, 10, 10, "mean_diff", "first", 5),
        "intra_impulse_event_detector": (price, volume, "up", "robust_z", 3.0, 1, 2, "count"),
        "intra_post_impulse_response": (price, volume, amount, "up", "robust_z", 3.0, 30, "retention"),
        "intra_probe_outcome_score": (price, volume, amount, "up", 3.0, 10, 30, "failure"),
        "intra_supply_absorption_score": (price, volume, amount, "all", 30, "absorption"),
        "intra_consolidation_quality": (price, volume, amount, "impulse", 3.0, 30, "tightness"),
        "intra_response_curve_features": (price, volume, "impulse", 30, "price", "slope"),
        "intra_liquidity_resilience_curve_fit": (price, volume, 2.5, 30, "half_life"),
        "erd_half_life_decay_count": (event, 10.0, 2),
        "erd_recency_decay": (event, 10.0),
        "erd_event_rate_decay_slope": (event, 40, 4),
        "erd_marked_event_decay": (mark, 10.0, 2),
        "erd_sign_consistent_decay": (mark, 1, 10.0, 2),
        "erd_cross_events_spacing": (event, 60, 3),
        "erd_post_event_hazard": (event, 60, 5, 4),
        "erd_event_window_return_gradient": (ret, event, 10, 10, 2),
        "erd_event_response_amplitude": (mark, 10.0, 2),
        "erd_burst_duration": (event,),
        "cs1_weekday_lag_ratio": (x, 5, 1),
        "cs1_weekday_lag_zscore": (x, 5, 3),
        "cs1_monthly_lag_ratio": (x, 20, 1),
        "cs1_seasonal_relative_rank": (x, 5, 3),
        "cs1_weekday_effect_strength": (x, 5, 40, 8),
        "cs1_weekday_anomaly": (x, 5, 2),
        "cs1_week_cycle_phase": (x, 5, 6),
        "cs1_weekly_harmonic_power": (x, 5, 6),
        "cs1_seasonal_residual_smoothness": (x, 5, 40, 8),
    }


def test_all_28_finalized_canonicals_finite_causal_and_positional_keyword_equivalent():
    calls = _calls()
    assert set(calls) == set(INTRA + ERD + SEASONAL)
    for name, args in calls.items():
        op = OperatorRegistry.get(name)
        kwargs = dict(zip(op.metadata.param_names, args))
        positional = op.calculate(*args)
        keyword = op.calculate(**kwargs)
        pd.testing.assert_frame_equal(positional, keyword, check_dtype=False, obj=name)
        assert np.isfinite(positional.to_numpy()).any(), name
        if name in INTRA:
            assert positional.shape == (2, 1)
            prefix_args = tuple(a.iloc[:240] if isinstance(a, pd.DataFrame) else a for a in args)
            expected = positional.iloc[:1]
        else:
            assert positional.shape == (100, 1)
            prefix_args = tuple(a.iloc[:70] if isinstance(a, pd.DataFrame) else a for a in args)
            expected = positional.iloc[:70]
        pd.testing.assert_frame_equal(op.calculate(*prefix_args), expected, check_dtype=False, obj=name)


def test_every_scalar_has_a_bounded_defaulted_contract():
    panel_names = {"x", "event_mask", "price", "volume", "amount", "activity", "event", "mark", "ret"}
    # `event` is scalar only for the two intra detectors.
    for name in INTRA + ERD + SEASONAL:
        op = OperatorRegistry.get(name)
        panels = panel_names - ({"event"} if name in {"intra_impulse_event_detector", "intra_supply_absorption_score"} else set())
        scalars = set(op.metadata.param_names) - panels
        assert scalars == set(op.metadata.param_specs), name
        for spec in op.metadata.param_specs.values():
            assert spec.default is not None
            assert spec.min is not None or spec.choices is not None


@pytest.mark.parametrize("name,args", [
    ("intra_event_window_reduce", (None, None, -1, 15, "mean", "first", 3)),
    ("intra_liquidity_resilience_curve_fit", (None, None, float("nan"), 30, "half_life")),
    ("erd_half_life_decay_count", (None, float("nan"), 2)),
    ("erd_sign_consistent_decay", (None, 0, 10.0, 2)),
    ("cs1_weekday_lag_zscore", (None, 1, 3)),
    ("cs1_weekday_effect_strength", (None, 5, 2, 8)),
])
def test_invalid_boundaries_fail_before_kernel(name, args):
    with pytest.raises((TypeError, ValueError)):
        OperatorRegistry.get(name).calculate(*args)
