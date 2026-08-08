# -*- coding: utf-8 -*-
"""Regression tests for the 2026-08 final pack (61 atomics).

Covers:
- Registration: every final-pack canonical has a pandas_numpy runtime.
- Surface: every final-pack canonical classifies ``daily`` (migrated) and
  remains a member of EXTENDED_ONLY (static partition contract).
- Policy: every final-pack canonical has an explicit PIT-safe policy with the
  intended scope (ts / cs / group / session_intraday).
- Semantic fail-closed: constant windows / short samples return NaN for a
  representative sample of the families, never an invented zero.
- Determinism + axes for a representative sample across all six groups.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.operator_policy import infer_operator_policy
from cleaned_operators.operator_surface import classify_canonical
import cleaned_operators.operator_surface as _surface_mod

ensure_cleaned_loaded()

FINAL_PACK = frozenset({
    # group 1 — robust tail
    "ts_lower_partial_moment", "ts_upper_partial_moment", "ts_expected_shortfall",
    "ts_quantile_skew", "ts_quantile_kurtosis", "ts_tail_ratio", "ts_extreme_cluster_ratio",
    # group 2 — nonlinear dependence
    "ts_distance_corr", "ts_distance_cov", "ts_mutual_information",
    "ts_lagged_mutual_information", "ts_upper_tail_dependence", "ts_lower_tail_dependence",
    # group 3 — complexity / long memory
    "ts_permutation_entropy", "ts_weighted_permutation_entropy",
    "ts_permutation_transition_entropy", "ts_sample_entropy", "ts_hurst_dfa",
    "ts_higuchi_fractal_dimension", "ts_variogram_slope", "ts_autocorr_decay_half_life",
    # group 4 — A-share limit/suspension state machine
    "ashare_limit_up_streak", "ashare_limit_down_streak", "ashare_days_since_limit_up",
    "ashare_days_since_limit_down", "ashare_limit_touch_count", "ashare_failed_limit_count",
    "ashare_one_price_limit_streak", "ashare_limit_event_density", "ashare_limit_asymmetry",
    "ashare_suspension_episode_length", "ashare_limit_open_up_streak",
    "ashare_limit_open_down_streak", "ashare_limit_up_volume_ratio",
    "ashare_limit_down_volume_ratio",
    # group 5 — relation / group distribution
    "relation_topk_concentration", "relation_distribution_skew", "relation_distribution_kurtosis",
    "relation_hhi_change", "relation_entropy_change", "relation_concentration_acceleration",
    "relation_rank_mobility", "relation_share_mobility", "group_skewness", "group_kurtosis",
    "group_quantile_spread", "group_tail_ratio",
    # group 6 — intraday time-structure v2
    "intra_bar_range_persistence", "intra_bar_range_deviation", "intra_tail_volume_share",
    "intra_volume_price_alignment", "intra_ute_high", "intra_ute_low",
    "intra_slot_volume_surprise", "intra_slot_amount_surprise", "intra_slot_volatility_surprise",
    "intra_market_lead_lag_ex_self", "intra_industry_lead_lag_ex_self",
    "intra_session_return_asymmetry", "intra_close_participation", "intra_high_low_affinity",
})

EXPECTED_SCOPE = {
    "ts_lower_partial_moment": "ts", "ts_upper_partial_moment": "ts",
    "ts_expected_shortfall": "ts", "ts_quantile_skew": "ts",
    "ts_quantile_kurtosis": "ts", "ts_tail_ratio": "ts", "ts_extreme_cluster_ratio": "ts",
    "ts_distance_corr": "ts", "ts_distance_cov": "ts", "ts_mutual_information": "ts",
    "ts_lagged_mutual_information": "ts", "ts_upper_tail_dependence": "ts",
    "ts_lower_tail_dependence": "ts",
    "ts_permutation_entropy": "ts", "ts_weighted_permutation_entropy": "ts",
    "ts_permutation_transition_entropy": "ts", "ts_sample_entropy": "ts",
    "ts_hurst_dfa": "ts", "ts_higuchi_fractal_dimension": "ts",
    "ts_variogram_slope": "ts", "ts_autocorr_decay_half_life": "ts",
    "ashare_limit_up_streak": "ts", "ashare_limit_down_streak": "ts",
    "ashare_days_since_limit_up": "ts", "ashare_days_since_limit_down": "ts",
    "ashare_limit_touch_count": "ts", "ashare_failed_limit_count": "ts",
    "ashare_one_price_limit_streak": "ts", "ashare_limit_event_density": "ts",
    "ashare_limit_asymmetry": "ts", "ashare_suspension_episode_length": "ts",
    "ashare_limit_open_up_streak": "ts", "ashare_limit_open_down_streak": "ts",
    "ashare_limit_up_volume_ratio": "ts", "ashare_limit_down_volume_ratio": "ts",
    "relation_topk_concentration": "cs", "relation_distribution_skew": "cs",
    "relation_distribution_kurtosis": "cs", "relation_hhi_change": "cs",
    "relation_entropy_change": "cs", "relation_concentration_acceleration": "cs",
    "relation_rank_mobility": "cs", "relation_share_mobility": "cs",
    "group_skewness": "group", "group_kurtosis": "group",
    "group_quantile_spread": "group", "group_tail_ratio": "group",
    "intra_bar_range_persistence": "session_intraday",
    "intra_bar_range_deviation": "session_intraday",
    "intra_tail_volume_share": "session_intraday",
    "intra_volume_price_alignment": "session_intraday",
    "intra_ute_high": "session_intraday", "intra_ute_low": "session_intraday",
    "intra_slot_volume_surprise": "session_intraday",
    "intra_slot_amount_surprise": "session_intraday",
    "intra_slot_volatility_surprise": "session_intraday",
    "intra_market_lead_lag_ex_self": "session_intraday",
    "intra_industry_lead_lag_ex_self": "session_intraday",
    "intra_session_return_asymmetry": "session_intraday",
    "intra_close_participation": "session_intraday",
    "intra_high_low_affinity": "session_intraday",
}


def _daily_panels(n: int = 120, cols: int = 4):
    dates = pd.bdate_range("2024-01-02", periods=n)
    assets = [f"S{i}" for i in range(cols)]
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, (n, cols))
    close = pd.DataFrame(np.exp(np.cumsum(r, axis=0)) * 10.0, index=dates, columns=assets)
    high = close * (1 + np.abs(rng.normal(0, 0.004, (n, cols))))
    low = close * (1 - np.abs(rng.normal(0, 0.004, (n, cols))))
    volume = pd.DataFrame(rng.lognormal(0, 0.3, (n, cols)) * 1e5, index=dates, columns=assets)
    return close, high, low, volume


def _minute_panels(days: int = 8, cols: int = 3):
    dates = pd.bdate_range("2024-01-02", periods=days)
    assets = [f"M{i}" for i in range(cols)]
    minutes = [570 + m for m in range(240)]
    idx = pd.DatetimeIndex(
        [d + pd.Timedelta(minutes=int(m)) for d in dates for m in minutes]
    )
    rng = np.random.default_rng(5)
    out = {}
    for inst in assets:
        r = rng.normal(0, 0.001, len(idx))
        close = np.exp(np.cumsum(r)) * 20.0
        vol = rng.lognormal(0, 0.3, len(idx)) * 1e4
        out[inst] = {
            "close": close, "volume": vol, "amount": vol * close,
            "high": close * (1 + np.abs(rng.normal(0, 0.0008, len(idx)))),
            "low": close * (1 - np.abs(rng.normal(0, 0.0008, len(idx)))),
        }
    panels = {k: pd.DataFrame({i: out[i][k] for i in assets}, index=idx)
              for k in ("close", "volume", "amount", "high", "low")}
    panels["industry"] = pd.DataFrame(
        {assets[0]: 1.0, assets[1]: 1.0, assets[2]: 2.0}, index=dates
    )
    return panels


def test_final_pack_all_registered_and_daily():
    missing = [c for c in FINAL_PACK if OperatorRegistry.get(c, "pandas_numpy") is None]
    assert not missing, f"missing runtimes: {missing}"
    not_daily = [c for c in sorted(FINAL_PACK) if classify_canonical(c) != "daily"]
    assert not not_daily, f"not daily surface: {not_daily}"
    # Read the live module attribute AFTER load_all (modules union into it at
    # import time; importing the name at module top would capture the empty set).
    live_extended = frozenset(_surface_mod.EXTENDED_ONLY_CANONICALS)
    not_extended = sorted(FINAL_PACK - live_extended)
    assert not not_extended, f"not in EXTENDED_ONLY (partition contract): {not_extended}"


def test_final_pack_policies_pit_safe_with_intended_scope():
    bad = []
    for canonical in sorted(FINAL_PACK):
        op = OperatorRegistry.get(canonical, "pandas_numpy")
        policy = infer_operator_policy(op, canonical=canonical)
        if not policy.pit_safe:
            bad.append(f"{canonical}: pit_safe=False")
        if policy.scope != EXPECTED_SCOPE[canonical]:
            bad.append(f"{canonical}: scope={policy.scope} expected={EXPECTED_SCOPE[canonical]}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("canonical", sorted(FINAL_PACK))
def test_final_pack_deterministic_and_axes(canonical):
    close, high, low, volume = _daily_panels()
    minute = _minute_panels()
    op = OperatorRegistry.get(canonical, "pandas_numpy")
    kw = {}
    if canonical.startswith("intra_"):
        m = minute
        panels = {
            "intra_bar_range_persistence": (m["high"], m["low"]),
            "intra_bar_range_deviation": (m["high"], m["low"]),
            "intra_tail_volume_share": (m["close"], m["volume"]),
            "intra_volume_price_alignment": (m["close"], m["volume"]),
            "intra_ute_high": (m["close"],),
            "intra_ute_low": (m["close"],),
            "intra_slot_volume_surprise": (m["volume"],),
            "intra_slot_amount_surprise": (m["amount"],),
            "intra_slot_volatility_surprise": (m["high"], m["low"]),
            "intra_market_lead_lag_ex_self": (m["close"],),
            "intra_industry_lead_lag_ex_self": (m["close"], m["industry"]),
            "intra_session_return_asymmetry": (m["close"],),
            "intra_close_participation": (m["volume"],),
            "intra_high_low_affinity": (m["high"], m["low"]),
        }[canonical]
        # Minute kernels emit a daily panel (one scalar per TradeDate), so the
        # axes template is the daily index derived from the minute grid.
        out_dates = pd.DatetimeIndex(sorted(set(m["close"].index.normalize())))
        template = pd.DataFrame(index=out_dates, columns=m["close"].columns, dtype=float)
    elif canonical.startswith("ashare_"):
        panels = _ashare_panels(close, high, low, volume)[canonical]
        template = close
        kw = {"tick_tolerance": 0.005, "window": 20}
    elif canonical in {"group_skewness", "group_kurtosis", "group_quantile_spread", "group_tail_ratio"}:
        group = pd.DataFrame(
            np.tile(["g0", "g1"], (close.shape[0], 2)),
            index=close.index, columns=close.columns,
        )
        panels = (close, group)
        template = close
    elif canonical.startswith("relation_"):
        template = close
        if canonical in {"relation_hhi_change", "relation_entropy_change",
                         "relation_concentration_acceleration"}:
            panels = (close,)
            kw = {"window": 5}
        elif canonical == "relation_distribution_skew":
            panels = tuple(close for _ in range(3))
            kw = {}
        elif canonical == "relation_distribution_kurtosis":
            panels = tuple(close for _ in range(4))
            kw = {}
        elif canonical == "relation_topk_concentration":
            panels = tuple(close for _ in range(3))
            kw = {"k": 3}
        else:  # rank/share mobility
            panels = tuple(close for _ in range(3))
            kw = {"window": 5}
    else:
        kw = {}
        if canonical in {
            "ts_distance_corr", "ts_distance_cov", "ts_mutual_information",
            "ts_lagged_mutual_information", "ts_upper_tail_dependence", "ts_lower_tail_dependence",
        }:
            panels = (close, low)
            kw = {"window": 40}
        elif canonical == "ts_lagged_mutual_information":
            panels = (close, low)
            kw = {"window": 40, "lag": 1}
        elif canonical in {"ts_permutation_entropy", "ts_weighted_permutation_entropy", "ts_permutation_transition_entropy"}:
            panels = (close,)
            kw = {"window": 30, "order": 3}
        elif canonical in {"ts_sample_entropy", "ts_hurst_dfa", "ts_higuchi_fractal_dimension", "ts_variogram_slope"}:
            panels = (close,)
            kw = {"window": 60}
        else:
            panels = (close,)
            kw = {"window": 60}
        template = close

    first = op.calculate(*panels, **kw)
    second = op.calculate(*panels, **kw)
    assert first.index.equals(template.index), f"{canonical}: index changed"
    assert first.columns.equals(template.columns), f"{canonical}: columns changed"
    assert np.allclose(
        first.fillna(-1).to_numpy(),
        second.fillna(-1).to_numpy(),
        equal_nan=True,
    ), f"{canonical}: non-deterministic"


def _ashare_panels(close, high, low, volume):
    open_ = close * (1 + 0.001 * np.arange(close.shape[0])[:, None] / 100.0)
    limit_up = close * 1.10
    limit_down = close * 0.90
    valid = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    up_event = pd.DataFrame(
        (high.to_numpy() >= limit_up.to_numpy() * 0.995).astype(float),
        index=close.index, columns=close.columns,
    )
    down_event = pd.DataFrame(
        (low.to_numpy() <= limit_down.to_numpy() * 1.005).astype(float),
        index=close.index, columns=close.columns,
    )
    return {
        "ashare_limit_up_streak": (close, limit_up, valid),
        "ashare_limit_down_streak": (close, limit_down, valid),
        "ashare_days_since_limit_up": (up_event,),
        "ashare_days_since_limit_down": (down_event,),
        "ashare_limit_touch_count": (high, low, limit_up, limit_down),
        "ashare_failed_limit_count": (high, low, close, limit_up, limit_down),
        "ashare_one_price_limit_streak": (open_, high, low, close, limit_up, limit_down, valid),
        "ashare_limit_event_density": (up_event,),
        "ashare_limit_asymmetry": (up_event, down_event),
        "ashare_suspension_episode_length": (valid,),
        "ashare_limit_open_up_streak": (high, limit_up, valid),
        "ashare_limit_open_down_streak": (low, limit_down, valid),
        "ashare_limit_up_volume_ratio": (volume, up_event),
        "ashare_limit_down_volume_ratio": (volume, down_event),
    }


def test_constant_window_returns_nan():
    """Representative fail-closed behaviour: degenerate ratio/correlation inputs -> NaN, never 0."""
    n = 60
    dates = pd.bdate_range("2024-01-02", periods=n)
    const = pd.DataFrame(np.ones((n, 2)), index=dates, columns=["A", "B"])
    const_high = const * 1.01
    const_low = const * 0.99
    cases = [
        ("ts_tail_ratio", (const,), {"window": 20}),
        ("ts_distance_corr", (const, const_high), {"window": 20}),
        ("ts_hurst_dfa", (const,), {"window": 20}),
        # Note: ts_permutation_entropy of a constant series is 0 (a single
        # permutation state), which is mathematically correct orderedness and is
        # intentionally NOT NaN.
        # Note: ts_lower_partial_moment / ts_expected_shortfall are NOT here —
        # R5 P1-38(a): a constant window has a well-defined deterministic value
        # (LPM = max(thr-c,0)^order = 0; ES = the sample value), so only the
        # sample-size floor applies and they must NOT be NaN.
    ]
    for canonical, panels, kw in cases:
        out = OperatorRegistry.get(canonical, "pandas_numpy").calculate(*panels, **kw)
        # Constant window must fail closed: fully NaN in the mature window rows.
        assert out.iloc[30:].isna().all().all(), f"{canonical}: constant window not NaN"

    # R5 P1-38(a): partial moment / ES of a constant window are the deterministic
    # values, not NaN.
    lpm = OperatorRegistry.get("ts_lower_partial_moment", "pandas_numpy").calculate(const, window=20)
    assert (lpm.iloc[30:] == 0.0).all().all(), "LPM of a constant window is 0"
    es = OperatorRegistry.get("ts_expected_shortfall", "pandas_numpy").calculate(const, window=20)
    assert np.allclose(es.iloc[30:].to_numpy(), 1.0), "ES of a constant window is the sample value"
