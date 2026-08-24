# -*- coding: utf-8 -*-
"""Regression tests for the 2026-08 alpha-language operator expansion.

Covers:
- Registration: every alpha-language canonical has a pandas_numpy runtime.
- Surface: every canonical classifies ``daily`` (migrated) and remains a member
  of EXTENDED_ONLY (static partition contract).
- Policy: every canonical has an explicit PIT-safe policy with the intended
  scope (ts / cs / group / fundamental_period).
- Semantic fail-closed: constant / degenerate windows return NaN for a
  representative sample, never an invented value.
- Determinism + axes preservation on a representative sample.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.operator_policy import infer_operator_policy
from factor_engine.cleaned_operators.operator_surface import classify_canonical
import factor_engine.cleaned_operators.operator_surface as _surface_mod

ensure_cleaned_loaded()

ALPHA_LANGUAGE = frozenset({
    # Module 1 — temporal state + signed pattern
    "ts_run_strength", "ts_run_efficiency", "ts_run_concentration",
    "ts_hysteresis_state", "ts_hysteresis_age", "ts_state_integral",
    "ts_state_entry_strength", "ts_transition_intensity", "ts_sign_persistence",
    "ts_sign_cluster_index",
    # Module 2 — path / shape geometry
    "ts_monotonicity", "ts_turning_rate", "ts_turning_intensity",
    "ts_path_efficiency", "ts_roughness", "ts_trend_break_score",
    "ts_weighted_time_centroid", "ts_endpoint_deviation", "ts_mass_concentration",
    # Module 3 — distribution shape / shift
    "ts_tail_imbalance", "ts_expected_shortfall_asymmetry", "ts_wasserstein_shift",
    "ts_ks_shift", "ts_location_shift", "ts_scale_shift",
    # Module 4 — volatility structure
    "ts_vol_of_vol", "ts_vol_acceleration", "ts_vol_term_structure",
    "ts_semivariance_balance", "ts_realized_quarticity", "ts_vol_clustering",
    "ts_leverage_effect", "ts_jump_bipower_proxy",
    # Module 5 — cross-sectional locality + group ex-self
    "cs_neighbor_gap", "cs_local_density", "cs_isolation", "cs_local_curvature",
    "group_ex_self_std", "group_ex_self_mad", "group_ex_self_quantile",
    "relation_weighted_std_ex_self",
    # Module 6 — events + fundamental report wrappers
    "event_frequency", "event_cluster_count", "event_cluster_mean_size",
    "report_rolling_mean", "report_yoy_lag",
})

EXPECTED_SCOPE = {
    "ts_run_strength": "ts", "ts_run_efficiency": "ts", "ts_run_concentration": "ts",
    "ts_hysteresis_state": "ts", "ts_hysteresis_age": "ts", "ts_state_integral": "ts",
    "ts_state_entry_strength": "ts", "ts_transition_intensity": "ts",
    "ts_sign_persistence": "ts", "ts_sign_cluster_index": "ts",
    "ts_monotonicity": "ts", "ts_turning_rate": "ts", "ts_turning_intensity": "ts",
    "ts_path_efficiency": "ts", "ts_roughness": "ts", "ts_trend_break_score": "ts",
    "ts_weighted_time_centroid": "ts", "ts_endpoint_deviation": "ts",
    "ts_mass_concentration": "ts",
    "ts_tail_imbalance": "ts", "ts_expected_shortfall_asymmetry": "ts",
    "ts_wasserstein_shift": "ts", "ts_ks_shift": "ts",
    "ts_location_shift": "ts", "ts_scale_shift": "ts",
    "ts_vol_of_vol": "ts", "ts_vol_acceleration": "ts", "ts_vol_term_structure": "ts",
    "ts_semivariance_balance": "ts", "ts_realized_quarticity": "ts",
    "ts_vol_clustering": "ts", "ts_leverage_effect": "ts", "ts_jump_bipower_proxy": "ts",
    "cs_neighbor_gap": "cs", "cs_local_density": "cs", "cs_isolation": "cs",
    "cs_local_curvature": "cs",
    "group_ex_self_std": "group", "group_ex_self_mad": "group",
    "group_ex_self_quantile": "group", "relation_weighted_std_ex_self": "group",
    "event_frequency": "ts", "event_cluster_count": "ts", "event_cluster_mean_size": "ts",
    "report_rolling_mean": "fundamental_period", "report_yoy_lag": "fundamental_period",
}


def _daily_panels(n: int = 120, cols: int = 4):
    dates = pd.bdate_range("2024-01-02", periods=n)
    assets = [f"S{i}" for i in range(cols)]
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, (n, cols))
    close = pd.DataFrame(np.exp(np.cumsum(r, axis=0)) * 10.0, index=dates, columns=assets)
    volume = pd.DataFrame(rng.lognormal(0, 0.3, (n, cols)) * 1e5, index=dates, columns=assets)
    industry = pd.DataFrame(
        {assets[0]: 1.0, assets[1]: 1.0, assets[2]: 2.0, assets[3]: 2.0}, index=dates
    )
    state = pd.DataFrame(
        np.sign(r), index=dates, columns=assets
    )
    return close, volume, industry, state


def test_alpha_language_all_registered_and_daily():
    missing = [c for c in ALPHA_LANGUAGE if OperatorRegistry.get(c, "pandas_numpy") is None]
    assert not missing, f"missing runtimes: {missing}"
    not_daily = [c for c in sorted(ALPHA_LANGUAGE) if classify_canonical(c) != "daily"]
    assert not not_daily, f"not daily surface: {not_daily}"
    live_extended = frozenset(_surface_mod.EXTENDED_ONLY_CANONICALS)
    not_extended = sorted(ALPHA_LANGUAGE - live_extended)
    assert not not_extended, f"not in EXTENDED_ONLY (partition contract): {not_extended}"


def test_alpha_language_policies_pit_safe_with_intended_scope():
    bad = []
    for c in sorted(ALPHA_LANGUAGE):
        pol = infer_operator_policy(c)
        if not pol.pit_safe:
            bad.append((c, "not pit_safe"))
        elif pol.scope != EXPECTED_SCOPE.get(c):
            bad.append((c, f"scope={pol.scope} != {EXPECTED_SCOPE.get(c)}"))
    assert not bad, f"policy violations: {bad}"


def test_alpha_language_dsl_usable_on_daily_surface():
    from factor_engine.cleaned_operators.operator_surface import is_dsl_name_allowed
    blocked = [
        c for c in sorted(ALPHA_LANGUAGE)
        if not is_dsl_name_allowed(c, c, surface="daily")
    ]
    assert not blocked, f"blocked from daily DSL: {blocked}"


def test_alpha_language_constant_window_is_fail_closed():
    """Constant / zero-variance windows never return ±Inf or huge fabricated
    values.  Ops whose value is legitimately 0 on a constant window (uniform
    concentration / zero path / zero flip count) keep 0; variance-degenerate
    ops (corr / MAD-thresholded) return NaN."""
    dates = pd.bdate_range("2024-01-02", periods=40)
    const = pd.DataFrame(np.ones((40, 2)), index=dates, columns=["S0", "S1"])
    cases = {
        "ts_monotonicity": {"window": 10},
        "ts_path_efficiency": {"window": 10},
        "ts_mass_concentration": {"window": 10},
        "ts_semivariance_balance": {"window": 10},
        "ts_vol_clustering": {"window": 10, "min_periods": 3},
        "ts_tail_imbalance": {"window": 10, "min_periods": 4},
        "ts_scale_shift": {"recent_window": 5, "old_window": 5, "min_periods": 3},
        "ts_turning_rate": {"window": 10, "min_periods": 2},
        "ts_endpoint_deviation": {"window": 10, "min_periods": 3},
        "ts_sign_persistence": {"window": 10, "min_periods": 3},
        "ts_roughness": {"window": 10, "min_periods": 3},
    }
    unbounded = []
    for op, kw in cases.items():
        inst = OperatorRegistry.get(op, "pandas_numpy")
        out = inst.calculate(const, **kw)
        tail = out.iloc[-1].to_numpy(dtype=float)
        if not np.all(np.isnan(tail) | np.isfinite(tail)):
            unbounded.append((op, tail))
    assert not unbounded, f"±Inf leaked from constant window: {unbounded}"
    # variance-degenerate ops must be NaN, not an invented 0
    for op, kw in (
        ("ts_vol_clustering", {"window": 10, "min_periods": 3}),
        ("ts_tail_imbalance", {"window": 10, "min_periods": 4}),
    ):
        inst = OperatorRegistry.get(op, "pandas_numpy")
        out = inst.calculate(const, **kw)
        assert np.all(np.isnan(out.iloc[-1].to_numpy(dtype=float))), f"{op} not NaN on constant window"


def test_alpha_language_determinism_and_axes():
    close, volume, industry, state = _daily_panels()
    runs = {
        "ts_run_strength": {"x": close, "state": state, "max_run": 20},
        "ts_hysteresis_state": {"z": close.pct_change(), "upper": 0.03, "lower": 0.01},
        "ts_monotonicity": {"x": close, "window": 20},
        "ts_vol_of_vol": {"ret": close.pct_change(), "inner_window": 5, "outer_window": 20},
        "event_frequency": {"condition": close > close.shift(1), "window": 20},
        "cs_neighbor_gap": {"x": close, "k": 5},
        "group_ex_self_std": {"x": close, "group": industry},
    }
    for op, kw in runs.items():
        inst = OperatorRegistry.get(op, "pandas_numpy")
        out1 = inst.calculate(**kw)
        out2 = inst.calculate(**kw)
        pd.testing.assert_frame_equal(out1, out2)
        assert out1.index.equals(close.index)
        assert list(out1.columns) == list(close.columns)
        assert out1.dtypes.apply(lambda d: d == float).all()


def test_report_operators_need_period_id_panel():
    """report_* wrappers are fundamental_period ops: they accept (x, period_id)."""
    dates = pd.bdate_range("2024-01-02", periods=120)
    assets = ["S0", "S1"]
    x = pd.DataFrame(np.random.default_rng(1).normal(1, 0.1, (120, 2)), index=dates, columns=assets)
    period_id = pd.DataFrame(
        np.repeat(np.arange(1, 31), 4)[:120].reshape(-1, 1)
        * np.ones((1, 2)),
        index=dates, columns=assets,
    )
    for op in ("report_rolling_mean", "report_yoy_lag"):
        inst = OperatorRegistry.get(op, "pandas_numpy")
        out = inst.calculate(x, period_id, periods=4) if op == "report_rolling_mean" else inst.calculate(x, period_id)
        assert out.shape == x.shape
        assert out.dtypes.apply(lambda d: d == float).all()
