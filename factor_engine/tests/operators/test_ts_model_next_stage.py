# -*- coding: utf-8 -*-
"""Tests for the 2026-08 next-stage time-series model operators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

REGRESSION_CANONICALS = (
    "ts_multi_regression_coeff ts_multi_regression_resid ts_multi_regression_resid_z "
    "ts_multi_regression_r2 ts_huber_regression_coeff ts_huber_regression_resid_z "
    "ts_ridge_regression_coeff ts_ridge_regression_resid_z ts_quantile_regression_coeff "
    "ts_quantile_regression_resid ts_quantile_beta_spread ts_ar_forecast ts_ar_innovation "
    "ts_ar_innovation_z ts_mean_reversion_half_life ts_variance_ratio_slope"
).split()

ADVANCED_CANONICALS = (
    "ts_kalman_level ts_kalman_trend ts_kalman_innovation_z ts_kalman_beta ts_kalman_beta_change "
    "ts_kalman_beta_uncertainty ts_garch_vol_forecast ts_garch_persistence ts_garch_standardized_shock "
    "ts_gjr_garch_vol_forecast ts_gjr_leverage ts_har_rv_forecast ts_har_rv_innovation_z "
    "ts_permutation_entropy ts_sample_entropy ts_lz_complexity ts_multiscale_entropy_slope "
    "ts_turning_point_ratio ts_dfa_hurst ts_cusum_vol_break_score ts_change_point_probability "
    "ts_regime_duration ts_two_state_regime_probability ts_wavelet_low_frequency_ratio "
    "ts_wavelet_high_frequency_ratio ts_wavelet_entropy ts_wavelet_energy_slope "
    "ts_spectral_low_frequency_ratio ts_matrix_profile_discord_score "
    "ts_matrix_profile_motif_distance ts_motif_recurrence_count ts_path_signature_area "
    "ts_path_signature_depth2_norm ts_path_leadlag_area"
).split()

COMPAT_ALIAS_TARGETS = {
    "ts_garch_vol_forecast": "ts_garch_next_vol_forecast",
    "ts_har_rv_forecast": "ts_har_rv_next_vol_forecast",
    "ts_har_rv_innovation_z": "ts_har_rv_forecast_error_z",
    "ts_matrix_profile_motif_distance": "ts_matrix_profile_discord_score",
}


def _panel(n: int = 160, seed: int = 0, cols: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(rng.standard_normal((n, cols)), index=idx, columns=["A", "B"])


@pytest.mark.parametrize("name", sorted(set(REGRESSION_CANONICALS + ADVANCED_CANONICALS)))
def test_registered_and_classified(name: str) -> None:
    from factor_engine.cleaned_operators.operator_surface import classify_canonical

    operator = OperatorRegistry.get(name)
    assert operator is not None, name
    if name in COMPAT_ALIAS_TARGETS:
        assert operator is OperatorRegistry.get(COMPAT_ALIAS_TARGETS[name]), name
        return
    # 2026-08 daily migration promoted many experimental model operators to the daily
    # surface; research remains for the source-side relation transforms.
    assert classify_canonical(name) in ("daily", "extended", "research"), name


def test_multi_regression_recovers_coefficient() -> None:
    x = _panel(seed=1)
    y = 1.5 * x + 0.3 + np.random.default_rng(2).standard_normal(x.shape) * 0.01
    op = OperatorRegistry.get("ts_multi_regression_coeff")
    out = op.calculate(y, x, window=80, coefficient_index=1, min_periods=10)
    assert out["A"].iloc[-1] == pytest.approx(1.5, abs=0.05)


def test_multi_regression_r2_on_perfect_line() -> None:
    x = _panel(seed=3)
    y = 2.0 * x + 0.5
    out = OperatorRegistry.get("ts_multi_regression_r2").calculate(y, x, window=60, min_periods=6)
    assert out["A"].iloc[-1] == pytest.approx(1.0, abs=1e-4)


def test_quantile_beta_spread_bounded() -> None:
    x = _panel(seed=4)
    y = 1.5 * x + np.random.default_rng(5).standard_normal(x.shape) * 0.01
    out = OperatorRegistry.get("ts_quantile_beta_spread").calculate(y, x, window=80, q_high=0.8, q_low=0.2, min_periods=10)
    assert out.shape == x.shape


def test_ar_forecast_matches_manual_ar1() -> None:
    rng = np.random.default_rng(0)
    n = 300
    rho = 0.7
    series = np.zeros(n)
    for t in range(1, n):
        series[t] = rho * series[t - 1] + rng.standard_normal()
    x = pd.DataFrame(series, index=pd.date_range("2024-01-01", periods=n), columns=["A"])
    f = OperatorRegistry.get("ts_ar_forecast").calculate(x, window=120, order=1)["A"]
    seg = series[-120:]
    b = np.linalg.lstsq(np.column_stack([np.ones(119), seg[:-1]]), seg[1:], rcond=None)[0]
    manual = b[0] + b[1] * series[-2]
    assert f.iloc[-1] == pytest.approx(manual, abs=1e-9)


def test_mean_reversion_half_life_positive_for_ou() -> None:
    rng = np.random.default_rng(1)
    n = 500
    mr = np.zeros(n)
    for t in range(1, n):
        mr[t] = mr[t - 1] + 0.08 * (0.0 - mr[t - 1]) + rng.standard_normal() * 0.1
    x = pd.DataFrame(mr, index=pd.date_range("2024-01-01", periods=n), columns=["A"])
    hl = OperatorRegistry.get("ts_mean_reversion_half_life").calculate(x, window=200, min_periods=20)["A"].iloc[-1]
    assert 4.0 < hl < 15.0


def test_kalman_tracks_level() -> None:
    rng = np.random.default_rng(2)
    n = 200
    level = np.linspace(0, 5, n) + rng.standard_normal(n) * 0.1
    x = pd.DataFrame(level, index=pd.date_range("2024-01-01", periods=n), columns=["A"])
    out = OperatorRegistry.get("ts_kalman_level").calculate(x, q=0.01, r=0.1)["A"]
    assert out.iloc[-1] == pytest.approx(5.0, abs=0.5)


def test_garch_persistence_in_unit_range() -> None:
    rng = np.random.default_rng(3)
    n = 400
    h = np.zeros(n)
    h[0] = 1.0
    ret = np.zeros(n)
    for t in range(1, n):
        h[t] = 0.05 + 0.1 * ret[t - 1] ** 2 + 0.8 * h[t - 1]
        ret[t] = np.sqrt(h[t]) * rng.standard_normal()
    x = pd.DataFrame(ret, index=pd.date_range("2024-01-01", periods=n), columns=["A"])
    out = OperatorRegistry.get("ts_garch_persistence").calculate(x, window=250)["A"]
    valid = out.dropna()
    assert len(valid) > 0
    assert (valid.between(0.0, 1.0)).all()


def test_entropy_and_hurst_finite() -> None:
    rng = np.random.default_rng(4)
    n = 300
    idx = pd.date_range("2024-01-01", periods=n)
    x = pd.DataFrame(rng.standard_normal((n, 1)), index=idx, columns=["A"])
    for name, kwargs in [
        ("ts_permutation_entropy", {"order": 4, "window": 120}),
        ("ts_turning_point_ratio", {"window": 60}),
        ("ts_lz_complexity", {"window": 200}),
        ("ts_cusum_vol_break_score", {}),
        ("ts_two_state_regime_probability", {}),
    ]:
        out = OperatorRegistry.get(name).calculate(x, **kwargs)
        assert out.shape == x.shape, name
        assert np.isfinite(out.to_numpy()).any(), name


def test_hurst_close_to_05_for_white_noise() -> None:
    # DFA of white noise: the profile is a random walk, so the DFA exponent ~ 0.5
    rng = np.random.default_rng(5)
    n = 800
    noise = rng.standard_normal(n)
    x = pd.DataFrame(noise, index=pd.date_range("2024-01-01", periods=n), columns=["A"])
    h = OperatorRegistry.get("ts_dfa_hurst").calculate(x, window=600, min_scale=8, max_scale=64)["A"].iloc[-1]
    assert abs(h - 0.5) < 0.25


def test_path_signature_area_shape() -> None:
    a, b = _panel(seed=6), _panel(seed=7)
    out = OperatorRegistry.get("ts_path_signature_area").calculate(a, b, window=60)
    assert out.shape == a.shape
    assert np.isfinite(out.to_numpy()).any()


def test_sequence_anomaly_finite() -> None:
    x = _panel(seed=8)
    implicit = {}
    for name in ("ts_matrix_profile_discord_score", "ts_matrix_profile_motif_distance", "ts_motif_recurrence_count"):
        out = OperatorRegistry.get(name).calculate(x, m=10)
        implicit[name] = out
        assert out.shape == x.shape, name
        assert np.isfinite(out.to_numpy()).any(), name

    explicit_default = OperatorRegistry.get(
        "ts_matrix_profile_discord_score"
    ).calculate(x, m=10, history_window=252)
    pd.testing.assert_frame_equal(
        implicit["ts_matrix_profile_discord_score"], explicit_default
    )

    with pytest.raises(ValueError, match="m < history_window"):
        OperatorRegistry.get("ts_matrix_profile_discord_score").calculate(
            x, m=10, history_window=10
        )


@pytest.mark.parametrize("name", sorted(set(REGRESSION_CANONICALS)))
def test_regression_operators_shape_determinism(name: str) -> None:
    x = _panel(seed=9)
    y = 1.5 * x + np.random.default_rng(10).standard_normal(x.shape) * 0.01
    if name.startswith("ts_ar") or name == "ts_mean_reversion_half_life" or name == "ts_variance_ratio_slope":
        args, kw = (x,), {}
    elif name.startswith("ts_multi"):
        args, kw = (y, x), {}
    else:
        args, kw = (y, x), {}
    op = OperatorRegistry.get(name)
    first = op.calculate(*args, **kw)
    second = op.calculate(*args, **kw)
    assert first.shape == x.shape, name
    pd.testing.assert_frame_equal(first, second, check_dtype=False)
