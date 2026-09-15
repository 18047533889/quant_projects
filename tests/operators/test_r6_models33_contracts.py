from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.base import MISSING
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.cross_section import panel_model
from factor_engine.cleaned_operators.ts_model import complexity, volatility


PANEL = tuple(panel_model._CANONICALS)
VOL = tuple(volatility._CANONICALS)
COMPLEXITY = (
    "ts_change_point_probability", "ts_cusum_vol_break_score", "ts_dfa_hurst",
    "ts_lz_complexity", "ts_multiscale_entropy_slope",
    "ts_pseudocount_sample_entropy", "ts_regime_duration",
    "ts_turning_point_ratio", "ts_two_state_regime_probability",
)


def _panels(n=150, cols=5):
    rng = np.random.default_rng(6033)
    idx = pd.date_range("2025-01-01", periods=n)
    x1 = pd.DataFrame(rng.normal(size=(n, cols)), index=idx, columns=list("ABCDE")[:cols])
    x2 = pd.DataFrame(rng.normal(size=(n, cols)), index=idx, columns=x1.columns)
    state = pd.DataFrame(np.tile(np.linspace(-2, 2, n)[:, None], (1, cols)), index=idx, columns=x1.columns)
    y = 0.8 * x1 - 0.35 * x2 + pd.DataFrame(rng.normal(scale=.03, size=(n, cols)), index=idx, columns=x1.columns)
    group = pd.DataFrame("g", index=idx, columns=x1.columns)
    return y, x1, x2, state, group


def _call_panel(name, y, x1, x2, state, group):
    op = OperatorRegistry.get(name, "pandas_numpy", mode="any")
    if name == "panel_rolling_pca_loading":
        return op.calculate(x1, window=24, component=0)
    if name in {"panel_rolling_pca_resid", "panel_rolling_pca_explained_ratio",
                "panel_rolling_pca_resid_vol", "panel_rolling_pca_resid_momentum"}:
        return op.calculate(x1, 24, n_components=2)
    if name == "industry_rolling_pca_loading":
        return op.calculate(x1, group, window=24, component=0)
    if name in {"panel_rolling_pcr_forecast", "panel_rolling_pls_forecast"}:
        return op.calculate(y, x1, x2, window=36, n_components=2, label_horizon=1)
    if name == "panel_rolling_elastic_net_forecast":
        return op.calculate(y=y, x1=x1, x2=x2, window=36, alpha=.01, l1_ratio=.5, label_horizon=1)
    if name == "panel_regime_conditioned_forecast":
        return op.calculate(y, x1, x2, market_state=state, window=60, n_regimes=2, label_horizon=1)
    if name == "panel_mixture_of_experts_score":
        return op.calculate(y, x1, x2, market_state=state, window=60, n_experts=2, label_horizon=1)
    if name == "ts_feature_pca_reconstruction_error":
        return op.calculate(x1, x2, window=24, n_components=1)
    if name == "cs_autoencoder_reconstruction_error":
        return op.calculate(f1=x1, f2=x2, window=24)
    raise AssertionError(name)


def test_exact_unique_inventory_and_scalar_contracts():
    assert len(PANEL) == len(set(PANEL)) == 13
    assert len(VOL) == len(set(VOL)) == 11
    assert len(COMPLEXITY) == len(set(COMPLEXITY)) == 9
    assert len(set(PANEL + VOL + COMPLEXITY)) == 33
    for name in PANEL + VOL + COMPLEXITY:
        op = OperatorRegistry.get(name, "pandas_numpy", mode="any")
        assert op is not None
        md = op.metadata
        assert md.panel_params
        for scalar, spec in md.param_specs.items():
            assert spec.default is not MISSING
            assert spec.param_role is not None
            if scalar not in md.panel_params:
                assert scalar in md.scalar_params


@pytest.mark.parametrize("name", PANEL)
def test_panel_each_executes_finite_preserves_shape_and_prefix(name):
    y, x1, x2, state, group = _panels()
    out = _call_panel(name, y, x1, x2, state, group)
    assert out.index.equals(y.index) and out.columns.equals(y.columns)
    assert np.isfinite(out.to_numpy()).any(), name
    cut = 115
    short = _call_panel(name, y.iloc[:cut], x1.iloc[:cut], x2.iloc[:cut], state.iloc[:cut], group.iloc[:cut])
    np.testing.assert_allclose(out.iloc[:cut], short, equal_nan=True, atol=1e-10, rtol=1e-10)


def _series(n=150):
    rng = np.random.default_rng(933)
    idx = pd.date_range("2025-01-01", periods=n)
    ret = pd.DataFrame({"A": rng.normal(scale=.018, size=n)}, index=idx)
    rv = pd.DataFrame({"A": .0002 + .3 * ret["A"].to_numpy() ** 2 + rng.uniform(0, .0002, n)}, index=idx)
    return ret, rv


@pytest.mark.parametrize("name", VOL)
def test_volatility_each_executes_finite_shape_prefix_and_negative(name):
    ret, rv = _series()
    x = rv if name.startswith("ts_har_rv") else ret
    window = 90 if name.startswith("ts_har") else 36
    op = OperatorRegistry.get(name, "pandas_numpy", mode="any")
    out = op.calculate(x, window=window)
    assert out.index.equals(x.index) and out.columns.equals(x.columns)
    assert np.isfinite(out.to_numpy()).any(), name
    short = op.calculate(x.iloc[:125], window=window)
    np.testing.assert_allclose(out.iloc[:125], short, equal_nan=True, atol=1e-10, rtol=1e-10)
    with pytest.raises((TypeError, ValueError)):
        op.calculate(x, window=-1)


def _complexity_kwargs(name):
    return {
        "ts_change_point_probability": {"window": 50, "transition_prob": .05},
        "ts_cusum_vol_break_score": {"window": 50, "min_periods": 10},
        "ts_dfa_hurst": {"window": 80, "min_scale": 4, "max_scale": 12},
        "ts_lz_complexity": {"window": 50, "bins": 4},
        "ts_multiscale_entropy_slope": {"max_scale": 4, "window": 80},
        "ts_pseudocount_sample_entropy": {"m": 2, "r": .3, "window": 50},
        "ts_regime_duration": {"window": 50, "transition_prob": .05},
        "ts_turning_point_ratio": {"window": 50},
        "ts_two_state_regime_probability": {"window": 50, "transition_prob": .05},
    }[name]


@pytest.mark.parametrize("name", COMPLEXITY)
def test_complexity_each_executes_finite_shape_prefix_and_negative(name):
    x, _ = _series(100)
    op = OperatorRegistry.get(name, "pandas_numpy", mode="any")
    out = op.calculate(x, **_complexity_kwargs(name))
    assert out.index.equals(x.index) and out.columns.equals(x.columns)
    assert np.isfinite(out.to_numpy()).any(), name
    short = op.calculate(x.iloc[:85], **_complexity_kwargs(name))
    np.testing.assert_allclose(out.iloc[:85], short, equal_nan=True, atol=1e-10, rtol=1e-10)
    bad = dict(_complexity_kwargs(name)); bad["window"] = -1
    with pytest.raises((TypeError, ValueError)):
        op.calculate(x, **bad)


def test_independent_oracles_and_forecast_label_maturity():
    x, _ = _series(60)
    turns = OperatorRegistry.get("ts_turning_point_ratio", "pandas_numpy", mode="any").calculate(x, window=20)
    d = np.diff(x.iloc[-20:, 0].to_numpy())
    assert turns.iloc[-1, 0] == pytest.approx(np.sum(d[1:] * d[:-1] < 0) / (len(d) - 1))
    y, x1, x2, state, group = _panels()
    base = _call_panel("panel_rolling_pcr_forecast", y, x1, x2, state, group)
    shocked = y.copy(); shocked.iloc[-1] = 1e9
    alt = _call_panel("panel_rolling_pcr_forecast", shocked, x1, x2, state, group)
    np.testing.assert_allclose(base.iloc[-1], alt.iloc[-1], equal_nan=True)
