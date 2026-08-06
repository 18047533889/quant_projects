# -*- coding: utf-8 -*-
"""Tests for the 2026-08 next-stage cross-sectional and peer operators."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

PEER_CANONICALS = (
    "group_peer_beta_deviation group_peer_deviation_index group_peer_information_diffusion "
    "group_leader_laggard_exposure group_return_dispersion_exposure group_multi_level_rank_consistency "
    "ts_market_liquidity_beta ts_industry_liquidity_beta"
).split()

CS_CANONICALS = (
    "cs_ridge_resid cs_quantile_resid cs_spline_resid cs_mahalanobis_distance "
    "cs_knn_distance cs_local_density_score cs_autoencoder_reconstruction_error"
).split()

PANEL_CANONICALS = (
    "panel_rolling_pca_loading panel_rolling_pca_resid panel_rolling_pca_resid_vol "
    "panel_rolling_pca_resid_momentum panel_rolling_pca_explained_ratio "
    "industry_rolling_pca_loading panel_rolling_pcr_forecast panel_rolling_pls_forecast "
    "panel_rolling_elastic_net_forecast panel_regime_conditioned_forecast "
    "panel_mixture_of_experts_score"
).split()


def _panel(n: int = 120, seed: int = 0, cols: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(rng.standard_normal((n, cols)), index=idx, columns=[f"C{i}" for i in range(cols)])


@pytest.mark.parametrize("name", sorted(set(PEER_CANONICALS + CS_CANONICALS + PANEL_CANONICALS)))
def test_registered(name: str) -> None:
    assert OperatorRegistry.get(name) is not None, name


def test_peer_beta_deviation_ex_self() -> None:
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    beta = pd.DataFrame({"A": np.linspace(0.5, 1.5, 60), "B": np.linspace(0.8, 0.9, 60)}, index=dates)
    grp = pd.DataFrame({"A": "g1", "B": "g1"}, index=dates, dtype=object)
    w = pd.DataFrame({"A": 1e8, "B": 2e8}, index=dates)
    out = OperatorRegistry.get("group_peer_beta_deviation").calculate(beta, grp, w)
    # last row: A peer (weighted ex-self) = B's beta = 0.9 ; B peer = A's beta = 1.5
    assert out["A"].iloc[-1] == pytest.approx(1.5 - 0.9, abs=1e-6)
    assert out["B"].iloc[-1] == pytest.approx(0.9 - 1.5, abs=1e-6)


def test_peer_deviation_index_zero_center() -> None:
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    rng = np.random.default_rng(0)
    a = pd.DataFrame(rng.standard_normal((30, 4)), index=dates, columns=[f"C{i}" for i in range(4)])
    b = pd.DataFrame(rng.standard_normal((30, 4)), index=dates, columns=[f"C{i}" for i in range(4)])
    out = OperatorRegistry.get("group_peer_deviation_index").calculate(a, b)
    assert out.shape == a.shape
    # cross-section mean of the index should be ~0 per day (z-score sum)
    assert out.mean(axis=1).abs().max() < 1e-6


def test_cs_ridge_resid_on_linear_model() -> None:
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    rng = np.random.default_rng(1)
    x = pd.DataFrame(rng.standard_normal((60, 10)), index=dates, columns=[f"C{i}" for i in range(10)])
    y = 2.0 * x.mean(axis=1).to_frame().values[:, 0] * 0 + 1.0  # constant -> residual ~0
    y = pd.DataFrame(3.0 * x["C0"] + 0.5, index=dates, columns=[f"C{i}" for i in range(10)]) + x * 0.0 + x * 0.0
    y = pd.DataFrame((3.0 * x).to_numpy() + 0.5, index=dates, columns=[f"C{i}" for i in range(10)])
    out = OperatorRegistry.get("cs_ridge_resid").calculate(y, x, alpha=1e-8)
    assert out.iloc[-1].abs().max() < 1e-3


def test_cs_mahalanobis_shape() -> None:
    x = _panel(seed=2, cols=8)
    out = OperatorRegistry.get("cs_mahalanobis_distance").calculate(x, _panel(seed=3, cols=8))
    assert out.shape == x.shape
    assert np.isfinite(out.to_numpy()).any()


def test_pca_resid_removes_common_factor() -> None:
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    rng = np.random.default_rng(4)
    common = rng.standard_normal((100, 1))
    ret = common @ rng.standard_normal((1, 10)) * 0.5 + rng.standard_normal((100, 10)) * 0.1
    ret = pd.DataFrame(ret, index=dates, columns=[f"C{i}" for i in range(10)])
    resid = OperatorRegistry.get("panel_rolling_pca_resid").calculate(ret, window=60, n_components=2)
    # residual cross-sectional variance should be below the raw variance on average
    assert resid.var(axis=1).mean() < ret.var(axis=1).mean()


def test_pcr_forecast_shape() -> None:
    x1 = _panel(seed=5, cols=8)
    x2 = _panel(seed=6, cols=8)
    y = pd.DataFrame(np.roll(x1.to_numpy(), -1, axis=0), index=x1.index, columns=x1.columns)  # next-day label
    out = OperatorRegistry.get("panel_rolling_pcr_forecast").calculate(y, x1, x2, window=60, n_components=3)
    assert out.shape == y.shape


def test_autoencoder_error_nonneg() -> None:
    x = _panel(seed=7, cols=8)
    out = OperatorRegistry.get("cs_autoencoder_reconstruction_error").calculate(x, window=40)
    valid = out.dropna().to_numpy()
    assert (valid >= 0).all()
