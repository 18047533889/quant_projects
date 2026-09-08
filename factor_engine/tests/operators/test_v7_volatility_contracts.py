from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _gjr_oracle(seg, omega, alpha, gamma, beta):
    fit_seg = seg[:-1]
    h = float(np.var(fit_seg))
    for shock in seg[:-1]:
        h = omega + (alpha + (gamma if shock < 0.0 else 0.0)) * shock**2 + beta * h
    shock = seg[-1]
    return np.sqrt(omega + (alpha + (gamma if shock < 0.0 else 0.0)) * shock**2 + beta * h)


def test_v7_gjr_forecast_uses_same_leverage_step_as_history(monkeypatch):
    import factor_engine.cleaned_operators.ts_model.volatility as vol

    params = (0.02, 0.08, 0.21, 0.63)
    monkeypatch.setattr(vol, "_fit_gjr_cached", lambda _: params)
    base = np.array([0.08, -0.04, 0.03, -0.02, 0.07, -0.05, 0.01,
                     -0.06, 0.04, -0.03, 0.05, -0.02], dtype=float)
    for current in (-0.17, 0.17, 0.0, -0.0):
        seg = np.r_[base, current]
        got = vol._garch_path(seg, 13, "forecast", True, 0.0)
        np.testing.assert_allclose(got, _gjr_oracle(seg, *params), rtol=1e-13, atol=1e-13)


def test_v7_gjr_gamma_zero_degenerates_to_garch(monkeypatch):
    import factor_engine.cleaned_operators.ts_model.volatility as vol

    garch = (0.02, 0.08, 0.63)
    monkeypatch.setattr(vol, "_fit_garch_cached", lambda _: garch)
    monkeypatch.setattr(vol, "_fit_gjr_cached", lambda _: (garch[0], garch[1], 0.0, garch[2]))
    seg = np.array([-0.12, 0.03, -0.05, 0.09, -0.02, 0.07, -0.08,
                    0.01, 0.04, -0.06, 0.11, -0.03, 0.02])
    np.testing.assert_allclose(
        vol._garch_path(seg, 13, "forecast", True, 0.0),
        vol._garch_path(seg, 13, "forecast", False, 0.0),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("current", [-0.017, 0.017, 0.0, -0.0])
def test_v7_gjr_fixed_parameters_match_arch_7_2(monkeypatch, current):
    arch = pytest.importorskip("arch")
    from arch.univariate import GARCH
    import factor_engine.cleaned_operators.ts_model.volatility as vol

    params = np.array([1.0e-5, 0.08, 0.21, 0.63])
    base = np.array([0.018, -0.014, 0.013, -0.012, 0.017, -0.015,
                     0.011, -0.016, 0.014, -0.013, 0.015, -0.012])
    seg = np.r_[base, current]
    monkeypatch.setattr(vol, "_fit_gjr_cached", lambda _: tuple(params))
    got_var = vol._garch_path(seg, 13, "forecast", True, 0.0) ** 2

    process = GARCH(p=1, o=1, q=1)
    initial_variance = float(np.var(seg[:-1]))
    # arch seeds sigma2[0] by applying omega plus persistence to `backcast`.
    # Invert that initialization so both implementations start with the exact
    # same sigma2[0] used by this module's explicit variance-path convention.
    arch_backcast = (initial_variance - params[0]) / (
        params[1] + 0.5 * params[2] + params[3]
    )
    bounds = process.variance_bounds(seg)
    forecast = process.forecast(
        params, seg, arch_backcast, bounds, start=len(seg) - 1, horizon=1
    ).forecasts[-1, 0]
    assert arch.__version__ == "7.2.0"
    np.testing.assert_allclose(got_var, forecast, rtol=1e-13, atol=1e-13)


def test_v7_garch_forecast_nonfinite_current_shock_fails_closed(monkeypatch):
    import factor_engine.cleaned_operators.ts_model.volatility as vol

    monkeypatch.setattr(vol, "_fit_gjr_cached", lambda _: (0.02, 0.08, 0.21, 0.63))
    base = np.linspace(-0.08, 0.09, 12)
    for bad in (np.nan, np.inf, -np.inf):
        assert np.isnan(vol._garch_path(np.r_[base, bad], 13, "forecast", True, 0.0))


@pytest.mark.parametrize(
    "suffix_values",
    [
        np.linspace(20.0, 200.0, 40),
        np.full(40, np.nan),
        np.r_[np.full(20, np.inf), np.full(20, -np.inf)],
    ],
    ids=["extreme_level_like", "long_missing", "infinite"],
)
def test_v7_registered_gjr_prefix_is_not_rejected_by_future_suffix(
    monkeypatch, suffix_values
):
    import factor_engine.cleaned_operators.ts_model.volatility as vol
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    rng = np.random.default_rng(731)
    prefix = pd.DataFrame({"A": rng.normal(0.0, 0.012, 90)})
    suffix = pd.DataFrame({"A": suffix_values})
    monkeypatch.setattr(vol, "_fit_gjr_cached", lambda _: (0.00001, 0.08, 0.12, 0.75))
    op = OperatorRegistry.get("ts_gjr_garch_vol_forecast")
    expected = op.calculate(prefix, window=40)["A"]
    extended = op.calculate(pd.concat([prefix, suffix], ignore_index=True), window=40)["A"]
    np.testing.assert_allclose(extended.iloc[: len(prefix)], expected, equal_nan=True)


def _har_innovation_oracle(rv: np.ndarray, window: int) -> float:
    seg = rv[-window:]
    n = len(seg)
    daily = seg
    weekly = pd.Series(seg).rolling(5).mean().to_numpy()
    monthly = pd.Series(seg).rolling(22).mean().to_numpy()
    X = np.column_stack([np.ones(n), daily, weekly, monthly])
    rows = np.arange(n - 2)
    mask = np.all(np.isfinite(X[rows]), axis=1) & np.isfinite(seg[rows + 1])
    Xs, y = X[rows][mask], seg[rows + 1][mask]
    beta = np.linalg.lstsq(Xs, y, rcond=None)[0]
    pred = float(X[n - 2] @ beta)
    sd = float(np.std(y - Xs @ beta))
    return float((seg[-1] - pred) / sd)


def test_v7_har_innovation_pairs_features_and_labels_on_original_clock():
    import factor_engine.cleaned_operators.ts_model.volatility as vol

    rng = np.random.default_rng(912)
    rv = np.exp(rng.normal(-4.0, 0.35, 120))
    rv[70] = np.nan
    expected = _har_innovation_oracle(rv, 120)
    got = vol._har_rv(rv, 120, "innovation_z")
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)
