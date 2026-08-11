# -*- coding: utf-8 -*-
"""R35 model-audit regression tests (M-040 / M-043 / M-044 / M-045 / M-050 /
M-051 / M-052 / M-055 / M-057 / M-060).

These pin the KERNEL-side fixes for the model-operator full audit.  They do NOT
touch the shared reconciler-owned files (model_timing.py / model_contract.py /
model_lane.py / operator_surface.py / layer_governance.py) — reconciler
obligations are asserted only where the kernel itself is responsible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from cleaned_operators import load_all  # noqa: E402
from cleaned_operators.registry import OperatorRegistry  # noqa: E402
from cleaned_operators.regression_models import (  # noqa: E402
    TsLoMackinlayVr,
    TsLoMackinlayZ,
    TsVarianceRatioProxy,
)
from cleaned_operators.ts_model import _rolling_core as _rc  # noqa: E402


def _load():
    load_all()


def _panel(n, cols=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(rng.standard_normal((n, cols)), index=idx, columns=list("ABC")[:cols])


def _get(name):
    _load()
    op = OperatorRegistry.get(name, "pandas_numpy")
    assert op is not None, f"{name} not registered"
    return op


# ---------------------------------------------------------------------------
# M-040: ts_ar_coefficient is IN-SAMPLE descriptive (kernel documents it)
# ---------------------------------------------------------------------------

def test_m040_ts_ar_coefficient_documented_in_sample():
    op = _get("ts_ar_coefficient")
    desc = op.metadata.description
    assert "IN-SAMPLE" in desc or "in-sample" in desc.lower(), desc
    # the strictly-prior replacement is named
    assert "ts_ar_prior_coeff" in desc, desc
    # kernel math is unchanged: the operator output at the last row must equal
    # the manual in-sample OLS slope of x_t on x_{t-1} over the same trailing
    # window INCLUDING the current row (cov/var, the kernel's formula).
    rng = np.random.default_rng(11)
    n = 500
    rho = 0.6
    series = np.zeros(n)
    for t in range(1, n):
        series[t] = rho * series[t - 1] + rng.standard_normal()
    x = pd.DataFrame(series, index=pd.date_range("2024-01-01", periods=n), columns=["A"])
    out = op.calculate(x, window=250, lag=1, min_periods=10)["A"]
    w = 250
    seg = series[-w:]
    current = seg[1:]
    lagged = seg[:-1]
    manual = float(
        np.mean((current - np.mean(current)) * (lagged - np.mean(lagged)))
        / np.var(lagged)
    )
    assert out.iloc[-1] == pytest.approx(manual, abs=1e-12)
    # a positive-phi AR(1) yields a positive in-sample coefficient
    assert out.iloc[-1] > 0.0


# ---------------------------------------------------------------------------
# M-043: ts_ar_innovation_z is IN-SAMPLE (fit_lag=0) despite its _innovation name
# ---------------------------------------------------------------------------

def test_m043_ts_ar_innovation_z_documented_in_sample():
    op = _get("ts_ar_innovation_z")
    desc = op.metadata.description
    assert "in-sample" in desc.lower(), desc
    # the strictly-prior z-variant is a distinct operator
    assert _get("ts_ar_prior_innovation_z") is not None


# ---------------------------------------------------------------------------
# M-044: mean-reversion half-life input semantic (stationary/spread/residual)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    ["ts_mean_reversion_half_life", "ts_mean_reversion_ou_approx_half_life"],
)
def test_m044_half_life_input_units_declared(name):
    op = _get(name)
    assert op.metadata.input_units, name
    assert "spread" in op.metadata.input_units["x"], op.metadata.input_units
    assert "stationary" in op.metadata.input_units["x"], op.metadata.input_units
    assert "residual" in op.metadata.input_units["x"], op.metadata.input_units
    assert "spread/residual/stationary" in op.metadata.description, op.metadata.description


@pytest.mark.parametrize(
    "name",
    ["ts_mean_reversion_half_life", "ts_mean_reversion_ou_approx_half_life"],
)
def test_m044_half_life_warns_on_trending_input(name):
    op = _get(name)
    rng = np.random.default_rng(12)
    # monotone-rising level: drift dominates, half-life is meaningless
    trend = pd.DataFrame(
        np.linspace(0.0, 100.0, 240) + rng.standard_normal(240) * 0.05,
        index=pd.date_range("2024-01-01", periods=240, freq="B"),
        columns=["A"],
    )
    with pytest.warns(UserWarning, match="TRENDING"):
        op.calculate(trend, window=120, min_periods=20)


def test_m044_half_life_stationary_input_no_warning():
    op = _get("ts_mean_reversion_half_life")
    # a stationary OU path should NOT trigger the trending warning
    rng = np.random.default_rng(13)
    phi = 0.9
    x = np.zeros(400)
    for t in range(1, 400):
        x[t] = phi * x[t - 1] + rng.standard_normal() * 0.1
    s = pd.DataFrame(x, index=pd.date_range("2024-01-01", periods=400, freq="B"), columns=["A"])
    import warnings as _w

    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        op.calculate(s, window=120, min_periods=20)
    assert not any("TRENDING" in str(r.message) for r in rec), [str(r.message) for r in rec]


# ---------------------------------------------------------------------------
# M-045: Lo-MacKinlay sign convention is VR>1 -> trending, VR<1 -> mean-reversion
# ---------------------------------------------------------------------------

def test_m045_lo_mackinlay_z_sign_doc_corrected():
    doc = (TsLoMackinlayZ.__doc__ or "")
    assert "z > 0" in doc and "TRENDING" in doc.upper(), doc
    assert "MEAN-REVERSION" in doc.upper(), doc
    # the inverted wording ("positive -> mean reversion") must be gone
    assert "positive values indicate mean reversion" not in doc.lower(), doc


def test_m045_lo_mackinlay_vr_sign_doc_present():
    doc = (TsLoMackinlayVr.__doc__ or "") + "\n" + (TsVarianceRatioProxy.__doc__ or "")
    assert "TRENDING" in doc.upper(), doc
    assert "MEAN-REVERSION" in doc.upper(), doc


def test_m045_sign_direction_empirical():
    """Empirical sign check: a trending walk gives VR>1 (z>0); an OU path gives
    VR<1 (z<0)."""
    rng = np.random.default_rng(14)
    n = 2000
    # trending: random walk with positive drift -> positive serial correlation
    drift = np.cumsum(rng.standard_normal(n) * 0.1 + 0.01) + 100.0
    # mean-reverting: stationary OU
    phi = 0.95
    ou = np.zeros(n)
    for t in range(1, n):
        ou[t] = phi * ou[t - 1] + rng.standard_normal() * 0.1
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    op = _get("ts_lo_mackinlay_z")
    z_trend = op.calculate(pd.DataFrame(drift, index=idx, columns=["A"]), window=800, q=10, min_periods=30)["A"].iloc[-1]
    z_mr = op.calculate(pd.DataFrame(ou, index=idx, columns=["A"]), window=800, q=10, min_periods=30)["A"].iloc[-1]
    assert np.isfinite(z_trend) and np.isfinite(z_mr)
    assert z_trend > 0.0, f"trending walk should give z>0, got {z_trend}"
    assert z_mr < 0.0, f"OU path should give z<0, got {z_mr}"


# ---------------------------------------------------------------------------
# M-050: in-sample beta-spread regressions tagged diagnostic_only
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["ts_quantile_beta_spread", "ts_expectile_beta_spread"])
def test_m050_beta_spread_diagnostic_only(name):
    op = _get(name)
    assert "diagnostic_only" in op.metadata.tags, name
    assert "in-sample" in op.metadata.description.lower(), op.metadata.description


# ---------------------------------------------------------------------------
# M-051: prior/forecast-error variants are strict t-1 fits (NOT diagnostic)
# ---------------------------------------------------------------------------

_PRIOR_VARIANTS = [
    "ts_multi_regression_coeff_prior",
    "ts_multi_regression_forecast_error",
    "ts_multi_regression_forecast_error_z",
    "ts_multi_regression_r2_prior",
    "ts_multi_regression_adjusted_r2_prior",
    "ts_huber_regression_coeff_prior",
    "ts_huber_regression_forecast_error",
    "ts_huber_regression_forecast_error_z",
    "ts_ridge_regression_coeff_prior",
    "ts_ridge_regression_forecast_error",
    "ts_ridge_regression_forecast_error_z",
    "ts_expectile_regression_coeff_prior",
    "ts_expectile_regression_forecast_error",
    "ts_quantile_regression_coeff_prior",
]


@pytest.mark.parametrize("name", _PRIOR_VARIANTS)
def test_m051_prior_variants_not_diagnostic(name):
    op = _get(name)
    assert "diagnostic_only" not in op.metadata.tags, name


# ---------------------------------------------------------------------------
# M-052: coeff_stability is a causal model-state alpha, not a diagnostic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["ts_multi_regression_coeff_stability", "ts_ar_coeff_stability"])
def test_m052_coeff_stability_role_documented(name):
    op = _get(name)
    desc = op.metadata.description
    assert "diagnostic" not in desc.lower() or "非诊断" in desc, desc
    assert "model-state" in desc or "model state" in desc or "因果" in desc, desc
    assert "diagnostic_only" not in op.metadata.tags, name


# ---------------------------------------------------------------------------
# M-055: Huber delta is a VERSIONED constant (not free-searched)
# ---------------------------------------------------------------------------

def test_m055_huber_delta_versioned_constant():
    assert _rc._HUBER_DELTA == pytest.approx(1.345)
    # the kernel default binds to the constant (behaviour identical to the old
    # literal 1.345), so it cannot drift out of sync with the versioned policy.
    import inspect

    sig = inspect.signature(_rc.huber_fit)
    assert sig.parameters["delta"].default is _rc._HUBER_DELTA


def test_m055_huber_fit_produces_sensible_coefficients():
    rng = np.random.default_rng(15)
    x = rng.standard_normal(200)
    y = 2.0 * x + 0.5 + rng.standard_normal(200) * 0.1
    # inject a couple of extreme outliers
    y[:5] += 1e3
    design = np.column_stack([np.ones(200), x])
    beta = _rc.huber_fit(design, y)
    assert beta is not None
    assert beta[1] == pytest.approx(2.0, abs=0.2)


# ---------------------------------------------------------------------------
# M-057: ts_quantile_beta_spread_prior exists and is a strict t-1 fit
# ---------------------------------------------------------------------------

def test_m057_quantile_beta_spread_prior_registered():
    op = _get("ts_quantile_beta_spread_prior")
    assert "diagnostic_only" not in op.metadata.tags
    assert "prior" in op.metadata.description.lower(), op.metadata.description
    assert _get("ts_quantile_beta_spread") is not None  # sibling still present


def test_m057_quantile_beta_spread_prior_is_causal():
    """Perturbing rows AFTER the decision row must not change outputs at or
    before that row (strict prior fit_lag=1)."""
    n = 220
    split = 160
    base = _get("ts_quantile_beta_spread_prior").calculate(
        *_reg_panels(n, seed=16), window=90, q_high=0.9, q_low=0.1, min_periods=20
    )
    y, x = _reg_panels(n, seed=16)
    xm = x.copy()
    for col in xm.columns:
        xm.iloc[split + 1 :, xm.columns.get_loc(col)] = 1e100
    ym = y.copy()
    for col in ym.columns:
        ym.iloc[split + 1 :, ym.columns.get_loc(col)] = -1e100
    poisoned = _get("ts_quantile_beta_spread_prior").calculate(
        ym, xm, window=90, q_high=0.9, q_low=0.1, min_periods=20
    )
    # outputs at rows <= split-1 (windows that never touch poisoned rows) match
    a = base.to_numpy(dtype=float)[: split - 1]
    b = poisoned.to_numpy(dtype=float)[: split - 1]
    mask = np.isfinite(a) & np.isfinite(b)
    assert mask.sum() > 0
    np.testing.assert_allclose(a[mask], b[mask], rtol=1e-9, atol=1e-12)


def test_m057_prior_differs_from_in_sample():
    """The prior variant is NOT the in-sample spread on the same window."""
    y, x = _reg_panels(220, seed=17)
    prior = _get("ts_quantile_beta_spread_prior").calculate(
        y, x, window=90, q_high=0.9, q_low=0.1, min_periods=20
    ).to_numpy(dtype=float)
    insample = _get("ts_quantile_beta_spread").calculate(
        y, x, window=90, q_high=0.9, q_low=0.1, min_periods=20
    ).to_numpy(dtype=float)
    both = np.isfinite(prior) & np.isfinite(insample)
    assert both.sum() > 20
    # they are not bit-identical (strict-prior fit_lag=1 vs in-sample fit_lag=0)
    assert not np.allclose(prior[both], insample[both], rtol=1e-12, atol=1e-12)


def _reg_panels(n, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    x = pd.DataFrame(rng.standard_normal((n, 3)), index=idx, columns=list("ABC"))
    y = 1.5 * x + 0.3 + rng.standard_normal((n, 3)) * 0.01
    return y, x


# ---------------------------------------------------------------------------
# M-060: iterative-convergence telemetry (fail-closed + last_fit_status)
# ---------------------------------------------------------------------------

def test_m060_expectile_fail_closed_on_non_convergence():
    rng = np.random.default_rng(18)
    x = rng.standard_normal(120)
    y = 1.5 * x + 0.3 + rng.standard_normal(120) * 0.01
    design = np.column_stack([np.ones(120), x])
    # zero-iteration budget cannot converge -> None + non_converged telemetry
    assert _rc.quantile_fit(design, y, 0.5, iterations=0) is None
    assert _rc.last_fit_status() == {"converged": False, "reason": "non_converged"}


def test_m060_expectile_converged_telemetry():
    rng = np.random.default_rng(19)
    x = rng.standard_normal(120)
    y = 1.5 * x + 0.3 + rng.standard_normal(120) * 0.01
    design = np.column_stack([np.ones(120), x])
    beta = _rc.quantile_fit(design, y, 0.5)
    assert beta is not None
    assert _rc.last_fit_status()["converged"] is True
    assert _rc.last_fit_status()["reason"] == "converged"
    # the honest alias is the same kernel
    assert _rc.expectile_fit is _rc.quantile_fit


def test_m060_huber_failure_reasons():
    # singular design -> singular
    bad = np.column_stack([np.ones(10), np.zeros(10)])
    assert _rc.huber_fit(bad, np.arange(10.0)) is None
    assert _rc.last_fit_status()["reason"] == "singular"
    # insufficient sample -> insufficient_sample
    assert _rc.huber_fit(np.ones((3, 5)), np.arange(3.0)) is None
    assert _rc.last_fit_status()["reason"] == "insufficient_sample"
    # well-posed -> converged
    rng = np.random.default_rng(20)
    x = rng.standard_normal(100)
    design = np.column_stack([np.ones(100), x])
    assert _rc.huber_fit(design, 2.0 * x + rng.standard_normal(100)) is not None
    assert _rc.last_fit_status()["reason"] == "converged"


def test_m060_pinball_quantile_failure_reasons():
    # insufficient sample -> insufficient_sample
    assert _rc.pinball_quantile_fit(np.ones((3, 5)), np.arange(3.0), 0.5) is None
    assert _rc.last_fit_status()["reason"] == "insufficient_sample"
    # well-posed -> converged
    rng = np.random.default_rng(21)
    x = rng.standard_normal(100)
    design = np.column_stack([np.ones(100), x])
    assert _rc.pinball_quantile_fit(design, 2.0 * x + rng.standard_normal(100), 0.5) is not None
    assert _rc.last_fit_status()["reason"] == "converged"


def test_m060_expectile_singular_design_reason():
    # degenerate feature (zero variance) -> singular
    bad = np.column_stack([np.ones(10), np.zeros(10)])
    assert _rc.quantile_fit(bad, np.arange(10.0), 0.5) is None
    assert _rc.last_fit_status()["reason"] == "singular"
