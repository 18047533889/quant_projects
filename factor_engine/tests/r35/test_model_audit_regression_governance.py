# -*- coding: utf-8 -*-
"""P1 regression / AR / GARCH / HAR window-semantics & parameter-governance audit.

Audit items (parent task: 回归/AR/GARCH/HAR 算子窗口语义与参数治理 P0/P1):

  1. AR / rolling-regression ``window`` is a MAX LOOKBACK, not a strict full
     window; the operator contract explicitly carries ``min_history`` /
     ``min_effective_obs`` and ``warmup_policy = expanding | full``.  Short
     series behave per ``warmup_policy``: expanding emits early (once the
     minimum effective observation count is present), full degrades to NaN when
     the window cannot be filled.
  2. Rolling-regression no-intercept R² definition: versioned CENTERED
     ``SS_tot = sum((y - mean(y))**2)`` for both intercept and no-intercept
     (no separate through-origin uncentered R² is exposed); adjusted-R² degrees
     of freedom pair with the intercept convention (``n - p - 1`` with an
     intercept, ``n - p`` without).
  3. Huber delta / Ridge alpha estimator policy: fixed values are declared
     NUMERICAL / ESTIMATOR policy, ``searchable=False``, semantic-versioned;
     an exposed alpha is never a continuous AlphaProbe/LLM search dimension.
  4. GARCH / GJR ``window`` parameter domain: a below-minimum window (guaranteed
     all-NaN by construction) is rejected at binding time.
  5. HAR sample-coverage double constraint: ``N_effective >= N_min`` AND
     ``N_effective / window >= coverage_min`` (25/120 rejected, 90/120 normal).

NOTE on imports: these tests import the operator modules DIRECTLY instead of
calling ``load_all()`` because the repo-wide registration audit currently fails
on an UNRELATED ``ts_first_passage_bias/polars`` backend-arity mismatch
(pre-existing, outside this task's file ownership).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cleaned_operators.base import ParamRole  # noqa: E402
from cleaned_operators.registry import OperatorRegistry  # noqa: E402

# Direct module imports register each family's canonicals + aliases (no load_all).
import cleaned_operators.regression_models as RM  # noqa: E402
import cleaned_operators.ts_model.ar_meanrev as AM  # noqa: E402
import cleaned_operators.ts_model.dynamic_regression as DR  # noqa: E402
import cleaned_operators.ts_model.volatility as V  # noqa: E402


def _get(name: str):
    op = OperatorRegistry.get(name, "pandas_numpy")
    assert op is not None, f"{name} not registered"
    return op


def _series(n: int, seed: int = 0, cols: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(rng.standard_normal((n, cols)), index=idx,
                        columns=list("ABCD")[:cols])


# ---------------------------------------------------------------------------
# Audit 1 — window = max lookback + warmup_policy contract (AR & rolling reg)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    ["ts_ar_prior_forecast", "ts_ar_prior_innovation_z", "ts_ar_coeff_stability"],
)
def test_ar_window_contract_documented(name):
    op = _get(name)
    assert op.metadata.window_semantics == "max_lookback", name
    assert "warmup_policy" in op.metadata.param_names, name
    spec = op.metadata.param_specs["warmup_policy"]
    assert spec.searchable is False, name
    assert spec.param_role == ParamRole.POLICY, name
    assert set(spec.choices) == {"expanding", "full"}, name
    # description carries the max-lookback / warmup contract
    desc = op.metadata.description
    assert "max lookback" in desc, desc
    assert "warmup_policy" in desc, desc


def test_ar_short_series_expanding_vs_full():
    """A series shorter than ``window``: expanding emits once the minimum
    effective observation count (order+2) is present; full is all-NaN."""
    x = _series(24, seed=1)
    op = _get("ts_ar_prior_forecast")
    exp = op.calculate(x, window=60, order=1, warmup_policy="expanding")["A"].to_numpy()
    full = op.calculate(x, window=60, order=1, warmup_policy="full")["A"].to_numpy()
    assert np.isnan(full).all(), "full warmup on a short series must be all-NaN"
    assert np.isfinite(exp).sum() > 0, "expanding must emit early output"
    assert np.isnan(exp[:3]).all(), "early rows have too few effective obs"


def test_ar_full_floor_first_finite_row_equals_window():
    """``warmup_policy='full'`` on a series LONGER than window: first finite row
    is exactly ``window`` (fit_lag=1 -> fit_end = row-1 >= window-1)."""
    x = _series(80, seed=2)
    op = _get("ts_ar_prior_forecast")
    full = op.calculate(x, window=20, order=1, warmup_policy="full")["A"].to_numpy()
    first = int(np.argmax(np.isfinite(full)))
    assert np.isnan(full[:20]).all()
    assert first == 20
    # expanding emits strictly earlier than the full-history floor
    exp = op.calculate(x, window=20, order=1, warmup_policy="expanding")["A"].to_numpy()
    assert int(np.argmax(np.isfinite(exp)) < 20


def test_ar_invalid_warmup_policy_rejected():
    op = _get("ts_ar_prior_forecast")
    with pytest.raises(ValueError):
        op.calculate(_series(20, seed=3), window=60, warmup_policy="bogus")


@pytest.mark.parametrize(
    "name",
    ["ts_multi_regression_coeff_prior", "ts_multi_regression_r2_prior",
     "ts_multi_regression_adjusted_r2_prior"],
)
def test_multi_regression_window_contract_documented(name):
    op = _get(name)
    assert op.metadata.window_semantics == "max_lookback", name
    assert "warmup_policy" in op.metadata.param_names, name
    spec = op.metadata.param_specs["warmup_policy"]
    assert spec.searchable is False and spec.param_role == ParamRole.POLICY, name
    assert "max lookback" in op.metadata.description


def test_multi_regression_short_series_expanding_vs_full():
    y = _series(30, seed=4)
    x = _series(30, seed=5)
    op = _get("ts_multi_regression_coeff_prior")
    exp = op.calculate(y, x, window=60, min_periods=10, warmup_policy="expanding")["A"].to_numpy()
    full = op.calculate(y, x, window=60, min_periods=10, warmup_policy="full")["A"].to_numpy()
    assert np.isnan(full).all(), "full warmup on a short series must be all-NaN"
    assert np.isfinite(exp).sum() > 0, "expanding must emit early output"
    assert np.isnan(exp[:9]).all(), "early rows have too few effective obs"


def test_ts_ar_coefficient_warmup_policy():
    op = _get("ts_ar_coefficient")
    assert op.metadata.window_semantics == "max_lookback"
    assert "warmup_policy" in op.metadata.param_names
    x = _series(24, seed=6)
    exp = op.calculate(x, window=60, lag=1, min_periods=5, warmup_policy="expanding")["A"].to_numpy()
    full = op.calculate(x, window=60, lag=1, min_periods=5, warmup_policy="full")["A"].to_numpy()
    assert np.isnan(full).all()
    assert np.isfinite(exp).sum() > 0


# ---------------------------------------------------------------------------
# Audit 2 — no-intercept R² definition (versioned centered SS_tot) + adj-R² df
# ---------------------------------------------------------------------------

def test_r2_definition_versioned_centered_ss_tot():
    assert DR._R2_DEFINITION == "centered_ss_tot"
    assert DR._R2_ADJ_TOTAL_DF_EXPR == "n - 1"
    assert DR._R2_ADJ_ERROR_DF_EXPR == "n - p - (1 if add_intercept else 0)"


def test_no_intercept_r2_uses_centered_ss_tot_and_matches_manual():
    """y = 10 + 2x with a no-intercept (through-the-origin) fit: the reported R²
    must equal the manual CENTERED definition and must differ from the uncentered
    through-origin convention ``1 - sum(e**2)/sum(y**2)``."""
    n = 40
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    x = pd.DataFrame(np.arange(1.0, n + 1), index=idx, columns=["A"])
    y = pd.DataFrame(10.0 + 2.0 * np.arange(1.0, n + 1), index=idx, columns=["A"])
    op = _get("ts_multi_regression_r2_prior")
    out = op.calculate(y, x, window=30, min_periods=5,
                       add_intercept=False, coefficient_index=0)["A"].iloc[-1]
    w = 30
    yv = y["A"].to_numpy()[n - w - 1 : n - 1]
    xv = x["A"].to_numpy()[n - w - 1 : n - 1]
    b = float(np.dot(xv, yv) / np.dot(xv, xv))          # through-origin OLS slope
    e = yv - b * xv
    ss_res = float(np.sum(e * e))
    ss_tot_centered = float(np.sum((yv - np.mean(yv)) ** 2))
    manual_centered = 1.0 - ss_res / ss_tot_centered
    manual_uncentered = 1.0 - ss_res / float(np.sum(yv * yv))
    assert out == pytest.approx(manual_centered, abs=1e-12)
    # the two conventions genuinely differ on this data -> proves the definition
    assert out != pytest.approx(manual_uncentered, abs=1e-6)


def test_adjusted_r2_df_pairs_with_intercept():
    """p = 1 feature: intercept=True -> df_error = n - 2; intercept=False ->
    df_error = n - 1, so adjusted R² equals R² exactly."""
    n = 40
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    x = pd.DataFrame(rng.standard_normal(n), index=idx, columns=["A"])
    y = pd.DataFrame(1.5 * x["A"] + 0.3 + rng.standard_normal(n) * 0.01, index=idx, columns=["A"])
    op_r2 = _get("ts_multi_regression_r2_prior")
    op_adj = _get("ts_multi_regression_adjusted_r2_prior")
    n_obs = 30  # trailing window rows in the fit

    r2_t = op_r2.calculate(y, x, window=30, min_periods=5, add_intercept=True)["A"].iloc[-1]
    adj_t = op_adj.calculate(y, x, window=30, min_periods=5, add_intercept=True)["A"].iloc[-1]
    # intercept=True: df_total = n-1, df_error = n - p - 1 = n - 2
    assert adj_t == pytest.approx(1 - (1 - r2_t) * (n_obs - 1) / (n_obs - 2), abs=1e-12)

    r2_f = op_r2.calculate(y, x, window=30, min_periods=5,
                           add_intercept=False, coefficient_index=0)["A"].iloc[-1]
    adj_f = op_adj.calculate(y, x, window=30, min_periods=5,
                             add_intercept=False, coefficient_index=0)["A"].iloc[-1]
    # intercept=False: df_error = n - p = n - 1 == df_total -> adj R² == R²
    assert adj_f == pytest.approx(r2_f, abs=1e-12)


# ---------------------------------------------------------------------------
# Audit 3 — Huber delta / Ridge alpha estimator policy (fixed / non-searchable)
# ---------------------------------------------------------------------------

def test_huber_delta_is_fixed_numerical_policy():
    assert RM._HUBER_DELTA_POLICY == pytest.approx(1.345)
    assert RM._HUBER_DELTA_ROLE == ParamRole.NUMERICAL
    # the operators do not expose delta as a searchable/declared parameter
    for name in ("ts_huber_regression_in_sample_resid", "ts_huber_regression_predictive_resid"):
        op = _get(name)
        assert "delta" not in op.metadata.param_names, name


def test_ridge_alpha_is_numerical_and_not_searchable():
    for name in ("ts_ridge_regression_in_sample_resid", "ts_ridge_regression_predictive_resid"):
        op = _get(name)
        spec = op.metadata.param_specs["alpha"]
        assert spec.searchable is False, name
        assert spec.param_role == ParamRole.NUMERICAL, name
        assert spec.min == 0.0, name
        desc = op.metadata.description
        assert "NUMERICAL" in desc and "searchable" in desc, desc


def test_ridge_alpha_certified_presets_documented():
    assert RM._RIDGE_ALPHA_CERTIFIED_PRESETS == (0.0, 0.1, 0.5, 1.0)


def test_ridge_alpha_hardcoded_family_exposes_no_alpha_param():
    """ts_ridge_regression_coeff_prior hard-codes alpha=0.1 inside the kernel
    (never a user parameter) — passing alpha externally is rejected as an
    undeclared keyword."""
    op = _get("ts_ridge_regression_coeff_prior")
    assert "alpha" not in op.metadata.param_names
    y = _series(30, seed=8)
    x = _series(30, seed=9)
    with pytest.raises(ValueError):
        op.calculate(y, x, window=30, alpha=0.5)


# ---------------------------------------------------------------------------
# Audit 4 — GARCH/GJR window parameter domain (below-min -> binding-time raise)
# ---------------------------------------------------------------------------

def test_garch_min_window_contract_constants():
    assert V._GARCH_MIN_WINDOW == 13
    assert V._GARCH_MIN_FIT_OBS == 12
    assert V._GARCH_PARAM_SPECS["window"].min == 13


@pytest.mark.parametrize(
    "name",
    ["ts_garch_next_vol_forecast", "ts_garch_vol_surprise", "ts_garch_persistence",
     "ts_garch_standardized_shock", "ts_gjr_garch_vol_forecast", "ts_gjr_leverage"],
)
def test_garch_window_below_min_rejected_at_binding_time(name):
    op = _get(name)
    rets = _series(60, seed=10)
    with pytest.raises(ValueError, match="window"):
        op.calculate(rets, window=12)
    # the minimum window is accepted (and is not all-NaN by construction)
    out = op.calculate(rets, window=13)["A"]
    assert np.isfinite(out.to_numpy()).any()


def test_garch_kernel_raises_on_below_min_window():
    rng = np.random.default_rng(11)
    rets = rng.standard_normal(60) * 0.02
    with pytest.raises(ValueError):
        V._garch_path(rets, 12, "forecast", False, 0.0)
    with pytest.raises(ValueError):
        V._garch_vol_surprise(rets, 12)
    with pytest.raises(ValueError):
        V._gjr_leverage(rets, 12)


def test_garch_window_semantics_documented():
    op = _get("ts_garch_persistence")
    assert op.metadata.window_semantics == "max_lookback"
    assert "max lookback" in op.metadata.description


# ---------------------------------------------------------------------------
# Audit 5 — HAR sample-coverage double constraint
# ---------------------------------------------------------------------------

def test_har_coverage_constants():
    assert V._HAR_MIN_TRAIN_OBS == 25
    assert V._HAR_MIN_COVERAGE == pytest.approx(0.6)
    assert V._HAR_MIN_WINDOW == 30
    assert V._HAR_PARAM_SPECS["window"].min == 30


def test_har_thin_coverage_rejected():
    """window=120 with only ~25 effective training rows (coverage ~1/5) must be
    rejected (NaN) by the double gate N_eff >= 25 AND N_eff >= 0.6*120."""
    rng = np.random.default_rng(12)
    rv = pd.DataFrame(np.abs(rng.standard_normal(300)) * 1e-4 + 1e-4, columns=["A"])
    arr = rv["A"].to_numpy()
    arr[:255] = np.nan          # leave a trailing finite run of ~45 rows
    rv["A"] = arr
    op = _get("ts_har_rv_next_vol_forecast")
    out = op.calculate(rv, window=120)["A"]
    assert np.isnan(out.to_numpy()[-1]), "thin-coverage HAR must fail closed to NaN"
    assert np.isnan(out.to_numpy()).all(), "every row with thin coverage must be NaN"


def test_har_good_coverage_emits():
    """window=120 on a full 300-row panel (~97 effective rows, coverage ~0.81)
    is a normal case and must produce a finite forecast."""
    rng = np.random.default_rng(13)
    rv = pd.DataFrame(np.abs(rng.standard_normal(300)) * 1e-4 + 1e-4, columns=["A"])
    op = _get("ts_har_rv_next_vol_forecast")
    out = op.calculate(rv, window=120)["A"]
    assert np.isfinite(out.to_numpy()[-1])


def test_har_kernel_double_gate():
    """Direct-kernel check of the double gate: 25/120 coverage -> NaN; a
    full-length 120-window (normal coverage) -> finite."""
    rng = np.random.default_rng(14)
    full = np.abs(rng.standard_normal(300)) + 1.0
    assert np.isfinite(V._har_rv(full, 120, "forecast"))
    thin = full.copy()
    thin[:255] = np.nan  # ~25 effective training rows in a 120 window
    assert np.isnan(V._har_rv(thin, 120, "forecast"))


def test_har_window_below_min_rejected():
    op = _get("ts_har_rv_next_vol_forecast")
    rv = _series(60, seed=15)
    with pytest.raises(ValueError, match="window"):
        op.calculate(rv, window=29)
    with pytest.raises(ValueError):
        V._har_rv(np.abs(np.random.default_rng(1).standard_normal(40)) + 1.0, 29, "forecast")


def test_har_window_semantics_documented():
    op = _get("ts_har_rv_next_vol_forecast")
    assert op.metadata.window_semantics == "max_lookback"
    assert "max lookback" in op.metadata.description
    assert "0.6" in op.metadata.description
