import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.ts_model import dynamic_regression as dr
from factor_engine.cleaned_operators.ts_model import sequence_anomaly as sa
from factor_engine.cleaned_operators.ts_model import volatility as vol
from factor_engine.cleaned_operators.candle_state_space import _matrix_profile_series
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _frame(values):
    return pd.DataFrame({"A": np.asarray(values, dtype=float)}, index=pd.date_range("2024-01-01", periods=len(values)))


@pytest.mark.parametrize("fit_fn,extra", [(dr.ols_fit, None), (dr.huber_fit, None), (dr.ridge_fit, 0.1)])
@pytest.mark.parametrize("stat", ["resid", "resid_z"])
def test_h18_prediction_requires_finite_current_features_and_output(fit_fn, extra, stat):
    t = np.arange(40.0)
    x = np.linspace(-2.0, 3.0, 40)
    y = 3.0 + 2.0 * x + 0.1 * np.sin(t)
    normal = dr._multi_regression(_frame(y), [_frame(x)], 20, 10, True, fit_fn, extra, stat, 1, fit_lag=1)
    broken_x = x.copy()
    broken_x[-1] = np.inf
    broken = dr._multi_regression(_frame(y), [_frame(broken_x)], 20, 10, True, fit_fn, extra, stat, 1, fit_lag=1)
    assert np.isfinite(normal.iloc[-1, 0])
    assert np.isnan(broken.iloc[-1, 0])
    assert broken.attrs["prediction_status_counts"]["PREDICTION_INPUT_NONFINITE"] >= 1
    pd.testing.assert_series_equal(normal.iloc[:-1, 0], broken.iloc[:-1, 0])


def test_h19_fixed_five_fit_support_and_single_fit_per_cutoff():
    t = np.arange(40.0)
    x = np.linspace(-2.0, 3.0, 40)
    y = 3.0 + 2.0 * x + 0.1 * np.sin(t)
    calls = []

    def counted(design, target):
        calls.append(len(target))
        return dr.ols_fit(design, target)

    out = dr._multi_regression(
        _frame(y), [_frame(x)], 20, 10, True, counted, None,
        "coeff", 1, fit_lag=1, stability_k=5,
    ).iloc[:, 0]
    assert out.first_valid_index() == out.index[14]
    assert len(calls) == 30
    assert np.isfinite(out.iloc[-1])


def test_h18_zero_residual_scale_has_separate_prediction_reason():
    x = np.arange(30.0)
    y = 3.0 + 2.0 * x
    out = dr._multi_regression(
        _frame(y), [_frame(x)], 20, 10, True, dr.ols_fit, None,
        "resid_z", 1, fit_lag=1,
    )
    assert np.isnan(out.iloc[-1, 0])
    assert out.attrs["prediction_status_counts"]["PREDICTION_ZERO_SCALE"] >= 1


def test_h18_public_calculate_preserves_queryable_prediction_reason():
    x = np.linspace(-2.0, 3.0, 40)
    y = 3.0 + 2.0 * x + 0.1 * np.sin(np.arange(40.0))
    x[-1] = np.inf
    op = OperatorRegistry.get("ts_multi_regression_forecast_error")
    out = op.calculate(_frame(y), _frame(x), window=20, min_periods=10)
    assert out.attrs["prediction_status_counts"] == {"PREDICTION_INPUT_NONFINITE": 1}


def test_h20_recurrence_is_actual_match_count_not_frequency():
    rng = np.random.default_rng(9)
    values = rng.normal(size=35)
    raw = values[:, None]
    result = _matrix_profile_series(raw, 35, 4, 35, return_counts=True)
    matches, eligible = result[4], result[5]
    got = sa._mp_stats(values, 4, "recurrence", 35)
    assert eligible[-1, 0] >= 3
    assert got == matches[-1, 0]
    assert got == float(int(got))
    assert sa.OperatorRegistry._catalog["ts_motif_recurrence_count"]["semantic_version"] == "2.0"


def test_h21_static_matrix_profile_geometry_rejects_before_scan():
    values = np.linspace(0.0, 1.0, 45)
    with pytest.raises(ValueError, match="INFEASIBLE_PARAMETER_DOMAIN"):
        sa._mp_stats(values, 4, "discord", 5)
    with pytest.raises(ValueError, match="INFEASIBLE_PARAMETER_DOMAIN"):
        sa._mp_stats(values, 4, "recurrence", 7)
    with pytest.raises(ValueError, match="INFEASIBLE_PARAMETER_DOMAIN"):
        sa._mp_stats(values, 16, "discord", 20)
    assert np.isfinite(sa._mp_stats(values + 0.1 * np.sin(np.arange(45)), 4, "recurrence", 8))
    expressions = [spec.expression for spec in sa._MP_RELATIONAL_SPECS]
    assert "history_window >= m + m // 4 + 1" in expressions


def test_h31_har_derived_window_boundaries():
    rng = np.random.default_rng(19)
    rv = np.exp(rng.normal(scale=0.3, size=80))
    with pytest.raises(ValueError, match="derived minimum 55"):
        vol._har_rv(rv[-54:], 54, "forecast")
    assert np.isfinite(vol._har_rv(rv[-55:], 55, "forecast"))
    with pytest.raises(ValueError, match="derived minimum 58"):
        vol._har_rv(rv[-57:], 57, "innovation_z")
    assert np.isfinite(vol._har_rv(rv[-58:], 58, "innovation_z"))
    assert vol._HAR_FORECAST_PARAM_SPECS["window"].min == 55
    assert vol._HAR_ERROR_PARAM_SPECS["window"].min == 58
    assert vol._derive_har_min_window(error=False) == 55
    assert vol._derive_har_min_window(error=True) == 58
