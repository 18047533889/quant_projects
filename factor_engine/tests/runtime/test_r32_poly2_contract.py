"""Independent contracts for the repaired R32 polynomial delegates and specs."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.contracts import ExecutionKind
from factor_engine.backend.polars_backend_kind import get_physical_spec
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


REPAIRED = (
    "ts_ar_fitted_value", "ts_ar_forecast", "ts_ar_innovation",
    "ts_ar_innovation_z", "ts_ar_in_sample_resid", "ts_ar_coeff_stability",
    "ts_ar_prior_coeff", "ts_ar_prior_forecast", "ts_ar_prior_innovation",
    "ts_ar_prior_innovation_z", "ts_huber_regression_in_sample_resid",
    "ts_ridge_regression_in_sample_resid", "ts_quantile_regression_coeff_prior",
    "ts_poly2_forecast_error", "ts_poly2_forecast_error_z",
    "ts_poly2_prior_coeff",
)


def _oracle(values, d, kind):
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    bars = np.arange(d, dtype=float)
    for row in range(d, len(values)):
        past = values[row - d:row]
        valid = np.isfinite(past)
        minimum = 4 if kind == "z" else 3
        if valid.sum() < minimum:
            continue
        design = np.column_stack((bars[valid] ** 2, bars[valid], np.ones(valid.sum())))
        coeff = np.linalg.lstsq(design, past[valid], rcond=None)[0]
        if kind == "coeff":
            out[row] = coeff[0]
            continue
        prediction = np.array([d * d, d, 1.0]) @ coeff
        error = values[row] - prediction
        if kind == "error":
            out[row] = error
            continue
        residual = past[valid] - design @ coeff
        scale = residual.std(ddof=1)
        if np.isfinite(scale) and scale > 0:
            out[row] = error / scale
    return out


@pytest.mark.parametrize(
    "name,kind",
    (("ts_poly2_prior_coeff", "coeff"),
     ("ts_poly2_forecast_error", "error"),
     ("ts_poly2_forecast_error_z", "z")),
)
def test_poly2_matches_lstsq_oracle_with_missing_rows_and_irregular_dates(name, kind):
    load_all()
    d = 11
    bar = np.arange(48, dtype=float)
    values = 0.7 - 0.08 * bar + 0.004 * bar ** 2 + 0.13 * np.sin(bar / 2.3)
    values[[5, 14, 22, 35]] = np.nan
    increments = np.resize(np.array([1, 4, 2, 7, 1]), len(values))
    dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(np.cumsum(increments), unit="D")
    pandas_input = pd.DataFrame({"A": values}, index=dates)
    polars_input = pl.DataFrame({"timestamp": dates, "A": values})
    expected = _oracle(values, d, kind)

    pandas_result = OperatorRegistry.get(name, "pandas_numpy", mode="any").calculate(
        pandas_input, d=d
    ).to_numpy()[:, 0]
    polars_result = OperatorRegistry.get(name, "polars", mode="any").calculate(
        polars_input, d=d
    )

    assert polars_result["timestamp"].to_list() == polars_input["timestamp"].to_list()
    assert np.isnan(polars_result["A"].to_numpy()[:d]).all()
    np.testing.assert_allclose(pandas_result, expected, rtol=1e-11, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(
        polars_result["A"].to_numpy(), expected, rtol=1e-11, atol=1e-12, equal_nan=True
    )


def test_poly2_prior_timing_and_current_value_perturbation():
    load_all()
    d, row = 12, 31
    bar = np.arange(50, dtype=float)
    values = 1.1 + 0.02 * bar + 0.003 * bar ** 2 + 0.2 * np.cos(bar / 3.0)
    base = pl.DataFrame({"A": values})
    changed_values = values.copy()
    changed_values[row] += 37.5
    changed = pl.DataFrame({"A": changed_values})

    coeff_op = OperatorRegistry.get("ts_poly2_prior_coeff", "polars", mode="any")
    error_op = OperatorRegistry.get("ts_poly2_forecast_error", "polars", mode="any")
    coeff_base = coeff_op.calculate(base, d=d)["A"].to_numpy()
    coeff_changed = coeff_op.calculate(changed, d=d)["A"].to_numpy()
    error_base = error_op.calculate(base, d=d)["A"].to_numpy()
    error_changed = error_op.calculate(changed, d=d)["A"].to_numpy()

    assert coeff_base[row] == coeff_changed[row]
    assert (error_changed[row] - error_base[row]) == pytest.approx(37.5)
    assert np.isnan(coeff_base[:d]).all() and np.isnan(error_base[:d]).all()


def test_repaired_polars_operators_declare_truthful_cpu_delegate_specs():
    load_all()
    for name in REPAIRED:
        operator = OperatorRegistry.get(name, "polars", mode="any")
        spec = get_physical_spec(operator)
        assert spec is not None
        assert spec.canonical == name
        assert spec.backend == "polars"
        assert spec.execution_kind is ExecutionKind.DELEGATE_PYTHON
        assert not spec.supports_lazy and not spec.supports_streaming
        assert spec.materializes_full_panel and spec.requires_sorted
        assert spec.supports_nulls and spec.supports_nan and not spec.supports_inf
        assert not spec.is_native_execution()
        assert not spec.is_production_eligible()
        assert "CPU" in spec.notes and "Polars" in spec.notes


@pytest.mark.parametrize(
    "name,params",
    (("ts_ridge_regression_in_sample_resid", {"window": 8, "alpha": 0.1, "min_periods": 4}),
     ("ts_huber_regression_in_sample_resid", {"window": 8, "min_periods": 4}),
     ("ts_quantile_regression_coeff_prior", {"window": 12, "q": 0.4, "min_periods": 4})),
)
def test_two_input_repairs_preserve_timestamp_and_reject_axis_mismatch(name, params):
    load_all()
    timestamp = pd.date_range("2025-03-01", periods=24, freq="2D")
    x = pl.DataFrame({"timestamp": timestamp, "A": np.linspace(-1, 2, 24), "B": np.linspace(3, 1, 24)})
    y = pl.DataFrame({"timestamp": timestamp, "A": np.linspace(0, 4, 24), "B": np.linspace(2, -2, 24)})
    operator = OperatorRegistry.get(name, "polars", mode="any")
    result = operator.calculate(y, x, **params)
    assert result["timestamp"].to_list() == y["timestamp"].to_list()
    shifted = x.with_columns((pl.col("timestamp") + pl.duration(days=1)).alias("timestamp"))
    with pytest.raises(ValueError):
        operator.calculate(y, shifted, **params)
    reordered = x.select("timestamp", "B", "A")
    with pytest.raises(ValueError):
        operator.calculate(y, reordered, **params)
