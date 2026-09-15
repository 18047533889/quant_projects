import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.contracts import ExecutionKind
from factor_engine.backend.polars_backend_kind import get_physical_spec
from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.base import MISSING
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.ts_model import dynamic_regression as _dynamic


CANONICALS = (
    "ts_expectile_beta_spread",
    "ts_huber_regression_coeff", "ts_huber_regression_coeff_prior",
    "ts_huber_regression_forecast_error", "ts_huber_regression_forecast_error_z",
    "ts_huber_regression_resid_z",
    "ts_multi_regression_adjusted_r2_prior", "ts_multi_regression_coeff",
    "ts_multi_regression_coeff_prior", "ts_multi_regression_coeff_stability",
    "ts_multi_regression_forecast_error", "ts_multi_regression_forecast_error_z",
    "ts_multi_regression_r2", "ts_multi_regression_r2_prior",
    "ts_multi_regression_resid", "ts_multi_regression_resid_z",
    "ts_quantile_beta_spread", "ts_quantile_regression_coeff",
    "ts_quantile_regression_resid", "ts_ridge_regression_coeff",
    "ts_ridge_regression_coeff_prior", "ts_ridge_regression_forecast_error",
    "ts_ridge_regression_forecast_error_z", "ts_ridge_regression_resid_z",
)


@pytest.fixture(scope="module", autouse=True)
def _fresh_final_registry():
    load_all()


def _fixture(rows=96, cols=2):
    index = pd.date_range("2024-01-01", periods=rows)
    t = np.arange(rows, dtype=float)[:, None]
    c = np.arange(cols, dtype=float)[None, :]
    x = np.sin(t / 5.0 + c * 0.4) + 0.025 * t + 0.1 * c
    # Deterministic heteroskedastic, asymmetric residuals keep robust and tail
    # fits identifiable without relying on random state.
    noise = (0.08 + 0.002 * t) * np.cos(t / 3.0 + c * 0.7)
    y = 1.25 + 2.5 * x + noise
    return (
        pd.DataFrame(y, index=index, columns=[f"S{i}" for i in range(cols)]),
        pd.DataFrame(x, index=index, columns=[f"S{i}" for i in range(cols)]),
    )


def _calculate(canonical, y, x):
    op = OperatorRegistry._operators[canonical]["pandas_numpy"]
    kwargs = {"window": 30, "min_periods": 10}
    if "coeff" in canonical and any(family in canonical for family in ("multi_regression", "huber", "ridge")):
        kwargs["coefficient_index"] = 1
    return op.calculate(y=y, x1=x, **kwargs) if "multi_regression" in canonical or "huber" in canonical or "ridge" in canonical else op.calculate(y=y, x=x, **kwargs)


def test_dynamic_regression_inventory_has_authoritative_contracts():
    assert set(CANONICALS).issubset(_dynamic._CANONICALS)
    for canonical in CANONICALS:
        meta = OperatorRegistry._operators[canonical]["pandas_numpy"].metadata
        assert set(meta.panel_params) | set(meta.param_specs) == set(meta.param_names)
        assert set(meta.panel_params).isdisjoint(meta.param_specs)
        assert all(spec.default is not MISSING for spec in meta.param_specs.values())


@pytest.mark.parametrize("canonical", CANONICALS)
def test_each_dynamic_regression_is_finite_and_prefix_causal(canonical):
    y, x = _fixture()
    full = _calculate(canonical, y, x)
    prefix = _calculate(canonical, y.iloc[:70].copy(), x.iloc[:70].copy())
    assert full.shape == y.shape
    assert np.isfinite(full.to_numpy(dtype=float)).any(), canonical
    np.testing.assert_allclose(full.iloc[:70], prefix, equal_nan=True, rtol=1e-9, atol=1e-9)


def test_simple_ols_outputs_match_independent_lstsq_oracle():
    y, x = _fixture(cols=1)
    window, row = 30, 80
    ys = y.iloc[row - window + 1:row + 1, 0].to_numpy()
    xs = x.iloc[row - window + 1:row + 1, 0].to_numpy()
    design = np.column_stack([np.ones(window), xs])
    beta, *_ = np.linalg.lstsq(design, ys, rcond=None)
    residual = ys[-1] - design[-1] @ beta
    ss_res = np.sum((ys - design @ beta) ** 2)
    ss_tot = np.sum((ys - ys.mean()) ** 2)

    coeff = _calculate("ts_multi_regression_coeff", y, x)
    resid = _calculate("ts_multi_regression_resid", y, x)
    r2 = _calculate("ts_multi_regression_r2", y, x)
    assert coeff.iloc[row, 0] == pytest.approx(beta[1], rel=1e-10, abs=1e-10)
    assert resid.iloc[row, 0] == pytest.approx(residual, rel=1e-10, abs=1e-10)
    assert r2.iloc[row, 0] == pytest.approx(1.0 - ss_res / ss_tot, rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("canonical", [
    "ts_multi_regression_coeff_prior", "ts_multi_regression_r2_prior",
    "ts_multi_regression_adjusted_r2_prior", "ts_multi_regression_coeff_stability",
    "ts_huber_regression_coeff_prior", "ts_ridge_regression_coeff_prior",
])
def test_prior_fit_state_ignores_current_row(canonical):
    y, x = _fixture(cols=1)
    row = 80
    baseline = _calculate(canonical, y, x)
    changed_y, changed_x = y.copy(), x.copy()
    changed_y.iloc[row, 0] += 7.0
    changed_x.iloc[row, 0] -= 3.0
    perturbed = _calculate(canonical, changed_y, changed_x)
    assert perturbed.iloc[row, 0] == pytest.approx(baseline.iloc[row, 0])


@pytest.mark.parametrize("canonical", [
    "ts_multi_regression_forecast_error", "ts_multi_regression_forecast_error_z",
    "ts_huber_regression_forecast_error", "ts_huber_regression_forecast_error_z",
    "ts_ridge_regression_forecast_error", "ts_ridge_regression_forecast_error_z",
])
def test_prior_forecast_uses_current_observation_without_fitting_it(canonical):
    y, x = _fixture(cols=1)
    row = 80
    baseline = _calculate(canonical, y, x)
    changed = y.copy()
    changed.iloc[row, 0] += 7.0
    perturbed = _calculate(canonical, changed, x)
    assert perturbed.iloc[row, 0] != pytest.approx(baseline.iloc[row, 0])


def test_in_sample_diagnostic_remains_current_observation_sensitive():
    y, x = _fixture(cols=1)
    row = 80
    baseline = _calculate("ts_multi_regression_coeff", y, x)
    changed = y.copy()
    changed.iloc[row, 0] += 7.0
    perturbed = _calculate("ts_multi_regression_coeff", changed, x)
    assert perturbed.iloc[row, 0] != pytest.approx(baseline.iloc[row, 0])
    assert "diagnostic_only" in OperatorRegistry._operators["ts_multi_regression_coeff"]["pandas_numpy"].metadata.tags


@pytest.mark.parametrize("canonical", [
    "ts_multi_regression_coeff", "ts_huber_regression_coeff", "ts_ridge_regression_coeff",
])
def test_unidentifiable_design_fails_closed_without_fabricated_fit(canonical):
    y, x = _fixture(cols=1)
    constant_x = x * 0.0 + 1.0
    result = _calculate(canonical, y, constant_x)
    assert result.isna().all().all()


def test_full_load_polars_fallback_kwargs_and_positional_match_pandas():
    pl = pytest.importorskip("polars")
    y, x = _fixture(rows=72, cols=1)
    py = pl.DataFrame({"S0": y["S0"].to_list()})
    px = pl.DataFrame({"S0": x["S0"].to_list()})
    for canonical in CANONICALS:
        pandas_result = _calculate(canonical, y, x).to_numpy(dtype=float)
        op = OperatorRegistry.get(canonical, "polars", mode="research")
        assert op is not None, f"{canonical}: final registry lacks a Polars backend"
        spec = get_physical_spec(op)
        assert spec is not None, f"{canonical}: Polars backend lacks an explicit physical spec"
        assert spec.execution_kind in {
            ExecutionKind.POLARS_PANDAS_DELEGATE,
            ExecutionKind.POLARS_NUMPY_KERNEL,
        }, canonical
        assert len(spec.implementation_source_hash) == 64, canonical
        int(spec.implementation_source_hash, 16)
        if spec.execution_kind is ExecutionKind.POLARS_PANDAS_DELEGATE:
            assert spec.materializes_full_panel
            assert not spec.supports_lazy and not spec.supports_streaming
        if any(family in canonical for family in ("multi_regression", "huber", "ridge")):
            kwargs = dict(y=py, x1=px, x2=None, x3=None, x4=None, window=30,
                          coefficient_index=1, min_periods=10,
                          add_intercept=True, warmup_policy="expanding")
            positional = (py, px, None, None, None, 30, 1, 10, True, "expanding")
        elif "beta_spread" in canonical:
            kwargs = dict(y=py, x=px, window=30, q_high=0.9, q_low=0.1, min_periods=10)
            positional = (py, px, 30, 0.9, 0.1, 10)
        else:
            kwargs = dict(y=py, x=px, window=30, q=0.5, min_periods=10)
            positional = (py, px, 30, 0.5, 10)
        by_kwargs = op.calculate(**kwargs)
        by_position = op.calculate(*positional)
        kw_values = np.asarray(by_kwargs["S0"].to_list(), dtype=float)[:, None]
        pos_values = np.asarray(by_position["S0"].to_list(), dtype=float)[:, None]
        np.testing.assert_allclose(kw_values, pandas_result, equal_nan=True, rtol=1e-8, atol=1e-8,
                                   err_msg=f"{canonical} kwargs")
        np.testing.assert_allclose(pos_values, pandas_result, equal_nan=True, rtol=1e-8, atol=1e-8,
                                   err_msg=f"{canonical} positional")
