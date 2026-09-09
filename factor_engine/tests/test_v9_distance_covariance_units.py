import numpy as np
import pandas as pd
import pytest


def _kernel():
    from factor_engine.cleaned_operators.nonlinear_dependence import _distance_corr
    return _distance_corr


def _full_reference(x, y):
    """Independent full-matrix biased double-centering reference."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = np.abs(x[:, None] - x[None, :])
    dy = np.abs(y[:, None] - y[None, :])
    A = dx - dx.mean(axis=0) - dx.mean(axis=1)[:, None] + dx.mean()
    B = dy - dy.mean(axis=0) - dy.mean(axis=1)[:, None] + dy.mean()
    dcov2 = np.mean(A * B)
    dvar_x2 = np.mean(A * A)
    dvar_y2 = np.mean(B * B)
    dcov = np.sqrt(max(0.0, dcov2))
    corr = dcov / np.sqrt(np.sqrt(dvar_x2) * np.sqrt(dvar_y2))
    return float(corr), float(dcov)


def test_independent_full_matrix_reference_and_block_error_bound():
    rng = np.random.default_rng(22)
    x = rng.normal(size=137)
    y = x ** 2 + 0.2 * rng.normal(size=137)
    expected = _full_reference(x, y)
    for block in (1, 7, 32, 137):
        actual = _kernel()(x, y, _block_size=block)
        np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)


@pytest.mark.parametrize("factor", [1e-300, 1e-6, 1e6, 1e300, 1e308])
def test_common_unit_scaling_preserves_corr_and_scales_covariance(factor):
    x = np.linspace(-1.0, 1.0, 20)
    expected_corr, expected_cov = _full_reference(x, x)
    corr, covariance = _kernel()(x * factor, x * factor)
    assert corr == pytest.approx(expected_corr, rel=1e-13, abs=1e-13)
    assert covariance == pytest.approx(expected_cov * abs(factor), rel=2e-13, abs=0.0)


def test_independent_marginal_units_restore_sqrt_product_unit():
    rng = np.random.default_rng(220)
    x = rng.normal(size=64)
    y = np.sin(x) + 0.1 * rng.normal(size=64)
    expected_corr, expected_cov = _kernel()(x, y)
    corr, covariance = _kernel()(x * 1e300, y * 1e-300)
    assert corr == pytest.approx(expected_corr, rel=2e-13, abs=2e-13)
    assert covariance == pytest.approx(expected_cov, rel=2e-13, abs=2e-13)


def test_identity_opposite_sign_constant_and_short_policy():
    x = np.linspace(-2.0, 3.0, 20)
    corr, covariance = _kernel()(x, x)
    flipped_corr, flipped_covariance = _kernel()(x, -x)
    assert corr == pytest.approx(1.0)
    assert flipped_corr == pytest.approx(1.0)
    assert flipped_covariance == pytest.approx(covariance)
    for a, b in ((np.ones(20), x), (x, np.ones(20)), (x[:3], x[:3])):
        actual = _kernel()(a, b)
        assert np.isnan(actual[0]) and np.isnan(actual[1])
    for a, b in ((x[:, None], x), (x, x[:, None])):
        actual = _kernel()(a, b)
        assert np.isnan(actual[0]) and np.isnan(actual[1])


def test_public_aligned_pair_mask_and_prefix():
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    ensure_cleaned_loaded()
    rng = np.random.default_rng(222)
    x_values = rng.normal(size=70)
    y_values = np.sin(x_values) + 0.1 * rng.normal(size=70)
    x_values[[42, 61]] = np.nan
    y_values[[45, 61]] = np.nan
    x = pd.DataFrame({"A": x_values})
    y = pd.DataFrame({"A": y_values})
    corr_op = OperatorRegistry.get("ts_distance_corr", "pandas_numpy")
    cov_op = OperatorRegistry.get("ts_distance_cov", "pandas_numpy")
    kwargs = {"window": 30, "min_periods": 10}
    corr = corr_op.calculate(x, y, **kwargs)
    covariance = cov_op.calculate(x, y, **kwargs)
    paired = np.isfinite(x_values[-30:]) & np.isfinite(y_values[-30:])
    expected = _kernel()(x_values[-30:][paired], y_values[-30:][paired])
    assert corr.iloc[-1, 0] == pytest.approx(expected[0])
    assert covariance.iloc[-1, 0] == pytest.approx(expected[1])
    pd.testing.assert_frame_equal(
        corr.iloc[:55], corr_op.calculate(x.iloc[:55], y.iloc[:55], **kwargs)
    )
