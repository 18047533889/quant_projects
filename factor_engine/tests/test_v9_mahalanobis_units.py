import numpy as np
import pandas as pd
import pytest


def _kernel():
    from factor_engine.cleaned_operators.candle_state_space import _mahalanobis_series
    return _mahalanobis_series


def _arrays(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return tuple(matrix[:, j : j + 1] for j in range(4))


def _oracle_last(matrix, window, shrinkage):
    """Direct original-coordinate solve, independent of kernel eigenspaces."""
    matrix = np.asarray(matrix, dtype=float)
    history = matrix[max(0, len(matrix) - window) : -1]
    history = history[np.all(np.isfinite(history), axis=1)]
    query = matrix[-1]
    keep = np.asarray([not np.all(history[:, j] == history[0, j]) for j in range(4)])
    history = history[:, keep]
    query = query[keep]
    if len(history) < 5 * history.shape[1]:
        return np.nan
    center = history.mean(axis=0)
    covariance = np.atleast_2d(np.cov(history, rowvar=False, ddof=1))
    covariance = ((1.0 - shrinkage) * covariance
                  + shrinkage * np.diag(np.diag(covariance)))
    delta = query - center
    return float(np.sqrt(delta @ np.linalg.solve(covariance, delta)))


def test_matches_independent_diagonal_shrinkage_oracle_and_feature_units():
    rng = np.random.default_rng(10)
    matrix = rng.normal(size=(81, 4))
    matrix[:, 1] += 0.7 * matrix[:, 0]
    expected = _oracle_last(matrix, 60, 0.2)
    actual = _kernel()(*_arrays(matrix), 60, 0.2)[-1, 0]
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)

    # Change each feature's unit independently, including extreme finite units.
    units = np.array([1e-300, 1e300, 1e-180, 1e180])
    converted = matrix * units
    converted_actual = _kernel()(*_arrays(converted), 60, 0.2)[-1, 0]
    assert converted_actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_large_translation_does_not_erase_representable_variation():
    rng = np.random.default_rng(110)
    matrix = rng.normal(scale=4.0, size=(81, 4))
    baseline = _kernel()(*_arrays(matrix), 60, 0.2)[-1, 0]
    translated = matrix + np.array([1e14, -1e14, 5e13, -5e13])
    actual = _kernel()(*_arrays(translated), 60, 0.2)[-1, 0]
    # Input rounding at the translated magnitude prevents bit identity, but
    # the varying dimensions must remain present and the geometry stable.
    assert actual == pytest.approx(baseline, rel=3e-3, abs=3e-3)


def test_constant_dimension_is_dropped_without_unit_dependent_ridge():
    rng = np.random.default_rng(1010)
    matrix = rng.normal(size=(81, 4))
    matrix[:, 3] = 7e250
    expected = _oracle_last(matrix, 60, 0.3)
    actual = _kernel()(*_arrays(matrix), 60, 0.3)[-1, 0]
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_rank_deficient_training_accepts_column_space_query():
    latent = np.linspace(-3.0, 4.0, 81)
    loadings = np.array([1.0, 2.0, -3.0, 0.5])
    matrix = latent[:, None] * loadings
    history = latent[21:-1]
    expected = abs(latent[-1] - history.mean()) / np.std(history, ddof=1)
    actual = _kernel()(*_arrays(matrix), 60, 0.0)[-1, 0]
    assert actual == pytest.approx(expected, rel=1e-11, abs=1e-11)


def test_query_component_in_unlearned_nullspace_fails_closed():
    from factor_engine.cleaned_operators.candle_state_space import last_mahalanobis_telemetry

    rng = np.random.default_rng(310)
    matrix = rng.normal(size=(81, 4))
    matrix[:, 1] = 2.0 * matrix[:, 0]
    matrix[-1, 1] += 1.0
    actual = _kernel()(*_arrays(matrix), 60, 0.0)[-1, 0]
    assert np.isnan(actual)
    assert last_mahalanobis_telemetry()["failure_reason"] == "query_in_covariance_nullspace"


def test_nonfinite_covariance_is_not_coerced_to_zero(monkeypatch):
    from factor_engine.cleaned_operators import candle_state_space as module

    matrix = np.random.default_rng(410).normal(size=(81, 4))
    monkeypatch.setattr(module.np, "cov", lambda *_args, **_kwargs: np.full((4, 4), np.inf))
    actual = module._mahalanobis_series(*_arrays(matrix), 60, 0.2)[-1, 0]
    assert np.isnan(actual)
    assert module.last_mahalanobis_telemetry()["failure_reason"] == "nonfinite_covariance"


def test_effective_sample_to_dimension_floor():
    rng = np.random.default_rng(101)
    insufficient = rng.normal(size=(20, 4))  # final query has only 19 history rows
    exact = rng.normal(size=(21, 4))         # final query has exactly N=5p=20
    assert np.isnan(_kernel()(*_arrays(insufficient), 21, 0.2)[-1, 0])
    assert np.isfinite(_kernel()(*_arrays(exact), 21, 0.2)[-1, 0])


def test_large_but_representable_distance_does_not_overflow_squared_form():
    matrix = np.random.default_rng(411).normal(size=(81, 4))
    matrix[-1, 0] = 1e200
    actual = _kernel()(*_arrays(matrix), 60, 0.2)[-1, 0]
    assert np.isfinite(actual)
    assert actual > 1e199


def test_public_prefix_causality():
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    ensure_cleaned_loaded()
    matrix = np.random.default_rng(510).normal(size=(95, 4))
    frames = [pd.DataFrame({"A": matrix[:, j]}) for j in range(4)]
    op = OperatorRegistry.get("ts_vector_state_mahalanobis", "pandas_numpy")
    kwargs = {"window": 60, "shrinkage": 0.2}
    full = op.calculate(*frames, **kwargs)
    prefix = op.calculate(*(frame.iloc[:80] for frame in frames), **kwargs)
    pd.testing.assert_frame_equal(prefix, full.iloc[:80])
    assert np.isfinite(full.to_numpy()).any()
