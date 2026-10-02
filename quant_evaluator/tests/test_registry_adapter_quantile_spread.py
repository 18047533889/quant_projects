"""Finite-observation reduction and empty-axis shape contracts."""
import warnings

import numpy as np

from quant_evaluator.metrics import registry_adapters as adapters


def test_spread_mean_excludes_nonfinite_and_emits_no_empty_warning(monkeypatch):
    returns = np.full((3, 5, 4), np.nan)
    returns[:, 0, :] = 0.0
    returns[:, -1, 0] = [2.0, 3.0, np.nan]
    returns[:, -1, 2] = [np.inf, np.nan, -np.inf]
    returns[:, -1, 3] = [2.0, 3.0, np.inf]
    monkeypatch.setattr(adapters, "compute_quantile_returns_fast",
                        lambda *args, **kwargs: (returns, None))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = adapters.compute_quantile_spread_value(None, None, min_periods=2)
    np.testing.assert_allclose(result, [2.5, np.nan, np.nan, 2.5], equal_nan=True)


def test_spread_empty_time_axis_preserves_factor_shape(monkeypatch):
    returns = np.empty((0, 5, 2))
    monkeypatch.setattr(adapters, "compute_quantile_returns_fast",
                        lambda *args, **kwargs: (returns, None))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = adapters.compute_quantile_spread_value(None, None)
    assert result.shape == (2,)
    assert np.isnan(result).all()


def test_spread_insufficient_finite_periods_remains_nan(monkeypatch):
    returns = np.zeros((2, 5, 1))
    returns[:, -1, 0] = [2.0, np.nan]
    monkeypatch.setattr(adapters, "compute_quantile_returns_fast",
                        lambda *args, **kwargs: (returns, None))
    assert np.isnan(adapters.compute_quantile_spread_value(None, None, min_periods=2)[0])
