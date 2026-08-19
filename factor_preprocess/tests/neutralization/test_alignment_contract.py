"""
Alignment guards for per-date neutralizers.

Neutralizers merge ``values`` with ``exposures`` and must return residuals
in the original ``values`` row order.  These tests pin that contract under
shuffled input order, duplicate (date, asset) exposure keys, and a
non-unique ``values`` index — the cases a merged-frame reindex gets wrong.
"""
import numpy as np
import pandas as pd
import pytest

from factor_preprocess.neutralization.regularized import (
    elastic_net_neutralize,
    lasso_neutralize,
    ridge_neutralize,
)
from factor_preprocess.neutralization.advanced.pca_neutralization import pca_neutralize
from factor_preprocess.neutralization.advanced.quantile_regression import quantile_neutralize
from factor_preprocess.neutralization.advanced.robust_regression import huber_neutralize
from factor_preprocess.neutralization.advanced.kernel_regression import kernel_neutralize

N_PER, N_DATES = 12, 3
N = N_PER * N_DATES

NEUTRALIZERS = [
    (ridge_neutralize, {}),
    (lasso_neutralize, {}),
    (elastic_net_neutralize, {}),
    (quantile_neutralize, {}),
    (huber_neutralize, {}),
    (pca_neutralize, {}),
    (kernel_neutralize, {}),
]


def _make_data():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=N_DATES, freq="D").repeat(N_PER)
    vals = pd.DataFrame(
        {
            "date": dates,
            "asset_id": [f"a{i % N_PER}" for i in range(N)],
            "value": rng.normal(size=N),
        }
    )
    vals.index = [f"r{i}" for i in range(N)]
    expo = pd.DataFrame(
        {
            "date": dates,
            "asset_id": [f"a{i % N_PER}" for i in range(N)],
            "size": rng.normal(size=N),
        }
    )
    return vals, expo


@pytest.mark.parametrize("fn,kwargs", NEUTRALIZERS, ids=lambda f: getattr(f, "__name__", f))
def test_shuffled_input_order_preserves_per_row_residuals(fn, kwargs):
    """Residual for row i must depend on (date, asset), not row position."""
    vals, expo = _make_data()
    expected = fn(vals, expo, min_observations=10, **kwargs)
    shuffled = vals.sample(frac=1.0, random_state=1)
    got = fn(shuffled, expo, min_observations=10, **kwargs).reindex(vals.index)
    assert len(got) == N
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), equal_nan=True)


@pytest.mark.parametrize("fn,kwargs", NEUTRALIZERS, ids=lambda f: getattr(f, "__name__", f))
def test_duplicate_exposure_key_keeps_one_row_per_input(fn, kwargs):
    """A duplicated (date, asset) exposure row must not corrupt alignment."""
    vals, expo = _make_data()
    expo_dup = pd.concat([expo, expo.iloc[[0]]], ignore_index=True)
    out = fn(vals, expo_dup, min_observations=10, **kwargs)
    assert len(out) == N
    assert out.index.tolist() == vals.index.tolist()


def test_non_unique_values_index_returns_all_rows():
    """A non-unique values index must not raise or drop rows."""
    vals, expo = _make_data()
    vals.index = np.zeros(N, dtype=int)
    out = ridge_neutralize(vals, expo, min_observations=10)
    assert len(out) == N
