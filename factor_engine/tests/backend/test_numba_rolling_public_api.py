import numpy as np
import pandas as pd

from factor_engine.backend.numba_kernels import (
    rolling_corr_panel,
    rolling_mean_panel,
    rolling_rank_pct_panel,
    rolling_std_panel,
)


def test_public_numba_rolling_api_nan_and_single_factor_parity():
    x = np.array([[1.0], [np.nan], [3.0], [4.0], [np.inf], [6.0]])
    y = np.array([[2.0], [2.0], [6.0], [8.0], [10.0], [12.0]])
    finite_x = np.where(np.isfinite(x[:, 0]), x[:, 0], np.nan)
    finite_y = np.where(np.isfinite(y[:, 0]), y[:, 0], np.nan)
    expected_mean = pd.Series(finite_x).rolling(3, min_periods=1).mean().to_numpy()[:, None]
    expected_std = pd.Series(finite_x).rolling(3, min_periods=2).std().to_numpy()[:, None]
    expected_corr = pd.Series(finite_x).rolling(3, min_periods=2).corr(pd.Series(finite_y)).to_numpy()[:, None]
    np.testing.assert_allclose(rolling_mean_panel(x, 3, 1), expected_mean, equal_nan=True)
    np.testing.assert_allclose(rolling_std_panel(x, 3, 2), expected_std, equal_nan=True)
    np.testing.assert_allclose(rolling_corr_panel(x, y, 3, 2), expected_corr, equal_nan=True)
    rank = rolling_rank_pct_panel(x, 3, 1)
    assert rank is not None and rank.shape == x.shape
    assert np.isnan(rank[1, 0]) and np.isnan(rank[4, 0])
