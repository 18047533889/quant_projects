"""R30 regression for the canonical single-lag AR coefficient contract."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _oracle(values, window, lag, min_periods, warmup_policy):
    values = np.asarray(values, dtype=float)
    out = np.full_like(values, np.nan)
    for row in range(len(values)):
        if warmup_policy == "full" and row < window - 1:
            continue
        start = max(0, row - window + 1)
        segment = values[start : row + 1]
        if len(segment) <= lag:
            continue
        current, lagged = segment[lag:], segment[:-lag]
        valid = np.isfinite(current) & np.isfinite(lagged)
        if valid.sum() < max(3, min_periods):
            continue
        current, lagged = current[valid], lagged[valid]
        if np.std(lagged) <= 0:
            continue
        # Independent QR/SVD least-squares oracle rather than reproducing the
        # kernel's covariance/variance formula.
        centered = lagged - lagged.mean()
        design = np.column_stack([np.ones(len(centered)), centered])
        out[row] = np.linalg.lstsq(
            design, current - current.mean(), rcond=None
        )[0][1]
    return out


@pytest.mark.parametrize("lag", [1, 2])
@pytest.mark.parametrize("warmup_policy", ["expanding", "full"])
def test_ar_coefficient_matches_intercept_ols_oracle_and_prefix(lag, warmup_policy):
    load_all()
    rng = np.random.default_rng(3017 + lag)
    values = np.empty(64)
    values[0] = 0.4
    for row in range(1, len(values)):
        values[row] = 0.3 + 0.72 * values[row - 1] + rng.normal(0, 0.08)
    values[[14, 27]] = np.nan
    frame = pd.DataFrame({"A": values})
    params = dict(window=20, lag=lag, min_periods=8, warmup_policy=warmup_policy)
    expected = _oracle(values, **params)
    arrays = {}
    for backend in ("pandas_numpy", "polars"):
        op = OperatorRegistry.get("ts_ar_coefficient", backend, mode="research")
        source = pl.from_pandas(frame) if backend == "polars" else frame
        actual = op.calculate(source, **params).to_numpy()[:, 0]
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
        future = frame.copy()
        future.iloc[49:, 0] += 1000.0
        future_source = pl.from_pandas(future) if backend == "polars" else future
        changed = op.calculate(future_source, **params).to_numpy()[:, 0]
        np.testing.assert_allclose(actual[:49], changed[:49], rtol=0, atol=0, equal_nan=True)
        arrays[backend] = actual
    np.testing.assert_allclose(arrays["pandas_numpy"], arrays["polars"], rtol=1e-12, atol=1e-12, equal_nan=True)
