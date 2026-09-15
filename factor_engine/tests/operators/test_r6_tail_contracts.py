from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry


ensure_cleaned_loaded()


def _op(name: str, backend: str = "pandas_numpy"):
    op = OperatorRegistry.get(name, backend)
    assert op is not None
    return op


def test_pocket_contract_oracle_and_polars_parity() -> None:
    idx = pd.date_range("2024-01-01", periods=3)
    factor = pd.DataFrame([[3, 2, 1]] * 3, index=idx, columns=list("ABC"), dtype=float)
    ret = pd.DataFrame([[3, 2, 1], [1, 3, 2], [4, 2, 3]], index=idx, columns=list("ABC"), dtype=float)
    expected = np.array([[np.nan] * 3, [0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
    pandas_out = _op("panel_factor_pocket_strength").calculate(
        ret, factor, window=2, min_periods=2, threshold=0.5
    )
    np.testing.assert_allclose(pandas_out, expected, equal_nan=True)
    polars_out = _op("panel_factor_pocket_strength", "polars").calculate(
        pl.from_pandas(ret), pl.from_pandas(factor), window=2, min_periods=2, threshold=0.5
    )
    np.testing.assert_allclose(polars_out.to_pandas(), pandas_out, equal_nan=True)
    meta = _op("panel_factor_pocket_strength").metadata
    assert meta.panel_params == ("ret", "factor")
    assert meta.param_specs["threshold"].dtype is float
    with pytest.raises(Exception, match="min_periods"):
        _op("panel_factor_pocket_strength").calculate(ret, factor, window=2, min_periods=3)


def test_local_moran_contract_oracle_and_polars_parity() -> None:
    cols = list("ABCDE")
    target = pd.DataFrame([[10, 20, 30, 40, 50]], columns=cols, dtype=float)
    features = [pd.DataFrame([[1, 2, 3, 4, 5]], columns=cols, dtype=float) for _ in range(3)]
    pandas_out = _op("cs_knn_local_moran").calculate(target, *features, k=2)
    np.testing.assert_allclose(pandas_out, [[0.5, 0.5, 0.0, 0.5, 0.5]], atol=1e-12)
    polars_out = _op("cs_knn_local_moran", "polars").calculate(
        pl.from_pandas(target), *(pl.from_pandas(frame) for frame in features), k=2
    )
    np.testing.assert_allclose(polars_out.to_pandas(), pandas_out, atol=1e-12)
    meta = _op("cs_knn_local_moran").metadata
    assert meta.panel_params == ("target", "f1", "f2", "f3")
    assert meta.param_specs["k"].dtype is int
    with pytest.raises(Exception, match="must be an integer"):
        _op("cs_knn_local_moran").calculate(target, *features, k=2.5)
