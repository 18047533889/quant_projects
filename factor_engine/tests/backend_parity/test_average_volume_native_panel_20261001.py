"""Regression for the former Series-only native average-volume kernel."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.contracts import ExecutionKind
from factor_engine.cleaned_operators.polars_native import ts_advanced_batch1 as native


@pytest.mark.parametrize("panel", [False, True])
def test_average_volume_native_finite_window_and_panel_coordinates(panel):
    values = [1.0, 2.0, 3.0, np.inf, 5.0, np.nan, 7.0, 8.0, 9.0]
    expected = pd.Series(values).rolling(3, min_periods=3).mean().to_numpy()
    data = pl.Series("A", values)
    if panel:
        data = pl.DataFrame({"__fe_time__": range(len(values)), "A": values})
    result = native.TSAverageVolumePolarsNative()._calculate_series(data, window=3)
    if panel:
        assert result.columns == data.columns
        assert result["__fe_time__"].equals(data["__fe_time__"])
        result = result["A"]
    np.testing.assert_allclose(result.to_numpy(), expected, equal_nan=True)


def test_average_volume_spec_tracks_actual_kernel_source():
    spec = native.TSAverageVolumePolarsNative._physical_spec
    assert spec.execution_kind is ExecutionKind.POLARS_NATIVE_EXPR
