from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()


def _op(name):
    op = OperatorRegistry.get(name, "pandas_numpy")
    assert op is not None
    return op


def _panels():
    index = pd.date_range("2024-01-01", periods=8)
    x = pd.DataFrame({"A": [9, 10, 11, 12, 10, 8, 9, 13]}, index=index, dtype=float)
    mid = pd.DataFrame({"A": [10, 10, 10, 10, 10, 10, 10, 10]}, index=index, dtype=float)
    widths = np.array([6, 5, 4, 3, 2, 4, 1, 5], dtype=float)
    upper = mid + widths[:, None] / 2
    lower = mid - widths[:, None] / 2
    return x, upper, lower, mid


def test_envelope_operators_match_independent_oracles():
    x, upper, lower, mid = _panels()
    window = 4
    compression = _op("ts_envelope_compression").calculate(
        upper, lower, mid, window=window
    )
    pressure = _op("ts_envelope_pressure").calculate(
        x, upper, lower, window=window
    )
    dwell = _op("ts_envelope_boundary_dwell").calculate(
        x, upper, lower, window=window, quantile=0.8
    )

    width = ((upper - lower) / mid.abs())["A"].to_numpy()
    position = (2 * (x - lower) / (upper - lower) - 1)["A"].to_numpy()
    compression_oracle, pressure_oracle, dwell_oracle = [], [], []
    for row in range(len(x)):
        start = max(0, row - window + 1)
        widths = width[start:row + 1]
        compression_oracle.append(1 - np.mean(widths <= width[row]))
        positions = position[start:row + 1]
        weights = np.arange(1, len(positions) + 1, dtype=float)
        pressure_oracle.append(np.average(positions, weights=weights))
        dwell_oracle.append(np.mean(np.abs(positions) >= 0.8))

    np.testing.assert_allclose(compression["A"], compression_oracle)
    np.testing.assert_allclose(pressure["A"], pressure_oracle)
    np.testing.assert_allclose(dwell["A"], dwell_oracle)


@pytest.mark.parametrize(
    "name",
    ["ts_envelope_compression", "ts_envelope_pressure", "ts_envelope_boundary_dwell"],
)
def test_envelope_missing_panels_propagate_and_future_does_not_change_prefix(name):
    x, upper, lower, mid = _panels()
    lower.iloc[3, 0] = np.nan
    if name == "ts_envelope_compression":
        result = _op(name).calculate(upper, lower, mid, window=4)
    elif name == "ts_envelope_pressure":
        result = _op(name).calculate(x, upper, lower, window=4)
    else:
        result = _op(name).calculate(x, upper, lower, window=4, quantile=0.8)
    assert np.isnan(result.iloc[3, 0])

    changed = [frame.copy() for frame in (x, upper, lower, mid)]
    for frame in changed:
        frame.iloc[6:] += 100
    cx, cu, cl, cm = changed
    if name == "ts_envelope_compression":
        rerun = _op(name).calculate(cu, cl, cm, window=4)
    elif name == "ts_envelope_pressure":
        rerun = _op(name).calculate(cx, cu, cl, window=4)
    else:
        rerun = _op(name).calculate(cx, cu, cl, window=4, quantile=0.8)
    pdt.assert_frame_equal(result.iloc[:6], rerun.iloc[:6])


def test_envelope_all_nan_inputs_return_all_nan():
    x, upper, lower, mid = _panels()
    for frame in (x, upper, lower, mid):
        frame.iloc[:] = np.nan
    assert _op("ts_envelope_compression").calculate(
        upper, lower, mid, window=4
    ).isna().all().all()
    assert _op("ts_envelope_pressure").calculate(
        x, upper, lower, window=4
    ).isna().all().all()
    assert _op("ts_envelope_boundary_dwell").calculate(
        x, upper, lower, window=4, quantile=0.8
    ).isna().all().all()


def test_envelope_metadata_declares_real_panel_topology():
    expected = {
        "ts_envelope_compression": ("upper", "lower", "mid"),
        "ts_envelope_pressure": ("x", "upper", "lower"),
        "ts_envelope_boundary_dwell": ("x", "upper", "lower"),
    }
    for name, panels in expected.items():
        metadata = _op(name).metadata
        assert metadata.name == name
        assert metadata.panel_params == panels
