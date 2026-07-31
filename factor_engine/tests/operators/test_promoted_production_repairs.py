from __future__ import annotations

import numpy as np
import pandas as pd


def _panel(values: list[float], *, cols: int = 2) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="B")
    arr = np.repeat(np.asarray(values, dtype=float)[:, None], cols, axis=1)
    return pd.DataFrame(arr, index=idx, columns=[f"A{i}" for i in range(cols)])


def _op(name: str):
    from cleaned_operators import load_all
    from cleaned_operators.registry import OperatorRegistry

    load_all()
    op = OperatorRegistry.get(name, "pandas_numpy")
    assert op is not None
    return op


def test_ts_max_buildup_is_prefix_invariant_and_window_local() -> None:
    op = _op("ts_max_buildup")
    x = _panel([1, 2, 1, 3, 2, 4, 1, 5, 2, 6])

    full = op.calculate(x, 4)
    prefix = op.calculate(x.iloc[:7], 4)
    pd.testing.assert_frame_equal(full.iloc[:7], prefix)

    # At t=4, trailing window is [2,1,3,2]. Record highs are 2 and 3 => 2.
    assert float(full.iloc[4, 0]) == 2.0
    # At t=6, trailing window is [3,2,4,1]. Record highs are 3 and 4 => 2.
    assert float(full.iloc[6, 0]) == 2.0


def test_digital_count_continues_updating_after_lookback_length() -> None:
    op = _op("digital_count")
    # Every step changes by ~1%, under threshold 2%; d caps the streak at 3.
    x = _panel([100, 101, 102, 103, 104, 105, 106, 107])
    out = op.calculate(x, 3, 0.02, 2)

    # Legacy implementation stopped calculating after the first d rows.
    assert float(out.iloc[5, 0]) == 3.0
    assert float(out.iloc[7, 0]) == 3.0


def test_ts_regression_slope_explicit_intercept_contract() -> None:
    op = _op("ts_regression_slope")
    x = _panel([1, 2, 3, 4, 5, 6, 7, 8])
    y = x * 2.0 + 5.0

    with_intercept = op.calculate(x, y, 5, True)
    through_origin = op.calculate(x, y, 5, False)

    assert np.isclose(float(with_intercept.iloc[-1, 0]), 2.0, atol=1e-12)
    # Through-origin fit is intentionally different because y contains +5 intercept.
    assert not np.isclose(float(through_origin.iloc[-1, 0]), 2.0, atol=1e-6)


def test_promoted_repairs_are_final_registry_implementations() -> None:
    assert type(_op("ts_max_buildup")).__name__ == "ProductionTSMaxBuildup"
    assert type(_op("digital_count")).__name__ == "ProductionDigitalCount"
    assert type(_op("ts_regression_slope")).__name__ == "ProductionTSRegressionSlope"
