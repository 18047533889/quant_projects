from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators import load_all
from factor_engine.cleaned_operators.registry import OperatorRegistry


load_all()


def _frame(values, *, double: bool = True) -> pd.DataFrame:
    arr = np.asarray(values, dtype=float)
    return pd.DataFrame({"B": arr * 2.0 if double else arr, "A": arr},
                        index=pd.date_range("2024-01-02", periods=len(values), freq="B"))


def test_filing_delay_surprise_requires_real_nonconstant_event_history() -> None:
    delay = _frame([10, 12, 9, 15, 11, 18])
    event = _frame([1, 1, 1, 1, 1, 1], double=False)
    op = OperatorRegistry.get("report_filing_delay_surprise", backend="pandas_numpy")

    got = op.calculate(delay, event, window=4, min_periods=3)

    assert op.metadata.panel_params == ("delay", "filing_event")
    assert got.iloc[:3].isna().all().all()
    assert np.isfinite(got.iloc[3:].to_numpy()).all()
    assert got.loc[got.index[3], "A"] == pytest.approx(3.3724537973829762)


def test_revision_magnitude_requires_typed_same_period_revision_pairs() -> None:
    current = _frame([110, 125, 87, 160, 121, 210])
    previous = _frame([100, 120, 90, 150, 110, 200])
    period = _frame([20230331, 20230630, 20230930, 20231231, 20240331, 20240630], double=False)
    event = _frame([1, 1, 1, 1, 1, 1], double=False)
    op = OperatorRegistry.get("report_revision_magnitude", backend="pandas_numpy")

    got = op.calculate(current, previous, period, period.copy(), event, window=4, min_periods=3)

    assert op.metadata.panel_params == (
        "x", "prev_x", "current_period_id", "prev_period_id", "revision_event",
    )
    assert got.iloc[:3].isna().all().all()
    np.testing.assert_allclose(got["A"].iloc[3:].to_numpy(), [0.04, 0.05, 2 / 63])


def test_revision_delta_fires_on_changed_vintage_within_same_period() -> None:
    value = _frame([100, 105, 200, 190, 300, 315])
    period = _frame([20230331, 20230331, 20230630, 20230630, 20230930, 20230930], double=False)
    revision = _frame([0, 1, 0, 1, 0, 1], double=False)
    op = OperatorRegistry.get("revision_delta", backend="pandas_numpy")

    got = op.calculate(value, period, revision, mode="absolute")

    assert op.metadata.panel_params == ("x", "period_id", "revision_id")
    np.testing.assert_allclose(got["A"].iloc[[1, 3, 5]].to_numpy(), [5.0, -10.0, 15.0])
    assert got["A"].iloc[[0, 2, 4]].isna().all()
