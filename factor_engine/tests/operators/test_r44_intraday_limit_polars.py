"""Daily-output contracts for the final-batch Polars limit delegates."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from factor_engine.backend.contracts import ExecutionKind
from factor_engine.cleaned_operators.polars_native.intraday_final import (
    IntraLimitDurationPolarsNative,
    IntraLimitFirstHitTimePolarsNative,
    IntraLimitReopenCountPolarsNative,
)


def _panels():
    timestamps = pd.DatetimeIndex(
        [pd.Timestamp(day) + pd.Timedelta(hours=9, minutes=31 + minute)
         for day in ("2024-01-02", "2024-01-03") for minute in range(6)]
    )
    up = np.array([9.0, 10.0, 10.0, 9.0, 10.0, 9.0])
    down = np.array([6.0, 5.0, 4.9, 5.2, 5.0, 5.3])
    close = pl.DataFrame(
        {"QuoteTime": timestamps, "C0": np.r_[up, up], "C1": np.r_[down, down]}
    )
    high = close.with_columns(pl.col("C0"), pl.col("C1"))
    low = close.with_columns(pl.col("C0"), pl.col("C1"))
    days = pd.to_datetime(["2024-01-02", "2024-01-03"])
    high_limit = pl.DataFrame({"QuoteTime": days, "C0": [10.0, np.nan], "C1": [10.0, np.nan]})
    low_limit = pl.DataFrame({"QuoteTime": days, "C0": [5.0, np.nan], "C1": [5.0, np.nan]})
    return close, high, low, high_limit, low_limit


def _values(frame: pl.DataFrame) -> np.ndarray:
    assert frame.columns == ["date", "C0", "C1"]
    assert frame.height == 2
    return frame.select("C0", "C1").to_numpy()


def test_limit_duration_is_daily_and_uses_valid_observed_bars():
    close, _, _, high_limit, low_limit = _panels()
    up = _values(IntraLimitDurationPolarsNative().calculate(close, high_limit, low_limit, side="up"))
    down = _values(IntraLimitDurationPolarsNative().calculate(close, high_limit, low_limit, side="down"))
    np.testing.assert_allclose(up[0], [3 / 6, 0.0])
    np.testing.assert_allclose(down[0], [0.0, 3 / 6])
    assert np.isnan(up[1]).all() and np.isnan(down[1]).all()


def test_first_hit_uses_side_specific_bar_and_official_slot_fraction():
    close, high, low, high_limit, low_limit = _panels()
    up = _values(
        IntraLimitFirstHitTimePolarsNative().calculate(
            close, high, low, high_limit, low_limit, side="up"
        )
    )
    down = _values(
        IntraLimitFirstHitTimePolarsNative().calculate(
            close, high, low, high_limit, low_limit, side="down"
        )
    )
    assert up[0, 0] == pytest.approx(1 / 240)
    assert np.isnan(up[0, 1])
    assert np.isnan(down[0, 0])
    assert down[0, 1] == pytest.approx(1 / 240)
    assert np.isnan(up[1]).all() and np.isnan(down[1]).all()


@pytest.mark.parametrize("transition", ["open", "reseal"])
def test_reopen_count_counts_only_adjacent_known_slot_transitions(transition):
    close, _, _, high_limit, low_limit = _panels()
    up = _values(
        IntraLimitReopenCountPolarsNative().calculate(
            close, high_limit, low_limit, side="up", transition=transition
        )
    )
    down = _values(
        IntraLimitReopenCountPolarsNative().calculate(
            close, high_limit, low_limit, side="down", transition=transition
        )
    )
    np.testing.assert_allclose(up[0], [2.0, 0.0])
    np.testing.assert_allclose(down[0], [0.0, 2.0])
    assert np.isnan(up[1]).all() and np.isnan(down[1]).all()


def test_delegate_metadata_is_truthful_and_invalid_policies_reject():
    close, _, _, high_limit, low_limit = _panels()
    for cls in (
        IntraLimitDurationPolarsNative,
        IntraLimitFirstHitTimePolarsNative,
        IntraLimitReopenCountPolarsNative,
    ):
        assert cls._physical_spec.execution_kind is ExecutionKind.POLARS_PANDAS_DELEGATE
        assert "delegate:pandas_numpy" in cls.metadata.tags
        assert cls.metadata.output_grain == "daily"
    with pytest.raises(ValueError, match="side"):
        IntraLimitDurationPolarsNative().calculate(close, high_limit, low_limit, side="bad")
    with pytest.raises(ValueError, match="transition"):
        IntraLimitReopenCountPolarsNative().calculate(
            close, high_limit, low_limit, transition="bad"
        )


def test_limit_converter_preserves_utc_session_clock():
    close, high, low, high_limit, low_limit = _panels()
    # The same observed sessions represented by UTC must not move to 01:31 local.
    def utc_panel(frame):
        return frame.with_columns(
            pl.col("QuoteTime").dt.replace_time_zone("Asia/Shanghai").dt.convert_time_zone("UTC"))
    for cls in (IntraLimitDurationPolarsNative, IntraLimitReopenCountPolarsNative):
        baseline = cls().calculate(close, high_limit, low_limit, side="up")
        actual = cls().calculate(utc_panel(close), utc_panel(high_limit), utc_panel(low_limit), side="up")
        np.testing.assert_allclose(_values(actual), _values(baseline), equal_nan=True)
