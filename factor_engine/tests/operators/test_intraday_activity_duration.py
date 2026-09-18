"""Meaningful tests for activity-duration curvature on an official grid."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.runtime.session_calendar import SessionCalendar

ensure_cleaned_loaded()

CALENDAR = SessionCalendar.ashare(timestamp_convention="bar_start")


def _op():
    operator = OperatorRegistry.get("intraday_activity_duration_curvature", "pandas_numpy")
    assert operator is not None
    return operator


def _grid(day: str) -> pd.DatetimeIndex:
    morning = pd.date_range(f"{day} 09:30", periods=120, freq="min")
    afternoon = pd.date_range(f"{day} 13:00", periods=120, freq="min")
    return morning.append(afternoon)


def _oracle(values: np.ndarray, buckets: int) -> float:
    scaled = values / np.max(values)
    cumulative = np.cumsum(scaled)
    total = cumulative[-1]
    completion = np.array(
        [np.flatnonzero(cumulative >= k * total / buckets)[0] for k in range(1, buckets + 1)],
        dtype=float,
    )
    second_difference = completion[2:] - 2 * completion[1:-1] + completion[:-2]
    mad = np.median(np.abs(completion - np.median(completion)))
    return float(np.mean(second_difference) / mad)


def _panel():
    first = np.linspace(1.0, 5.0, 240) ** 2
    second = np.linspace(5.0, 1.0, 240) ** 2
    index = _grid("2024-01-02").append(_grid("2024-01-03"))
    return pd.DataFrame({"A": np.r_[first, second], "B": np.r_[second, first]}, index=index)


def test_curvature_matches_independent_bucket_completion_oracle():
    activity = _panel()
    result = _op().calculate(activity, buckets=6, calendar=CALENDAR)

    expected = pd.DataFrame(
        {
            "A": [_oracle(activity["A"].iloc[:240].to_numpy(), 6), _oracle(activity["A"].iloc[240:].to_numpy(), 6)],
            "B": [_oracle(activity["B"].iloc[:240].to_numpy(), 6), _oracle(activity["B"].iloc[240:].to_numpy(), 6)],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    assert np.isfinite(result.to_numpy()).all()
    pd.testing.assert_frame_equal(result, expected)


def test_missing_calendar_fails_closed_instead_of_self_certifying_grid(monkeypatch):
    from factor_engine.cleaned_operators import intraday_activity_duration
    monkeypatch.setattr(intraday_activity_duration, "_WARNED_NO_CALENDAR", False)
    with pytest.warns(RuntimeWarning, match="explicit.*calendar"):
        result = _op().calculate(_panel(), buckets=6)
    assert result.isna().all().all()


def test_missing_minute_and_nan_activity_fail_only_the_affected_day_symbol():
    activity = _panel()
    activity = activity.drop(pd.Timestamp("2024-01-02 10:30"))
    activity.loc[pd.Timestamp("2024-01-03 10:30"), "A"] = np.nan
    result = _op().calculate(activity, buckets=6, calendar=CALENDAR)

    assert result.loc[pd.Timestamp("2024-01-02")].isna().all()
    assert np.isnan(result.loc[pd.Timestamp("2024-01-03"), "A"])
    assert np.isfinite(result.loc[pd.Timestamp("2024-01-03"), "B"])


def test_completed_day_is_prefix_stable_when_later_day_is_appended():
    activity = _panel()
    prefix = _op().calculate(activity.iloc[:240], buckets=6, calendar=CALENDAR)
    full = _op().calculate(activity, buckets=6, calendar=CALENDAR)
    pd.testing.assert_frame_equal(prefix, full.loc[[pd.Timestamp("2024-01-02")]])


def test_curvature_metadata_declares_calendar_policy_and_daily_output():
    meta = _op().metadata
    assert meta.param_names == ["activity", "buckets", "calendar"]
    assert meta.input_grain == "minute"
    assert meta.output_grain == "daily"
    assert meta.available_at == "session_close"
    assert meta.same_session_usable is False
