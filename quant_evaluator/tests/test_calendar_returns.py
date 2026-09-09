from datetime import datetime, timezone

import numpy as np
import pytest

from data_access.r30.calendar_snapshot import CalendarSnapshot
from quant_evaluator.metrics.calendar_returns import (
    compute_worst_calendar_month,
    compute_worst_calendar_quarter,
    compute_worst_calendar_year,
    compute_worst_rolling_return,
)


def snapshot(days, *, tz="Asia/Shanghai"):
    return CalendarSnapshot(
        market="fixture",
        source_version="approved-fixture-v1",
        timezone=tz,
        trading_days=tuple(days),
        sessions=(),
        early_close=(),
        snapshot_id="calendar-fixture-sha256",
    )


def instants(days):
    return tuple(datetime.fromisoformat(f"{day}T04:00:00+00:00") for day in days)


def test_calendar_month_uses_actual_sessions_and_independent_month_counts():
    calendar = snapshot((
        "2025-01-31",  # coverage buffer
        "2025-02-03", "2025-02-04", "2025-02-05",  # holiday-short month fixture
        "2025-03-03", "2025-03-04", "2025-03-05", "2025-03-06",
        "2025-04-01",  # coverage buffer
    ))
    observed = calendar.trading_days[1:-1]
    returns = np.array([
        [0.10, 0.01], [0.00, 0.01], [0.00, 0.01],
        [-0.05, 0.02], [-0.05, 0.02], [0.00, 0.02], [0.00, 0.02],
    ])
    artifact = compute_worst_calendar_month(
        returns, instants(observed), ("alpha", "beta"), calendar
    )
    np.testing.assert_allclose(artifact.values, [0.9**0 * 0.95**2 - 1.0, 1.01**3 - 1.0])
    rows = {row["period_id"]: row for row in artifact.provenance["period_rows"]}
    assert rows["2025-02"]["expected_session_count"] == 3
    assert rows["2025-03"]["expected_session_count"] == 4
    assert rows["2025-02"]["complete_by_factor"] == (True, True)
    assert artifact.provenance["observation_counts"] == (2, 2)
    assert artifact.provenance["calendar_snapshot_id"] == calendar.snapshot_id
    assert artifact.provenance["calendar_timezone"] == "Asia/Shanghai"


def test_partial_month_excluded_by_default_and_explicitly_included_with_counts():
    calendar = snapshot(("2025-01-31", "2025-02-03", "2025-02-04", "2025-02-05", "2025-03-03"))
    observed = ("2025-02-03", "2025-02-05")
    returns = np.array([[-0.10], [-0.10]])
    excluded = compute_worst_calendar_month(returns, instants(observed), ("f",), calendar)
    assert np.isnan(excluded.values[0])
    included = compute_worst_calendar_month(
        returns, instants(observed), ("f",), calendar, partial_policy="include"
    )
    assert included.values[0] == pytest.approx(0.9**2 - 1.0)
    row = next(row for row in included.provenance["period_rows"] if row["period_id"] == "2025-02")
    assert row["expected_session_count"] == 3
    assert row["observed_session_count"] == 2
    assert row["partial"] is True


def test_unknown_return_is_never_silently_zero_and_bad_capital_return_rejected():
    calendar = snapshot(("2025-01-31", "2025-02-03", "2025-02-04", "2025-03-03"))
    times = instants(("2025-02-03", "2025-02-04"))
    artifact = compute_worst_calendar_month(
        np.array([[0.1], [np.nan]]), times, ("f",), calendar, partial_policy="include"
    )
    assert np.isnan(artifact.values[0])
    row = next(row for row in artifact.provenance["period_rows"] if row["period_id"] == "2025-02")
    assert row["compounded_returns"] == (None,)
    with pytest.raises(ValueError, match=">= -1"):
        compute_worst_calendar_month(np.array([[-1.01], [0.0]]), times, ("f",), calendar)


@pytest.mark.parametrize(
    ("function", "days", "observed", "expected"),
    [
        (
            compute_worst_calendar_quarter,
            ("2024-12-31", "2025-01-02", "2025-02-03", "2025-03-03", "2025-04-01"),
            ("2025-01-02", "2025-02-03", "2025-03-03"),
            0.9**3 - 1.0,
        ),
        (
            compute_worst_calendar_year,
            ("2024-12-31", "2025-01-02", "2025-06-02", "2025-12-31", "2026-01-02"),
            ("2025-01-02", "2025-06-02", "2025-12-31"),
            0.9**3 - 1.0,
        ),
    ],
)
def test_calendar_quarter_and_year_use_calendar_periods(function, days, observed, expected):
    artifact = function(np.full((3, 1), -0.1), instants(observed), ("f",), snapshot(days))
    assert artifact.values[0] == pytest.approx(expected)


def test_truncated_snapshot_cannot_certify_a_full_calendar_year():
    days = ("2025-01-02", "2025-06-02", "2025-12-31")
    artifact = compute_worst_calendar_year(
        np.full((3, 1), 0.01), instants(days), ("f",), snapshot(days)
    )
    assert np.isnan(artifact.values[0])
    row = artifact.provenance["period_rows"][0]
    assert row["sessions_complete"] is True
    assert row["calendar_coverage_bracketed"] is False
    assert row["complete_by_factor"] == (False,)


def test_timezone_is_applied_to_instants_and_non_calendar_dates_reject():
    calendar = snapshot(("2025-01-31", "2025-02-03", "2025-03-03"))
    # UTC Sunday is already Monday in the snapshot's Asia/Shanghai timezone.
    instant = datetime(2025, 2, 2, 16, 30, tzinfo=timezone.utc)
    artifact = compute_worst_calendar_month(np.array([[0.1]]), (instant,), ("f",), calendar)
    assert artifact.values[0] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="absent from calendar"):
        compute_worst_calendar_month(
            np.array([[0.1]]), (datetime(2025, 2, 3, 16, 30, tzinfo=timezone.utc),),
            ("f",), calendar,
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        compute_worst_calendar_month(
            np.array([[0.1]]), (datetime(2025, 2, 3, 12),), ("f",), calendar,
        )


def test_rolling_return_is_per_factor_worst_of_only_mature_full_finite_windows():
    # An adverse prefix is not available until the full window matures.
    assert np.isnan(compute_worst_rolling_return(np.array([[-0.5], [0.0]]), window=3)[0])
    returns = np.array([
        [0.10, 0.10], [0.00, np.nan], [-0.10, 0.10],
        [0.20, 0.00], [0.00, 0.00], [0.10, 0.20],
    ])
    result = compute_worst_rolling_return(returns, window=3, min_periods=1)
    assert result.shape == (2,)
    expected_windows = [1.1 * 1.0 * 0.9 - 1.0, 1.0 * 0.9 * 1.2 - 1.0,
                        0.9 * 1.2 * 1.0 - 1.0, 1.2 * 1.0 * 1.1 - 1.0]
    assert result[0] == pytest.approx(min(expected_windows))
    # Missingness invalidates only windows containing it; after it exits the
    # original grid, the final two full windows are valid again.
    assert result[1] == pytest.approx(0.10)
    assert np.isnan(compute_worst_rolling_return(returns, window=3, min_periods=3)[1])
    with pytest.raises(ValueError, match=">= -1"):
        compute_worst_rolling_return(np.array([[-1.1]]))
