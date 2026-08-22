from __future__ import annotations

import numpy as np
import pandas as pd


def _panel(values):
    index = pd.date_range("2024-01-01", periods=len(values), freq="B")
    return pd.DataFrame({"A": values}, index=index)


def test_event_surprise_history_counts_each_report_period_once():
    from cleaned_operators.fundamental.expectation_v2 import (
        fin_surprise_event_percentile,
        fin_surprise_event_zscore,
    )

    period_id = _panel(["Q1"] * 3 + ["Q2"] * 3 + ["Q3"] * 3 + ["Q4"] * 3)
    expected = _panel([10.0] * 12)
    actual = _panel([11.0] * 3 + [12.0] * 3 + [9.0] * 3 + [14.0] * 3)
    scale = _panel([10.0] * 12)

    percentile = fin_surprise_event_percentile(
        actual, expected, scale, period_id, 4
    )
    zscore = fin_surprise_event_zscore(actual, expected, scale, period_id, 4)

    # The same forward-filled event value must remain stable within a period.
    assert percentile.iloc[9:, 0].nunique(dropna=True) == 1
    assert zscore.iloc[9:, 0].nunique(dropna=True) == 1
    assert percentile.iloc[-1, 0] > 0.75
    assert np.isfinite(zscore.iloc[-1, 0])


def test_event_surprise_is_prefix_invariant():
    from cleaned_operators.fundamental.expectation_v2 import (
        fin_surprise_event_zscore,
    )

    period_id = _panel(np.repeat(np.arange(10), 3))
    expected = _panel(np.repeat(10.0, 30))
    actual = _panel(np.repeat(np.linspace(9.0, 14.0, 10), 3))
    scale = _panel(np.repeat(10.0, 30))

    full = fin_surprise_event_zscore(actual, expected, scale, period_id, 6)
    prefix = fin_surprise_event_zscore(
        actual.iloc[:24], expected.iloc[:24], scale.iloc[:24], period_id.iloc[:24], 6
    )
    np.testing.assert_allclose(
        full.iloc[:24, 0], prefix.iloc[:, 0], equal_nan=True
    )


def test_expectation_revision_ignores_target_roll_and_counts_same_target_changes():
    from cleaned_operators.fundamental.expectation_v2 import (
        fin_expectation_revision,
        fin_expectation_revision_count,
        fin_expectation_revision_magnitude,
    )

    target = _panel(["Q2", "Q2", "Q2", "Q3", "Q3", "Q3"])
    expected = _panel([10.0, 11.0, 11.0, 20.0, 18.0, 18.0])
    revision = fin_expectation_revision(expected, target)
    count = fin_expectation_revision_count(expected, target, 6)
    magnitude = fin_expectation_revision_magnitude(expected, target, 6)

    np.testing.assert_allclose(
        revision["A"].to_numpy(), [0.0, 1.0, 0.0, 0.0, -2.0, 0.0]
    )
    assert count.iloc[-1, 0] == 2.0
    assert magnitude.iloc[-1, 0] > 0.0


def test_days_since_revision_is_bounded_and_resets_only_on_revision():
    from cleaned_operators.fundamental.expectation_v2 import (
        fin_days_since_expectation_revision,
    )

    target = _panel(["Q2", "Q2", "Q3", "Q3", "Q3", "Q3"])
    expected = _panel([10.0, 11.0, 20.0, 20.0, 19.0, 19.0])
    age = fin_days_since_expectation_revision(expected, target, 3)
    # R23-105/106 left-censor: the FIRST observation has no history, so the
    # revision age is UNKNOWN (NaN), not "stale at max_days" — the old
    # ``age = max_days`` initial clock silently reported a cap as staleness.
    # The clock only starts at the first genuinely observed revision event.
    np.testing.assert_allclose(age["A"].to_numpy(), [np.nan, 0.0, 1.0, 2.0, 0.0, 1.0], equal_nan=True)
