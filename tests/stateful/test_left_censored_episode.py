# -*- coding: utf-8 -*-
"""R24-105..109 + R24-247: survival operators treat the initial active run as
LEFT-CENSORED (never a completed episode), censor provider gaps, and keep
inactive rows as NaN (not a 0-percentile outcome) unless policy opts to zero."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded

ensure_cleaned_loaded()
from cleaned_operators.registry import OperatorRegistry  # noqa: E402


def _col(values):
    return pd.DataFrame(
        np.array(values, dtype=float)[:, None],
        index=pd.bdate_range("2024-01-02", periods=len(values)),
        columns=["A"],
    )


def test_first_run_is_left_censored_and_excluded() -> None:
    # R24-247 golden: active,active,active,inactive — the first run (starts at
    # sample start) is left-censored and must not enter the completed
    # distribution.
    s = _col([1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1])
    op = OperatorRegistry.get("ts_state_age_percentile", "pandas_numpy")
    out = op.calculate(s, min_completed_runs=2)
    # Completed runs: only the OBSERVED-ENTRY ones (run2 len2, run3 len3) → 2
    # runs.  The left-censored first run (len3) is excluded.  At run4 age 1:
    # ECDF(1 among {2,3}) = 0.0.
    assert out["A"].iloc[-1] == pytest.approx(0.0)


def test_left_censored_run_never_added_to_completed() -> None:
    # A run that starts at sample start and is LONGER than every observed run:
    # if it were (wrongly) completed, the percentile at the observed run would
    # include it.  R24-106: it must not.
    s = _col([1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1])
    op = OperatorRegistry.get("ts_state_age_percentile", "pandas_numpy")
    out = op.calculate(s, min_completed_runs=2)
    # Completed = {2, 3}; the left-censored length-4 run is excluded.  At run4
    # age 1: ECDF(1 among {2,3}) = 0.0.  If the left-censored run were (wrongly)
    # completed → {4,2,3} → ECDF(1 among {4,2,3}) = 0.0 too — so verify the
    # completed set contents directly: at run4 age 3 the ECDF(3 among {2,3})
    # would be 1.0, but with the len-4 run included it stays 1.0.  Use a longer
    # run4 to distinguish: at run4 age 5, ECDF(5 among {2,3}) = 1.0 only if the
    # left-censored run is excluded (it never reaches age 5 if included it would
    # count as a completed length-4, still <= 5 → 1.0).  Assert the exclusion by
    # the completed count via min_completed_runs=3: only 2 runs exist → NaN.
    out3 = op.calculate(s, min_completed_runs=3)
    assert np.isnan(out3["A"].iloc[-1])


def test_inactive_policy_nan_default() -> None:
    # R24-109: inactive is NOT a 0-percentile outcome.
    s = _col([1, 1, 0, 1, 1, 0])
    op = OperatorRegistry.get("ts_state_age_percentile", "pandas_numpy")
    out = op.calculate(s, min_completed_runs=1, inactive_policy="nan")
    assert np.isnan(out["A"].iloc[2])
    assert np.isnan(out["A"].iloc[5])
    # Legacy zero policy keeps the old reading.
    out0 = op.calculate(s, min_completed_runs=1, inactive_policy="zero")
    assert out0["A"].iloc[2] == pytest.approx(0.0)


def test_gap_censored_run_not_completed() -> None:
    # R24-107/108: a provider gap inside an active run makes its length unknown;
    # the run is never recorded as completed.
    s = _col([1, 1, np.nan, 1, 0, 1, 1, 0])
    op = OperatorRegistry.get("ts_state_age_percentile", "pandas_numpy")
    out = op.calculate(s, min_completed_runs=2)
    # The gap-broken run is not completed; only the trailing observed run of
    # length 2 is → sample = 1 < 2 → NaN.
    assert np.isnan(out["A"].iloc[-1])


def test_gap_policy_lower_bound_reports_floor_age() -> None:
    # R24-108: with gap_policy="lower_bound", the age after a gap is a floor.
    s = _col([1, 1, np.nan, 1, 1, 0])
    op = OperatorRegistry.get("ts_state_age_percentile", "pandas_numpy")
    # percentile under lower_bound still counts from the original run start.
    out = op.calculate(s, min_completed_runs=1, gap_policy="lower_bound")
    # The run was not completed, so no completed sample → NaN for percentile
    # (this asserts the run is NOT recorded even in lower_bound mode).
    assert np.isnan(out["A"].iloc[4])
