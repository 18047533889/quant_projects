"""R21-091..118: DQ contract fixes.

- duplicate (timestamp, instrument) key fails even on a single day (R21-091)
- min_instruments_per_day / coverage use a single finite policy (R21-092)
- preserve_invalid_rows does not count Inf as a valid instrument (R21-093)
- role-aware domain contracts: Condition {0,1}, Rank/Pct [0,1] (R21-097..099)
- time-distribution: latest-day collapse / all-zero / variance collapse (R21-102..105)
- DataAccess ``column_null_ratio`` is null-ratio, converted with 1-x (R21-111..113)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.runtime.quality.dq_gates import (
    DQThresholds,
    DQRole,
    build_daily_coverage_profile,
    evaluate_factor_dq,
    role_domain_checks,
)
from factor_engine.runtime.quality.input_dq import (
    InputDQThresholds,
    adjust_input_dq_thresholds_from_stats,
)


def _series(values, *, dates=("2024-01-01", "2024-01-02"), instruments=("A", "B")):
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(list(dates)), list(instruments)],
        names=["timestamp", "instrument"],
    )
    return pd.Series(values[: len(idx)], index=idx, dtype=float)


class TestR21OutputDQ:
    def test_single_day_duplicate_key_fails(self):
        # Two rows on the SAME day share (ts, instrument) -> must fail even
        # though there is only one unique timestamp (R21-091).
        ts = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"])
        idx = pd.MultiIndex.from_product([ts, ["A"]], names=["timestamp", "instrument"])
        s = pd.Series([1.0, 2.0, 3.0], index=idx)
        report = evaluate_factor_dq(s)
        dup = next((c for c in report.checks if c.name == "unique_keys"), None)
        assert dup is not None and not dup.passed

    def test_min_instruments_uses_finite(self):
        # A day with only Inf values contributes 0 valid instruments (R21-092/093).
        s = _series([1.0, 2.0, np.inf, np.inf], dates=("2024-01-01", "2024-01-02"))
        report = evaluate_factor_dq(s, thresholds=DQThresholds(min_instruments_per_day=2))
        check = next(c for c in report.checks if c.name == "min_instruments_per_day")
        assert check.passed is False  # day 2 has 0 finite values
        assert check.value == 0

    def test_preserve_invalid_rows_still_excludes_inf_from_valid(self):
        s = _series([1.0, np.inf, 1.0, np.inf], dates=("2024-01-01", "2024-01-02"))
        report = evaluate_factor_dq(
            s, thresholds=DQThresholds(min_instruments_per_day=1), preserve_invalid_rows=True
        )
        check = next(c for c in report.checks if c.name == "min_instruments_per_day")
        assert check.passed is True  # each day has exactly 1 finite instrument

    def test_condition_domain_rejects_non_binary(self):
        s = _series([0.0, 1.0, 0.5, 1.0])
        report = evaluate_factor_dq(s, role=DQRole.CONDITION)
        check = next(c for c in report.checks if c.name == "condition_domain")
        assert check.passed is False

    def test_condition_domain_accepts_sparse_binary(self):
        s = _series([0.0, 1.0, np.nan, 1.0])
        report = evaluate_factor_dq(s, role=DQRole.CONDITION)
        check = next(c for c in report.checks if c.name == "condition_domain")
        assert check.passed is True

    def test_rank_range(self):
        s = _series([0.0, 0.5, 1.0, 0.25])
        checks = role_domain_checks(s, role="rank")
        rng = next(c for c in checks if c.name == "rank_range")
        assert rng.passed is True

    def test_rank_range_out_of_bounds(self):
        s = _series([0.0, 1.5, 2.0, 0.25])
        checks = role_domain_checks(s, role="rank")
        rng = next(c for c in checks if c.name == "rank_range")
        assert rng.passed is False

    def test_alpha_constant_detected(self):
        s = _series([3.0, 3.0, 3.0, 3.0])
        report = evaluate_factor_dq(s, role=DQRole.ALPHA)
        check = next(c for c in report.checks if c.name == "alpha_not_constant")
        assert check.passed is False

    def test_latest_day_collapse_detected(self):
        # history 2 days fine, latest day all-NaN -> latest_day_coverage fails.
        s = _series([1.0, 2.0, np.nan, np.nan], dates=("2024-01-01", "2024-01-02"))
        report = evaluate_factor_dq(s, time_aware=True)
        check = next(c for c in report.checks if c.name == "latest_day_coverage")
        assert check.passed is False

    def test_all_zero_detected(self):
        s = _series([0.0, 0.0, 0.0, 0.0])
        report = evaluate_factor_dq(s, time_aware=True)
        check = next(c for c in report.checks if c.name == "not_all_zero")
        assert check.passed is False

    def test_coverage_profile(self):
        s = _series([1.0, 2.0, 3.0, 4.0])
        profile = build_daily_coverage_profile(s)
        assert len(profile) == 2
        assert (profile == 1.0).all()


class _Stats:
    column_null_ratio = {"close": 0.9, "open": 0.5}
    column_non_null_ratio = None
    column_finite_ratio = None


class _StatsNonNull:
    column_null_ratio = None
    column_non_null_ratio = {"close": 0.2, "open": 0.6}
    column_finite_ratio = None


def test_null_ratio_converted_to_non_null_ratio():
    """R21-111..113: column_null_ratio is a NULL ratio; use 1 - null_ratio."""
    base = InputDQThresholds(min_non_null_ratio=0.01)
    adjusted = adjust_input_dq_thresholds_from_stats(base, _Stats(), ["close", "open"])
    # close null 0.9 -> non-null 0.1; slack 0.95 -> floor 0.095
    assert adjusted.min_non_null_ratio == pytest.approx(0.095, abs=1e-9)
    assert adjusted.min_non_null_ratio < 0.5  # NOT the buggy 0.855


def test_non_null_ratio_used_directly():
    adjusted = adjust_input_dq_thresholds_from_stats(
        InputDQThresholds(), _StatsNonNull(), ["close", "open"]
    )
    assert adjusted.min_non_null_ratio == pytest.approx(0.2 * 0.95, abs=1e-9)


def test_extreme_null_fixture():
    base = InputDQThresholds(min_non_null_ratio=0.01)
    # 100% null column -> non-null floor 0 -> default 0.01 keeps.
    all_null = type("S", (), {"column_null_ratio": {"x": 1.0}, "column_non_null_ratio": None, "column_finite_ratio": None})()
    adjusted = adjust_input_dq_thresholds_from_stats(base, all_null, ["x"])
    assert adjusted.min_non_null_ratio == pytest.approx(base.min_non_null_ratio, abs=1e-9)
