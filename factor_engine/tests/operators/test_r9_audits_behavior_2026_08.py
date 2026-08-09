# -*- coding: utf-8 -*-
"""R9-P0-028..032 behavioral regression tests for ``operator_audits``.

Verifies the five reworked audit primitives are behavior-based (a "fail" is a
real behavioral observation, never a source-text grep):

* ``audit_cohort_consistency``        — detects non-overlapping-cohort gaps
                                        (input A gap at rows 3-5, input B gap at
                                        rows 8-10); same-row gaps pass.
* ``audit_group_membership_vintage``  — flags historical-membership leaks by
                                        comparing against a uniform baseline.
* ``audit_zero_missing_invalid``      — compares zero vs NaN (not zero vs
                                        finite); a NaN input must not equal 0.
* ``audit_scale_invariance``          — invariant ops have output ratio ~1;
                                        scale-sensitive ops are flagged.
* ``audit_temporal_exclusion`` /      — warmup region must stay NaN; the
  ``audit_effective_sample_size``       effective-sample gate must fail closed
                                        on short data.

All tests use small synthetic operators (deterministic behavior), so they never
touch the operator registry / load_all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators.operator_audits import (
    AuditResult,
    audit_cohort_consistency,
    audit_effective_sample_size,
    audit_group_membership_vintage,
    audit_scale_invariance,
    audit_temporal_exclusion,
    audit_zero_missing_invalid,
)


def _frame(n: int = 24, cols: int = 3, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(100 + np.cumsum(rng.normal(0, 1, (n, cols)), axis=0),
                        index=idx, columns=[f"C{i}" for i in range(cols)])


def _group(n: int = 24, cols: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({f"C{i}": ["G0"] * n for i in range(cols)}, index=idx)


# ---------------------------------------------------------------------------
# Synthetic operators (deterministic behavior)
# ---------------------------------------------------------------------------

class CohortConsistentOp:
    """Elementwise sum — NaN propagates; output missing wherever ANY input is
    missing (cohort-consistent)."""

    def calculate(self, a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
        return a + b


class CohortMismatchedOp:
    """Repairs gaps (forward-fill / zero-fill) — emits finite values where an
    input is missing (cohort-inconsistent)."""

    def calculate(self, a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
        return a.ffill().fillna(0.0) + b.ffill().fillna(0.0)


class CurrentMembershipOp:
    """Group demean using ONLY today's group labels (current_members_retrospective)."""

    def calculate(self, x: pd.DataFrame, group_id: pd.DataFrame) -> pd.DataFrame:
        labels = group_id.iloc[-1]
        out = x.copy()
        for g in pd.unique(labels):
            cols = labels[labels == g].index
            if len(cols) > 1:
                out[cols] = x[cols].sub(x[cols].mean(axis=1), axis=0)
        return out


class VintageLeakingOp:
    """Group demean using the label AT EACH ROW — historical membership leaks."""

    def calculate(self, x: pd.DataFrame, group_id: pd.DataFrame) -> pd.DataFrame:
        out = x.copy()
        vals = x.values.astype(float)
        glab = group_id.values
        for t in range(len(x)):
            row_labels = glab[t]
            for g in np.unique(row_labels):
                cols = np.where(row_labels == g)[0]
                if len(cols) > 1:
                    out.iloc[t, cols] = vals[t, cols] - vals[t, cols].mean()
        return out


class TriStateOkOp:
    """Identity — NaN stays NaN, 0 stays 0 (distinct tri-state)."""

    def calculate(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x


class ConflateOp:
    """fillna(0) — treats NaN exactly like 0 (zero/missing conflation)."""

    def calculate(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.fillna(0.0)


class RankScaleOp:
    """Cross-sectional pct-rank — invariant to uniform input scaling."""

    def calculate(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.rank(axis=1, pct=True)


class LinearScaleOp:
    """x*2 — output scales linearly with input (NOT scale-invariant)."""

    def calculate(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x * 2.0


class StrictWarmupOp:
    """Rolling mean that stays NaN until a full window of data exists."""

    window = 8

    def calculate(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.rolling(self.window, min_periods=self.window).mean()


class RepairGapOp:
    """fillna(0) before a short rolling mean — fabricates values in the NaN
    warmup region."""

    def calculate(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.fillna(0.0).rolling(3, min_periods=1).mean()


class EssGatedOp:
    """Fail-closed on insufficient effective sample: NaN until `window` rows."""

    window = 12

    def calculate(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.rolling(self.window, min_periods=self.window).mean()


class NoEssGateOp:
    """No sample-size gate — returns input values immediately."""

    def calculate(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x


# ---------------------------------------------------------------------------
# 1. Cohort audit (R9-P0-028)
# ---------------------------------------------------------------------------

def test_cohort_audit_same_row_gap_passes() -> None:
    a, b = _frame(seed=1), _frame(seed=2)
    res = audit_cohort_consistency(CohortConsistentOp(), ["a", "b"],
                                   {"a": a, "b": b}, {})
    same = [r for r in res if r.metric == "same_row_gap_nan_coverage"]
    assert len(same) == 1
    assert same[0].status == "pass"
    assert same[0].expected == 1.0
    assert same[0].observed == 1.0


def test_cohort_audit_non_overlap_gap_passes_for_consistent_op() -> None:
    a, b = _frame(seed=1), _frame(seed=2)
    res = audit_cohort_consistency(CohortConsistentOp(), ["a", "b"],
                                   {"a": a, "b": b}, {})
    non = [r for r in res if r.metric == "non_overlap_gap_nan_coverage"]
    assert len(non) == 1
    assert non[0].status == "pass"
    assert non[0].observed == 1.0


def test_cohort_audit_detects_multi_input_misalignment() -> None:
    """Input A gap at rows 3-5, input B gap at rows 8-10 (DIFFERENT rows) must
    be reported as a fail by the non-overlapping-cohort metric."""
    a, b = _frame(seed=1), _frame(seed=2)
    res = audit_cohort_consistency(CohortMismatchedOp(), ["a", "b"],
                                   {"a": a, "b": b}, {})
    by_metric = {r.metric: r for r in res}
    assert "non_overlap_gap_nan_coverage" in by_metric
    assert by_metric["non_overlap_gap_nan_coverage"].status == "fail"
    assert by_metric["non_overlap_gap_nan_coverage"].observed < 1.0
    assert by_metric["same_row_gap_nan_coverage"].status == "fail"
    assert by_metric["same_row_gap_nan_coverage"].observed < 1.0


# ---------------------------------------------------------------------------
# 2. Group-membership vintage audit (R9-P0-029)
# ---------------------------------------------------------------------------

def test_group_vintage_audit_current_membership_passes() -> None:
    x, g = _frame(seed=3), _group()
    res = audit_group_membership_vintage(
        CurrentMembershipOp(), ["x", "group_id"], {"x": x, "group_id": g}, {})
    assert len(res) == 1
    assert res[0].status == "pass"
    assert res[0].expected == 0.0
    assert res[0].observed == 0.0


def test_group_vintage_audit_detects_historical_leak() -> None:
    x, g = _frame(seed=3), _group()
    res = audit_group_membership_vintage(
        VintageLeakingOp(), ["x", "group_id"], {"x": x, "group_id": g}, {})
    assert len(res) == 1
    assert res[0].status == "fail"
    assert res[0].observed > 0.0


# ---------------------------------------------------------------------------
# 3. Zero-vs-missing audit (R9-P0-030)
# ---------------------------------------------------------------------------

def test_zero_missing_audit_zero_vs_nan_distinct_passes() -> None:
    x = _frame(seed=4)
    res = audit_zero_missing_invalid(TriStateOkOp(), "x", {"x": x}, {})
    assert len(res) == 1
    assert res[0].status == "pass"
    assert res[0].expected == 0.0
    assert res[0].observed == 0.0


def test_zero_missing_audit_detects_zero_nan_conflation() -> None:
    """A NaN input must NOT be treated as 0: fillna(0) must be flagged."""
    x = _frame(seed=4)
    res = audit_zero_missing_invalid(ConflateOp(), "x", {"x": x}, {})
    assert len(res) == 1
    assert res[0].status == "fail"
    assert res[0].observed == 1.0


# ---------------------------------------------------------------------------
# 4. Scale-invariance audit (R9-P0-030)
# ---------------------------------------------------------------------------

def test_scale_audit_invariant_ratio_one_passes() -> None:
    x = _frame(seed=5)
    res = audit_scale_invariance(RankScaleOp(), "x", {"x": x}, {})
    assert len(res) == 1
    assert res[0].status == "pass"
    assert res[0].observed == pytest.approx(1.0)


def test_scale_audit_detects_scale_sensitive_op() -> None:
    x = _frame(seed=5)
    res = audit_scale_invariance(LinearScaleOp(), "x", {"x": x}, {})
    assert len(res) == 1
    assert res[0].status == "fail"
    assert res[0].observed == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 5. Temporal exclusion / ESS audits (R9-P0-031)
# ---------------------------------------------------------------------------

def test_temporal_exclusion_audit_warmup_nan_passes() -> None:
    x = _frame(seed=6)
    res = audit_temporal_exclusion(StrictWarmupOp(), ["x"], {"x": x}, {})
    assert len(res) == 1
    assert res[0].status == "pass"
    assert res[0].expected == 1.0
    assert res[0].observed == 1.0


def test_temporal_exclusion_audit_detects_missing_warmup_exclusion() -> None:
    x = _frame(seed=6)
    res = audit_temporal_exclusion(RepairGapOp(), ["x"], {"x": x}, {})
    assert len(res) == 1
    assert res[0].status == "fail"
    assert res[0].observed < 1.0


def test_effective_sample_audit_gate_behaves() -> None:
    x = _frame(seed=6)
    res = audit_effective_sample_size(EssGatedOp(), ["x"], {"x": x}, {})
    assert len(res) == 1
    assert res[0].status == "pass"


def test_effective_sample_audit_detects_missing_gate() -> None:
    x = _frame(seed=6)
    res = audit_effective_sample_size(NoEssGateOp(), ["x"], {"x": x}, {})
    assert len(res) == 1
    assert res[0].status == "fail"


# ---------------------------------------------------------------------------
# Result-shape contract
# ---------------------------------------------------------------------------

def test_audit_results_are_structured_records() -> None:
    """Every audit returns a non-empty list of AuditResult (never a bare bool)."""
    a, b = _frame(seed=1), _frame(seed=2)
    x, g = _frame(seed=3), _group()
    cases = [
        audit_cohort_consistency(CohortConsistentOp(), ["a", "b"],
                                 {"a": a, "b": b}, {}),
        audit_group_membership_vintage(
            CurrentMembershipOp(), ["x", "group_id"], {"x": x, "group_id": g}, {}),
        audit_zero_missing_invalid(TriStateOkOp(), "x", {"x": x}, {}),
        audit_scale_invariance(RankScaleOp(), "x", {"x": x}, {}),
        audit_temporal_exclusion(StrictWarmupOp(), ["x"], {"x": x}, {}),
        audit_effective_sample_size(EssGatedOp(), ["x"], {"x": x}, {}),
    ]
    for results in cases:
        assert isinstance(results, list)
        assert len(results) >= 1
        for r in results:
            assert isinstance(r, AuditResult)
            assert r.status in ("pass", "fail", "info")
            assert r.test_name
            assert r.canonical
            assert r.metric
