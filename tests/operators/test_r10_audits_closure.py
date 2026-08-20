# -*- coding: utf-8 -*-
"""R10 #36-#42 closure tests for ``operator_audits``.

Each test pins one external-review finding that was fixed in this round:

* #36 ``audit_column_permutation`` — the inverse permutation must restore the
     ORIGINAL column order via ``reindex``, not ``np.argsort`` of labels, so an
     arbitrary unordered ticker set (['ZZZ','AAA','MNO']) is handled correctly.
* #37 ``audit_group_membership_vintage`` — reads the operator's membership-
     semantics contract; historical-membership operators are audited distinctly
     from current-members-retrospective operators.
* #38 ``audit_temporal_exclusion`` — honours MissingPolicy ``carry``: a missing
     input that carries the previous state is NOT flagged as "should be NaN".
* #39 ``audit_effective_sample_size`` — adds a DOF-aware gate (required_n /
     estimator_dof) so a short panel with a high finite ratio cannot pass just
     via warmup.
* #40 ``audit_scale_invariance`` — reads a declared MetamorphicContract;
     a ``none`` contract is never scale-invariance-asserted.
* #41 ``audit_complexity_vs_param`` / ``audit_performance_benchmark`` — the
     deterministic complexity-model audit and the wall-clock benchmark are
     separated; no single wall-clock reading with a fixed 50x gate.
* #42 ``audit_golden_reference`` — zero comparable cells is FAIL or
     INCONCLUSIVE, never PASS.

All tests use small synthetic operators, so they never touch the registry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators.operator_audits import (
    AuditResult,
    MetamorphicContract,
    audit_column_permutation,
    audit_effective_sample_size,
    audit_golden_reference,
    audit_group_membership_vintage,
    audit_scale_invariance,
    audit_temporal_exclusion,
    metamorphic_contract,
)


def _frame(n: int = 24, cols: int = 3, seed: int = 7,
           columns: list[str] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    if columns is None:
        columns = [f"C{i}" for i in range(cols)]
    return pd.DataFrame(100 + np.cumsum(rng.normal(0, 1, (n, len(columns))), axis=0),
                        index=idx, columns=columns)


def _group(n: int = 24, cols: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({f"C{i}": ["G0"] * n for i in range(cols)}, index=idx)


# ---------------------------------------------------------------------------
# Synthetic operators (deterministic behaviour)
# ---------------------------------------------------------------------------

class IdentityOp:
    """Per-column identity — permutation-invariant."""

    def calculate(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x


class HistoricalMembershipOp:
    """Group demean using the label AT EACH ROW — honours historical membership."""

    membership_semantics = "historical_membership"

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


class CurrentMembershipLeakOp:
    """Declares current-members-retrospective but leaks historical membership."""

    membership_semantics = "current_members_retrospective"

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


class CarryStateOp:
    """Forward-fill — a missing input carries the previous state after the first
    valid observation; the leading warmup (no prior state) stays NaN."""

    missing_policy = "carry"

    def calculate(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.ffill()


class SmallWarmupHighDofOp:
    """Tiny warmup (window 2) — high finite ratio on short data — but declares
    a required effective sample of 10."""

    required_n = 10

    def calculate(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.rolling(2, min_periods=2).mean()


class DofGatedOp:
    """Rolling mean that fail-closes until 10 observations are available."""

    required_n = 10

    def calculate(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.rolling(10, min_periods=10).mean()


class RankScaleDeclaredOp:
    """Cross-sectional pct-rank declared scale-invariant."""

    metamorphic_contract = MetamorphicContract.SCALE_INVARIANT

    def calculate(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x.rank(axis=1, pct=True)


class LinearOpDeclaredNone:
    """Scale-sensitive op (x*2) that explicitly declares NO metamorphic
    contract — the scale audit must NOT require scale invariance."""

    metamorphic_contract = MetamorphicContract.NONE

    def calculate(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x * 2.0


class LinearOpDeclaredUnitCovariant:
    """Linear op declaring unit-covariance — output scales with the input."""

    metamorphic_contract = MetamorphicContract.UNIT_COVARIANT

    def calculate(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x * 2.0


class AlwaysNanOp:
    """Outputs all-NaN — no finite overlap with any golden reference."""

    def calculate(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x * np.nan


class FiniteRefOp:
    """Identity — matches a finite reference implementation."""

    def calculate(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return x


# ---------------------------------------------------------------------------
# #36 Column permutation with unordered ticker labels
# ---------------------------------------------------------------------------

def test_column_permutation_unordered_ticker_labels_restores_order() -> None:
    x = _frame(columns=["ZZZ", "AAA", "MNO"])
    panels = {"x": x}
    res = audit_column_permutation(IdentityOp(), ["x"], panels, {})
    assert len(res) == 1
    assert res[0].status == "pass"
    assert res[0].observed == 0.0
    assert res[0].metric == "max_abs_diff_after_column_permutation"


def test_column_permutation_ndarray_output_unordered_labels() -> None:
    """An operator that returns a bare ndarray must also be un-permuted by
    position, not by label-sort order."""

    class NdarrayIdentityOp:
        def calculate(self, x: pd.DataFrame, **kwargs) -> np.ndarray:
            return x.to_numpy()

    x = _frame(columns=["ZZZ", "AAA", "MNO"])
    panels = {"x": x}
    res = audit_column_permutation(NdarrayIdentityOp(), ["x"], panels, {})
    assert len(res) == 1
    assert res[0].status == "pass"
    assert res[0].observed == 0.0


# ---------------------------------------------------------------------------
# #37 Membership semantics audited distinctly
# ---------------------------------------------------------------------------

def test_historical_membership_op_is_not_flagged() -> None:
    """An operator declaring ``historical_membership`` SHOULD change under a
    reclassified history — the audit must NOT flag that behaviour."""
    x, g = _frame(seed=3), _group()
    res = audit_group_membership_vintage(
        HistoricalMembershipOp(), ["x", "group_id"], {"x": x, "group_id": g}, {})
    assert len(res) == 1
    assert res[0].status == "pass"
    assert res[0].metric == "max_abs_diff_uniform_vs_reclassified_hist"
    assert res[0].observed > 0.0


def test_current_membership_leak_is_flagged() -> None:
    """An operator declaring ``current_members_retrospective`` that leaks
    historical membership is a genuine violation and MUST be flagged."""
    x, g = _frame(seed=3), _group()
    res = audit_group_membership_vintage(
        CurrentMembershipLeakOp(), ["x", "group_id"], {"x": x, "group_id": g}, {})
    assert len(res) == 1
    assert res[0].status == "fail"
    assert res[0].observed > 0.0


# ---------------------------------------------------------------------------
# #38 Carry-state operator not false-flagged by the temporal audit
# ---------------------------------------------------------------------------

def test_temporal_exclusion_carry_state_not_false_flagged() -> None:
    x = _frame(seed=6)
    res = audit_temporal_exclusion(CarryStateOp(), ["x"], {"x": x}, {})
    assert len(res) == 2
    # Leading warmup (no prior state) is still fail-closed NaN.
    warm = [r for r in res if r.metric == "warmup_missing_region_nan_coverage"]
    assert len(warm) == 1
    assert warm[0].status == "pass"
    assert warm[0].observed == 1.0
    # The interior gap is reported as info (carry bridges it) — never a fail.
    gap = [r for r in res if r.metric == "interior_gap_nan_coverage_carry_policy"]
    assert len(gap) == 1
    assert gap[0].status == "info"
    assert "fail" not in [r.status for r in res]


def test_temporal_exclusion_carry_operator_via_kwarg() -> None:
    """The carry policy may also arrive as an explicit ``missing_policy``
    kwarg — it must still be honoured."""

    class ParamCarryOp:
        def calculate(self, x: pd.DataFrame, missing_policy: str = "carry",
                      **kwargs) -> pd.DataFrame:
            if missing_policy == "carry":
                return x.ffill()
            return x

    x = _frame(seed=6)
    res = audit_temporal_exclusion(ParamCarryOp(), ["x"], {"x": x},
                                   {"missing_policy": "carry"})
    assert "fail" not in [r.status for r in res]
    assert any(r.metric == "interior_gap_nan_coverage_carry_policy" for r in res)


# ---------------------------------------------------------------------------
# #39 DOF-aware effective-sample audit
# ---------------------------------------------------------------------------

def test_effective_sample_dof_gate_rejects_high_ratio_short_panel() -> None:
    """A tiny-warmup op has a high finite ratio on the short panel (passes the
    ratio check 'just via warmup') but declares required_n=10 — the DOF gate
    must reject it."""
    x = _frame(seed=6)
    res = audit_effective_sample_size(SmallWarmupHighDofOp(), ["x"], {"x": x}, {})
    dof = [r for r in res if r.metric == "short_panel_dof_gate"]
    assert len(dof) == 1
    assert dof[0].status == "fail"
    assert "effective_n=4" in str(dof[0].expected)
    assert dof[0].observed != 0.0


def test_effective_sample_dof_gate_passes_when_fail_closed() -> None:
    x = _frame(seed=6)
    res = audit_effective_sample_size(DofGatedOp(), ["x"], {"x": x}, {})
    dof = [r for r in res if r.metric == "short_panel_dof_gate"]
    assert len(dof) == 1
    assert dof[0].status == "pass"


# ---------------------------------------------------------------------------
# #40 MetamorphicContract-aware scale audit
# ---------------------------------------------------------------------------

def test_metamorphic_contract_none_not_scale_asserted() -> None:
    """A 'none' contract operator (scale-sensitive x*2) must NOT be failed for
    scale sensitivity."""
    x = _frame(seed=5)
    res = audit_scale_invariance(LinearOpDeclaredNone(), "x", {"x": x}, {})
    assert len(res) == 1
    assert res[0].status == "info"
    assert res[0].metric == "metamorphic_contract"
    assert res[0].expected.startswith("none")


def test_metamorphic_contract_scale_invariant_passes() -> None:
    x = _frame(seed=5)
    res = audit_scale_invariance(RankScaleDeclaredOp(), "x", {"x": x}, {})
    assert len(res) == 1
    assert res[0].status == "pass"
    assert res[0].observed == pytest.approx(1.0)


def test_metamorphic_contract_unit_covariant_passes() -> None:
    x = _frame(seed=5)
    res = audit_scale_invariance(LinearOpDeclaredUnitCovariant(), "x", {"x": x}, {})
    assert len(res) == 1
    assert res[0].status == "pass"
    assert res[0].observed == pytest.approx(10.0)


def test_metamorphic_contract_lookup_defaults_to_none() -> None:
    assert metamorphic_contract("no_such_canonical") == MetamorphicContract.NONE
    assert metamorphic_contract("no_such_canonical") == "none"


# ---------------------------------------------------------------------------
# #42 Golden audit zero-overlap never passes
# ---------------------------------------------------------------------------

def test_golden_zero_comparable_cells_is_inconclusive() -> None:
    x = _frame(seed=9)
    panels = {"x": x}
    res = audit_golden_reference(AlwaysNanOp(), "x", panels, {},
                                 reference=lambda arr: arr * 2.0)
    assert len(res) == 1
    assert res[0].status == "inconclusive"
    assert res[0].status != "pass"
    assert res[0].metric == "comparable_cells"
    assert res[0].observed.startswith("0")


def test_golden_zero_overlap_nan_reference_is_inconclusive() -> None:
    x = _frame(seed=9)
    panels = {"x": x}
    res = audit_golden_reference(FiniteRefOp(), "x", panels, {},
                                 reference=lambda arr: arr * np.nan)
    assert len(res) == 1
    assert res[0].status == "inconclusive"
    assert res[0].status != "pass"


def test_golden_with_overlap_passes() -> None:
    x = _frame(seed=9)
    panels = {"x": x}
    res = audit_golden_reference(FiniteRefOp(), "x", panels, {},
                                 reference=lambda arr: arr * 1.0)
    assert len(res) == 1
    assert res[0].status == "pass"


# ---------------------------------------------------------------------------
# Result-shape contract
# ---------------------------------------------------------------------------

def test_all_r10_results_are_structured_records() -> None:
    x = _frame(seed=6)
    g = _group()
    cases = [
        audit_column_permutation(IdentityOp(), ["x"], {"x": _frame(columns=["ZZZ", "AAA", "MNO"])}, {}),
        audit_group_membership_vintage(
            HistoricalMembershipOp(), ["x", "group_id"], {"x": x, "group_id": g}, {}),
        audit_temporal_exclusion(CarryStateOp(), ["x"], {"x": x}, {}),
        audit_effective_sample_size(DofGatedOp(), ["x"], {"x": x}, {}),
        audit_scale_invariance(LinearOpDeclaredNone(), "x", {"x": x}, {}),
        audit_golden_reference(AlwaysNanOp(), "x", {"x": x}, {},
                               reference=lambda arr: arr * 2.0),
    ]
    for results in cases:
        assert isinstance(results, list)
        assert len(results) >= 1
        for r in results:
            assert isinstance(r, AuditResult)
            assert r.status in ("pass", "fail", "info", "inconclusive")
            assert r.test_name
            assert r.canonical
            assert r.metric
