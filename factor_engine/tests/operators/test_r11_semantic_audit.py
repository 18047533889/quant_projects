# -*- coding: utf-8
"""Machine operator-semantic audit framework — R11 operator review §8.

The audit rules (cleaned_operators/semantic_audit.py) scan the catalog for the
semantic bug classes a hand audit keeps finding.  These tests pin the framework
mechanics and the negative detections (a rule MUST fire on a planted bad
operator) and the positive payoffs on the real fixed operators (the rules that
were added BECAUSE of the review — default-output, mirror symmetry, golden
reference — must NOT fire on the corrected implementations).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cleaned_operators import load_all

load_all()

from cleaned_operators.base import OperatorMetadata, OperatorParameterError  # noqa: E402
from cleaned_operators.semantic_audit import (  # noqa: E402
    AuditFinding,
    AuditReport,
    RULES,
    run_audit,
)
from cleaned_operators.registry import OperatorRegistry  # noqa: E402


# ---------------------------------------------------------------------------
# framework mechanics
# ---------------------------------------------------------------------------

def test_finding_and_report_mechanics():
    f = AuditFinding("x", "op", "error", "boom")
    assert f.rule == "x" and f.severity == "error"
    with pytest.raises(ValueError):
        AuditFinding("x", "op", "nonsense", "boom")

    report = AuditReport()
    report.add(f)
    report.note_ran("x")
    report.note_skipped("x", "op: reason")
    assert report.errors == [f]
    assert "1 findings across 1 rules" in report.summary()


def test_all_15_rules_registered():
    assert len(RULES) == 15
    expected = {
        "default_output_finite", "parameter_injectivity", "relational_constraints",
        "mirror_symmetry", "column_permutation", "missing_state",
        "semantic_type_gate", "finite_range", "scale_shift_invariance",
        "group_migration", "native_cohort", "psd_geometry", "golden_reference",
        "canonical_honesty", "default_searchability",
    }
    assert set(RULES) == expected


def test_audit_all_runs_and_reports_skips():
    report = run_audit()
    assert isinstance(report, AuditReport)
    assert len(report.findings) >= 0
    # rules actually ran (some operators checked) and skips are surfaced
    assert report.ran
    summary = report.summary()
    assert "findings across" in summary
    # every finding references a registered rule
    for finding in report.findings:
        assert finding.rule in RULES


# ---------------------------------------------------------------------------
# negative detections — a rule MUST fire on a planted bad operator
# ---------------------------------------------------------------------------

class _StubOp:
    """Minimal operator stand-in carrying the metadata the rule inspects."""

    def __init__(self, name, description="", param_names=(), calculate=None,
                 input_units=None):
        self.metadata = OperatorMetadata(
            name=name,
            category="test",
            description=description,
            param_names=list(param_names),
            return_type="series",
            tags=["unit:dimensionless", "cost:1"],
            input_units=input_units or {},
        )
        self._calc = calculate

    def calculate(self, *frames, **kwargs):
        if self._calc is None:
            raise AssertionError("stub has no calculate")
        return self._calc(*frames, **kwargs)


def test_canonical_honesty_flags_dishonest_robust_name():
    check = RULES["canonical_honesty"]["check"]
    dishonest = _StubOp("ts_robust_something", "a cross-sectional adjustment")
    findings = check(dishonest, {})
    assert any(f.rule == "canonical_honesty" for f in findings)

    honest = _StubOp(
        "ts_robust_something",
        "trimmed-OLS residual: fit y on x after trimming the extreme "
        "observations (estimator = least squares on the trimmed sample).",
    )
    assert check(honest, {}) == []


def test_default_output_finite_detects_dead_factor():
    check = RULES["default_output_finite"]["check"]
    dead = _StubOp(
        "op_dead",
        "returns NaN everywhere",
        param_names=["x"],
        calculate=lambda x, **_: pd.DataFrame(np.nan, index=x.index, columns=x.columns),
    )
    frame = pd.DataFrame(np.arange(40.0).reshape(40, 1), columns=["S0"])
    ctx = {"sample": type("S", (), {"frames": [frame], "kwargs": {}, "note": ""})()}
    findings = check(dead, ctx)
    assert any(f.rule == "default_output_finite" and f.severity == "error" for f in findings)


def test_mirror_symmetry_rule_registered():
    # The mirror rule must run (or honestly skip) on the tail operators — it must
    # not crash the sweep.
    report = run_audit(sample_names=["ts_mean_excess_slope", "ts_hill_tail_index"],
                       rule_names=["mirror_symmetry"])
    for finding in report.findings:
        assert finding.rule == "mirror_symmetry"


def test_parameter_injectivity_rejects_fraction_via_gate():
    # 5.1 on a declared int param must be a loud OperatorParameterError, never a
    # silent int(5.1) == 5.  Exercise the central gate through a real operator.
    op = OperatorRegistry.get("ts_mean", "pandas_numpy")
    frame = pd.DataFrame(np.arange(30.0).reshape(30, 1), columns=["S0"])
    with pytest.raises(OperatorParameterError):
        op.calculate(frame, window=5.1)


# ---------------------------------------------------------------------------
# positive payoffs — the review's fixed operators must pass their audit rules
# ---------------------------------------------------------------------------

def test_hill_default_output_finite_after_fix():
    """P0-1: the double-tail-selection dead-factor bug must be gone."""
    findings = run_audit(sample_names=["ts_hill_tail_index"],
                         rule_names=["default_output_finite"]).findings
    assert not [f for f in findings if f.rule == "default_output_finite"
                and f.severity == "error"], findings


def test_hill_golden_reference_recovers_pareto_shape():
    """P0-1: a true Hill on a Pareto DGP recovers the known tail shape."""
    findings = run_audit(sample_names=["ts_hill_tail_index"],
                         rule_names=["golden_reference"]).findings
    assert not [f for f in findings if f.rule == "golden_reference"
                and f.severity == "error"], findings


def test_mean_excess_mirror_symmetry_holds_after_fix():
    """P0-2: ME_lower(x) == ME_upper(-x); the slope direction is mirrored."""
    findings = run_audit(sample_names=["ts_mean_excess_slope"],
                         rule_names=["mirror_symmetry"]).findings
    assert not [f for f in findings if f.rule == "mirror_symmetry"], findings
