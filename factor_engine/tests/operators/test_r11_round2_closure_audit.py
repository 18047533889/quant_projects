# -*- coding: utf-8
"""Round-2 closure-audit engine (review §20): full-library Operator Closure Audit.

The engine iterates the whole registry and runs machine categories —
param-type fuzzing, parameter-grid dead regions, multi-input axis integrity,
missing-time topology, recursive re-warmup, ConditionBool contracts, unit
algebra, boundary behavior and semantic-duplicate clustering.  These tests pin
the framework (all categories registered, honest skip reasons, AUDIT_ERROR vs
NOT_APPLICABLE separation) and the payoff detections.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factor_engine.cleaned_operators as cleaned_operators
from factor_engine.cleaned_operators.base import SeriesOperator, register_operator
from factor_engine.cleaned_operators.closure_audit import (
    CLOSURE_CATEGORIES,
    _stem,
    run_closure_audit,
)

cleaned_operators.load_all()

# ---------------------------------------------------------------------------
# framework mechanics
# ---------------------------------------------------------------------------


def test_all_round2_categories_registered():
    for name in (
        "param_type_fuzzing",
        "parameter_grid_dead_region",
        "axis_integrity",
        "missing_time_topology",
        "recursive_rewarmup",
        "condition_bool_contract",
        "unit_algebra",
        "boundary_behavior",
        "semantic_duplicates",
    ):
        assert name in CLOSURE_CATEGORIES, name


def test_small_sweep_runs_without_audit_errors():
    report = run_closure_audit(
        canonical_names=["ts_count_if", "ts_downside_deviation"],
        categories=["param_type_fuzzing", "boundary_behavior", "unit_algebra"],
    )
    # no rule crashed on either operator
    assert report.audit_error_count == 0, report.audit_errors
    # honest accounting: every op is either checked or skipped-with-reason
    checked = sum(report.ran.values())
    skipped = sum(len(v) for v in report.skipped.values())
    assert checked + skipped == 2 * 3  # 2 ops * 3 categories


def test_skip_reasons_are_never_silent():
    # a single-panel op has no axis to check -> NOT_APPLICABLE with a reason
    report = run_closure_audit(
        canonical_names=["ts_count_if"], categories=["axis_integrity"]
    )
    assert report.ran.get("axis_integrity", 0) == 0
    reasons = report.skipped.get("axis_integrity", [])
    assert reasons, "single-panel op must record a NOT_APPLICABLE reason"
    assert "single-panel" in reasons[0]


def test_semantic_duplicates_stem():
    assert _stem("ts_variance_ratio") == "ts_variance"
    assert _stem("relation_distribution_kurtosis") == "relation_distribution_kurtosis"
    assert _stem("cs_robust_resid") == "cs_robust"
    assert _stem("piotroski_f_score") == "piotroski_f"


# ---------------------------------------------------------------------------
# planted-bad operator detections
# ---------------------------------------------------------------------------

class _SilentlyTruncatesInt(SeriesOperator):
    """Bad operator: an int-flavoured param (``lag_w``) silently truncates 5.1.

    ``lag_w`` is deliberately OUTSIDE the legacy int whitelist (so the central
    gate does not pre-reject it) but IS name-classified integer (contains
    ``lag``) — this is exactly the fake-parameter the fuzzing category exists to
    catch (review §1: ``int()`` truncation / bool coercion).
    """

    metadata = __import__("factor_engine.cleaned_operators.base", fromlist=["OperatorMetadata"]).OperatorMetadata(
        name="test_bad_silent_int",
        category="test",
        description="lag_w with no contract — accepts 5.1",
        param_names=["x", "lag_w"],
        panel_params=("x",),
        return_type="series",
        tags=["unit:dimensionless"],
    )

    def _calculate_series(self, x: pd.DataFrame, lag_w: float = 5, **_: object) -> pd.DataFrame:
        w = int(lag_w)  # silent truncation
        return x.rolling(w).mean()


class _BadUnitTstat(SeriesOperator):
    """Bad operator: a t-statistic labelled as 'level'."""

    metadata = __import__("factor_engine.cleaned_operators.base", fromlist=["OperatorMetadata"]).OperatorMetadata(
        name="test_bad_tstat_unit",
        category="test",
        description="tstat",
        param_names=["x"],
        panel_params=("x",),
        return_type="series",
        tags=["unit:level"],
    )

    def _calculate_series(self, x: pd.DataFrame, **_: object) -> pd.DataFrame:
        return x


def _check_direct(cls, category, frames, kwargs):
    """Run one closure-audit category against a raw operator instance.

    The registry is sealed after ``load_all`` (layer_governance), so planted
    test operators cannot be registered — call the category check directly with
    a hand-built context instead.
    """
    from factor_engine.cleaned_operators.closure_audit import CLOSURE_CATEGORIES as _CATS
    from factor_engine.cleaned_operators.semantic_audit import OperatorSample

    op = cls()
    sample = OperatorSample("test", frames, kwargs)
    ctx = {"sample": sample, "canonical": "test", "op": op}
    findings, _skip = _CATS[category]["check"](op, ctx)
    return findings


def _audit_panel():
    from factor_engine.cleaned_operators.semantic_audit import _panel

    return [_panel(seed=7)]


def test_fuzzing_detects_silent_int_truncation():
    hits = _check_direct(
        _SilentlyTruncatesInt, "param_type_fuzzing", _audit_panel(), {}
    )
    assert any("5.1" in f.message for f in hits), hits


def test_unit_algebra_detects_tstat_as_level():
    hits = _check_direct(_BadUnitTstat, "unit_algebra", _audit_panel(), {})
    assert hits and hits[0].severity == "error", hits


# ---------------------------------------------------------------------------
# payoff: the round-2 corrected operators are clean on the audit
# ---------------------------------------------------------------------------

def test_round2_corrected_operators_clean_on_core_categories():
    """Operators the round-2 batch corrected must not trip the audited contracts."""
    names = [
        "ts_count_if",          # ConditionBool family
        "ts_downside_deviation",
        "ts_hill_tail_index",
        "ts_mean_excess_slope",
    ]
    report = run_closure_audit(
        canonical_names=names,
        categories=["param_type_fuzzing", "unit_algebra", "boundary_behavior"],
    )
    assert report.audit_error_count == 0, report.audit_errors
    errors = [f for f in report.findings if f.severity == "error"]
    assert not errors, errors
