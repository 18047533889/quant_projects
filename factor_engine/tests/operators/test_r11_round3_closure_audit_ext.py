# -*- coding: utf-8
"""Round-3 closure-audit engine extensions (review §三 Audits 7/10/12).

Three new machine categories join the round-2 engine:

* ``semantic_input_type`` — a declared PositivePrice / NonNegativeVolume /
  ConditionBool input must actually fail closed on a violating value;
* ``null_calibration``    — an estimator knob must not move the null-level
  output dispersion >10x across its legal range;
* ``practical_usability`` — coverage / Inf / constant-column / trailing-NaN
  gates on a realistic 250×20 panel.

These tests pin the framework mechanics and the payoff detections with planted
operators (the registry is sealed after load_all, so category checks are called
directly with hand-built contexts).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import factor_engine.cleaned_operators as cleaned_operators
from factor_engine.cleaned_operators.base import OperatorMetadata, ParamRole, ParamSpec
from factor_engine.cleaned_operators.closure_audit import CLOSURE_CATEGORIES

cleaned_operators.load_all()

from factor_engine.cleaned_operators.semantic_audit import OperatorSample, _panel


def _check_direct(op, category, frames, kwargs):
    from factor_engine.cleaned_operators.closure_audit import CLOSURE_CATEGORIES as _CATS

    sample = OperatorSample("test", frames, kwargs)
    ctx = {"sample": sample, "canonical": "test", "op": op}
    findings, _skip = _CATS[category]["check"](op, ctx)
    return findings


# ---------------------------------------------------------------------------
# framework mechanics
# ---------------------------------------------------------------------------


def test_round3_categories_registered():
    for name in (
        "semantic_input_type",
        "null_calibration",
        "practical_usability",
        "param_search_grade",
    ):
        assert name in CLOSURE_CATEGORIES, name


def test_new_categories_can_skip_honestly():
    """Ops that cannot be exercised record a NOT_APPLICABLE skip, never a crash."""
    from factor_engine.cleaned_operators.closure_audit import run_closure_audit

    report = run_closure_audit(
        canonical_names=["ts_count_if"], categories=["practical_usability"]
    )
    assert report.audit_error_count == 0, report.audit_errors
    assert report.ran.get("practical_usability", 0) + len(
        report.skipped.get("practical_usability", [])
    ) == 1


# ---------------------------------------------------------------------------
# planted-bad detections
# ---------------------------------------------------------------------------


class _AcceptsNegativePrice(cleaned_operators.base.SeriesOperator):
    """Bad: declares input_units=price but passes -1.0 straight through."""

    metadata = OperatorMetadata(
        name="test_bad_accepts_negative_price",
        category="test",
        description="price input with no enforcement",
        param_names=["x"],
        panel_params=("x",),
        input_units={"x": "price"},
        return_type="series",
        tags=["unit:dimensionless"],
    )

    def _calculate_series(self, x: pd.DataFrame, **_: object) -> pd.DataFrame:
        return x


class _RejectsNegativePrice(cleaned_operators.base.SeriesOperator):
    """Good: declares input_units=price and poisons negative values to NaN."""

    metadata = OperatorMetadata(
        name="test_good_rejects_negative_price",
        category="test",
        description="price input enforced",
        param_names=["x"],
        panel_params=("x",),
        input_units={"x": "price"},
        return_type="series",
        tags=["unit:dimensionless"],
    )

    def _calculate_series(self, x: pd.DataFrame, **_: object) -> pd.DataFrame:
        arr = x.to_numpy(dtype=float).copy()
        arr = np.where(arr < 0, np.nan, arr)
        return pd.DataFrame(arr, index=x.index, columns=x.columns)


def test_semantic_input_type_flags_undeclared_enforcement():
    frames = [_panel(seed=3, positive=True)]
    hits = _check_direct(_AcceptsNegativePrice(), "semantic_input_type", frames, {})
    assert any("price" in f.message and "no fail-closed" in f.message for f in hits), hits


def test_semantic_input_type_passes_enforced():
    frames = [_panel(seed=3, positive=True)]
    hits = _check_direct(_RejectsNegativePrice(), "semantic_input_type", frames, {})
    assert not hits, hits


class _BinsAmplifiesNoise(cleaned_operators.base.SeriesOperator):
    """Bad: an ESTIMATOR_RESOLUTION knob whose null response scales with the knob."""

    metadata = OperatorMetadata(
        name="test_bad_bins_amplifies_null",
        category="test",
        description="null dispersion scales with bins",
        param_names=["x", "bins"],
        panel_params=("x",),
        scalar_params=("bins",),
        input_arity=1,
        param_specs={
            "bins": ParamSpec(
                dtype=int, min=1, max=64, default=8,
                param_role=ParamRole.ESTIMATOR_RESOLUTION,
            )
        },
        return_type="series",
        tags=["unit:dimensionless"],
    )

    def _calculate_series(self, x: pd.DataFrame, bins: int = 8, **_: object) -> pd.DataFrame:
        return x * float(bins)


class _BinsUnused(cleaned_operators.base.SeriesOperator):
    """Good: an ESTIMATOR_RESOLUTION knob that does not move the null baseline."""

    metadata = OperatorMetadata(
        name="test_good_bins_unused",
        category="test",
        description="null dispersion independent of bins",
        param_names=["x", "bins"],
        panel_params=("x",),
        scalar_params=("bins",),
        input_arity=1,
        param_specs={
            "bins": ParamSpec(
                dtype=int, min=1, max=64, default=8,
                param_role=ParamRole.ESTIMATOR_RESOLUTION,
            )
        },
        return_type="series",
        tags=["unit:dimensionless"],
    )

    def _calculate_series(self, x: pd.DataFrame, bins: int = 8, **_: object) -> pd.DataFrame:
        return x * 1.0


def test_null_calibration_flags_knob_that_moves_null():
    frames = [_panel(seed=5)]
    hits = _check_direct(_BinsAmplifiesNoise(), "null_calibration", frames,
                         {"bins": 8})
    assert any("null-level output dispersion" in f.message for f in hits), hits


def test_null_calibration_passes_invariant_knob():
    frames = [_panel(seed=5)]
    hits = _check_direct(_BinsUnused(), "null_calibration", frames, {"bins": 8})
    assert not hits, hits


class _ConstantOutput(cleaned_operators.base.SeriesOperator):
    """Bad: emits a constant factor regardless of input."""

    metadata = OperatorMetadata(
        name="test_bad_constant_output",
        category="test",
        description="constant factor",
        param_names=["x"],
        panel_params=("x",),
        return_type="series",
        tags=["unit:dimensionless"],
    )

    def _calculate_series(self, x: pd.DataFrame, **_: object) -> pd.DataFrame:
        return pd.DataFrame(np.full_like(x.to_numpy(dtype=float), 5.0),
                            index=x.index, columns=x.columns)


class _PassThrough(cleaned_operators.base.SeriesOperator):
    """Good: emits a usable (non-constant, full-coverage) factor."""

    metadata = OperatorMetadata(
        name="test_good_passthrough",
        category="test",
        description="usable factor",
        param_names=["x"],
        panel_params=("x",),
        return_type="series",
        tags=["unit:dimensionless"],
    )

    def _calculate_series(self, x: pd.DataFrame, **_: object) -> pd.DataFrame:
        return x


def test_practical_usability_flags_constant_factor():
    frames = [_panel(seed=9)]
    hits = _check_direct(_ConstantOutput(), "practical_usability", frames, {})
    assert any("constant" in f.message for f in hits), hits


def test_practical_usability_passes_usable_factor():
    frames = [_panel(seed=9)]
    hits = _check_direct(_PassThrough(), "practical_usability", frames, {})
    assert not hits, hits


class _UndeclaredKnob(cleaned_operators.base.SeriesOperator):
    """Bad: ``bins`` with no declared role silently becomes full ECONOMIC."""

    metadata = OperatorMetadata(
        name="test_bad_undeclared_knob",
        category="test",
        description="undeclared estimator knob",
        param_names=["x", "bins"],
        panel_params=("x",),
        scalar_params=("bins",),
        input_arity=1,
        param_specs={"bins": ParamSpec(dtype=int, min=2, max=32, default=8)},
        return_type="series",
        tags=["unit:dimensionless"],
    )

    def _calculate_series(self, x: pd.DataFrame, bins: int = 8, **_: object) -> pd.DataFrame:
        return x * float(bins)


class _ContradictoryRole(cleaned_operators.base.SeriesOperator):
    """Bad: a NUMERICAL role with searchable=True."""

    metadata = OperatorMetadata(
        name="test_bad_contradictory_role",
        category="test",
        description="numerical role but searchable",
        param_names=["x", "eps"],
        panel_params=("x",),
        scalar_params=("eps",),
        input_arity=1,
        param_specs={
            "eps": ParamSpec(
                dtype=float, default=1e-8, searchable=True,
                param_role=ParamRole.NUMERICAL,
            )
        },
        return_type="series",
        tags=["unit:dimensionless"],
    )

    def _calculate_series(self, x: pd.DataFrame, eps: float = 1e-8, **_: object) -> pd.DataFrame:
        return x * 1.0


def test_param_search_grade_flags_undeclared_knob():
    frames = [_panel(seed=11)]
    hits = _check_direct(_UndeclaredKnob(), "param_search_grade", frames, {})
    assert any("no declared role" in f.message for f in hits), hits


def test_param_search_grade_flags_role_searchable_contradiction():
    frames = [_panel(seed=11)]
    hits = _check_direct(_ContradictoryRole(), "param_search_grade", frames, {})
    assert any("contradiction" in f.message for f in hits), hits


def test_param_search_grade_clean_on_declared_knob():
    hits = _check_direct(_BinsUnused(), "param_search_grade", [_panel(seed=11)], {})
    # _BinsUnused declares ESTIMATOR_RESOLUTION -> no finding
    assert not hits, hits
