# -*- coding: utf-8 -*-
"""Round-13 P1 fixes (2026-08-09).

* P1-12: ``fillna(method="ffill"/"pad"/"forward_fill")`` must canonical-rewrite
  onto the single gated ``ffill`` implementation so the
  ``forward_fill_allowed`` / ``max_ffill_gap`` gate cannot be bypassed, and the
  PIT audit must treat ``fillna(method="ffill")`` as a forward fill.
* P1-13: the PIT audit's SourceRef table classification comes from the
  DataAccess contract registry (``LogicalTableContract``) instead of hand-written
  table sets; unknown tables fail closed under ``fail_on_missing=True``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.cleaned_operators.common.data_cleaning import FillForward, FillNA


# ---------------------------------------------------------------------------
# P1-12  fillna(method="ffill") is canonical-rewritten to the gated ffill
# ---------------------------------------------------------------------------
def _panel() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "A": [1.0, np.nan, np.nan, 4.0, np.nan],
            "B": [np.nan, 2.0, np.nan, np.nan, 5.0],
        },
        index=idx,
    )


@pytest.mark.parametrize("method", ["ffill", "pad", "forward_fill"])
def test_p12_fillna_ffill_class_matches_ffill(method):
    """``fillna(x, method="ffill")`` and ``ffill(x)`` produce identical output —
    the same single forward-fill implementation."""
    x = _panel()
    expected = FillForward()._calculate_series(x)
    got = FillNA()._calculate_series(x, method)
    pd.testing.assert_frame_equal(got, expected)


def test_p12_fillna_ffill_respects_forward_fill_gate():
    """The gate applies through the fillna spelling too (fail-closed on
    ``forward_fill_allowed=False``, bounded by ``max_ffill_gap``)."""
    x = _panel()
    blocked = FillNA()._calculate_series(x, "ffill", forward_fill_allowed=False)
    assert np.isnan(blocked["A"].iloc[1]) and np.isnan(blocked["A"].iloc[2])
    assert np.isnan(blocked["B"].iloc[0]) and np.isnan(blocked["B"].iloc[2])

    bounded = FillNA()._calculate_series(x, "ffill", max_ffill_gap=1)
    # A carries at most one bar: row2 stays NaN; B never back-fills.
    assert bounded["A"].iloc[0] == 1.0 and bounded["A"].iloc[1] == 1.0
    assert np.isnan(bounded["A"].iloc[2])
    assert np.isnan(bounded["B"].iloc[0])


def test_p12_audit_flags_fillna_ffill_as_forward_fill():
    """PIT audit with ``forbid_forward_fill=True`` must reject
    ``fillna(..., method="ffill")`` exactly like ``ffill``."""
    from factor_engine.ir.nodes import IRNode
    from factor_engine.runtime.quality.pit_audit import audit_ir

    # ``fillna(x, "ffill")`` lowers to a fillna node whose method literal is an
    # input child at index 1.
    node = IRNode(
        op="fillna",
        inputs=(
            IRNode(op="column", attrs={"name": "close"}),
            IRNode(op="literal", attrs={"value": "ffill"}),
        ),
    )
    report = audit_ir(node, forbid_forward_fill=True)
    assert not report.passed
    assert any("fillna(forward_fill)" in v for v in report.violations)


def test_p12_audit_does_not_flag_fillna_constant():
    """``fillna(x, "mean")`` is not a forward fill and must pass the gate."""
    from factor_engine.ir.nodes import IRNode
    from factor_engine.runtime.quality.pit_audit import audit_ir

    node = IRNode(
        op="fillna",
        inputs=(
            IRNode(op="column", attrs={"name": "close"}),
            IRNode(op="literal", attrs={"value": "mean"}),
        ),
    )
    report = audit_ir(node, forbid_forward_fill=True)
    assert report.passed or not any("forward_fill" in v for v in report.violations)


# ---------------------------------------------------------------------------
# P1-13  PIT authority single-sourced from the DataAccess contract registry
# ---------------------------------------------------------------------------
def _source_ref_name(table: str, field: str, *, transform: str | None = None) -> str:
    from factor_engine.api.source_ref import encode_source_ref, make_source_ref

    spec = make_source_ref(table, field, transform=transform)
    return encode_source_ref(spec)


def test_p13_unknown_table_fails_closed():
    """A table absent from the contract registry is fail-closed in production:
    ``fail_on_missing=True`` yields a violation, never a silent hardcoded pass."""
    from factor_engine.ir.nodes import IRNode
    from factor_engine.runtime.quality.pit_audit import audit_ir

    name = _source_ref_name("MysteryTableNotInRegistry", "Close")
    report = audit_ir(IRNode(op="column", attrs={"name": name}), fail_on_missing=True)
    assert not report.passed
    assert any("unknown_availability_contract" in v for v in report.violations)


def test_p13_contract_layer_not_hardcoded_branch():
    """The SourceRef validation path must consult the contract layer: a table
    that IS in the registry is validated by the contract's join policy, so an
    invalid transform produces the contract message (``unsupported_transform``),
    not the hardcoded one (``unexpected_transform``)."""
    from factor_engine.runtime.quality.pit_audit import _audit_source_ref

    name = _source_ref_name("DailyBar", "Close", transform="asof_backward")
    violations: list[str] = []
    checked: list[str] = []
    _audit_source_ref(
        name,
        forbid_forward_fill=False,
        fail_on_missing=True,
        violations=violations,
        checked=checked,
        source_guard=set(),
    )
    assert checked == ["SourceRef[DailyBar.Close]"]
    assert any("unsupported_transform=asof_backward" in v for v in violations), violations
    assert not any("unexpected_transform" in v for v in violations), violations


def test_p13_valid_contract_table_passes():
    """A well-formed contract SourceRef (anchor daily, no transform) passes
    without a violation."""
    from factor_engine.runtime.quality.pit_audit import _audit_source_ref

    name = _source_ref_name("DailyBar", "Close")
    violations: list[str] = []
    checked: list[str] = []
    _audit_source_ref(
        name,
        forbid_forward_fill=False,
        fail_on_missing=True,
        violations=violations,
        checked=checked,
        source_guard=set(),
    )
    assert violations == [], violations


def test_p13_required_parameter_conflict_from_contract():
    """The contract's ``required_parameter`` (IndexSymbol) is enforced by the
    audit: a conflicting canonical/legacy spelling is rejected."""
    from factor_engine.api.source_ref import make_source_ref
    from factor_engine.runtime.quality.pit_audit import _audit_source_ref

    spec = make_source_ref(
        "BenchmarkIndexDailyBar",
        "Close",
        params={"IndexSymbol": "000985.SH", "index": "000300.SH"},
    )
    from factor_engine.api.source_ref import encode_source_ref

    violations: list[str] = []
    checked: list[str] = []
    _audit_source_ref(
        encode_source_ref(spec),
        forbid_forward_fill=False,
        fail_on_missing=True,
        violations=violations,
        checked=checked,
        source_guard=set(),
    )
    assert any("conflicting_IndexSymbol" in v for v in violations), violations
