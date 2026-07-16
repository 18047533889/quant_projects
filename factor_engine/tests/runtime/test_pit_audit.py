# -*- coding: utf-8
"""PIT 安全审计测试。"""

from __future__ import annotations

import pytest

from api import ts_mean
from api.columns import col
from api.factor import Factor
from ir.analyzer import Analyzer
from runtime.pit_audit import PitSafetyError, assert_pit_safe, audit_ir


def test_audit_ir_passes_causal_factor():
    factor = Factor(name="mom", expr=ts_mean(col("close"), 5))
    ir = Analyzer().lower(factor.expr).ir
    report = audit_ir(ir)
    assert report.passed


def test_audit_ir_flags_negative_lag():
    from expr.cleaned_call import CleanedCall
    from expr.column import ColumnRef
    from expr.literal import Literal

    call = CleanedCall("ts_pct", (ColumnRef("close"), Literal(-1)))
    ir = Analyzer().lower(call).ir
    report = audit_ir(ir)
    assert not report.passed
    assert any(v.startswith("ts_pct(") for v in report.violations)


def test_assert_pit_safe_raises_when_enforced():
    from expr.cleaned_call import CleanedCall
    from expr.column import ColumnRef
    from expr.literal import Literal

    call = CleanedCall("ts_pct", (ColumnRef("close"), Literal(-1)))
    ir = Analyzer().lower(call).ir
    with pytest.raises(PitSafetyError):
        assert_pit_safe(ir, enforce=True)
