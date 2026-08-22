from __future__ import annotations

import pytest

from backend.parameter_aliases import ParameterAliasError
from expr.cleaned_call import CleanedCall
from expr.column import ColumnRef
from ir.analyzer import Analyzer


def _call(op: str, **kwargs) -> CleanedCall:
    return CleanedCall(op, (ColumnRef("close"),), tuple(kwargs.items()))


def test_analyzer_normalizes_window_alias_before_ir():
    ir = Analyzer().lower(_call("ts_mean", d=5)).ir
    assert ir.attrs == {"window": 5}


def test_analyzer_preserves_canonical_d_for_log_return():
    ir = Analyzer().lower(_call("ts_log_return", window=5)).ir
    assert ir.attrs == {"d": 5}


def test_analyzer_rejects_conflicting_alias_and_canonical():
    with pytest.raises(ParameterAliasError, match="conflicting parameters"):
        Analyzer().lower(_call("ts_mean", d=5, window=3))
