from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.cleaned_bridge import _call_cleaned_operator
from backend.parameter_aliases import (
    ParameterAliasError,
    normalize_parameter_aliases,
)


def test_analyzer_parameter_normalization_maps_only_declared_aliases():
    out = normalize_parameter_aliases("ts_mean", {"d": 5, "other": 1})
    assert out == {"other": 1, "window": 5}
    assert normalize_parameter_aliases("add", {"d": 5}) == {"d": 5}


def test_analyzer_parameter_normalization_rejects_conflict():
    with pytest.raises(ParameterAliasError, match="conflicting parameters"):
        normalize_parameter_aliases("ts_mean", {"d": 5, "window": 3})


def test_call_cleaned_operator_accepts_canonical_parameters():
    op = MagicMock()
    op.calculate.return_value = "ok"
    result = _call_cleaned_operator("ts_mean", op, [], {"window": 5})
    assert result == "ok"
    assert op.calculate.call_count == 1
    op.calculate.assert_called_with(window=5)


def test_call_cleaned_operator_rejects_runtime_aliases():
    op = MagicMock()
    with pytest.raises(ParameterAliasError, match="non-canonical runtime"):
        _call_cleaned_operator("ts_mean", op, [], {"d": 5})
    op.calculate.assert_not_called()


def test_call_cleaned_operator_reraises_unrelated_type_error():
    op = MagicMock()
    op.calculate.side_effect = TypeError("unsupported operand type(s)")
    with pytest.raises(TypeError, match="unsupported operand"):
        _call_cleaned_operator("ts_mean", op, [], {"window": 5})
