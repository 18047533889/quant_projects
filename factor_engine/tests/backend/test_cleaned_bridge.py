from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.cleaned_bridge import _call_cleaned_operator, _remap_d_to_window


def test_remap_d_to_window():
    out = _remap_d_to_window({"d": 5, "other": 1})
    assert out == {"other": 1, "window": 5}


def test_remap_d_to_window_skips_when_window_present():
    assert _remap_d_to_window({"d": 5, "window": 3}) is None


def test_call_cleaned_operator_retries_on_unexpected_keyword_d():
    op = MagicMock()
    op.calculate.side_effect = [
        TypeError("calculate() got an unexpected keyword argument 'd'"),
        "ok",
    ]
    result = _call_cleaned_operator(op, [], {"d": 5})
    assert result == "ok"
    assert op.calculate.call_count == 2
    op.calculate.assert_called_with(window=5)


def test_call_cleaned_operator_reraises_unrelated_type_error():
    op = MagicMock()
    op.calculate.side_effect = TypeError("unsupported operand type(s)")
    with pytest.raises(TypeError, match="unsupported operand"):
        _call_cleaned_operator(op, [], {"d": 5})
