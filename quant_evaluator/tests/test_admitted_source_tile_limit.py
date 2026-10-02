"""Contract tests for memory-admitted source tile hints."""
import numpy as np
import pytest

from quant_evaluator.runtime.source_memory_budget import admitted_source_tile_limit


@pytest.mark.parametrize("declared,hint,expected", [
    (16, None, 16), (16, 5, 5), (16, 1, 1), (16, 16, 16),
    (np.int64(16), np.int64(5), 5),
])
def test_admitted_limit_preserves_declared_cap_and_accepts_integral_hint(declared, hint, expected):
    result = admitted_source_tile_limit(declared, hint)
    assert type(result) is int
    assert result == expected


@pytest.mark.parametrize("declared", [0, -1, True, False, 16.0, "16", None])
def test_invalid_declared_cap_is_rejected(declared):
    with pytest.raises(ValueError, match="declared"):
        admitted_source_tile_limit(declared)


@pytest.mark.parametrize("hint", [0, -1, 17, True, False, 5.0, "5", np.nan, np.inf])
def test_invalid_memory_hint_is_rejected(hint):
    with pytest.raises(ValueError, match="admitted"):
        admitted_source_tile_limit(16, hint)
