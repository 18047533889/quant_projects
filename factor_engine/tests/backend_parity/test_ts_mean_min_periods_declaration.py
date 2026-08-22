"""Test min_periods and ddof parameter declarations for ts_mean and ts_std.

These tests verify that min_periods and ddof are declared in the operator spec.
If they are not declared, the tests will fail with a clear error message.
"""
from __future__ import annotations

import pytest

# Try to import from backend.operator_types
try:
    from backend.operator_types import OPERATOR_SIGNATURES
    from backend.operator_signatures_phase2 import phase2_operator_signatures
except ImportError as e:
    pytest.skip(f"Cannot import operator signatures: {e}", allow_module_level=True)


def _get_all_signatures() -> dict:
    """Merge base and phase2 signatures."""
    sigs = dict(OPERATOR_SIGNATURES)
    sigs.update(phase2_operator_signatures())
    return sigs


class TestTSMeanMinPeriodsDeclaration:
    """Verify ts_mean declares min_periods in its operator signature."""

    def test_min_periods_declared(self):
        """ts_mean must have min_periods as a declared parameter."""
        sigs = _get_all_signatures()
        if "ts_mean" not in sigs:
            pytest.skip("ts_mean not in operator signatures")

        ts_mean_sig = sigs["ts_mean"]
        param_names = [arg.name for arg in ts_mean_sig.inputs]

        assert "min_periods" in param_names, (
            f"ts_mean does not declare min_periods parameter. "
            f"Declared params: {param_names}. "
            f"This must be fixed before ts_mean(min_periods=N) can be used in production."
        )


class TestTSStdDdofDeclaration:
    """Verify ts_std declares ddof in its operator signature."""

    def test_ddof_declared(self):
        """ts_std must have ddof as a declared parameter."""
        sigs = _get_all_signatures()
        if "ts_std" not in sigs:
            pytest.skip("ts_std not in operator signatures")

        ts_std_sig = sigs["ts_std"]
        param_names = [arg.name for arg in ts_std_sig.inputs]

        assert "ddof" in param_names, (
            f"ts_std does not declare ddof parameter. "
            f"Declared params: {param_names}. "
            f"This must be fixed before ts_std(ddof=N) can be used in production."
        )
