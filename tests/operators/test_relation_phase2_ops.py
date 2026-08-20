# -*- coding: utf-8 -*-
"""Tests for Phase 2 relation/index/event operators (2026-08-12)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.cleaned_bridge import ensure_cleaned_loaded
from cleaned_operators.operator_surface import classify_canonical
from cleaned_operators.registry import OperatorRegistry

ensure_cleaned_loaded()

# Phase 2 新增：trading_day_diff
# 其余算子（relation_hhi, relation_entropy 等）已在 ops.py 中实现并测试
PHASE2_NEW_CANONICALS = frozenset([
    "trading_day_diff",
])


@pytest.mark.parametrize("name", sorted(PHASE2_NEW_CANONICALS))
def test_phase2_operator_registered_and_extended_surface(name: str) -> None:
    """Verify Phase 2 new operators are registered and on extended surface."""
    assert OperatorRegistry.get(name) is not None
    assert classify_canonical(name) in {"daily", "extended"}


def test_trading_day_diff_approximates_trading_days() -> None:
    """Test trading_day_diff returns approximate trading day difference."""
    idx = pd.date_range("2024-01-01", periods=2)
    # Date difference: 10 calendar days
    date1 = pd.DataFrame([
        [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
        [pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-01")],
    ], index=idx, columns=["A", "B"])

    date2 = pd.DataFrame([
        [pd.Timestamp("2024-01-11"), pd.Timestamp("2024-01-11")],
        [pd.Timestamp("2024-02-11"), pd.Timestamp("2024-02-11")],
    ], index=idx, columns=["A", "B"])

    op = OperatorRegistry.get("trading_day_diff")
    result = op.calculate(date1, date2)

    # 10 calendar days * 0.7 = 7 trading days (approximate)
    # Allow tolerance since this is a simplified implementation
    assert result.iloc[0, 0] == pytest.approx(7.0, abs=1.0)
    assert result.iloc[0, 1] == pytest.approx(7.0, abs=1.0)
    assert result.iloc[1, 0] == pytest.approx(7.0, abs=1.0)
    assert result.iloc[1, 1] == pytest.approx(7.0, abs=1.0)


def test_trading_day_diff_handles_nan() -> None:
    """Test trading_day_diff handles NaN dates correctly."""
    idx = pd.date_range("2024-01-01", periods=1)
    date1 = pd.DataFrame([[pd.Timestamp("2024-01-01"), pd.NaT]], index=idx, columns=["A", "B"])
    date2 = pd.DataFrame([[pd.Timestamp("2024-01-11"), pd.Timestamp("2024-01-11")]], index=idx, columns=["A", "B"])

    op = OperatorRegistry.get("trading_day_diff")
    result = op.calculate(date1, date2)

    # Column A should have valid difference
    assert not np.isnan(result.iloc[0, 0])
    # Column B should have NaN (date1 is NaT)
    assert np.isnan(result.iloc[0, 1])


def test_trading_day_diff_preserves_panel_shape() -> None:
    """Test that trading_day_diff preserves input panel shape."""
    idx = pd.date_range("2024-01-01", periods=5)
    date_panel = pd.DataFrame(
        [[pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)] * 3 for i in range(5)],
        index=idx,
        columns=["A", "B", "C"]
    )
    op = OperatorRegistry.get("trading_day_diff")
    result = op.calculate(date_panel, date_panel.shift(1))
    assert result.shape == date_panel.shape


def test_trading_day_diff_strict_alignment() -> None:
    """Test that trading_day_diff enforces strict panel alignment."""
    idx1 = pd.date_range("2024-01-01", periods=3)
    idx2 = pd.date_range("2024-01-02", periods=3)
    panel1 = pd.DataFrame(
        [[pd.Timestamp("2024-01-01")] * 2] * 3,
        index=idx1,
        columns=["A", "B"]
    )
    panel2 = pd.DataFrame(
        [[pd.Timestamp("2024-01-02")] * 2] * 3,
        index=idx2,
        columns=["A", "B"]
    )

    # Misaligned indices should raise
    op = OperatorRegistry.get("trading_day_diff")
    with pytest.raises(ValueError, match="index"):
        op.calculate(panel1, panel2)


def test_trading_day_diff_zero_difference() -> None:
    """Test trading_day_diff returns 0 for same dates."""
    idx = pd.date_range("2024-01-01", periods=2)
    date = pd.DataFrame([
        [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-15")],
        [pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-15")],
    ], index=idx, columns=["A", "B"])

    op = OperatorRegistry.get("trading_day_diff")
    result = op.calculate(date, date)

    # Same date should give 0 trading days
    assert result.iloc[0, 0] == pytest.approx(0.0)
    assert result.iloc[0, 1] == pytest.approx(0.0)
    assert result.iloc[1, 0] == pytest.approx(0.0)
    assert result.iloc[1, 1] == pytest.approx(0.0)


def test_trading_day_diff_negative_difference() -> None:
    """Test trading_day_diff handles negative differences (date1 > date2)."""
    idx = pd.date_range("2024-01-01", periods=1)
    date1 = pd.DataFrame([[pd.Timestamp("2024-01-15")]], index=idx, columns=["A"])
    date2 = pd.DataFrame([[pd.Timestamp("2024-01-01")]], index=idx, columns=["A"])

    op = OperatorRegistry.get("trading_day_diff")
    result = op.calculate(date1, date2)

    # 14 days * 0.7 = ~10 trading days, but negative
    assert result.iloc[0, 0] < 0
    assert result.iloc[0, 0] == pytest.approx(-10.0, abs=1.0)

