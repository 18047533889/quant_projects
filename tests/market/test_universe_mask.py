# -*- coding: utf-8 -*-
"""Universal UniverseMask contract (P1-001)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market.universe import (
    apply_universe_mask,
    universe_mask_contract,
)


def test_ashare_mask_contract() -> None:
    c = universe_mask_contract("ashare")
    assert "tradability_state" in c.required_fields  # NOT IsSuspend
    assert "public_status" in c.required_fields
    assert c.shape_preserving


def test_us_mask_contract() -> None:
    c = universe_mask_contract("us")
    assert any("stock_list" in f for f in c.required_fields)
    assert "universe_daily" in c.required_fields
    assert c.shape_preserving


def test_unknown_market_rejected() -> None:
    with pytest.raises(KeyError):
        universe_mask_contract("hk")


def test_apply_universe_mask_shape_preserving() -> None:
    panel = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        index=pd.date_range("2026-01-01", periods=2),
        columns=["A", "B", "C"],
    )
    mask = pd.DataFrame(
        [[True, True, False], [True, False, True]],
        index=panel.index,
        columns=panel.columns,
    )
    out = apply_universe_mask(panel, mask)
    assert out.shape == panel.shape  # shape preserved, nothing dropped
    assert np.isnan(out["C"].iloc[0])  # out-of-universe -> NaN
    assert out["A"].iloc[1] == 4.0
    assert out["A"].iloc[0] == 1.0
    assert np.isnan(out["B"].iloc[1])  # out-of-universe at row 1


def test_apply_universe_mask_shape_mismatch() -> None:
    panel = pd.DataFrame(np.ones((2, 3)))
    mask = np.ones((3, 2), dtype=bool)
    with pytest.raises(ValueError, match="shape mismatch"):
        apply_universe_mask(panel.to_numpy(), mask)


def test_apply_universe_mask_nan_is_out_of_universe() -> None:
    # Audit P0: bool(np.nan) is True, so an unknown universe state must never
    # silently become an in-pool member.
    panel = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        index=pd.date_range("2026-01-01", periods=2),
        columns=["A", "B", "C"],
    )
    mask = pd.DataFrame(
        [[True, np.nan, False], [np.nan, True, False]],
        index=panel.index,
        columns=panel.columns,
    )
    out = apply_universe_mask(panel, mask)
    # NaN mask cells fail closed -> out-of-universe (NaN), not in-pool.
    assert np.isnan(out["A"].iloc[1])
    assert np.isnan(out["B"].iloc[0])
    assert np.isnan(out["C"].iloc[0])
    assert out["A"].iloc[0] == 1.0
    assert out["B"].iloc[1] == 5.0


def test_apply_universe_mask_aligned_by_label_not_position() -> None:
    # Audit P0: same shape but a different column order must not pair A's data
    # with B's mask (positional masking would silently swap instruments).
    panel = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        index=pd.date_range("2026-01-01", periods=2),
        columns=["A", "B", "C"],
    )
    mask = pd.DataFrame(
        [[True, True, False], [True, False, True]],
        index=panel.index,
        columns=["C", "B", "A"],  # same values, different column order
    )
    out = apply_universe_mask(panel, mask)
    # Label-based alignment: mask["A"] governs panel["A"], mask["B"] panel["B"],
    # mask["C"] panel["C"].  After reindex the mask is
    #   A=[False, True], B=[True, False], C=[True, True].
    assert np.isnan(out["A"].iloc[0])  # mask A row0 = False
    assert out["A"].iloc[1] == 4.0  # mask A row1 = True
    assert out["B"].iloc[0] == 2.0  # mask B row0 = True
    assert np.isnan(out["B"].iloc[1])  # mask B row1 = False
    assert out["C"].iloc[0] == 3.0  # mask C row0 = True
    assert out["C"].iloc[1] == 6.0  # mask C row1 = True
