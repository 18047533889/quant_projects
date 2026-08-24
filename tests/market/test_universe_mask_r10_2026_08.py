# -*- coding: utf-8 -*-
"""Review-10 universe-mask contract fixes (R10-P0-010..013).

* R10-P0-010 — production rejects a mask that is missing a REQUIRED component
  (or has no components at all); research still allows a partial mask.
* R10-P0-011 — per-component predicates: ``is_suspend`` uses ``==0`` and
  ``stock_list.type`` uses ``=="CS"``, not the blanket ``!=0``.
* R10-P0-012 — ``is_market_eligible`` matches canonical field identity, never
  substring (``preclose`` must not satisfy ``close``).
* R10-P0-013 — production applies the mask with EXACT axis identity; a mask
  missing a column raises instead of silently reindexing to NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factor_engine.market.universe import (
    MaskComponentSpec,
    UniverseContractError,
    apply_mask_predicate,
    apply_universe_mask,
    apply_universe_mask_to_panel,
    is_market_eligible,
)


def _panel(rows: int = 3, cols: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=rows)
    return pd.DataFrame(
        np.arange(1.0, rows * cols + 1.0).reshape(rows, cols),
        index=idx,
        columns=["A", "B", "C"],
    )


# ---------------------------------------------------------------------------
# R10-P0-010: production fail-closed on missing components
# ---------------------------------------------------------------------------

def test_production_missing_component_raises():
    panel = _panel()
    ones = pd.DataFrame(np.ones(panel.shape), index=panel.index, columns=panel.columns)
    with pytest.raises(UniverseContractError, match="missing"):
        apply_universe_mask_to_panel(
            panel, "ashare",
            {"tradability_state": ones, "close": ones},
            mode="production",
        )


def test_production_no_components_raises():
    panel = _panel()
    with pytest.raises(UniverseContractError, match="missing"):
        apply_universe_mask_to_panel(panel, "ashare", None, mode="production")


def test_research_partial_mask_still_allowed():
    panel = _panel()
    ones = pd.DataFrame(np.ones(panel.shape), index=panel.index, columns=panel.columns)
    out, mask, cov = apply_universe_mask_to_panel(
        panel, "ashare", {"close": ones}
    )
    assert out.shape == panel.shape
    assert mask.shape == panel.shape
    assert cov == 1.0


# ---------------------------------------------------------------------------
# R10-P0-011: per-component predicates, not blanket !=0
# ---------------------------------------------------------------------------

def test_is_suspend_predicate_equals_zero():
    panel = _panel()
    # 1 = suspended, 0 = tradable.  B is suspended.
    suspend = pd.DataFrame(
        [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        index=panel.index, columns=panel.columns,
    )
    ones = pd.DataFrame(np.ones(panel.shape), index=panel.index, columns=panel.columns)
    _, mask_ok, _ = apply_universe_mask_to_panel(
        panel, "ashare",
        {"is_suspend": suspend, "tradability_state": ones, "public_status": ones, "close": ones},
        mode="production",
        predicates={"is_suspend": MaskComponentSpec("is_suspend", "==0").predicate},
    )
    # with ==0, the suspended column B is OUT (mask is a numpy array: col order A B C)
    assert mask_ok[0, 1] == 0.0
    assert mask_ok[0, 0] == 1.0
    assert mask_ok[0, 2] == 1.0

    # contrast: the old blanket !=0 predicate would have KEPT the suspended name
    _, mask_bad, _ = apply_universe_mask_to_panel(
        panel, "ashare",
        {"is_suspend": suspend, "tradability_state": ones, "public_status": ones, "close": ones},
        mode="production",
        predicates={"is_suspend": "!=0"},
    )
    assert mask_bad[0, 1] == 1.0  # wrongly in-pool under !=0


def test_stock_list_type_equals_cs():
    panel = _panel()
    typ = pd.DataFrame(
        [["CS", "NO", "CS"], ["CS", "NO", "CS"], ["CS", "NO", "CS"]],
        index=panel.index, columns=panel.columns, dtype=object,
    )
    ones = pd.DataFrame(np.ones(panel.shape), index=panel.index, columns=panel.columns)
    _, mask, _ = apply_universe_mask_to_panel(
        panel, "us",
        {"stock_list.type": typ, "universe_daily": ones, "close": ones},
        mode="production",
        predicates={"stock_list.type": '=="CS"'},
    )
    assert mask[0, 0] == 1.0
    assert mask[0, 1] == 0.0  # non-CS out
    assert mask[0, 2] == 1.0


def test_apply_mask_predicate_nan_fails_closed():
    arr = np.array([1.0, np.nan, 0.0])
    np.testing.assert_array_equal(
        apply_mask_predicate(arr, "!=0"), [True, False, False]
    )
    np.testing.assert_array_equal(
        apply_mask_predicate(arr, "==0"), [False, False, True]
    )


# ---------------------------------------------------------------------------
# R10-P0-012: canonical field identity, no substring
# ---------------------------------------------------------------------------

def test_is_market_eligible_no_substring_matching():
    # "preclose" must NOT satisfy "close"
    assert not is_market_eligible("ashare", ("preclose",))
    # a canonical FieldID / last-component spelling DOES satisfy it
    assert is_market_eligible(
        "ashare",
        ("StockDailyBar.Close", "StockStatus.public_status", "StockStatus.tradability_state"),
    )
    # "closing_price" also must not satisfy "close"
    assert not is_market_eligible("ashare", ("closing_price",))


# ---------------------------------------------------------------------------
# R10-P0-013: strict axis identity in production
# ---------------------------------------------------------------------------

def test_strict_alignment_rejects_incomplete_mask_columns():
    panel = _panel(rows=3, cols=3)
    mask = pd.DataFrame(
        np.ones((3, 2)), index=panel.index, columns=["A", "B"]
    )
    with pytest.raises(UniverseContractError, match="EXACTLY the panel"):
        apply_universe_mask(panel, mask, strict_alignment=True)


def test_strict_alignment_exact_axes_ok():
    panel = _panel(rows=3, cols=3)
    mask = pd.DataFrame(
        np.ones(panel.shape), index=panel.index, columns=panel.columns
    )
    out = apply_universe_mask(panel, mask, strict_alignment=True)
    assert out.shape == panel.shape


def test_production_strict_alignment_on_component():
    panel = _panel(rows=3, cols=3)
    ones = pd.DataFrame(np.ones((3, 2)), index=panel.index, columns=["A", "B"])
    with pytest.raises(UniverseContractError, match="exact axes"):
        apply_universe_mask_to_panel(
            panel, "ashare",
            {"tradability_state": ones, "public_status": ones, "close": ones},
            mode="production",
        )
