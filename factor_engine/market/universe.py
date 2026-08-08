# -*- coding: utf-8 -*-
"""Universal UniverseMask contract for cross-sectional operators (P1-001).

The audit found that cross-sectional operators (``rank``/``zscore``/
``cs_regression``/``group_*``/``cs_multi_robust_resid``/KNN/local-Moran/
transport/group-spectrum) each decided their own "universe" by "whatever is
finite".  That lets delisted / ST / suspended / non-CS names leak into the
cross-section and biases every rank / zscore / regression / size-neutralization.

This module fixes the ONE universe contract every cross-sectional pass must
apply BEFORE ranking, and provides the shape-preserving helper.  It does not
rewrite every operator: the runtime / mining pipeline applies
``apply_universe_mask`` to the input panel once, then feeds the masked panel to
the operator (operators keep their NaN fail-closed policy on top).

Composition per market (2026-08-08 COS dicts):

A-share (anchor = DailyBar/List member):
    ``tradability_state`` (NOT IsSuspend)      AND
    ``public_status`` listed policy            AND
    finite daily bar (Close/Ret present)

US (anchor = StockList.type == CS):
    ``stock_list.type == "CS"``                AND
    ``universe_daily`` membership              AND
    finite daily bar (Close/Ret present)

A share's trading panel is NOT identical to the Status universe: Status contains
terminated/ST/not-yet-listed codes, so the trading panel anchors on the daily
bar / list.  US DailyBar additionally contains codes outside the ordinary CS
universe and has PreClose/Ret gaps, hence the type + universe_daily + valid bar
triple.  ``is_ticker_halt`` is far too sparse to be an IsSuspend equivalent; it
is an optional minute-level enhancement only, never the A-side daily mask.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.universe_spec import UNIVERSE_SPEC, cross_section_shape_preserving

# Source concepts each market's mask is composed from.
A_SHARE_MASK_FIELDS: tuple[str, ...] = (
    "tradability_state",  # NOT IsSuspend (StockDailyBar.IsSuspend negated)
    "public_status",      # listed-state policy (ashare_stock_status)
    "close",              # finite daily bar anchors the trading panel
)
US_MASK_FIELDS: tuple[str, ...] = (
    "stock_list.type",    # == "CS"
    "universe_daily",     # universe membership
    "close",              # finite daily bar
)

_MASK_FIELDS = {
    "ashare": A_SHARE_MASK_FIELDS,
    "us": US_MASK_FIELDS,
}


@dataclass(frozen=True)
class UniverseMaskContract:
    """The documented universe composition for one market.

    ``required_fields`` are the source concepts the runtime must load to build
    the mask.  ``shape_preserving`` mirrors ``backend.universe_spec``: after the
    mask is applied the panel keeps its (date × instrument) shape, out-of-universe
    names become NaN — never dropped rows.
    """

    market: str
    required_fields: tuple[str, ...]
    notes: str = ""
    shape_preserving: bool = True


def universe_mask_contract(market: str) -> UniverseMaskContract:
    key = str(market).strip().lower()
    if key == "a_share" or key == "cn":
        key = "ashare"
    try:
        fields = _MASK_FIELDS[key]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"no universe mask contract for market {market!r}") from exc
    notes = (
        "A: DailyBar/List anchor AND NOT IsSuspend AND PublicStatus policy AND valid bar"
        if key == "ashare"
        else "US: StockList.type==CS AND universe_daily AND valid bar"
    )
    return UniverseMaskContract(
        market=key,
        required_fields=fields,
        notes=notes,
        shape_preserving=cross_section_shape_preserving(),
    )


def apply_universe_mask(
    panel: Any,
    mask: Any,
    *,
    shape_preserving: bool | None = None,
) -> Any:
    """Mask a (date × instrument) panel to its universe, keeping its shape.

    ``panel`` and ``mask`` are aligned boolean/numeric arrays: True = in-universe.
    Out-of-universe names become NaN (shape-preserving, matching
    ``backend.universe_spec``), never dropped rows.  ``None`` / all-NaN mask
    fail-closes the whole row/column via the NaN result.

    Accepts 2-D numpy arrays or pandas DataFrames (same axes).
    """
    keep_shape = UNIVERSE_SPEC.shape_preserving_cross_section if shape_preserving is None else shape_preserving
    _apply_pandas = True
    try:
        import pandas as pd  # noqa: F401

        if isinstance(panel, pd.DataFrame):
            _apply_pandas = True
    except Exception:  # pragma: no cover - optional pandas
        _apply_pandas = False
    if _apply_pandas:
        import pandas as pd

        if isinstance(panel, pd.DataFrame):
            m = mask.to_numpy(dtype=bool) if isinstance(mask, pd.DataFrame) else np.asarray(mask, dtype=bool)
            out = panel.to_numpy(dtype=float).copy()
            out[~m] = np.nan
            return pd.DataFrame(out, index=panel.index, columns=panel.columns, dtype=float)
    p = np.asarray(panel, dtype=float).copy()
    m = np.asarray(mask, dtype=bool)
    if p.shape != m.shape:
        raise ValueError(f"panel/mask shape mismatch: {p.shape} vs {m.shape}")
    p[~m] = np.nan
    return p


def is_market_eligible(market: str, field_names: tuple[str, ...]) -> bool:
    """True when every required universe field for ``market`` is available."""
    required = universe_mask_contract(market).required_fields
    return all(any(f in name for name in field_names) for f in required)


__all__ = [
    "A_SHARE_MASK_FIELDS",
    "US_MASK_FIELDS",
    "UniverseMaskContract",
    "apply_universe_mask",
    "is_market_eligible",
    "universe_mask_contract",
]
