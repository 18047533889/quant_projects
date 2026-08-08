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

    NaN/±Inf in the mask are treated as OUT-of-universe (fail-closed): an
    unknown universe state must never become an in-pool member
    (``bool(np.nan) is True`` — audit P0).  A DataFrame mask is aligned to the
    panel by label (index AND columns): same shape but a different column order
    can no longer pair A's data with B's mask.

    Accepts 2-D numpy arrays or pandas DataFrames (same axes).
    """
    keep_shape = UNIVERSE_SPEC.shape_preserving_cross_section if shape_preserving is None else shape_preserving
    del keep_shape  # shape is preserved by construction (never dropped rows)

    import pandas as pd

    if isinstance(panel, pd.DataFrame):
        if isinstance(mask, pd.DataFrame):
            # Label-based alignment: a mask with the same shape but a different
            # column order must NOT be applied positionally.  Reindex by label;
            # cells the mask does not cover become NaN -> out-of-universe.
            m = mask.reindex(index=panel.index, columns=panel.columns).to_numpy(
                dtype=float
            )
        else:
            m = np.asarray(mask, dtype=float)
        if m.shape != panel.shape:
            raise ValueError(
                f"panel/mask shape mismatch: {panel.shape} vs {m.shape}"
            )
        # Fail-closed truth: finite & nonzero = in-universe; NaN/Inf/0 = out.
        m_bool = np.isfinite(m) & (m != 0)
        out = panel.to_numpy(dtype=float).copy()
        out[~m_bool] = np.nan
        return pd.DataFrame(out, index=panel.index, columns=panel.columns, dtype=float)

    p = np.asarray(panel, dtype=float).copy()
    m = np.asarray(mask, dtype=float)
    if p.shape != m.shape:
        raise ValueError(f"panel/mask shape mismatch: {p.shape} vs {m.shape}")
    m_bool = np.isfinite(m) & (m != 0)
    p[~m_bool] = np.nan
    return p


def apply_universe_mask_to_panel(
    panel: Any,
    market: str,
    mask_columns: dict[str, Any] | None = None,
) -> tuple[Any, Any, float]:
    """Build and apply the universe mask for ``panel`` (P0-034).

    Returns ``(masked_panel, mask_df, coverage_ratio)`` where ``mask_df`` is the
    combined boolean mask (same axes as ``panel``) and ``coverage_ratio`` is the
    fraction of finite panel cells that stay in-universe after masking.

    ``mask_columns`` keys the per-market mask components by name — e.g.
    ``{"tradability_state": <panel>, "close": <panel>}`` for A-share or
    ``{"stock_list.type": <panel>, "universe_daily": <panel>, "close": <panel>}``
    for US (see ``universe_mask_contract``).  Each value is a 2-D array/DataFrame
    aligned to ``panel`` or a 1-D Series/array of per-date masks (broadcast
    across instruments).  A missing component contributes no constraint; a
    NaN/Inf/0 cell in any supplied component fails closed to OUT-of-universe
    (matching ``apply_universe_mask``'s NaN policy).

    This is the single integration point the CS materialization entry calls
    BEFORE ranking/regression so the universe mask is applied once and the
    coverage is recorded in the run lineage (P1-24).
    """
    import numpy as np
    import pandas as pd

    contract = universe_mask_contract(market)  # raises on unknown market
    shape = np.asarray(panel).shape
    if mask_columns is None or not mask_columns:
        mask = np.ones(shape, dtype=float)
    else:
        mask = np.ones(shape, dtype=float)
        for name, comp in mask_columns.items():
            if comp is None:
                continue
            if isinstance(comp, pd.DataFrame):
                comp_arr = comp.reindex(
                    index=getattr(panel, "index", None),
                    columns=getattr(panel, "columns", None),
                ).to_numpy(dtype=float)
            elif isinstance(comp, pd.Series):
                comp_arr = comp.reindex(getattr(panel, "index", None)).to_numpy(dtype=float)
                if comp_arr.shape[0] == shape[0] and shape[1] > 1:
                    comp_arr = np.tile(comp_arr.reshape(-1, 1), (1, shape[1]))
            else:
                comp_arr = np.asarray(comp, dtype=float)
                if comp_arr.ndim == 1 and comp_arr.shape[0] == shape[0] and shape[1] > 1:
                    comp_arr = np.tile(comp_arr.reshape(-1, 1), (1, shape[1]))
            if comp_arr.shape != shape:
                raise ValueError(
                    f"universe mask component {name!r} has shape {comp_arr.shape}, "
                    f"panel is {shape}"
                )
            # Fail-closed truth: finite & nonzero = in-universe; NaN/Inf/0 = out.
            mask = mask * (np.isfinite(comp_arr) & (comp_arr != 0)).astype(float)

    masked = apply_universe_mask(panel, mask)
    panel_finite = np.isfinite(np.asarray(panel, dtype=float))
    in_universe = np.isfinite(np.asarray(masked, dtype=float))
    total = int(panel_finite.sum())
    coverage_ratio = float(in_universe.sum() / total) if total else 0.0
    return masked, mask, coverage_ratio


def is_market_eligible(market: str, field_names: tuple[str, ...]) -> bool:
    """True when every required universe field for ``market`` is available."""
    required = universe_mask_contract(market).required_fields
    return all(any(f in name for name in field_names) for f in required)


__all__ = [
    "A_SHARE_MASK_FIELDS",
    "US_MASK_FIELDS",
    "UniverseMaskContract",
    "apply_universe_mask",
    "apply_universe_mask_to_panel",
    "is_market_eligible",
    "universe_mask_contract",
]
