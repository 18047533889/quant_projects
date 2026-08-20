# -*- coding: utf-8
"""空 universe 与过滤后截面 shape 契约。

Round-6 P0-27/28/30: the universe mask is the MANDATORY entry for every
cross-sectional runtime.  The DataAccess layer serves universe-filtered panels
(historical ``StockList`` / ``UniverseDaily`` membership — never a
current-universe backfill, P0-29); the FactorEngine side guarantees that all
panels reaching a CS operator share the SAME universe through the strict-axis
alignment gate (``cleaned_operators.base._validate_panel_axes`` +
``cleaned_operators.alignment.align_panel_inputs``): a delisted / suspended
name present in one input panel must be present in every other, or the call
raises — it can never silently drop out of one panel and bias the cross-section
(P0-30).  ``apply_universe_mask`` in ``cleaned_operators.alignment`` is the
public fail-closed mask application (NaN/Inf/False -> out; axis mismatch ->
error).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EmptyUniversePolicy = Literal["preserve_keys_null", "zero_rows"]
FilteredAllInvalidPolicy = Literal["preserve_keys_null"]


@dataclass(frozen=True)
class UniverseSpec:
    empty_universe: EmptyUniversePolicy = "preserve_keys_null"
    filtered_all_invalid: FilteredAllInvalidPolicy = "preserve_keys_null"
    shape_preserving_cross_section: bool = True
    # Round-6 P0-27: every CS operator consumes one common universe.  ``True``
    # here is the framework's promise that cross-sectional operators are
    # shape-preserving AND that the strict-axis gate rejects any input panel
    # whose instrument set diverges from the others (the "all CS operators use
    # exactly the same universe" guarantee, P0-28).
    strict_axes_mandatory: bool = True
    # Round-6 P0-30: a name that is not in today's universe (delisted,
    # suspended, bankrupt) must still be PRESENT in historical panels so the
    # historical cross-section / training set keeps it.  The runtime never drops
    # rows to match a current universe; it masks them out per-date.
    preserve_delisted_in_history: bool = True


UNIVERSE_SPEC = UniverseSpec()


def empty_universe_preserves_keys() -> bool:
    return UNIVERSE_SPEC.empty_universe == "preserve_keys_null"


def cross_section_shape_preserving() -> bool:
    return UNIVERSE_SPEC.shape_preserving_cross_section


def strict_axes_mandatory() -> bool:
    """CS runtime must reject axis-divergent input panels, never reindex them."""
    return UNIVERSE_SPEC.strict_axes_mandatory
