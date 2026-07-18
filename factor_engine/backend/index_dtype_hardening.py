# -*- coding: utf-8 -*-
"""Exact output-index normalization across Pandas and Polars panel paths.

Pandas 3 may infer string instrument labels as ``StringDtype`` while a Polars
round-trip reconstructs the same labels as an ``object`` index.  MultiIndex
label equality alone does not preserve that dtype contract.  After value
alignment, reuse the execution template index object exactly so all backends
return the same index names, level dtypes, ordering and metadata.
"""
from __future__ import annotations

from typing import Any


_APPLIED = False


def _exact_template_panel_to_series(panel, ctx, *, template):
    from backend.pandas_compat import pd

    if not isinstance(template, pd.Series):
        raise TypeError("panel_to_series requires a pandas Series template")
    if not isinstance(template.index, pd.MultiIndex):
        raise TypeError("panel_to_series template must use a MultiIndex")

    stacked = panel.stack(future_stack=True)
    stacked.index.names = [ctx.timestamp_col, ctx.instrument_col]

    # ``MultiIndex.equals`` intentionally ignores some dtype distinctions, so
    # even the fast path must replace the index with the exact template object.
    if len(stacked) == len(template.index) and stacked.index.equals(template.index):
        stacked.index = template.index
        return stacked

    aligned = stacked.reindex(template.index)
    # Reindexing may still preserve the source MultiIndex levels under newer
    # pandas versions.  The labels have now been aligned, so assigning the
    # immutable template index is safe and makes the cross-backend contract
    # exact rather than merely label-equivalent.
    aligned.index = template.index
    return aligned


def apply_index_dtype_hardening() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from backend import cleaned_bridge

    cleaned_bridge.panel_to_series = _exact_template_panel_to_series
    _APPLIED = True
