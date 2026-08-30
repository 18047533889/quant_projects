# -*- coding: utf-8 -*-
"""Shared helpers for ``DataAccessSource`` construction (no heavy imports)."""

from __future__ import annotations

from typing import Any

_KNOWN_INDUSTRY_DATASETS = frozenset({
    "ashare_stock_industry",
    "us_stock_industry",
})

#: Datasets that DO carry the ``IndustrySource`` dimension column.  A
#: ``semantic_filters`` entry for ``IndustrySource`` only makes sense on these
#: tables (the logical-source ``_child`` path applies it when reading
#: StockIndustry / IndustryDaily).
_INDUSTRY_DATASETS = frozenset({
    "ashare_stock_industry",
    "us_stock_industry",
})


def _prune_semantic_filters_for_dataset(
    semantic_filters: dict[str, Any],
    dataset: str,
) -> dict[str, Any]:
    """Drop ``semantic_filters`` entries whose filter column is not in ``dataset``.

    2026-08-29: the daily/adj panel has no ``IndustrySource`` column.  Pushing
    ``filters={"IndustrySource": "sw_l1"}`` into every anchor read makes DuckDB
    fail with ``Referenced column "IndustrySource" not found``.  The industry
    filter only applies when reading the industry classification table (whose
    child source carries its own required-filter contract).  Other filter keys
    (e.g. ``timeframe`` for US financials) are kept untouched.
    """
    if not semantic_filters:
        return dict(semantic_filters)
    out = dict(semantic_filters)
    if "IndustrySource" in out and dataset not in _INDUSTRY_DATASETS:
        out.pop("IndustrySource", None)
    return out


__all__ = ["_prune_semantic_filters_for_dataset", "_INDUSTRY_DATASETS"]
