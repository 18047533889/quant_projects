# -*- coding: utf-8 -*-
"""Daily return decomposition operators.

Split the close-to-close return into overnight / intraday / VWAP segments.
All operators are elementwise safe-division daily-panel transforms.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator


def _metadata(name: str, description: str, params: list[str], *, domain: str, unit: str, price_params: tuple[str, ...]) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="return_decomposition",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "return_decomposition", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", "cost:1",
        ],
        # R11 #164: every price input must share the same PriceBasis — the typed
        # search/compile layer reads these declarations so a raw open and an
        # adjusted (continuous) close can never be combined into a return.
        input_units={p: "price" for p in price_params},
        compatible_units={p: ("price",) for p in price_params},
    )


def _assert_same_axes(*frames: pd.DataFrame) -> None:
    """R11 #163: pandas ``/`` auto-aligns on labels; differing open/close/VWAP
    axes would be silently unioned/reindexed, breaking shape preservation.
    Require exact index/columns first and fail loudly on any mismatch."""
    if not frames:
        return
    ref = frames[0]
    for other in frames[1:]:
        if not ref.index.equals(other.index) or not ref.columns.equals(other.columns):
            raise ValueError(
                "all price inputs of a return decomposition must share the exact "
                f"same index/columns; found {ref.shape} vs {other.shape} — a "
                "silent label reindex would corrupt the return"
            )


def _price_basis_of_column(name: str) -> str | None:
    """Resolve a column's canonical price basis from the field-concept registry.

    ``None`` means the column carries no resolvable basis (e.g. a bare symbol
    name) — permissive, matching the analyzer's untyped-column stance.
    """
    from fields.concepts import concept_alias_map, get_concept

    try:
        alias_map = concept_alias_map()
    except Exception:
        return None
    concept_id = alias_map.get(str(name).lower())
    if not concept_id:
        return None
    try:
        concept = get_concept(concept_id)
    except Exception:
        return None
    if concept is not None and concept.price_basis:
        return concept.price_basis
    return None


def _assert_shared_price_basis(*frames: pd.DataFrame) -> None:
    """R11 #164: reject a mixed price basis across the inputs of a return
    decomposition.  A raw open and an adjusted (continuous) close must NEVER be
    combined into a return — the ratio would mix split-unadjusted and
    split-adjusted levels.  The typed-IR layer enforces this at compile time;
    this is the runtime defense-in-depth on resolvable column names."""
    resolved: set[str] = set()
    for frame in frames:
        for col in frame.columns:
            basis = _price_basis_of_column(str(col))
            if basis:
                resolved.add(basis)
    if len(resolved) > 1:
        raise ValueError(
            "mixed price basis across inputs "
            f"({sorted(resolved)}) — a return decomposition requires all price "
            "inputs to share the same PriceBasis (raw vs continuous must never "
            "combine into a return)"
        )


def _safe_ratio(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
    den = denominator.replace(0, np.nan) if hasattr(denominator, "replace") else denominator
    out = numerator / den
    return out.replace([np.inf, -np.inf], np.nan)


@register_operator(
    name="overnight_return",
    category="return_decomposition",
    business_category="return_decomposition",
    canonical="overnight_return",
    source="return_decomp",
    status="experimental",
)
class OvernightReturn(SeriesOperator):
    """隔夜收益：open / pre_close - 1。"""

    metadata = _metadata(
        "overnight_return",
        "隔夜收益 open/pre_close - 1。",
        ["open", "pre_close"],
        domain="return",
        unit="return",
        price_params=("open", "pre_close"),
    )

    def _calculate_series(self, open_px: pd.DataFrame, pre_close: pd.DataFrame, **_: Any) -> pd.DataFrame:
        _assert_same_axes(open_px, pre_close)
        _assert_shared_price_basis(open_px, pre_close)
        return _safe_ratio(open_px, pre_close) - 1.0


@register_operator(
    name="open_close_return",
    category="return_decomposition",
    business_category="return_decomposition",
    canonical="open_close_return",
    source="return_decomp",
    status="experimental",
)
class OpenCloseReturn(SeriesOperator):
    """日内收益（开盘→收盘）：close / open - 1。"""

    metadata = _metadata(
        "open_close_return",
        "开盘到收盘收益 close/open - 1。",
        ["open", "close"],
        domain="return",
        unit="return",
        price_params=("open", "close"),
    )

    def _calculate_series(self, open_px: pd.DataFrame, close: pd.DataFrame, **_: Any) -> pd.DataFrame:
        _assert_same_axes(open_px, close)
        _assert_shared_price_basis(open_px, close)
        return _safe_ratio(close, open_px) - 1.0


@register_operator(
    name="open_to_vwap_return",
    category="return_decomposition",
    business_category="return_decomposition",
    canonical="open_to_vwap_return",
    source="return_decomp",
    status="experimental",
)
class OpenToVwapReturn(SeriesOperator):
    """开盘到 VWAP 收益：vwap / open - 1。"""

    metadata = _metadata(
        "open_to_vwap_return",
        "开盘到 VWAP 收益 vwap/open - 1。",
        ["open", "vwap"],
        domain="return",
        unit="return",
        price_params=("open", "vwap"),
    )

    def _calculate_series(self, open_px: pd.DataFrame, vwap: pd.DataFrame, **_: Any) -> pd.DataFrame:
        _assert_same_axes(open_px, vwap)
        _assert_shared_price_basis(open_px, vwap)
        return _safe_ratio(vwap, open_px) - 1.0


@register_operator(
    name="vwap_to_close_return",
    category="return_decomposition",
    business_category="return_decomposition",
    canonical="vwap_to_close_return",
    source="return_decomp",
    status="experimental",
)
class VwapToCloseReturn(SeriesOperator):
    """VWAP 到收盘收益：close / vwap - 1。"""

    metadata = _metadata(
        "vwap_to_close_return",
        "VWAP 到收盘收益 close/vwap - 1。",
        ["vwap", "close"],
        domain="return",
        unit="return",
        price_params=("vwap", "close"),
    )

    def _calculate_series(self, vwap: pd.DataFrame, close: pd.DataFrame, **_: Any) -> pd.DataFrame:
        _assert_same_axes(vwap, close)
        _assert_shared_price_basis(vwap, close)
        return _safe_ratio(close, vwap) - 1.0
