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


def _metadata(name: str, description: str, params: list[str], *, domain: str, unit: str) -> OperatorMetadata:
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
    )

    def _calculate_series(self, open_px: pd.DataFrame, pre_close: pd.DataFrame, **_: Any) -> pd.DataFrame:
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
    )

    def _calculate_series(self, open_px: pd.DataFrame, close: pd.DataFrame, **_: Any) -> pd.DataFrame:
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
    )

    def _calculate_series(self, open_px: pd.DataFrame, vwap: pd.DataFrame, **_: Any) -> pd.DataFrame:
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
    )

    def _calculate_series(self, vwap: pd.DataFrame, close: pd.DataFrame, **_: Any) -> pd.DataFrame:
        return _safe_ratio(close, vwap) - 1.0
