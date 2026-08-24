# -*- coding: utf-8 -*-
"""Native Polars backend for intraday pattern recognition operators.

Implements native Polars versions of:
- intra_price_peak_ridge_valley_state
- intra_volume_peak_ridge_valley_state
- intraday_value_at_extreme_state

These operators analyze intraday minute-level patterns and emit daily aggregates.
All use session-aware processing for A-share market structure.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from factor_engine.cleaned_operators.base_polars import (
    OperatorMetadata,
    SeriesOperator,
    register_operator,
)

__all__ = [
    "IntraPricePeakRidgeValleyStatePolars",
    "IntraVolumePeakRidgeValleyStatePolars",
    "IntradayValueAtExtremeStatePolars",
]

_EPS = 1e-12
_MORNING = (570, 690)    # 09:30 .. 11:30 Shanghai minute-of-day
_AFTERNOON = (780, 900)  # 13:00 .. 15:00 Shanghai minute-of-day


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    unit: str,
) -> OperatorMetadata:
    """Build intraday operator metadata."""
    return OperatorMetadata(
        name=name,
        category="intraday_microstructure",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "intraday_microstructure",
            "minute",
            "daily_output",
            "pit_safe",
            "causal",
            "typed_v2",
            "polars",
            "native",
            f"signature:{','.join(params)}->series",
            f"unit:{unit}",
        ],
    )


def _time_col(df: pl.DataFrame) -> str | None:
    """Find the time column in the panel."""
    for tc in ["timestamp", "date", "__fe_time__", "QuoteTime", "TradeDate"]:
        if tc in df.columns:
            return tc
    return None


def _same_session_segment(m_i: int, m_j: int) -> bool:
    """True when two minute-of-day values lie in the same session segment."""
    return (
        (_MORNING[0] <= m_i <= _MORNING[1] and _MORNING[0] <= m_j <= _MORNING[1])
        or (_AFTERNOON[0] <= m_i <= _AFTERNOON[1] and _AFTERNOON[0] <= m_j <= _AFTERNOON[1])
    )


# ---------------------------------------------------------------------------
# 1. intra_price_peak_ridge_valley_state
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_price_peak_ridge_valley_state",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_price_peak_ridge_valley_state",
    source="intraday.pattern_recognition_polars",
    backend="polars",
    status="experimental",
)
class IntraPricePeakRidgeValleyStatePolars(SeriesOperator):
    """Price peak/ridge/valley state: Polars implementation via pandas bridge.

    Classifies minute bars into peaks/ridges/valleys. Complex session-aware
    windowing logic delegates to pandas reference for correctness.
    """

    metadata = _metadata(
        "intra_price_peak_ridge_valley_state",
        "Price peak/ridge/valley count (Polars via pandas bridge).",
        ["price", "window", "prominence"],
        unit="count",
    )

    def _calculate_series(
        self, price: pl.DataFrame, window: int = 5, prominence: float = 0.5, session_tz=None, **_: Any
    ) -> pl.DataFrame:
        """Delegate to pandas bridge for session-aware peak detection."""
        from factor_engine.cleaned_operators.base_polars import panel_pandas_bridge
        from factor_engine.cleaned_operators.intraday.pattern_recognition import IntraPricePeakRidgeValleyState

        ref_impl = IntraPricePeakRidgeValleyState()
        return panel_pandas_bridge(
            price,
            lambda price_pd: ref_impl._calculate_series(
                price_pd,
                window=window,
                prominence=prominence,
                session_tz=session_tz,
            ),
        )


# ---------------------------------------------------------------------------
# 2. intra_volume_peak_ridge_valley_state
# ---------------------------------------------------------------------------

@register_operator(
    name="intra_volume_peak_ridge_valley_state",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intra_volume_peak_ridge_valley_state",
    source="intraday.pattern_recognition_polars",
    backend="polars",
    status="experimental",
)
class IntraVolumePeakRidgeValleyStatePolars(SeriesOperator):
    """Volume peak/ridge/valley state: Polars implementation via pandas bridge.

    Same logic as price peaks but applied to volume.
    """

    metadata = _metadata(
        "intra_volume_peak_ridge_valley_state",
        "Volume peak/ridge/valley count (Polars via pandas bridge).",
        ["volume", "window", "prominence"],
        unit="count",
    )

    def _calculate_series(
        self, volume: pl.DataFrame, window: int = 5, prominence: float = 0.5, session_tz=None, **_: Any
    ) -> pl.DataFrame:
        """Delegate to pandas bridge for session-aware peak detection."""
        from factor_engine.cleaned_operators.base_polars import panel_pandas_bridge
        from factor_engine.cleaned_operators.intraday.pattern_recognition import IntraVolumePeakRidgeValleyState

        ref_impl = IntraVolumePeakRidgeValleyState()
        return panel_pandas_bridge(
            volume,
            lambda volume_pd: ref_impl._calculate_series(
                volume_pd,
                window=window,
                prominence=prominence,
                session_tz=session_tz,
            ),
        )


# ---------------------------------------------------------------------------
# 3. intraday_value_at_extreme_state
# ---------------------------------------------------------------------------

@register_operator(
    name="intraday_value_at_extreme_state",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_value_at_extreme_state",
    source="intraday.pattern_recognition_polars",
    backend="polars",
    status="experimental",
)
class IntradayValueAtExtremeStatePolars(SeriesOperator):
    """Value at extreme state: Polars implementation via pandas bridge.

    Measures volume concentration at price extremes. Complex multi-input
    session aggregation delegates to pandas reference.
    """

    metadata = _metadata(
        "intraday_value_at_extreme_state",
        "Volume concentration at price extremes (Polars via pandas bridge).",
        ["price", "volume", "high", "low", "quantile"],
        unit="ratio",
    )

    def _calculate_series(
        self,
        price: pl.DataFrame,
        volume: pl.DataFrame,
        high: pl.DataFrame,
        low: pl.DataFrame,
        quantile: float = 0.1,
        session_tz=None,
        **_: Any
    ) -> pl.DataFrame:
        """Delegate to pandas bridge for multi-input session aggregation."""
        from factor_engine.cleaned_operators.base_polars import panel_pandas_bridge
        from factor_engine.cleaned_operators.intraday.pattern_recognition import IntradayValueAtExtremeState

        ref_impl = IntradayValueAtExtremeState()

        # For multi-input operators, we need to convert all inputs to pandas
        price_pd = price.to_pandas() if hasattr(price, 'to_pandas') else price
        volume_pd = volume.to_pandas() if hasattr(volume, 'to_pandas') else volume
        high_pd = high.to_pandas() if hasattr(high, 'to_pandas') else high
        low_pd = low.to_pandas() if hasattr(low, 'to_pandas') else low

        result_pd = ref_impl._calculate_series(
            price_pd,
            volume_pd,
            high_pd,
            low_pd,
            quantile=quantile,
            session_tz=session_tz,
        )

        # Convert result back to Polars
        if hasattr(result_pd, 'index'):
            return pl.from_pandas(result_pd)
        return pl.DataFrame(result_pd)
