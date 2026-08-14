# -*- coding: utf-8 -*-
"""Native Polars implementations for promoted research operators.

This module deliberately contains no pandas conversion or registry bridge.  Operators
here operate directly on Polars expressions/dataframes and preserve trailing-window
semantics: every value at t depends only on rows <= t.
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl

from cleaned_operators.base_polars import OperatorMetadata, SeriesOperator, register_operator


def _numeric_cols(frame: pl.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in {"date", "timestamp", "stock_code", "instrument"}]


def _positive(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@register_operator(name="price_spread_deviation", category="time_series", business_category="time_series", canonical="price_spread_deviation", source="factor_dsl_polars", research_only=True)
class PriceSpreadDeviationPolars(SeriesOperator):
    metadata = OperatorMetadata(name="price_spread_deviation", category="time_series", description="Trailing price deviation", param_names=["x", "d"], tags=["polars", "pit_safe"])

    def _calculate_series(self, x: pl.DataFrame, d: int = 20, **kwargs) -> pl.DataFrame:
        window = _positive(kwargs.get("window", d), "d")
        cols = _numeric_cols(x)
        return x.with_columns([
            (pl.col(c) / pl.col(c).rolling_mean(window_size=window, min_samples=1)).sub(1.0).alias(c)
            for c in cols
        ])


@register_operator(name="ts_decay_exp_window", category="time_series", business_category="time_series", canonical="ts_decay_exp_window", source="factor_dsl_polars", research_only=True)
class TSDecayExpWindowNativePolars(SeriesOperator):
    metadata = OperatorMetadata(name="ts_decay_exp_window", category="time_series", description="Trailing exponential weighted mean", param_names=["x", "window", "alpha"], tags=["polars", "pit_safe"])

    def _calculate_series(self, x: pl.DataFrame, window: int = 10, alpha: float = 0.5, **kwargs) -> pl.DataFrame:
        window = _positive(window, "window")
        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be finite and between 0 and 1")
        # Polars rolling_map executes on trailing windows only; no pandas bridge.
        weights = [alpha ** i for i in range(window)][::-1]
        return x.with_columns([
            pl.col(c).rolling_map(
                lambda s: float((s * pl.Series(weights[-len(s):])).sum() / sum(weights[-len(s):])) if len(s) else None,
                window_size=window,
                min_samples=1,
            ).alias(c)
            for c in _numeric_cols(x)
        ])


@register_operator(name="ts_sum_decay", category="time_series", business_category="time_series", canonical="ts_sum_decay", source="factor_dsl_polars", research_only=True)
class TSSumDecayNativePolars(SeriesOperator):
    metadata = OperatorMetadata(name="ts_sum_decay", category="time_series", description="Trailing exponential weighted sum", param_names=["x", "window"], tags=["polars", "pit_safe"])

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, **kwargs) -> pl.DataFrame:
        window = _positive(window, "window")
        weights = [2.0 ** (i / window) for i in range(window)]

        def decay_sum(s):
            # 与 pandas 参考一致：加权平均（除以权重和），非原始加权和。
            if len(s) == 0:
                return None
            w = weights[:len(s)]
            return float((np.asarray(s, dtype=float) * w).sum() / sum(w))

        return x.with_columns([
            pl.col(c).rolling_map(
                decay_sum,
                window_size=window,
                min_samples=1,
            ).alias(c)
            for c in _numeric_cols(x)
        ])


@register_operator(
    name="ts_moment", category="time_series", business_category="time_series",
    canonical="ts_moment", source="factor_dsl_polars",
    replace=True, replacement_reason="3-arg central moment overrides the 2-arg colwise bridge (round-7 P0 chain pinning)",
    expected_old_source="factor_dsl_polars_bridge",
    research_only=True,
)
class TSMomentNativePolars(SeriesOperator):
    metadata = OperatorMetadata(name="ts_moment", category="time_series", description="Trailing central moment", param_names=["x", "d", "k"], tags=["polars", "pit_safe"])

    def _calculate_series(self, x: pl.DataFrame, d: int = 20, k: int = 3, **kwargs) -> pl.DataFrame:
        d = _positive(d, "d")
        k = _positive(k, "k")
        # pandas ts_moment (ts_moment_ kernel) requires the full window; partial
        # windows are NaN.  min_samples=d reproduces that reference semantics
        # (the previous min_samples=1 emitted partial-window moments).
        return x.with_columns([
            pl.col(c).rolling_map(
                lambda s: float(((s - s.mean()) ** k).mean()) if len(s) else None,
                window_size=d,
                min_samples=d,
            ).alias(c)
            for c in _numeric_cols(x)
        ])


@register_operator(name="saturate", category="signal", business_category="technical_signal", canonical="saturate", source="factor_dsl_polars", research_only=True)
class SaturateNativePolars(SeriesOperator):
    metadata = OperatorMetadata(name="saturate", category="signal", description="Elementwise clip to [0,1]", param_names=["x"], tags=["polars", "pit_safe"])

    def _calculate_series(self, x: pl.DataFrame, **kwargs) -> pl.DataFrame:
        return x.with_columns([pl.col(c).clip(0.0, 1.0).alias(c) for c in _numeric_cols(x)])


@register_operator(name="signed_power", category="signal", business_category="technical_signal", canonical="signed_power", source="factor_dsl_polars", research_only=True)
class SignedPowerNativePolars(SeriesOperator):
    metadata = OperatorMetadata(name="signed_power", category="signal", description="Sign-preserving power", param_names=["x", "c"], tags=["polars", "pit_safe"])

    def _calculate_series(self, x: pl.DataFrame, c: float = 2.0, **kwargs) -> pl.DataFrame:
        c = float(c)
        if not math.isfinite(c):
            raise ValueError("c must be finite")
        return x.with_columns([(pl.col(col).sign() * pl.col(col).abs().pow(c)).alias(col) for col in _numeric_cols(x)])
