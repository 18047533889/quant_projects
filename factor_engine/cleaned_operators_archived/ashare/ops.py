# -*- coding: utf-8 -*-
"""A 股特色的可复用日频算子。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator


def _safe_div(num, den) -> pd.DataFrame:
    denominator = den.replace(0, np.nan) if hasattr(den, "replace") else den
    out = num / denominator
    return out.replace([np.inf, -np.inf], np.nan)


def _metadata(name: str, description: str, params: list[str], *, domain: str, unit: str) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="ashare",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "ashare", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", "cost:1",
        ],
    )


class _RatioOp(SeriesOperator):
    def _calculate_series(self, numerator, denominator, **kwargs):
        return _safe_div(numerator, denominator)


@register_operator(name="earnings_yield", category="ashare", business_category="valuation", canonical="earnings_yield", source="ashare.ops", status="experimental")
class EarningsYield(_RatioOp):
    metadata = _metadata("earnings_yield", "盈利收益率：1 / PE；非正 PE 视为无效。", ["pe"], domain="valuation", unit="ratio")

    def _calculate_series(self, pe, **kwargs):
        valid_pe = pe.where(pe > 0)
        return _safe_div(1.0, valid_pe)


@register_operator(name="book_to_price", category="ashare", business_category="valuation", canonical="book_to_price", source="ashare.ops", status="experimental")
class BookToPrice(_RatioOp):
    metadata = _metadata("book_to_price", "账面市值比：1 / PB；非正 PB 视为无效。", ["pb"], domain="valuation", unit="ratio")

    def _calculate_series(self, pb, **kwargs):
        return _safe_div(1.0, pb.where(pb > 0))


@register_operator(name="float_share_ratio", category="ashare", business_category="capital_structure", canonical="float_share_ratio", source="ashare.ops", status="experimental")
class FloatShareRatio(_RatioOp):
    metadata = _metadata("float_share_ratio", "流通股本占总股本比例。", ["float_shares", "total_shares"], domain="capital", unit="ratio")


@register_operator(name="free_float_share_ratio", category="ashare", business_category="capital_structure", canonical="free_float_share_ratio", source="ashare.ops", status="experimental")
class FreeFloatShareRatio(_RatioOp):
    metadata = _metadata("free_float_share_ratio", "自由流通股本占总股本比例。", ["free_float_shares", "total_shares"], domain="capital", unit="ratio")


@register_operator(name="true_turnover_rate", category="ashare", business_category="liquidity", canonical="true_turnover_rate", source="ashare.ops", status="experimental")
class TrueTurnoverRate(_RatioOp):
    metadata = _metadata("true_turnover_rate", "真实换手率：成交量 / 自由流通股本。", ["volume", "free_float_shares"], domain="liquidity", unit="ratio")


@register_operator(name="limit_up_close", category="ashare", business_category="trading_state", canonical="limit_up_close", source="ashare.ops", status="experimental")
class LimitUpClose(SeriesOperator):
    metadata = _metadata("limit_up_close", "收盘价在 tick 容差内达到实际涨停价。", ["close", "upper_limit", "tick_tolerance"], domain="trading_state", unit="boolean")

    def _calculate_series(self, close, upper_limit, tick_tolerance=0.005, **kwargs):
        tolerance = float(tick_tolerance)
        if tolerance < 0:
            raise ValueError("tick_tolerance must be non-negative")
        valid = close.notna() & upper_limit.notna()
        return (close >= upper_limit - tolerance).astype(float).where(valid)


@register_operator(name="limit_up_state", category="ashare", business_category="trading_state", canonical="limit_up_close", source="ashare.ops", status="deprecated")
class LimitUpState(LimitUpClose):
    metadata = _metadata("limit_up_state", "Deprecated alias of limit_up_close.", ["close", "upper_limit", "tick_tolerance"], domain="trading_state", unit="boolean")


@register_operator(name="limit_down_close", category="ashare", business_category="trading_state", canonical="limit_down_close", source="ashare.ops", status="experimental")
class LimitDownClose(SeriesOperator):
    metadata = _metadata("limit_down_close", "收盘价在 tick 容差内达到实际跌停价。", ["close", "lower_limit", "tick_tolerance"], domain="trading_state", unit="boolean")

    def _calculate_series(self, close, lower_limit, tick_tolerance=0.005, **kwargs):
        tolerance = float(tick_tolerance)
        if tolerance < 0:
            raise ValueError("tick_tolerance must be non-negative")
        valid = close.notna() & lower_limit.notna()
        return (close <= lower_limit + tolerance).astype(float).where(valid)


@register_operator(name="limit_down_state", category="ashare", business_category="trading_state", canonical="limit_down_close", source="ashare.ops", status="deprecated")
class LimitDownState(LimitDownClose):
    metadata = _metadata("limit_down_state", "Deprecated alias of limit_down_close.", ["close", "lower_limit", "tick_tolerance"], domain="trading_state", unit="boolean")


@register_operator(name="tradable_state", category="ashare", business_category="trading_state", canonical="tradable_state", source="ashare.ops", status="experimental")
class TradableState(SeriesOperator):
    metadata = _metadata("tradable_state", "可交易状态：上市、未停牌、非一字涨跌停。", ["listed", "suspended", "limit_up", "limit_down"], domain="trading_state", unit="boolean")

    def _calculate_series(self, listed, suspended, limit_up, limit_down, **kwargs):
        valid = listed.notna() & suspended.notna() & limit_up.notna() & limit_down.notna()
        out = (listed > 0) & (suspended <= 0) & (limit_up <= 0) & (limit_down <= 0)
        return out.astype(float).where(valid)


@register_operator(name="benchmark_excess_return", category="ashare", business_category="benchmark_relative", canonical="benchmark_excess_return", source="ashare.ops", status="experimental")
class BenchmarkExcessReturn(SeriesOperator):
    metadata = _metadata("benchmark_excess_return", "个股收益减基准收益。", ["ret", "benchmark_ret"], domain="benchmark", unit="return")

    def _calculate_series(self, ret, benchmark_ret, **kwargs):
        return ret - benchmark_ret


@register_operator(name="benchmark_relative_price", category="ashare", business_category="benchmark_relative", canonical="benchmark_relative_price", source="ashare.ops", status="experimental")
class BenchmarkRelativePrice(_RatioOp):
    metadata = _metadata("benchmark_relative_price", "个股价格相对基准点位。", ["price", "benchmark_price"], domain="benchmark", unit="ratio")
