# -*- coding: utf-8 -*-
"""
基本面与财报衍生算子。

语义
----
对财报时序字段做 TTM、同比、季度化、两期平均等变换，例如：
- ``ttm``：滚动十二个月；
- ``yoy``：同比增速；
- ``quarter`` / ``avg2``：季度化或两期平均。

依赖 canonical 基本面字段（见 ``docs/canonical_data_fields.md``）；与价量时序算子分文件维护。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from cleaned_operators.base import (
    Operator,
    OperatorMetadata,
    SeriesOperator,
    ScalarOperator,
    TwoVarOperator,
    register_operator,
)


@register_operator(name="ttm", category="fundamental", business_category="fundamental", canonical="ttm", source="factor_dsl_np", status="deprecated")
class LqtpTtmOp(SeriesOperator):
    """Deprecated 滚动十二个月（TTM）累加；可选 fiscal_quarter(1-4) 对齐报告期。

    旧实现按 trading-day row 滚动（daily as-of ffill 下 ttm=最近4个交易日相加），
    与严格 fiscal-ordinal 语义不一致。请改用 strict operators:
    ``ttm_from_quarterly`` / ``ttm_from_cumulative``（compatibility-only）。
    """
    metadata = OperatorMetadata(
        name="ttm",
        category="fundamental",
        description="Deprecated: 按交易日行滚动 TTM，ffill 下语义错误；改用 ttm_from_quarterly / ttm_from_cumulative。",
        param_names=["x", "fiscal_quarter"],
        return_type="series",
        tags=["fundamental", "pit_safe", "deprecated", "compatibility_only"],
    )

    def _calculate_series(self, x: pd.DataFrame, fiscal_quarter: pd.DataFrame | None = None, **kwargs):
        from cleaned_operators.fundamental.period_helpers import compute_ttm

        return compute_ttm(x, fiscal_quarter)


@register_operator(name="quarter", category="fundamental", business_category="fundamental", canonical="quarter", source="factor_dsl_np", status="deprecated")
class LqtpQuarterOp(SeriesOperator):
    """Deprecated 累计值转单季度；可选 fiscal_quarter。

    旧实现按日频行走 Q1/跨年边界，daily as-of ffill 下季度化语义错误。
    请改用 ``quarter_from_cumulative``（compatibility-only）。
    """
    metadata = OperatorMetadata(
        name="quarter",
        category="fundamental",
        description="Deprecated: 日频行季度化在 ffill 下语义错误；改用 quarter_from_cumulative。",
        param_names=["x", "fiscal_quarter"],
        return_type="series",
        tags=["fundamental", "pit_safe", "deprecated", "compatibility_only"],
    )

    def _calculate_series(self, x: pd.DataFrame, fiscal_quarter: pd.DataFrame | None = None, **kwargs):
        from cleaned_operators.fundamental.period_helpers import compute_quarter

        return compute_quarter(x, fiscal_quarter)


@register_operator(name="yoy", category="fundamental", business_category="fundamental", canonical="yoy", source="factor_dsl_np", status="deprecated")
class LqtpYoyOp(SeriesOperator):
    """Deprecated 同比增速；默认 lag=4 行。

    旧实现按 trading-day row 找去年同日（ffill 下 yoy fallback=i-4 交易日错误）。
    请改用 ``yoy_by_period``（compatibility-only）。
    """
    metadata = OperatorMetadata(
        name="yoy",
        category="fundamental",
        description="Deprecated: 交易日行同比在 ffill 下语义错误；改用 yoy_by_period。",
        param_names=["x", "fiscal_quarter"],
        return_type="series",
        tags=["fundamental", "pit_safe", "deprecated", "compatibility_only"],
    )

    def _calculate_series(self, x: pd.DataFrame, fiscal_quarter: pd.DataFrame | None = None, **kwargs):
        from cleaned_operators.fundamental.period_helpers import compute_yoy

        return compute_yoy(x, fiscal_quarter)


def _period_op_factory(helper_name: str, description: str, *, canonical: str):
    """period_helpers 显式算子的轻量 SeriesOperator 工厂。"""

    @register_operator(
        name=canonical,
        category="fundamental",
        business_category="fundamental",
        canonical=canonical,
        source="factor_dsl_np",
        status="experimental",
    )
    class _PeriodOp(SeriesOperator):
        metadata = OperatorMetadata(
            name=canonical,
            category="fundamental",
            description=description,
            param_names=["x", "fiscal_quarter"],
            return_type="series",
            tags=["fundamental", "pit_safe", "period_aware"],
        )

        def _calculate_series(
            self, x: pd.DataFrame, fiscal_quarter: pd.DataFrame | None = None, **kwargs
        ):
            from cleaned_operators.fundamental import period_helpers as ph

            fn = getattr(ph, helper_name)
            return fn(x, fiscal_quarter)

    _PeriodOp.__doc__ = description
    return _PeriodOp


QuarterFromCumulativeOp = _period_op_factory(
    "quarter_from_cumulative",
    "累计值转单季（显式 period-aware；等同 quarter）",
    canonical="quarter_from_cumulative",
)
TtmFromQuarterlyOp = _period_op_factory(
    "ttm_from_quarterly",
    "单季值滚动 4 期 TTM（输入须为单季口径）",
    canonical="ttm_from_quarterly",
)
TtmFromCumulativeOp = _period_op_factory(
    "ttm_from_cumulative",
    "累计值 → 单季 → 滚动 4 季 TTM",
    canonical="ttm_from_cumulative",
)
YoyByPeriodOp = _period_op_factory(
    "yoy_by_period",
    "按报告期计算同比增速（显式 period-aware；等同 yoy）",
    canonical="yoy_by_period",
)


@register_operator(
    name="operating_margin",
    category="fundamental",
    business_category="fundamental",
    canonical="operating_margin",
    source="factor_dsl_np",
    status="experimental",
)
class OperatingMarginOp(TwoVarOperator):
    """营业利润率：operating_income / revenue（分母为 0 时 NaN）"""
    metadata = OperatorMetadata(
        name="operating_margin",
        category="fundamental",
        description="营业利润率：operating_income / revenue（分母为 0 时 NaN）",
        param_names=["operating_income", "revenue"],
        return_type="series",
        tags=["fundamental", "ratio", "pit_safe"],
        input_units={"operating_income": "flow", "revenue": "flow"},
    )

    def _calculate_series(self, operating_income: pd.DataFrame, revenue: pd.DataFrame, **kwargs) -> pd.DataFrame:
        den = revenue.replace(0, np.nan) if hasattr(revenue, "replace") else revenue
        out = operating_income / den
        return out.replace([np.inf, -np.inf], np.nan)


@register_operator(
    name="current_ratio",
    category="fundamental",
    business_category="fundamental",
    canonical="current_ratio",
    source="factor_dsl_np",
    status="experimental",
)
class CurrentRatioOp(TwoVarOperator):
    """流动比率：current_assets / current_liabilities（分母为 0 时 NaN）"""
    metadata = OperatorMetadata(
        name="current_ratio",
        category="fundamental",
        description="流动比率：current_assets / current_liabilities（分母为 0 时 NaN）",
        param_names=["current_assets", "current_liabilities"],
        return_type="series",
        tags=["fundamental", "ratio", "pit_safe"],
        input_units={"current_assets": "stock", "current_liabilities": "stock"},
    )

    def _calculate_series(
        self, current_assets: pd.DataFrame, current_liabilities: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        den = (
            current_liabilities.replace(0, np.nan)
            if hasattr(current_liabilities, "replace")
            else current_liabilities
        )
        out = current_assets / den
        return out.replace([np.inf, -np.inf], np.nan)


@register_operator(
    name="quick_ratio",
    category="fundamental",
    business_category="fundamental",
    canonical="quick_ratio",
    source="factor_dsl_np",
    status="experimental",
)
class QuickRatioOp(SeriesOperator):
    """速动比率：(current_assets - inventory) / current_liabilities"""
    metadata = OperatorMetadata(
        name="quick_ratio",
        category="fundamental",
        description="速动比率：(current_assets - inventory) / current_liabilities",
        param_names=["current_assets", "inventory", "current_liabilities"],
        return_type="series",
        tags=["fundamental", "ratio", "pit_safe"],
        input_units={
            "current_assets": "stock",
            "inventory": "stock",
            "current_liabilities": "stock",
        },
    )

    def _calculate_series(
        self,
        current_assets: pd.DataFrame,
        inventory: pd.DataFrame,
        current_liabilities: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        num = current_assets - inventory
        den = (
            current_liabilities.replace(0, np.nan)
            if hasattr(current_liabilities, "replace")
            else current_liabilities
        )
        out = num / den
        return out.replace([np.inf, -np.inf], np.nan)


@register_operator(
    name="debt_to_equity",
    category="fundamental",
    business_category="fundamental",
    canonical="debt_to_equity",
    source="factor_dsl_np",
    status="experimental",
)
class DebtToEquityOp(TwoVarOperator):
    """负债权益比：total_debt / total_equity（分母为 0 时 NaN）"""
    metadata = OperatorMetadata(
        name="debt_to_equity",
        category="fundamental",
        description="负债权益比：total_debt / total_equity（分母为 0 时 NaN）",
        param_names=["total_debt", "total_equity"],
        return_type="series",
        tags=["fundamental", "ratio", "pit_safe"],
        input_units={"total_debt": "stock", "total_equity": "stock"},
    )

    def _calculate_series(
        self, total_debt: pd.DataFrame, total_equity: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        den = total_equity.replace(0, np.nan) if hasattr(total_equity, "replace") else total_equity
        out = total_debt / den
        return out.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# P1-129: legacy row-wise ttm/quarter/yoy marked deprecated / compatibility-only.
# The strict fiscal-ordinal replacements are registered by fiscal_strict.py
# (quarter_from_cumulative / ttm_from_quarterly / ttm_from_cumulative /
# yoy_by_period); the old row-walking names stay for back-compat but are
# advertised as superseded.
# ---------------------------------------------------------------------------
from cleaned_operators.registry import OperatorRegistry as _registry  # noqa: E402

for _old, _new in (
    ("ttm", "ttm_from_quarterly"),
    ("quarter", "quarter_from_cumulative"),
    ("yoy", "yoy_by_period"),
):
    _entry = _registry._catalog.get(_old)
    if _entry is not None:
        _entry["compatibility_only"] = True
        _entry["preferred_replacements"] = [_new]
        _entry.setdefault(
            "semantic_note",
            "旧 ttm/quarter/yoy 按交易日行滚动，在 daily as-of forward-fill 下语义错误"
            "（ttm=最近4个交易日相加、yoy fallback=i-4 交易日、quarter=日频行季度化）；"
            "改用 strict fiscal-ordinal 算子。",
        )
