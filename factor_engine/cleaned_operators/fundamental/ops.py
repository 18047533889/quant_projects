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


@register_operator(name="ttm", category="fundamental", business_category="fundamental", canonical="ttm", source="factor_dsl_np", status="experimental")
class LqtpTtmOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="ttm",
        category="fundamental",
        description="滚动十二个月（TTM）累加；可选 fiscal_quarter(1-4) 对齐报告期",
        param_names=["x", "fiscal_quarter"],
        return_type="series",
        tags=["fundamental", "pit_safe"],
    )

    def _calculate_series(self, x: pd.DataFrame, fiscal_quarter: pd.DataFrame | None = None, **kwargs):
        from cleaned_operators.fundamental.period_helpers import compute_ttm

        return compute_ttm(x, fiscal_quarter)


@register_operator(name="quarter", category="fundamental", business_category="fundamental", canonical="quarter", source="factor_dsl_np", status="experimental")
class LqtpQuarterOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="quarter",
        category="fundamental",
        description="累计值转单季度；Q1/跨年用当期值，其余期用差分；可选 fiscal_quarter",
        param_names=["x", "fiscal_quarter"],
        return_type="series",
        tags=["fundamental", "pit_safe"],
    )

    def _calculate_series(self, x: pd.DataFrame, fiscal_quarter: pd.DataFrame | None = None, **kwargs):
        from cleaned_operators.fundamental.period_helpers import compute_quarter

        return compute_quarter(x, fiscal_quarter)


@register_operator(name="yoy", category="fundamental", business_category="fundamental", canonical="yoy", source="factor_dsl_np", status="experimental")
class LqtpYoyOp(SeriesOperator):
    metadata = OperatorMetadata(
        name="yoy",
        category="fundamental",
        description="同比增速；默认 lag=4，可选 fiscal_quarter 找同季度去年同期",
        param_names=["x", "fiscal_quarter"],
        return_type="series",
        tags=["fundamental", "pit_safe"],
    )

    def _calculate_series(self, x: pd.DataFrame, fiscal_quarter: pd.DataFrame | None = None, **kwargs):
        from cleaned_operators.fundamental.period_helpers import compute_yoy

        return compute_yoy(x, fiscal_quarter)


@register_operator(name="avg2", category="fundamental", business_category="fundamental", canonical="avg2", source="factor_dsl_np")
class LqtpAvg2Op(SeriesOperator):
    metadata = OperatorMetadata(
        name="avg2",
        category="fundamental",
        description="当期与上期均值；可选 fiscal_quarter 限制连续报告期",
        param_names=["x", "fiscal_quarter"],
        return_type="series",
        tags=["fundamental", "pit_safe"],
    )

    def _calculate_series(self, x: pd.DataFrame, fiscal_quarter: pd.DataFrame | None = None, **kwargs):
        from cleaned_operators.fundamental.period_helpers import compute_avg2

        return compute_avg2(x, fiscal_quarter)


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
    metadata = OperatorMetadata(
        name="operating_margin",
        category="fundamental",
        description="营业利润率：operating_income / revenue（分母为 0 时 NaN）",
        param_names=["operating_income", "revenue"],
        return_type="series",
        tags=["fundamental", "ratio", "pit_safe"],
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
    metadata = OperatorMetadata(
        name="current_ratio",
        category="fundamental",
        description="流动比率：current_assets / current_liabilities（分母为 0 时 NaN）",
        param_names=["current_assets", "current_liabilities"],
        return_type="series",
        tags=["fundamental", "ratio", "pit_safe"],
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
    metadata = OperatorMetadata(
        name="quick_ratio",
        category="fundamental",
        description="速动比率：(current_assets - inventory) / current_liabilities",
        param_names=["current_assets", "inventory", "current_liabilities"],
        return_type="series",
        tags=["fundamental", "ratio", "pit_safe"],
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
    metadata = OperatorMetadata(
        name="debt_to_equity",
        category="fundamental",
        description="负债权益比：total_debt / total_equity（分母为 0 时 NaN）",
        param_names=["total_debt", "total_equity"],
        return_type="series",
        tags=["fundamental", "ratio", "pit_safe"],
    )

    def _calculate_series(
        self, total_debt: pd.DataFrame, total_equity: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        den = total_equity.replace(0, np.nan) if hasattr(total_equity, "replace") else total_equity
        out = total_debt / den
        return out.replace([np.inf, -np.inf], np.nan)
