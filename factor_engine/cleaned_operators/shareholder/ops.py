# -*- coding: utf-8 -*-
"""股东集中度与变化算子。"""
from __future__ import annotations

import numpy as np

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator


def _meta(name: str, description: str, params: list[str], *, unit: str = "ratio") -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="shareholder",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "shareholder", "ashare", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", "domain:shareholder",
            f"unit:{unit}", "cost:1",
        ],
    )


def _safe_div(num, den):
    out = num / den.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


@register_operator(name="holder_concentration", category="shareholder", business_category="shareholder", canonical="holder_concentration", source="shareholder.ops", status="experimental")
class HolderConcentration(SeriesOperator):
    metadata = _meta("holder_concentration", "前 N 大股东持股数占总股本比例。", ["top_holder_shares", "total_shares"])

    def _calculate_series(self, top_holder_shares, total_shares, **kwargs):
        return _safe_div(top_holder_shares, total_shares)


@register_operator(name="holder_concentration_change", category="shareholder", business_category="shareholder", canonical="holder_concentration_change", source="shareholder.ops", status="experimental")
class HolderConcentrationChange(SeriesOperator):
    metadata = _meta("holder_concentration_change", "股东集中度较上一已披露时点的变化。", ["concentration", "lag"], unit="ratio_change")

    def _calculate_series(self, concentration, lag=1, **kwargs):
        return concentration - concentration.shift(int(lag))


@register_operator(name="holder_count_change_rate", category="shareholder", business_category="shareholder", canonical="holder_count_change_rate", source="shareholder.ops", status="experimental")
class HolderCountChangeRate(SeriesOperator):
    metadata = _meta("holder_count_change_rate", "股东户数较上一已披露时点的变化率。", ["holder_count", "lag"], unit="return")

    def _calculate_series(self, holder_count, lag=1, **kwargs):
        previous = holder_count.shift(int(lag))
        return _safe_div(holder_count - previous, previous)
