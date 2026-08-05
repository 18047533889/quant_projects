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


@register_operator(name="holder_concentration_change", category="shareholder", business_category="shareholder", canonical="holder_concentration_change", source="shareholder.ops", status="deprecated")
class HolderConcentrationChange(SeriesOperator):
    metadata = _meta("holder_concentration_change", "Deprecated: use source-side relation_snapshot_change on distinct snapshots.", ["concentration", "lag"], unit="ratio_change")

    def _calculate_series(self, concentration, lag=1, **kwargs):
        raise ValueError(
            "holder_concentration_change cannot operate on daily panels; use "
            "storage.sources.relation.relation_snapshot_change"
        )


@register_operator(name="holder_count_change_rate", category="shareholder", business_category="shareholder", canonical="holder_count_change_rate", source="shareholder.ops", status="deprecated")
class HolderCountChangeRate(SeriesOperator):
    metadata = _meta("holder_count_change_rate", "Deprecated: TopTen rows are not total shareholder count; use snapshot entity-count metrics.", ["holder_count", "lag"], unit="return")

    def _calculate_series(self, holder_count, lag=1, **kwargs):
        raise ValueError(
            "holder_count_change_rate is undefined for TopTen rows; use distinct holder "
            "snapshot metrics from storage.sources.relation"
        )
