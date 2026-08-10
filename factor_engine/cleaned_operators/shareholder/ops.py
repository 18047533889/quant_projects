# -*- coding: utf-8 -*-
"""股东集中度与变化算子。"""
from __future__ import annotations

import numpy as np
import pandas as pd

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


@register_operator(
    name="holder_company_ownership_hhi",
    category="shareholder",
    business_category="shareholder",
    canonical="holder_company_ownership_hhi",
    source="shareholder.ops",
    status="experimental",
)
class HolderCompanyOwnershipHhi(SeriesOperator):
    """R24-011..013: 公司总股本口径股东集中度 HHI = Σ ownership_ratio_i²。

    权重分母是公司总股本（每个 s_i 本身已是公司所有权比例，不做 top-k
    再归一化）。与 ``holder_observed_topk_hhi``（分母=观测 top-k 合计）是
    两个不同的经济定义，R24-014 禁止二者 alias。
    """

    metadata = _meta(
        "holder_company_ownership_hhi",
        "公司总股本口径 HHI：Σ ownership_ratio_i²（s_i 已是公司所有权比例）。",
        ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"],
        unit="ratio",
    )

    def _calculate_series(self, *args, **kwargs):
        if len(args) < 2:
            raise ValueError("holder_company_ownership_hhi requires at least two ratio panels")
        base = args[0]
        stacked = np.stack([np.asarray(p.to_numpy(dtype=float)) for p in args], axis=0)
        finite = np.isfinite(stacked)
        squares = np.where(finite, stacked * stacked, np.nan)
        hhi = np.nansum(squares, axis=0)
        hhi = np.where(np.isfinite(squares).sum(axis=0) > 0, hhi, np.nan)
        out = pd.DataFrame(hhi, index=base.index, columns=base.columns, dtype=float)
        return out


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


# R24-011: the company-ownership HHI canonical lives on the extended surface.
import cleaned_operators.operator_surface as _surface  # noqa: E402

_surface.extend_extended_only({"holder_company_ownership_hhi"})
