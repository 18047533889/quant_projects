# -*- coding: utf-8 -*-
"""Group ex-self and hierarchical neutralization operators.

``group`` panels carry a per-cell group label (string or numeric code) and
must share the exact shape of ``x``.  All operators are causal daily-panel
transforms evaluated independently per trading row.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.common.daily_panel import _aligned


def _metadata(name: str, description: str, params: list[str], *, domain: str, unit: str) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="group_neutralization",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "group_neutralization", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", "cost:1",
        ],
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _group_ex_self_mean_row(x_row: np.ndarray, g_row: np.ndarray, finite: np.ndarray) -> np.ndarray:
    out = np.full(x_row.shape, np.nan, dtype=float)
    labels = pd.unique(g_row)
    for label in labels:
        idx = (g_row == label) & finite
        count = int(np.sum(idx))
        if count <= 1:
            continue
        total = float(np.sum(x_row[idx]))
        for j in np.flatnonzero(g_row == label):
            if not finite[j]:
                continue
            out[j] = (total - x_row[j]) / (count - 1.0)
    return out


@register_operator(
    name="group_ex_self_mean",
    category="group_neutralization",
    business_category="group_neutralization",
    canonical="group_ex_self_mean",
    source="group_ext",
    status="experimental",
)
class GroupExSelfMean(SeriesOperator):
    """组内除自身外其余成员的均值（leave-one-out peer mean）。"""

    metadata = _metadata(
        "group_ex_self_mean",
        "组内除自身外其余成员的均值。",
        ["x", "group"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, group: pd.DataFrame, **_: Any) -> pd.DataFrame:
        x, group = _aligned(x, group)
        xv = x.to_numpy(dtype=float)
        gv = group.to_numpy()
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        finite = np.isfinite(xv)
        for row in range(rows):
            out[row] = _group_ex_self_mean_row(xv[row], gv[row], finite[row])
        return _frame_like(x, out)


@register_operator(
    name="group_ex_self_weighted_mean",
    category="group_neutralization",
    business_category="group_neutralization",
    canonical="group_ex_self_weighted_mean",
    source="group_ext",
    status="experimental",
)
class GroupExSelfWeightedMean(SeriesOperator):
    """组内除自身外其余成员的权重加权均值。"""

    metadata = _metadata(
        "group_ex_self_weighted_mean",
        "组内除自身外其余成员的权重加权均值。",
        ["x", "weight", "group"],
        domain="price_volume",
        unit="ratio",
    )

    def _calculate_series(self, x: pd.DataFrame, weight: pd.DataFrame, group: pd.DataFrame, **_: Any) -> pd.DataFrame:
        x, weight, group = _aligned(x, weight, group)
        xv = x.to_numpy(dtype=float)
        wv = weight.to_numpy(dtype=float)
        gv = group.to_numpy()
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for row in range(rows):
            g_row = gv[row]
            w_row = wv[row]
            x_row = xv[row]
            labels = pd.unique(g_row)
            for label in labels:
                idx = g_row == label
                # R5 P1-37(a): numerator and denominator MUST use the *same*
                # mask `finite(x) & finite(w) & w >= 0`.  The previous code
                # included peers whose x was missing in the denominator, making
                # `sum(w_j x_j) / sum(w_j)` inconsistent (denominator counted
                # missing-x peers, numerator silently dropped them to NaN).
                mask = idx & np.isfinite(w_row) & np.isfinite(x_row) & (w_row >= 0)
                total_w = float(np.sum(w_row[mask]))
                if not np.isfinite(total_w) or total_w <= 0.0:
                    continue
                weighted = float(np.sum(w_row[mask] * x_row[mask]))
                for j in np.flatnonzero(idx):
                    if not mask[j]:
                        continue
                    denom = total_w - w_row[j]
                    if denom <= 0.0:
                        continue
                    out[row][j] = (weighted - w_row[j] * x_row[j]) / denom
        return _frame_like(x, out)


@register_operator(
    name="hierarchical_group_neutralize",
    category="group_neutralization",
    business_category="group_neutralization",
    canonical="hierarchical_group_neutralize",
    source="group_ext",
    status="experimental",
)
class HierarchicalGroupNeutralize(SeriesOperator):
    """分级中性化：先在 subgroup 内减均值，再在 group 内减均值。

    LIMITATION (R5 P1-37(b))：连续对子组/父组逐次 demean 只是近似，不是一次
    真正的 nested-exposure 中性化 —— 先减 subgroup 均值、再减 group 均值，无法
    严格消除「subgroup 效应」与「group 效应」的全部联合暴露（第二步会把第一步
    已部分抵消的 group 均值重新带回）。若需要严格 nested exposure 中性化，应改用
    (1) 以 ``group + subgroup`` 组合键（composite group key）做单次 demean，
    或 (2) 对 group/subgroup 哑变量做横截面回归后取残差（dummy regression）。
    本算子保留逐次 demean 语义，仅在此明确标注局限。
    """

    metadata = _metadata(
        "hierarchical_group_neutralize",
        "先去除 subgroup 均值，再去除 group 均值（近似 nested 中性化，见局限说明）。",
        ["x", "group", "subgroup"],
        domain="price_volume",
        unit="ratio",
    )

    @staticmethod
    def _demean_row(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
        out = values.copy()
        for label in pd.unique(labels):
            idx = np.flatnonzero(labels == label)
            group_values = values[idx]
            finite = group_values[np.isfinite(group_values)]
            if finite.size == 0:
                continue
            mean = float(np.mean(finite))
            out[idx] = values[idx] - mean
        return out

    def _calculate_series(self, x: pd.DataFrame, group: pd.DataFrame, subgroup: pd.DataFrame, **_: Any) -> pd.DataFrame:
        x, group, subgroup = _aligned(x, group, subgroup)
        xv = x.to_numpy(dtype=float)
        gv = group.to_numpy()
        sv = subgroup.to_numpy()
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for row in range(rows):
            # Sequential demeaning is an *approximation* of nested neutralization
            # (R5 P1-37(b)) — see the class LIMITATION note.  Strict nested
            # exposure requires a composite group key or a dummy regression.
            stage1 = self._demean_row(xv[row].copy(), sv[row])
            out[row] = self._demean_row(stage1, gv[row])
        return _frame_like(x, out)


@register_operator(
    name="cs_robust_resid",
    category="cross_sectional",
    business_category="cross_sectional_regression",
    canonical="cs_robust_resid",
    source="group_ext",
    status="experimental",
)
class CsRobustResid(SeriesOperator):
    """横截面残差：先按 trim_ratio 截尾样本拟合 y=a+bx（trimmed-OLS），再输出当前残差。

    注意 (R5 P1-37(c))：本算子是「截尾后 OLS」（trimmed OLS），不是 Huber/LAD
    等真正的稳健回归。它对两端离群点做硬截断，但截尾后仍用最小二乘，对剩余
    样本内的强影响点不稳健；保留此名仅因历史兼容。如需真稳健回归请使用 Huber
    或 LAD 估计量。
    """

    metadata = OperatorMetadata(
        name="cs_robust_resid",
        category="cross_sectional",
        description="横截面残差（trimmed-OLS：截尾两端后最小二乘，非 Huber/LAD 稳健回归）。",
        param_names=["y", "x", "trim_ratio", "add_intercept"],
        return_type="series",
        tags=[
            "cross_sectional", "daily", "pit_safe", "causal", "typed_v2",
            "signature:y,x,trim_ratio,add_intercept->series", "domain:price_volume",
            "unit:ratio", "cost:2",
        ],
        output_unit="same_as:target",
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, trim_ratio: float = 0.1, add_intercept: bool = True, **_: Any) -> pd.DataFrame:
        y, x = _aligned(y, x)
        trim = float(trim_ratio)
        if not (0.0 <= trim < 0.5):
            raise ValueError("cs_robust_resid requires 0 <= trim_ratio < 0.5")
        yv = y.to_numpy(dtype=float)
        xv = x.to_numpy(dtype=float)
        rows, cols = yv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for row in range(rows):
            valid = np.isfinite(xv[row]) & np.isfinite(yv[row])
            if valid.sum() < 3:
                continue
            xs = xv[row][valid]
            ys = yv[row][valid]
            cut = int(np.floor(trim * xs.size))
            if cut > 0:
                keep = np.ones(xs.size, dtype=bool)
                keep[np.argsort(xs)[:cut]] = False
                keep[np.argsort(xs)[-cut:]] = False
                xs = xs[keep]
                ys = ys[keep]
            if xs.size < 2 or np.std(xs) == 0:
                continue
            if add_intercept:
                coeffs = np.polyfit(xs, ys, 1)
                fitted = np.polyval(coeffs, xv[row])
            else:
                slope = float(np.sum(xs * ys) / np.sum(xs * xs))
                fitted = slope * xv[row]
            out[row] = yv[row] - fitted
        return _frame_like(y, out)
