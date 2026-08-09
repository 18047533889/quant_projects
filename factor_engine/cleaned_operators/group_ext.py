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

# R11 #137: cross-section minimum breadth.  An industry/group regression on
# ~3 names is not a meaningful fit — require at least ``_MIN_BREADTH`` valid
# observations AND at least ``_BREADTH_PARAM_RATIO`` observations per estimated
# parameter (slope + optional intercept).  Below the gate the operator emits
# NaN (fail closed) instead of letting a handful of stocks drive the fit.
_MIN_BREADTH = 10
_BREADTH_PARAM_RATIO = 5.0


def _metadata(name: str, description: str, params: list[str], *, domain: str, unit: str) -> OperatorMetadata:
    # R11 #135/#136: unit-algebra honesty.  Algebraic units (``same_as:`` /
    # ``unit(...)`` / ``dimensionless``) are propagated to ``output_unit`` so
    # the catalog / typed search see the real output dimension instead of an
    # opaque ``ratio`` tag.
    output_unit = unit if (unit.startswith("same_as:") or unit.startswith("unit(") or unit == "dimensionless") else None
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
        output_unit=output_unit,
    )


def _valid_membership_label(label: Any) -> bool:
    """True when ``label`` is a concrete group/subgroup membership label.

    Missing/unknown labels (NaN, +/-inf, ``None``, empty string) are NOT real
    memberships: a cell carrying one must keep its residual NaN and must never
    join a composite ``(group, subgroup)`` demeaning key (P0-9).  Mirrors
    ``cleaned_operators.common.polars_group._valid_membership_label``.
    """
    if label is None:
        return False
    if isinstance(label, str):
        return label != ""
    if isinstance(label, bool):
        return True
    try:
        return bool(np.isfinite(float(label)))
    except (TypeError, ValueError):
        return True


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
        # R11 #135: a leave-one-out mean of x carries x's unit — NOT ratio.
        unit="same_as:x",
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
        # R11 #135: a weighted mean of x carries x's unit — NOT ratio.
        unit="same_as:x",
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
    """分级中性化：按 ``(group, subgroup)`` 组合键去均值（真 nested 中性化）。

    语义 (R11 #134)：一次 demean 使用复合键 ``(group, subgroup)``，保证
    subgroup 均值只在「同一父组内」计算 —— 若 subgroup 编码跨父组重复，裸
    subgroup demeaning 会把不同父组的成员混进同一个均值。复合键 demean 后，
    每个 (group, subgroup) 格的有限残差均值为 0，故每个父组内的残差均值亦为 0
    —— 先前的「第二步 group demean」因此冗余（除非 subgroup 码不严格嵌套）。
    """

    metadata = _metadata(
        "hierarchical_group_neutralize",
        "按 (group, subgroup) 组合键去均值（真 nested 中性化；子组码跨父组重复时不再串组）。",
        ["x", "group", "subgroup"],
        domain="price_volume",
        # demeaning subtracts within-group means -> output carries x's unit.
        unit="same_as:x",
    )

    @staticmethod
    def _demean_composite_row(
        values: np.ndarray,
        group_labels: np.ndarray,
        subgroup_labels: np.ndarray,
    ) -> np.ndarray:
        """Demean by the composite ``(group, subgroup)`` key — truly nested.

        A bare subgroup label that repeats across parent groups must NOT be
        merged (that would mix members of different parents); the composite key
        keeps subgroup means within their own parent group.  If subgroup codes
        are globally unique the composite key collapses to the subgroup key and
        the parent group mean is already zeroed (a further group demean would be
        a no-op).

        P0-9: an UNKNOWN / missing group or subgroup label is not a membership.
        A cell whose group or subgroup label is invalid (NaN / ±inf / None /
        empty) keeps its residual NaN and never joins a key — an unknown-group
        cell alone in its own (invalid) key must NOT be demeaned to 0
        (``mean == self`` manufactures a perfectly-neutralized residual).
        """
        out = np.full(len(values), np.nan, dtype=float)
        n = len(values)
        keys: list[tuple[Any, Any] | None] = [None] * n
        for i in range(n):
            if not _valid_membership_label(group_labels[i]) \
                    or not _valid_membership_label(subgroup_labels[i]):
                continue  # unknown group/subgroup -> residual stays NaN
            keys[i] = (group_labels[i], subgroup_labels[i])
        # Deduplicate with Python tuple equality (a numpy object-array ``==``
        # would broadcast a 2-tuple against the whole column and raise).
        seen: list[tuple[Any, Any]] = []
        for key in keys:
            if key is not None and key not in seen:
                seen.append(key)
        for key in seen:
            idx = np.flatnonzero(np.fromiter((k == key for k in keys), dtype=bool, count=n))
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
            # R11 #134: single composite-key demean replaces the old sequential
            # subgroup-then-group approximation.  The composite key is the exact
            # nested exposure neutralizer; the second group demean is redundant.
            out[row] = self._demean_composite_row(xv[row].copy(), gv[row], sv[row])
        return _frame_like(x, out)


@register_operator(
    name="cs_trimmed_ols_resid",
    category="cross_sectional",
    business_category="cross_sectional_regression",
    canonical="cs_trimmed_ols_resid",
    source="group_ext",
    status="experimental",
)
class CsRobustResid(SeriesOperator):
    """横截面残差：先按 trim_ratio 截尾样本拟合 y=a+bx（trimmed-OLS），再输出当前残差。

    注意 (R5 P1-37(c))：本算子是「截尾后 OLS」（trimmed OLS），不是 Huber/LAD
    等真正的稳健回归。它对两端离群点做硬截断，但截尾后仍用最小二乘，对剩余
    样本内的强影响点不稳健。R11 诚实命名：canonical 名现为 cs_trimmed_ols_resid，
    cs_robust_resid 仅作 deprecated alias 保留（历史兼容）。如需真稳健回归请使用
    Huber 或 LAD 估计量。
    """

    metadata = OperatorMetadata(
        name="cs_trimmed_ols_resid",
        category="cross_sectional",
        description=(
            "横截面残差（trimmed-OLS：截尾两端后最小二乘，非 Huber/LAD 稳健回归）。"
            "R11 #136：canonical 名现为 cs_trimmed_ols_resid（诚实命名），"
            "cs_robust_resid 为 deprecated alias；"
            "输出残差单位为 unit(y)（§37-D unit-algebra）。"
        ),
        param_names=["y", "x", "trim_ratio", "add_intercept"],
        return_type="series",
        tags=[
            "cross_sectional", "daily", "pit_safe", "causal", "typed_v2",
            "signature:y,x,trim_ratio,add_intercept->series", "domain:price_volume",
            "unit:same_as:y", "cost:2",
        ],
        output_unit="same_as:y",
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, trim_ratio: float = 0.1, add_intercept: bool = True, **_: Any) -> pd.DataFrame:
        y, x = _aligned(y, x)
        trim = float(trim_ratio)
        if not (0.0 <= trim < 0.5):
            raise ValueError("cs_trimmed_ols_resid requires 0 <= trim_ratio < 0.5")
        yv = y.to_numpy(dtype=float)
        xv = x.to_numpy(dtype=float)
        rows, cols = yv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        # R11 #137: cross-section minimum breadth — 3 stocks must not drive an
        # industry/group regression.  The gate is on the RAW aligned breadth;
        # after trimming we still require the parameter-count DOF margin.
        param_count = 2 if add_intercept else 1
        min_breadth = max(_MIN_BREADTH, int(_BREADTH_PARAM_RATIO * param_count))
        for row in range(rows):
            valid = np.isfinite(xv[row]) & np.isfinite(yv[row])
            if valid.sum() < min_breadth:
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
            if xs.size < (param_count + 1) or np.std(xs) == 0:
                continue
            if add_intercept:
                coeffs = np.polyfit(xs, ys, 1)
                fitted = np.polyval(coeffs, xv[row])
            else:
                slope = float(np.sum(xs * ys) / np.sum(xs * xs))
                fitted = slope * xv[row]
            out[row] = yv[row] - fitted
        return _frame_like(y, out)


def _register_trimmed_ols_honest_rename() -> None:
    """R11: keep the historical ``cs_robust_resid`` resolvable as a deprecated
    alias of the honest canonical ``cs_trimmed_ols_resid``.

    The implementation is genuinely trimmed-OLS (trim extreme ``x``, then
    ordinary least squares) — not a robust regression (no Huber/LAD/Theil-Sen) —
    so the canonical name must be honest.  ``rename_canonical``/``register_alias``
    preserve the old DSL name.  The static surface partition is updated in sync
    (new canonical -> extended; retired spelling -> removed) so
    ``layer_governance``'s exact-classification check still holds at finalize.
    """
    from cleaned_operators.registry import OperatorRegistry

    if "cs_robust_resid" in OperatorRegistry._operators:
        # A backend registered the historical spelling directly; migrate it.
        OperatorRegistry.rename_canonical("cs_robust_resid", "cs_trimmed_ols_resid")
    elif "cs_trimmed_ols_resid" in OperatorRegistry._operators:
        OperatorRegistry.register_alias("cs_robust_resid", "cs_trimmed_ols_resid")
    try:
        from cleaned_operators.operator_surface import extend_extended_only, retract_extended_only

        extend_extended_only(["cs_trimmed_ols_resid"])
        retract_extended_only(["cs_robust_resid"])
    except ImportError:  # pragma: no cover - surface module always present in-tree
        pass


_register_trimmed_ols_honest_rename()
