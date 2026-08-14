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
    research_only=True,
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
    research_only=True,
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
    research_only=True,
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
    research_only=True,
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

        extend_extended_only(
            ["cs_trimmed_ols_resid", "cs_huber_resid", "cs_lad_resid"]
        )
        retract_extended_only(["cs_robust_resid"])
    except ImportError:  # pragma: no cover - surface module always present in-tree
        pass


_register_trimmed_ols_honest_rename()


# ---------------------------------------------------------------------------
# P1-24: genuinely robust cross-sectional residuals (Huber / LAD).
# ---------------------------------------------------------------------------
# ``cs_robust_resid`` is a deprecated alias for trimmed-OLS — honest but NOT a
# robust regression (hard-trim extreme x, then ordinary least squares).  These
# two new operators are true M-estimators: ``cs_huber_resid`` (Huber loss with
# delta ~ 1.345) and ``cs_lad_resid`` (L1 / median regression).  Both are
# per-column (each row is a cross-section over instruments), shape-preserving,
# and emit NaN for a statistical failure (degenerate x / non-convergence).


def _huber_irls_fit(xs: np.ndarray, ys: np.ndarray, add_intercept: bool, max_iter: int = 60, tol: float = 1e-8) -> tuple[float, float] | None:
    """Huber M-estimator fit of one cross-section via IRLS.

    Minimises ``sum rho(r)`` with Huber's ``rho`` (``delta = 1.345 * scale``,
    scale = MAD/0.6745 of the residuals).  Returns ``(intercept, slope)``
    (``intercept == 0`` when ``add_intercept`` is False) or ``None`` when the
    fit cannot be established (degenerate design / non-convergence).
    """
    n = xs.size
    if n < 2:
        return None
    design = np.column_stack([np.ones(n), xs]) if add_intercept else xs.reshape(-1, 1)
    try:
        beta = np.linalg.lstsq(design, ys, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(beta)):
        return None
    for _ in range(max_iter):
        fitted = design @ beta
        r = ys - fitted
        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med)) / 0.6745)
        std = float(np.std(r))
        scale = mad if (np.isfinite(mad) and mad > 0.0) else (std if np.isfinite(std) and std > 0.0 else 1.0)
        delta = 1.345 * scale
        w = np.where(np.abs(r) <= delta, 1.0, delta / np.maximum(np.abs(r), 1e-12))
        sqrt_w = np.sqrt(np.maximum(w, 1e-12))
        try:
            beta_new = np.linalg.lstsq(design * sqrt_w[:, None], ys * sqrt_w, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(beta_new)):
            return None
        if np.max(np.abs(beta_new - beta)) <= tol * max(1.0, float(np.max(np.abs(beta)))):
            beta = beta_new
            break
        beta = beta_new
    else:  # no convergence in max_iter -> statistical failure -> NaN
        return None
    if add_intercept:
        return float(beta[1]), float(beta[0])
    return 0.0, float(beta[0])


def _lad_coordinate_descent(xs: np.ndarray, ys: np.ndarray, add_intercept: bool, max_iter: int = 300, tol: float = 1e-9) -> tuple[float, float] | None:
    """L1 (median) fit via coordinate descent / iterated weighted median.

    Alternates ``intercept = median(y - slope*x)`` and
    ``slope = weighted median of (y - intercept)/x with weights |x|`` — the
    exact minimiser of ``sum |y - a - b x|`` for a fixed ``a``.  Returns
    ``(intercept, slope)`` or ``None`` on statistical failure.
    """
    n = xs.size
    if n < 2:
        return None
    if np.all(np.abs(xs) < 1e-12):
        return None
    try:
        b_ols = np.polyfit(xs, ys, 1)
        slope, intercept = float(b_ols[0]), float(b_ols[1])
    except Exception:
        return None
    if not np.all(np.isfinite([slope, intercept])):
        return None
    for _ in range(max_iter):
        intercept_new = np.median(ys - slope * xs) if add_intercept else 0.0
        ratios = (ys - intercept_new) / xs
        weights = np.abs(xs)
        order = np.argsort(ratios)
        cw = np.cumsum(weights[order])
        mid = 0.5 * cw[-1]
        slope_new = float(ratios[order[min(int(np.searchsorted(cw, mid)), n - 1)]])
        if (
            abs(slope_new - slope) <= tol * max(1.0, abs(slope))
            and abs(intercept_new - intercept) <= tol * max(1.0, abs(intercept))
        ):
            return float(intercept_new), float(slope_new)
        slope, intercept = slope_new, intercept_new
    return None  # non-convergence -> NaN


def _lad_fit(xs: np.ndarray, ys: np.ndarray, add_intercept: bool) -> tuple[float, float] | None:
    """L1 regression fit of one cross-section.

    Uses ``scipy.optimize.minimize`` (Nelder-Mead on ``sum |resid|``) with a
    median-based init when scipy is available, else the iterated weighted
    median.  Returns ``(intercept, slope)`` or ``None`` on statistical failure.
    """
    n = xs.size
    if n < 2:
        return None
    if not add_intercept:
        if np.all(np.abs(xs) < 1e-12):
            return None
        ratios = ys / xs
        weights = np.abs(xs)
        order = np.argsort(ratios)
        cw = np.cumsum(weights[order])
        mid = 0.5 * cw[-1]
        return 0.0, float(ratios[order[min(int(np.searchsorted(cw, mid)), n - 1)]])
    try:
        b_ols = np.polyfit(xs, ys, 1)
        beta0 = np.array([float(b_ols[1]), float(b_ols[0])])  # (intercept, slope)
    except Exception:
        return None
    if not np.all(np.isfinite(beta0)):
        return None
    try:
        from scipy.optimize import minimize

        def _obj(beta: np.ndarray) -> float:
            return float(np.sum(np.abs(ys - (beta[0] + beta[1] * xs))))

        res = minimize(_obj, beta0, method="Nelder-Mead", options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-10})
        if res.success and np.all(np.isfinite(res.x)):
            return float(res.x[0]), float(res.x[1])
        return _lad_coordinate_descent(xs, ys, add_intercept)
    except ImportError:  # scipy unavailable -> pure-numpy iterated median
        return _lad_coordinate_descent(xs, ys, add_intercept)


def _cs_robust_resid(y: pd.DataFrame, x: pd.DataFrame, fit_fn: Any, add_intercept: bool) -> pd.DataFrame:
    """Shared per-column cross-sectional residual driver (P1-24).

    ``fit_fn(xs, ys, add_intercept) -> (intercept, slope) | None``.  The same
    minimum-breadth gate as ``cs_trimmed_ols_resid`` keeps ~3-stock regressions
    from driving an industry/group fit; a statistical failure keeps the cell
    NaN (fail closed).
    """
    y, x = _aligned(y, x)
    yv = y.to_numpy(dtype=float)
    xv = x.to_numpy(dtype=float)
    rows, cols = yv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    param_count = 2 if add_intercept else 1
    min_breadth = max(_MIN_BREADTH, int(_BREADTH_PARAM_RATIO * param_count))
    for row in range(rows):
        valid = np.isfinite(xv[row]) & np.isfinite(yv[row])
        if valid.sum() < min_breadth:
            continue
        xs = xv[row][valid]
        ys = yv[row][valid]
        if np.std(xs) == 0:
            continue
        coefs = fit_fn(xs, ys, add_intercept)
        if coefs is None:
            continue  # statistical failure -> NaN for this row
        intercept, slope = coefs
        if add_intercept:
            fitted = intercept + slope * xv[row]
        else:
            fitted = slope * xv[row]
        out[row] = yv[row] - fitted
    return _frame_like(y, out)


@register_operator(
    name="cs_huber_resid",
    category="cross_sectional",
    business_category="cross_sectional_regression",
    canonical="cs_huber_resid",
    source="group_ext",
    status="experimental",
    research_only=True,
)
class CsHuberResid(SeriesOperator):
    """横截面残差：Huber M-估计量回归（Huber loss, delta≈1.345·MAD）对离群点稳健。

    与 trimmed-OLS 的硬截尾不同，Huber 用平方-线性混合损失给大残差降权，
    对强影响点不敏感（P1-24 真稳健回归）。逐行（横截面跨 instruments）拟合，
    形状保持；退化 x / 不收敛 -> NaN。
    """

    metadata = OperatorMetadata(
        name="cs_huber_resid",
        category="cross_sectional",
        description=(
            "横截面残差（Huber M-估计量，IRLS，delta≈1.345·MAD；对离群点稳健，"
            "非 trimmed-OLS）。输出残差单位为 unit(y)（§37-D unit-algebra）。"
        ),
        param_names=["y", "x", "add_intercept"],
        return_type="series",
        tags=[
            "cross_sectional", "daily", "pit_safe", "causal", "typed_v2",
            "signature:y,x,add_intercept->series", "domain:price_volume",
            "unit:same_as:y", "cost:3",
        ],
        output_unit="same_as:y",
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, add_intercept: bool = True, **_: Any) -> pd.DataFrame:
        return _cs_robust_resid(y, x, _huber_irls_fit, bool(add_intercept))


@register_operator(
    name="cs_lad_resid",
    category="cross_sectional",
    business_category="cross_sectional_regression",
    canonical="cs_lad_resid",
    source="group_ext",
    status="experimental",
    research_only=True,
)
class CsLadResid(SeriesOperator):
    """横截面残差：LAD（L1 / 中位数）回归残差，对离群点稳健。

    LAD 最小化绝对残差和，对 y 端离群点不敏感（P1-24 真稳健回归）。逐行
    （横截面跨 instruments）拟合，形状保持；退化 x / 不收敛 -> NaN。
    """

    metadata = OperatorMetadata(
        name="cs_lad_resid",
        category="cross_sectional",
        description=(
            "横截面残差（LAD / L1 中位数回归，对离群点稳健；非 trimmed-OLS）。"
            "输出残差单位为 unit(y)（§37-D unit-algebra）。"
        ),
        param_names=["y", "x", "add_intercept"],
        return_type="series",
        tags=[
            "cross_sectional", "daily", "pit_safe", "causal", "typed_v2",
            "signature:y,x,add_intercept->series", "domain:price_volume",
            "unit:same_as:y", "cost:3",
        ],
        output_unit="same_as:y",
    )

    def _calculate_series(self, y: pd.DataFrame, x: pd.DataFrame, add_intercept: bool = True, **_: Any) -> pd.DataFrame:
        return _cs_robust_resid(y, x, _lad_fit, bool(add_intercept))


def _group_multi_ols_fit(
    ys: np.ndarray,
    xs_list: list[np.ndarray],
    add_intercept: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Multi-feature OLS fit for one group cross-section.

    Fits ``y ~ x1 + x2 + ... + xN`` (with optional intercept) via normal
    equations. Returns ``(coeffs, fitted)`` or ``None`` on statistical failure
    (insufficient samples, singular matrix).

    Args:
        ys: dependent variable (valid samples only)
        xs_list: list of independent variable arrays (valid samples only)
        add_intercept: whether to include intercept term

    Returns:
        (coeffs, fitted) where fitted = X @ coeffs, or None on failure
    """
    n = len(ys)
    n_features = len(xs_list)

    if n < 2:
        return None

    # Build design matrix
    if add_intercept:
        X = np.column_stack([np.ones(n)] + xs_list)
    else:
        X = np.column_stack(xs_list) if n_features > 1 else xs_list[0].reshape(-1, 1)

    # Need at least as many observations as parameters (n >= n_params for exact/over-determined)
    if n < X.shape[1]:
        return None

    # Check for rank deficiency (collinear features)
    rank = np.linalg.matrix_rank(X)
    if rank < X.shape[1]:
        return None

    # Solve normal equations
    try:
        coeffs = np.linalg.lstsq(X, ys, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    if not np.all(np.isfinite(coeffs)):
        return None

    fitted = X @ coeffs
    return coeffs, fitted


def _group_multi_resid_row(
    y_row: np.ndarray,
    x_rows: list[np.ndarray],
    g_row: np.ndarray,
    add_intercept: bool,
    min_obs: int | None,
) -> np.ndarray:
    """Compute group-wise multi-feature regression residuals for one trading day.

    Each group independently fits ``y ~ x1 + x2 + ... + xN`` and returns
    residuals (y - y_pred). Groups with insufficient samples, singular matrices,
    or all-NaN values emit NaN.

    Args:
        y_row: dependent variable for one row (N_stocks,)
        x_rows: list of independent variable arrays, each (N_stocks,)
        g_row: group labels for one row (N_stocks,)
        add_intercept: whether to include intercept
        min_obs: minimum observations per group (default: n_params where n_params = n_features + 1 if intercept else n_features)

    Returns:
        residuals array (N_stocks,)
    """
    out = np.full(len(y_row), np.nan, dtype=float)
    n_features = len(x_rows)

    # Default min_obs: number of parameters (allows perfect fit with 0 DOF)
    if min_obs is None:
        n_params = (n_features + 1) if add_intercept else n_features
        min_obs = n_params

    # Get unique groups (filter out NaN/None for object arrays)
    if g_row.dtype == object:
        valid_mask = pd.notna(g_row)
        labels = pd.unique(g_row[valid_mask])
    else:
        labels = pd.unique(g_row[np.isfinite(g_row)])

    for label in labels:
        if not _valid_membership_label(label):
            continue

        # Find group members
        idx = g_row == label

        # Joint missing value filter: valid if y and ALL x's are finite
        valid = np.isfinite(y_row[idx])
        for x_row in x_rows:
            valid = valid & np.isfinite(x_row[idx])

        if valid.sum() < min_obs:
            continue

        # Extract valid samples
        ys = y_row[idx][valid]
        xs_list = [x_row[idx][valid] for x_row in x_rows]

        # Fit OLS (rank check inside will detect singular matrices)
        result = _group_multi_ols_fit(ys, xs_list, add_intercept)
        if result is None:
            continue

        coeffs, fitted_valid = result

        # Compute fitted values for ALL group members (including those with missing data)
        group_indices = np.flatnonzero(idx)
        if add_intercept:
            X_full = np.column_stack([np.ones(len(group_indices))] + [x_row[idx] for x_row in x_rows])
        else:
            X_full = np.column_stack([x_row[idx] for x_row in x_rows]) if n_features > 1 else x_rows[0][idx].reshape(-1, 1)

        fitted_full = X_full @ coeffs

        # Only write residuals where all inputs (y and x's) are valid
        resid_full = y_row[idx] - fitted_full
        valid_full = np.isfinite(y_row[idx])
        for x_row in x_rows:
            valid_full = valid_full & np.isfinite(x_row[idx])

        # Write residuals only for valid positions
        out_idx = group_indices[valid_full]
        out[out_idx] = resid_full[valid_full]

    return out


@register_operator(
    name="group_multi_resid",
    category="group_neutralization",
    business_category="group_neutralization",
    canonical="group_multi_resid",
    source="group_ext",
    status="experimental",
    research_only=True,
)
class GroupMultiResid(SeriesOperator):
    """组内多元线性回归残差（多自变量 OLS）。

    每个 group 独立拟合 ``y ~ x1 + x2 + ... + xN``，返回残差 (y - y_pred)。
    ``add_intercept=True`` 时自动添加截距项。``min_obs`` 默认为参数个数（n_features + 1
    含截距，或 n_features 不含截距），允许精确拟合。样本不足、矩阵奇异、全 NaN 组返回 NaN（fail-closed）。

    参数说明：
        y: 因变量
        group: 分组标签
        x1, x2, ..., x5: 自变量（variadic 参数）
        add_intercept: 是否添加截距（默认 True）
        min_obs: 最小观测数（默认等于参数个数，允许精确拟合）

    语义：
        - 联合过滤缺失值（y 和所有 x 必须同时有限）
        - 每组独立做 OLS，矩阵奇异或样本不足时该组全 NaN
        - 残差单位与 y 相同（unit:same_as:y）
    """

    metadata = _metadata(
        "group_multi_resid",
        "组内多元线性回归残差（支持多个自变量的组内 OLS 残差）",
        ["y", "group", "x1", "x2", "x3", "x4", "x5", "add_intercept", "min_obs"],
        domain="price_volume",
        unit="same_as:y",
    )

    def _calculate_series(
        self,
        y: pd.DataFrame,
        group: pd.DataFrame,
        x1: pd.DataFrame,
        x2: pd.DataFrame | None = None,
        x3: pd.DataFrame | None = None,
        x4: pd.DataFrame | None = None,
        x5: pd.DataFrame | None = None,
        add_intercept: bool = True,
        min_obs: int | None = None,
        **_: Any
    ) -> pd.DataFrame:
        # Collect non-None features
        features = [x1]
        for x in [x2, x3, x4, x5]:
            if x is not None:
                features.append(x)

        # Align all inputs
        aligned = _aligned(y, group, *features)
        y_aligned = aligned[0]
        group_aligned = aligned[1]
        features_aligned = aligned[2:]

        # Convert to numpy
        yv = y_aligned.to_numpy(dtype=float)
        gv = group_aligned.to_numpy()
        xvs = [f.to_numpy(dtype=float) for f in features_aligned]

        rows, cols = yv.shape
        out = np.full((rows, cols), np.nan, dtype=float)

        # Process each row
        for row in range(rows):
            out[row] = _group_multi_resid_row(
                yv[row],
                [xv[row] for xv in xvs],
                gv[row],
                add_intercept,
                min_obs,
            )

        return _frame_like(y_aligned, out)


# Register group_multi_resid to extended surface
try:
    from cleaned_operators.operator_surface import extend_extended_only

    extend_extended_only(["group_multi_resid"])
except ImportError:  # pragma: no cover
    pass

