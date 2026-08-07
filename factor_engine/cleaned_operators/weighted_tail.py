# -*- coding: utf-8 -*-
"""Stratified-conditional and weight-aware downside/tail-risk primitives (2026-08).

These generalise the fixed formulas Gemini proposed into weight- and
sorter-parameterised primitives that AlphaMiner / AlphaProbe / GP can recombine:

* ``ts_stratified_mean_spread``      — mean of ``target`` on the high-sorter tail
  minus the mean on the low-sorter tail inside a trailing window (``target`` /
  ``sorter`` are parameters, so ``return x volume`` is just one recipe).
* ``ts_weighted_semivariance``       — sqrt of the weight-normalised mean of
  squared downside deviations below a target (weights default to turnover).
* ``ts_weighted_expected_shortfall`` — weight-normalised mean of the tail beyond
  a weighted quantile.
* ``ts_weighted_drawdown_area``      — weight-normalised integral of drawdown
  depth from the running peak.

Every operator is prefix-causal: the window only ever reads rows ``<= t``, and
missing values use aligned-pair / drop-valid policy inside the window.  The
output shares the input panel axes.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import SeriesOperator, register_operator
from cleaned_operators.rolling_pack import aligned_pairs, frame_like, map_pair_rolling, valid_values

_EPS = 1e-12


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    """Linear-interpolated weighted quantile over (value, weight) pairs."""
    qq = float(quantile)
    order = np.argsort(values)
    cs = values[order]
    cw = weights[order]
    cdf = np.cumsum(cw)
    total = float(cdf[-1])
    if total <= _EPS:
        return np.nan
    cdf = cdf / total
    if qq <= 0.0:
        return float(cs[0])
    if qq >= 1.0:
        return float(cs[-1])
    idx = int(np.searchsorted(cdf, qq, side="left"))
    idx = min(max(idx, 1), cs.shape[0] - 1)
    span = float(cdf[idx] - cdf[idx - 1])
    if span <= _EPS:
        return float(cs[idx])
    frac = (qq - float(cdf[idx - 1])) / span
    return float(cs[idx - 1] + frac * (cs[idx] - cs[idx - 1]))


def _metadata(name: str, description: str, params: list[str], *, unit: str) -> Any:
    from cleaned_operators.base import OperatorMetadata

    return OperatorMetadata(
        name=name,
        category="time_series_risk",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "time_series_risk", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", f"unit:{unit}", "cost:1",
        ],
    )


@register_operator(
    name="ts_stratified_mean_spread",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_stratified_mean_spread",
    source="weighted_tail",
)
class TsStratifiedMeanSpread(SeriesOperator):
    """Mean of ``target`` on the high-``sorter`` tail minus the low-sorter tail.

    Inside each trailing window the aligned (``target``, ``sorter``) pairs are
    sorted by ``sorter``; the result is the mean of ``target`` over the top
    ``quantile`` fraction minus the mean over the bottom ``quantile`` fraction.
    ``target = return, sorter = volume`` reproduces the volume-stratified return
    spread, but the primitive is generic: ``return x amount``, ``return x
    turnover``, ``fundamental_change x turnover`` etc. are all the same call.
    """

    metadata = _metadata(
        "ts_stratified_mean_spread",
        "按 sorter 分层的 target 高低尾均值差。",
        ["target", "sorter", "window", "quantile", "min_periods"],
        unit="level",
    )

    def _calculate_series(
        self,
        target: pd.DataFrame,
        sorter: pd.DataFrame,
        window: int = 60,
        quantile: float = 0.2,
        min_periods: Any = None,
        **_: Any,
    ) -> pd.DataFrame:
        w = max(2, int(window))
        q = float(quantile)
        if not 0.0 < q < 1.0:
            raise ValueError("quantile must be in (0, 1)")
        mp = int(min_periods) if min_periods is not None else max(5, w // 4)
        mp = max(2, mp)

        def _fn(a: np.ndarray, b: np.ndarray) -> float:
            x, s = aligned_pairs(a, b)
            if x.size < mp:
                return np.nan
            order = np.argsort(s)
            xs = x[order]
            k = max(1, int(round(q * x.size)))
            k = min(k, x.size - 1)
            return float(np.mean(xs[-k:]) - np.mean(xs[:k]))

        return frame_like(
            target,
            map_pair_rolling(target.to_numpy(dtype=float), sorter.to_numpy(dtype=float), w, _fn),
        )


@register_operator(
    name="ts_weighted_semivariance",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_weighted_semivariance",
    source="weighted_tail",
)
class TsWeightedSemivariance(SeriesOperator):
    """Weight-normalised downside deviation below ``target``.

    ``sqrt( sum w * max(target - x, 0)^2 / sum w )`` over aligned (``x``,
    ``weight``) pairs in the trailing window.  With ``weight = turnover`` this is
    the turnover-weighted analogue of ``ts_downside_deviation``; the weight is a
    free parameter, so volume / amount / 1 are all searchable.
    """

    metadata = _metadata(
        "ts_weighted_semivariance",
        "加权下半方差 sqrt(sum w*max(target-x,0)^2 / sum w)。",
        ["x", "weight", "window", "target", "min_periods"],
        unit="ratio",
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        weight: pd.DataFrame,
        window: int = 20,
        target: float = 0.0,
        min_periods: Any = None,
        **_: Any,
    ) -> pd.DataFrame:
        w = max(2, int(window))
        tgt = float(target)
        mp = int(min_periods) if min_periods is not None else 2
        mp = max(2, mp)

        def _fn(a: np.ndarray, b: np.ndarray) -> float:
            xv, wv = aligned_pairs(a, b)
            if xv.size < mp:
                return np.nan
            total = float(wv.sum())
            if total <= _EPS:
                return np.nan
            below = np.maximum(tgt - xv, 0.0)
            return float(np.sqrt(np.sum(wv * below * below) / total))

        return frame_like(
            x,
            map_pair_rolling(x.to_numpy(dtype=float), weight.to_numpy(dtype=float), w, _fn),
        )


@register_operator(
    name="ts_weighted_expected_shortfall",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_weighted_expected_shortfall",
    source="weighted_tail",
)
class TsWeightedExpectedShortfall(SeriesOperator):
    """Weight-normalised mean of the tail beyond a weighted quantile.

    For ``side='lower'`` the tail is ``{x <= Q_q}`` for the weighted quantile
    ``Q_q``; the result is ``sum(w*x) / sum(w)`` over that tail.  ``side='upper'``
    mirrors at ``Q_{1-q}``.  This answers "how severe is the volume that actually
    participated in the extreme move" and is distinct from the equal-weighted
    ``ts_expected_shortfall``.
    """

    metadata = _metadata(
        "ts_weighted_expected_shortfall",
        "加权期望损失: 加权分位数之外尾部的加权均值。",
        ["x", "weight", "window", "quantile", "side", "min_tail_count"],
        unit="level",
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        weight: pd.DataFrame,
        window: int = 60,
        quantile: float = 0.05,
        side: str = "lower",
        min_tail_count: Any = None,
        **_: Any,
    ) -> pd.DataFrame:
        w = max(2, int(window))
        q = float(quantile)
        if not 0.0 < q < 1.0:
            raise ValueError("quantile must be in (0, 1)")
        kind = str(side).lower()
        if kind not in {"lower", "upper"}:
            raise ValueError("side must be 'lower' or 'upper'")
        if min_tail_count is not None:
            min_tail = max(2, int(min_tail_count))
        else:
            min_tail = max(2, 3)

        def _fn(a: np.ndarray, b: np.ndarray) -> float:
            xv, wv = aligned_pairs(a, b)
            if xv.size < max(min_tail, 3):
                return np.nan
            total = float(wv.sum())
            if total <= _EPS:
                return np.nan
            thr = _weighted_quantile(xv, wv, q if kind == "lower" else 1.0 - q)
            if not np.isfinite(thr):
                return np.nan
            if kind == "lower":
                tail = xv <= thr
            else:
                tail = xv >= thr
            w_tail = wv[tail]
            x_tail = xv[tail]
            if w_tail.size < min_tail or float(w_tail.sum()) <= _EPS:
                return np.nan
            return float(np.sum(w_tail * x_tail) / np.sum(w_tail))

        return frame_like(
            x,
            map_pair_rolling(x.to_numpy(dtype=float), weight.to_numpy(dtype=float), w, _fn),
        )


@register_operator(
    name="ts_weighted_drawdown_area",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_weighted_drawdown_area",
    source="weighted_tail",
)
class TsWeightedDrawdownArea(SeriesOperator):
    """Weight-normalised drawdown-depth area from the running peak.

    With ``DD_s = max(0, 1 - x_s / Peak_s)`` (``Peak_s`` the running max inside
    the window), the output is ``sum(w * DD) / sum(w)``.  ``weight = volume``
    recovers Gemini's ``ts_volume_underwater`` idea, but the weight is a free
    parameter (amount / turnover / 1 are all searchable).
    """

    metadata = _metadata(
        "ts_weighted_drawdown_area",
        "加权回撤深度面积(运行峰值起) sum(w*DD)/sum(w)。",
        ["x", "weight", "window"],
        unit="ratio",
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        weight: pd.DataFrame,
        window: int = 60,
        **_: Any,
    ) -> pd.DataFrame:
        w = max(2, int(window))

        def _fn(a: np.ndarray, b: np.ndarray) -> float:
            xv, wv = aligned_pairs(a, b)
            pos = (xv > 0.0) & np.isfinite(xv)
            if int(pos.sum()) < 2:
                return np.nan
            p = xv[pos]
            wt = wv[pos]
            total = float(wt.sum())
            if total <= _EPS:
                return np.nan
            running_max = np.maximum.accumulate(p)
            dd = np.maximum(0.0, 1.0 - p / running_max)
            return float(np.sum(wt * dd) / total)

        return frame_like(
            x,
            map_pair_rolling(x.to_numpy(dtype=float), weight.to_numpy(dtype=float), w, _fn),
        )


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS)
        | {
            "ts_stratified_mean_spread",
            "ts_weighted_semivariance",
            "ts_weighted_expected_shortfall",
            "ts_weighted_drawdown_area",
        }
    )
    from cleaned_operators.rolling_pack import register_polars_bridge

    for _canon in (
        "ts_stratified_mean_spread",
        "ts_weighted_semivariance",
        "ts_weighted_expected_shortfall",
        "ts_weighted_drawdown_area",
    ):
        register_polars_bridge(_canon)


_register_surface()
