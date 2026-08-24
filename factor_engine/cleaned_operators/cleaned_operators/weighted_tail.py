# -*- coding: utf-8 -*-
"""Stratified-conditional and weight-aware downside/tail-risk primitives (2026-08).

These generalise the fixed formulas Gemini proposed into weight- and
sorter-parameterised primitives that AlphaMiner / AlphaProbe / GP can recombine:

* ``ts_stratified_mean_spread``      — mean of ``target`` on the high-sorter tail
  minus the mean on the low-sorter tail inside a trailing window (``quantile``
  must be ``<= 0.5`` so the two strata never overlap; ``target`` / ``sorter``
  are parameters, so ``return x volume`` is just one recipe).  Cutoff sorter
  ties enter with FRACTIONAL participation (R3-116).
* ``ts_weighted_semivariance``       — weight-normalised mean of squared
  downside deviations below a target (NO sqrt — the honest semivariance).
* ``ts_weighted_downside_deviation`` — sqrt of the weight-normalised mean of
  squared downside deviations (the historical "semivariance" formula, renamed
  honestly; R3-113).
* ``ts_weighted_expected_shortfall`` — weight-normalised mean of the tail beyond
  a weighted ECDF-inverse quantile (no interpolation; R3-115).
* ``ts_weighted_drawdown_area``      — weight-normalised integral of drawdown
  depth from the running peak.

R3-113 naming: ``sqrt(sum w (target-x)_+^2 / sum w)`` is the weighted DOWNside
DEVIATION, not the semivariance.  The semivariance keeps the square;
``ts_weighted_semivariance`` is the no-sqrt name, the sqrt variant lives at
``ts_weighted_downside_deviation`` (back-compat alias
``ts_weighted_semivariance_sqrt`` preserves the old sqrt behaviour).

Every operator is prefix-causal: the window only ever reads rows ``<= t``, and
missing values use aligned-pair / drop-valid policy inside the window.  The
output shares the input panel axes.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import SeriesOperator, register_operator
from factor_engine.cleaned_operators.rolling_pack import aligned_pairs, frame_like, map_pair_rolling, valid_values

_EPS = 1e-12


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    """Weighted ECDF-inverse quantile — an OBSERVED value, never a fabricated
    interpolated threshold (R3-115).

    ``q`` maps to the smallest observed value whose cumulative weight reaches
    ``q`` (``np.searchsorted(cdf, q, side="left")``).  With
    ``10@w0.4, 20@w0.6`` the q=0.5 quantile is ``20`` — NOT the made-up ``14.7``
    linear interpolation previously emitted.  ``q <= cdf[0]`` clamps to
    ``cs[0]`` and ``q >= cdf[-1]`` clamps to ``cs[-1]`` (P0-17 / P0-16: no
    below-first-value extrapolation).
    """
    qq = float(quantile)
    order = np.argsort(values, kind="stable")
    cs = values[order]
    cw = weights[order]
    cdf = np.cumsum(cw)
    total = float(cdf[-1])
    if total <= _EPS:
        return np.nan
    cdf = cdf / total
    if qq <= float(cdf[0]):
        return float(cs[0])
    if qq >= float(cdf[-1]):
        return float(cs[-1])
    idx = int(np.searchsorted(cdf, qq, side="left"))
    idx = min(max(idx, 0), cs.shape[0] - 1)
    return float(cs[idx])


def _weighted_es_tail(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
    side: str,
    min_tail: int,
) -> float:
    """Weighted mean of the tail beyond a weighted ECDF quantile (R3-115).

    The boundary is an OBSERVED value (weighted ECDF inverse — no linear
    interpolation).  When the cutoff lands inside a group of identical values,
    the tied group enters with FRACTIONAL weight so the tail holds exactly the
    requested mass fraction (mirrors ``group_topk_mean``).  Returns NaN when the
    effective tail has fewer than ``min_tail`` members.
    """
    qq = float(quantile)
    order = np.argsort(values, kind="stable")
    sv = values[order]
    sw = weights[order]
    total = float(sw.sum())
    if total <= _EPS:
        return np.nan
    cdf = np.cumsum(sw)
    if side == "lower":
        target = qq * total
        idx = int(np.searchsorted(cdf, target, side="left"))
        idx = min(idx, sv.size - 1)
        vstar = sv[idx]
        keep_mask = sv < vstar
        need = target - float(sw[keep_mask].sum())
    else:
        target = (1.0 - qq) * total
        idx = int(np.searchsorted(cdf, target, side="left"))
        idx = min(idx, sv.size - 1)
        vstar = sv[idx]
        keep_mask = sv > vstar
        need = qq * total - float(sw[keep_mask].sum())
    tie_mask = sv == vstar
    w_tie = float(sw[tie_mask].sum())
    frac = (need / w_tie) if w_tie > _EPS else 0.0
    frac = min(max(frac, 0.0), 1.0)
    n_eff = float(keep_mask.sum()) + frac * float(tie_mask.sum())
    if n_eff < min_tail:
        return np.nan
    num = float(np.sum(sw[keep_mask] * sv[keep_mask])) + frac * float(np.sum(sw[tie_mask] * sv[tie_mask]))
    den = float(sw[keep_mask].sum()) + frac * w_tie
    if den <= _EPS:
        return np.nan
    return num / den


def _stratified_stratum_mean(xs: np.ndarray, ss: np.ndarray, n_take: int, *, top: bool) -> float:
    """Mean of ``xs`` over the top/bottom ``n_take`` members ranked by ``ss``.

    R3-116: a cutoff that lands inside a group of identical sorter values is
    resolved with FRACTIONAL participation — the tied group enters with weight
    ``(n_take - n_strict) / n_tie`` (mirrors ``group_topk_mean``), never by
    ``argsort`` position, so a column/row permutation cannot change the spread.
    """
    n = xs.size
    if top:
        kth = ss[n - n_take]
        strict = ss > kth
    else:
        kth = ss[n_take - 1]
        strict = ss < kth
    tie = ss == kth
    n_strict = int(strict.sum())
    n_tie = int(tie.sum())
    n_take_tie = min(max(n_take - n_strict, 0), n_tie)
    frac = (n_take_tie / n_tie) if n_tie > 0 else 0.0
    return float((np.sum(xs[strict]) + frac * np.sum(xs[tie])) / n_take)


def _validate_nonneg_weight(weight: pd.DataFrame, name: str) -> None:
    """Reject negative weights *at the current window* before risk statistics.

    The check runs inside the trailing-window kernel (``wv < 0 -> NaN``), never
    as a whole-panel pre-scan: scanning the entire DataFrame would let a future
    bad weight invalidate every earlier row — program success/failure looking
    ahead (round-7 P0).  Kept as a module helper for direct kernel tests; the
    operators enforce the same rule per window.
    """
    arr = weight.to_numpy(dtype=float)
    if np.any(arr < 0.0):
        raise ValueError(f"{name}: weights must be non-negative (got a negative weight)")


def _metadata(
    name: str, description: str, params: list[str], *, unit: str, output_unit: str | None = None,
    cost: str = "cost:1",
) -> Any:
    from factor_engine.cleaned_operators.base import OperatorMetadata

    if output_unit is None:
        # R11 §37-D unit-algebra honesty: algebraic units (``same_as:`` /
        # ``unit(...)``) are propagated to the ``output_unit`` field; a fixed
        # "ratio"/"level" label is a declared dimensionless-ish fixed unit and
        # is left unset.
        output_unit = unit if (unit.startswith("same_as:") or unit.startswith("unit(")) else None
    return OperatorMetadata(
        name=name,
        category="time_series_risk",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "time_series_risk", "daily", "pit_safe", "causal", "typed_v2",
            f"signature:{','.join(params)}->series", f"unit:{unit}", cost,
        ],
        output_unit=output_unit,
    )


@register_operator(
    name="ts_stratified_mean_spread",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_stratified_mean_spread",
    source="weighted_tail")
class TsStratifiedMeanSpread(SeriesOperator):
    """Mean of ``target`` on the high-``sorter`` tail minus the low-sorter tail.

    Inside each trailing window the aligned (``target``, ``sorter``) pairs are
    sorted by ``sorter``; the result is the mean of ``target`` over the top
    ``quantile`` fraction minus the mean over the bottom ``quantile`` fraction.
    ``quantile`` must be ``<= 0.5`` so the top and bottom strata stay disjoint
    (P1-83).  When the cutoff lands inside a group of identical ``sorter``
    values the tied group participates FRACTIONALLY (R3-116, mirrors
    ``group_topk_mean``) — never by ``argsort`` position, so a column
    permutation cannot change the spread.  The output carries the unit of
    ``target`` (R3-117).  ``target = return, sorter = volume`` reproduces the
    volume-stratified return spread, but the primitive is generic: ``return x
    amount``, ``return x turnover``, ``fundamental_change x turnover`` etc. are
    all the same call.
    """

    metadata = _metadata(
        "ts_stratified_mean_spread",
        "按 sorter 分层的 target 高低尾均值差（分位数边界并列按比例计入）。",
        ["target", "sorter", "window", "quantile", "min_periods"],
        unit="same_as:target",
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
        if not 0.0 < q <= 0.5:
            # P1-83: with q > 0.5 the top and bottom strata overlap and the
            # spread becomes degenerate, so reject it loudly.
            raise ValueError("quantile must be in (0, 0.5] so the top and bottom strata are disjoint")
        mp = int(min_periods) if min_periods is not None else max(5, w // 4)
        mp = max(2, mp)

        def _fn(a: np.ndarray, b: np.ndarray) -> float:
            x, s = aligned_pairs(a, b)
            if x.size < mp:
                return np.nan
            order = np.argsort(s, kind="stable")
            xs = x[order]
            ss = s[order]
            # R16-094: ``k = round(q * N)`` OVERLAPPED the strata on odd N
            # (N=7, q=.5 -> round(3.5)=4, so top-4 and bottom-4 overlap).  Use
            # ``floor(q * N)`` and REQUIRE ``2*k <= N`` (guaranteed for q<=0.5
            # with floor) plus a min-stratum floor; a degenerate/overlapping
            # stratum is NaN, never a silent adjustment.
            k = int(np.floor(q * x.size))
            if k < 1 or 2 * k > x.size:
                return np.nan
            top = _stratified_stratum_mean(xs, ss, k, top=True)
            bot = _stratified_stratum_mean(xs, ss, k, top=False)
            if not np.isfinite(top) or not np.isfinite(bot):
                return np.nan
            return float(top - bot)

        return frame_like(
            target,
            map_pair_rolling(target.to_numpy(dtype=float), sorter.to_numpy(dtype=float), w, _fn),
        )


@register_operator(
    name="ts_weighted_semivariance",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_weighted_semivariance",
    source="weighted_tail")
class TsWeightedSemivariance(SeriesOperator):
    """Weight-normalised SEMIvariance below ``target`` (NO sqrt; R3-113).

    ``sum w * max(target - x, 0)^2 / sum w`` over aligned (``x``, ``weight``)
    pairs in the trailing window.  The name is honest: this is the variance of
    the downside deviations, carrying ``unit(x)^2``.  With ``weight = turnover``
    this is the turnover-weighted analogue of the semivariance; the weight is a
    free parameter, so volume / amount / 1 are all searchable.  For the sqrt'd
    variant (the historical "semivariance" formula, which is really the
    downside DEVIATION) use ``ts_weighted_downside_deviation``.
    """

    metadata = _metadata(
        "ts_weighted_semivariance",
        "加权下半方差 sum w*max(target-x,0)^2 / sum w（无 sqrt）。",
        ["x", "weight", "window", "target", "min_periods"],
        unit="unit(x)^2",
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
            # Per-window (prefix-safe) non-negativity: a negative weight in the
            # current trailing window makes the risk statistic ill-defined, so
            # fail closed for this row.  Never scan the future panel.
            if np.any(wv < 0.0):
                return np.nan
            if xv.size < mp:
                return np.nan
            total = float(wv.sum())
            if total <= _EPS:
                return np.nan
            below = np.maximum(tgt - xv, 0.0)
            return float(np.sum(wv * below * below) / total)

        return frame_like(
            x,
            map_pair_rolling(x.to_numpy(dtype=float), weight.to_numpy(dtype=float), w, _fn),
        )


@register_operator(
    name="ts_weighted_downside_deviation",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_weighted_downside_deviation",
    source="weighted_tail")
class TsWeightedDownsideDeviation(SeriesOperator):
    """Weight-normalised DOWNside DEVIATION below ``target`` (sqrt; R3-113).

    ``sqrt( sum w * max(target - x, 0)^2 / sum w )`` over aligned (``x``,
    ``weight``) pairs in the trailing window.  This is the formula the
    historical ``ts_weighted_semivariance`` computed under a misnomer; the
    honest name is downside deviation and the output carries the unit of ``x``
    (``same_as:x``, R3-114).  The no-sqrt variant (true semivariance) is
    ``ts_weighted_semivariance``; back-compat alias
    ``ts_weighted_semivariance_sqrt`` also resolves here.
    """

    metadata = _metadata(
        "ts_weighted_downside_deviation",
        "加权下行偏离 sqrt(sum w*max(target-x,0)^2 / sum w)，输出同 x 单位。",
        ["x", "weight", "window", "target", "min_periods"],
        unit="same_as:x",
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
            # Per-window (prefix-safe) non-negativity; see round-7 P0.
            if np.any(wv < 0.0):
                return np.nan
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
    source="weighted_tail")
class TsWeightedExpectedShortfall(SeriesOperator):
    """Weight-normalised mean of the tail beyond a weighted quantile.

    For ``side='lower'`` the tail is the ``q`` mass fraction of the smallest
    ``x``; the boundary is a weighted ECDF inverse (an OBSERVED value — no
    interpolation, R3-115) and a cutoff that lands inside a group of identical
    values enters with FRACTIONAL weight.  ``side='upper'`` mirrors at the top
    ``q`` fraction.  This answers "how severe is the volume that actually
    participated in the extreme move" and is distinct from the equal-weighted
    ``ts_expected_shortfall``.
    """

    metadata = _metadata(
        "ts_weighted_expected_shortfall",
        "加权期望损失: 加权分位数之外尾部的加权均值。",
        ["x", "weight", "window", "quantile", "side", "min_tail_count"],
        # R16-096: ES is the WEIGHTED MEAN of x over the tail — its unit is
        # ``same_as:x``, not a generic level.
        unit="same_as:x",
        # R16-098: weighted quantile + per-window sort is NOT a linear rolling;
        # the real cost is W log W.
        cost="cost:3",
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
        # Expected shortfall's ``quantile`` is a *tail fraction*: values above
        # 0.5 are not tail-risk semantics (a q=0.8 "lower tail" is the bottom
        # 80% of the distribution).  Restrict to (0, 0.5]; production search
        # should use a small grid like {0.01, 0.025, 0.05, 0.10, 0.20} (round-7
        # P0).
        if not 0.0 < q <= 0.5:
            raise ValueError("quantile must be in (0, 0.5] for expected-shortfall tail semantics")
        kind = str(side).lower()
        if kind not in {"lower", "upper"}:
            raise ValueError("side must be 'lower' or 'upper'")
        if min_tail_count is not None:
            min_tail = max(2, int(min_tail_count))
        else:
            min_tail = max(2, 3)

        def _fn(a: np.ndarray, b: np.ndarray) -> float:
            xv, wv = aligned_pairs(a, b)
            # Per-window (prefix-safe) non-negativity; see round-7 P0.
            if np.any(wv < 0.0):
                return np.nan
            if xv.size < max(min_tail, 3):
                return np.nan
            # R16-097: member-count alone cannot gate a WEIGHTED tail — one
            # huge weight + many tiny weights passes a size gate while carrying
            # ~no effective tail mass.  Kish effective-N ``(sum w)^2 / sum w^2``
            # is the honest sample-size; a too-small Kish is NaN.
            sw = float(wv.sum())
            sw2 = float(np.sum(wv * wv))
            kish = (sw * sw) / sw2 if sw2 > 0.0 else 0.0
            if kish < min_tail:
                return np.nan
            return _weighted_es_tail(xv, wv, q, kind, min_tail)

        return frame_like(
            x,
            map_pair_rolling(x.to_numpy(dtype=float), weight.to_numpy(dtype=float), w, _fn),
        )


@register_operator(
    name="ts_weighted_drawdown_area",
    category="time_series_risk",
    business_category="time_series_risk",
    canonical="ts_weighted_drawdown_area",
    source="weighted_tail")
class TsWeightedDrawdownArea(SeriesOperator):
    """Weight-normalised drawdown-depth area from the running peak.

    With ``DD_s = max(0, 1 - x_s / Peak_s)`` (``Peak_s`` the running max inside
    the window), the output is ``sum(w * DD) / sum(w)``.  The time axis is
    preserved: a missing price resets the running peak so the drawdown never
    reconnects across a data gap (P1-85).  ``weight = volume`` recovers
    Gemini's ``ts_volume_underwater`` idea, but the weight is a free parameter
    (amount / turnover / 1 are all searchable).
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
            # P1-85: do NOT compress the time axis with aligned_pairs before
            # computing the running peak.  ``a`` / ``b`` keep their positions, so
            # a missing price (NaN) breaks the path: the running peak resets at
            # each contiguous valid run and never reconnects across a gap.  A
            # missing *weight* does not break the price path — its drawdown is
            # simply excluded from the weighted mean.
            if np.any(b < 0.0):
                # Per-window (prefix-safe) non-negativity; round-7 P0.
                return np.nan
            price_ok = np.isfinite(a) & (a > 0.0)
            if int(price_ok.sum()) < 2:
                return np.nan
            dd = np.full(a.shape[0], np.nan)
            run_start = None
            peak = 0.0
            for i in range(a.shape[0]):
                if not price_ok[i]:
                    run_start = None
                    continue
                if run_start is None:
                    run_start = i
                    peak = a[i]
                else:
                    peak = max(peak, a[i])
                dd[i] = max(0.0, 1.0 - a[i] / peak)
            wv = np.where(np.isfinite(b), b, 0.0)
            valid = price_ok & np.isfinite(b)
            total = float(wv[valid].sum())
            if total <= _EPS:
                return np.nan
            return float(np.sum(wv[valid] * dd[valid]) / total)

        return frame_like(
            x,
            map_pair_rolling(x.to_numpy(dtype=float), weight.to_numpy(dtype=float), w, _fn),
        )


def _register_surface() -> None:
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({
            "ts_stratified_mean_spread",
            "ts_weighted_semivariance",
            "ts_weighted_downside_deviation",
            "ts_weighted_expected_shortfall",
            "ts_weighted_drawdown_area",
        })
    from factor_engine.cleaned_operators.rolling_pack import register_polars_bridge

    for _canon in (
        "ts_stratified_mean_spread",
        "ts_weighted_semivariance",
        "ts_weighted_downside_deviation",
        "ts_weighted_expected_shortfall",
        "ts_weighted_drawdown_area",
    ):
        register_polars_bridge(_canon)

    # R3-113 back-compat: the historical sqrt'd "semivariance" behaviour now
    # lives at ``ts_weighted_downside_deviation``; keep it resolvable under a
    # legacy spelling so old recipes/strategies can opt back in explicitly.
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    OperatorRegistry.register_compat_alias(
        "ts_weighted_semivariance_sqrt",
        "ts_weighted_downside_deviation",
        migration_reason=(
            "R3-113 naming split: the sqrt of the weight-normalised squared "
            "downside deviation is the weighted downside deviation; the honest "
            "semivariance (no sqrt) keeps the name ts_weighted_semivariance"
        ),
        deprecated_since="0.11.0",
        removal_version="0.13.0",
    )


_register_surface()
