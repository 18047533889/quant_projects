# -*- coding: utf-8 -*-
"""Quantile-hit and extreme-dependence dynamics (2026-08 market language, P1/P2).

Four families share hit-sequence kernels:

* ``ts_quantilogram`` / ``ts_cross_quantilogram`` — autocorrelation of the
  *hit process* ``H_t(q) = I(x_t <= Q_q) - q`` (lower) or the upper-tail
  mirror.  Ordinary ACF asks "does ``x_t`` correlate with ``x_{t-k}``"; the
  quantilogram asks "does *being in an extreme state* carry memory", and the
  cross-quantilogram whether a source being extreme at ``t-lag`` makes a
  target extreme at ``t`` more likely.
* ``ts_quantile_crossing_spectral_concentration`` — spectral concentration of a
  quantile-regime hit series: is the extreme regime random or periodic?
* ``ts_extremogram`` / ``ts_cross_extremogram`` / ``ts_extremal_dependence_decay``
  — excess dependence ``P(E_{t+h}=1 | E_t=1) - P(E)`` (extremogram minus the
  baseline exceedance probability) and its exponential decay time constant.

All kernels are prefix-causal: the quantile threshold is estimated from the
trailing window ending at the current row, and every lagged term combines only
past observations.  By default the threshold is re-estimated every day, so a
*past* hit label is repainted as the window rolls; ``fixed_threshold=True``
switches to the strict-past / fixed-event-threshold mode (threshold estimated
once from the leading strictly-past rows and frozen for the whole window) —
R5 P1-40(b).  Fail-closed to NaN on degenerate / constant / too-short windows;
deterministic (no randomness).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import (
    frame_like,
    map_pair_rolling,
    map_rolling,
    register_polars_udf,
)

_EPS = 1e-12


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    domain: str,
    unit: str,
    cost: int,
    extra_tags: tuple[str, ...] = (),
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="quantile_dynamics",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "quantile_dynamics", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic", *extra_tags,
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _check_q(q: float) -> float:
    qq = float(q)
    # Round-7 P0: these operators model *extreme/tail states* — the hit process
    # of a tail event.  A "tail" cannot cover more than half the distribution:
    # q > 0.5 as a lower-tail threshold means "the bottom 80% is extreme", a
    # semantic contradiction (the upper side mirrors at 1-q, so (0, 0.5] spans
    # every meaningful tail fraction on both sides).
    if not (0.0 < qq <= 0.5):
        raise ValueError("tail-state quantile must be in (0, 0.5]")
    return qq


def _check_side(side: str) -> None:
    if side not in ("lower", "upper"):
        raise ValueError("side must be 'lower' or 'upper'")


def _side_threshold(chunk: np.ndarray, q: float, side: str) -> float:
    """Quantile threshold for the requested tail: lower uses Q_q, upper Q_{1-q}."""
    return float(np.nanquantile(chunk, q if side == "lower" else 1.0 - q))


def _hit_series(
    chunk: np.ndarray, q: float, side: str, demean: bool, fixed_threshold: bool = False
) -> np.ndarray:
    """1/0 hit indicator (optionally demeaned by ``q``); NaN preserved.

    R5 P1-40(b): by default the quantile is re-estimated from the full trailing
    window every day, so a *past* hit label is repainted as the window rolls (a
    documented property).  ``fixed_threshold=True`` switches to the strict-past /
    fixed-event-threshold mode: the threshold is estimated ONCE from the
    strictly-past leading rows (excluding the current row) and frozen for the
    whole window, so no past label is repainted by today's data.
    """
    if fixed_threshold:
        lead = chunk[:-1] if chunk.shape[0] > 1 else chunk
        thr = _side_threshold(lead, q, side)
    else:
        thr = _side_threshold(chunk, q, side)
    if not np.isfinite(thr):
        return np.full(chunk.shape, np.nan)
    if side == "lower":
        h = (chunk <= thr).astype(float)
    else:
        h = (chunk >= thr).astype(float)
    h = np.where(np.isfinite(chunk), h, np.nan)
    if demean:
        h = h - q
    return h


def _pearson_lag(a: np.ndarray, lag: int) -> float:
    """Pearson correlation of ``a[lag:]`` vs ``a[:-lag]`` (jointly finite)."""
    m = a.shape[0]
    if m <= lag + 2:
        return np.nan
    x = a[lag:]
    y = a[: m - lag]
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < lag + 2:
        return np.nan
    xx = x[ok]
    yy = y[ok]
    vx = float(np.var(xx))
    vy = float(np.var(yy))
    if vx <= _EPS or vy <= _EPS:
        return np.nan
    return float(np.corrcoef(xx, yy)[0, 1])


# --------------------------------------------------------------------------
# quantilogram
# --------------------------------------------------------------------------
def _quantilogram_chunk(
    chunk: np.ndarray, q: float, lag: int, side: str, fixed_threshold: bool = False
) -> float:
    H = _hit_series(chunk, q, side, demean=True, fixed_threshold=fixed_threshold)
    return _pearson_lag(H, lag)


@register_operator(
    name="ts_quantilogram",
    category="quantile_dynamics",
    business_category="quantile_dynamics",
    canonical="ts_quantilogram",
    source="advanced_quantile_dynamics",
)
class TsQuantilogram(SeriesOperator):
    """分位命中序列的自相关（extreme state 的记忆）。

    先取窗口内 ``Q_q``（lower）或 ``Q_{1-q}``（upper），构造命中过程
    ``H_t = I(x_t 越界) - q``，输出 ``Corr(H_t, H_{t-lag})``。回答"跌进历史
    bottom 10% 后，第二天是不是更容易继续处于极差状态"这类问题，与普通
    autocorrelation 不同。PIT 安全：阈值用 trailing 窗口估计，滞后项只取过去。
    P1。
    """

    metadata = _metadata(
        "ts_quantilogram",
        "分位命中序列自相关 Corr(H_t, H_{t-lag})（extreme-state 记忆）。",
        ["x", "window", "quantile", "lag", "side", "fixed_threshold"],
        domain="price_volume",
        unit="corr",
        cost=4,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 120,
        quantile: float = 0.1,
        lag: int = 1,
        side: str = "lower",
        fixed_threshold: bool = False,
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        q = _check_q(quantile)
        lg = int(lag)
        _check_side(side)
        if lg < 1:
            raise ValueError("ts_quantilogram requires lag >= 1")
        if w < lg + 3:
            raise ValueError("ts_quantilogram requires window >= lag + 3")
        return frame_like(
            x,
            map_rolling(
                x.to_numpy(dtype=float),
                w,
                lambda c: _quantilogram_chunk(c, q, lg, side, fixed_threshold),
            ),
        )


def _cross_quantilogram_chunk(
    tc: np.ndarray,
    sc: np.ndarray,
    tq: float,
    sq: float,
    lag: int,
    ts: str,
    ss: str,
    fixed_threshold: bool = False,
) -> float:
    Ht = _hit_series(tc, tq, ts, demean=True, fixed_threshold=fixed_threshold)
    Hs = _hit_series(sc, sq, ss, demean=True, fixed_threshold=fixed_threshold)
    m = Ht.shape[0]
    if m <= lag + 2:
        return np.nan
    x = Ht[lag:]  # target hit at t
    y = Hs[: m - lag]  # source hit at t-lag
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < lag + 2:
        return np.nan
    xx = x[ok]
    yy = y[ok]
    vx = float(np.var(xx))
    vy = float(np.var(yy))
    if vx <= _EPS or vy <= _EPS:
        return np.nan
    return float(np.corrcoef(xx, yy)[0, 1])


@register_operator(
    name="ts_cross_quantilogram",
    category="quantile_dynamics",
    business_category="quantile_dynamics",
    canonical="ts_cross_quantilogram",
    source="advanced_quantile_dynamics",
)
class TsCrossQuantilogram(SeriesOperator):
    """跨序列分位命中相关（source 极值状态对 target 极值状态的预测）。

    ``Corr(H_target_t, H_source_{t-lag})``：``turnover/vol/估值`` 处于某分位
    后，``return`` 是否更易落入其对应分位。可自由混搭 volume→return、
    volatility→return、market→stock、industry→stock 等。自动因子挖掘的理想
    primitive。P1。
    """

    metadata = _metadata(
        "ts_cross_quantilogram",
        "跨序列分位命中相关 Corr(H_target_t, H_source_{t-lag})。",
        ["target", "source", "window", "target_q", "source_q", "lag", "target_side", "source_side", "fixed_threshold"],
        domain="price_volume",
        unit="corr",
        cost=5,
    )

    def _calculate_series(
        self,
        target: pd.DataFrame,
        source: pd.DataFrame,
        window: int = 120,
        target_q: float = 0.1,
        source_q: float = 0.1,
        lag: int = 1,
        target_side: str = "lower",
        source_side: str = "lower",
        fixed_threshold: bool = False,
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        tq = _check_q(target_q)
        sq = _check_q(source_q)
        lg = int(lag)
        _check_side(target_side)
        _check_side(source_side)
        if lg < 1:
            raise ValueError("ts_cross_quantilogram requires lag >= 1")
        if w < lg + 3:
            raise ValueError("ts_cross_quantilogram requires window >= lag + 3")
        return frame_like(
            target,
            map_pair_rolling(
                target.to_numpy(dtype=float),
                source.to_numpy(dtype=float),
                w,
                lambda a, b: _cross_quantilogram_chunk(
                    a, b, tq, sq, lg, target_side, source_side, fixed_threshold
                ),
            ),
        )


def _hit_spectral_concentration_chunk(
    chunk: np.ndarray, q: float, side: str, fixed_threshold: bool = False
) -> float:
    H = _hit_series(chunk, q, side, demean=False, fixed_threshold=fixed_threshold)
    fin = np.isfinite(H)
    n = chunk.shape[0]
    # R5 P1-40(a): censoring, not manufacture — missing hits are excluded from
    # the periodogram entirely (they must NOT contribute a zero "no-event"
    # sample), and mostly-missing windows fail closed via the coverage gate.
    if int(fin.sum()) < max(8, int(np.ceil(0.5 * n))):
        return np.nan
    h = H[fin]  # observed hit process only (missing gaps compressed)
    mu = float(h.mean())
    h = h - mu
    powers = np.abs(np.fft.rfft(h)) ** 2.0
    powers = powers[1:]  # drop DC
    total = float(powers.sum())
    if total <= _EPS:
        return np.nan
    return float(powers.max() / total)


@register_operator(
    name="ts_quantile_crossing_spectral_concentration",
    category="quantile_dynamics",
    business_category="quantile_dynamics",
    canonical="ts_quantile_crossing_spectral_concentration",
    source="advanced_quantile_dynamics",
    status="experimental",
)
class TsQuantileCrossingSpectralConcentration(SeriesOperator):
    """分位 regime 命中序列的谱集中度（极端状态是否周期性）。

    对命中序列做 periodogram（rfft 功率谱，去掉 DC），输出最大非零频点功率占比
    ``max P_f / sum P_f``。白噪声命中 → 谱平坦、集中度低；有重复周期（如隔周
    放量、定期爆雷）→ 集中度高。普通频谱分析作用于原始 x，这里作用于
    "某 quantile regime" 本身。R5 P1-40(a)：missing 命中被 censor（从谱估计中
    排除，而不是填 0 制造"无事件"）；缺失过多的窗口经 coverage gate fail-closed
    → NaN。P2 / Research。
    """

    metadata = _metadata(
        "ts_quantile_crossing_spectral_concentration",
        "分位命中序列谱集中度 max P_f / sum P_f（极端状态周期性）。",
        ["x", "window", "quantile", "side", "fixed_threshold"],
        domain="price_volume",
        unit="ratio",
        cost=5,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 120,
        quantile: float = 0.1,
        side: str = "lower",
        fixed_threshold: bool = False,
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        q = _check_q(quantile)
        _check_side(side)
        if w < 8:
            raise ValueError("ts_quantile_crossing_spectral_concentration requires window >= 8")
        return frame_like(
            x,
            map_rolling(
                x.to_numpy(dtype=float),
                w,
                lambda c: _hit_spectral_concentration_chunk(c, q, side, fixed_threshold),
            ),
        )


# --------------------------------------------------------------------------
# extremogram
# --------------------------------------------------------------------------
def _extremogram_excess_chunk(
    chunk: np.ndarray, q: float, lag: int, side: str, fixed_threshold: bool = False
) -> float:
    E = _hit_series(chunk, q, side, demean=False, fixed_threshold=fixed_threshold)
    m = E.shape[0]
    if m <= lag + 1:
        return np.nan
    x = E[lag:]  # E_{t+lag}
    y = E[: m - lag]  # E_t
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 3:
        return np.nan
    yy = y[ok]
    xx = x[ok]
    denom = float(yy.sum())
    if denom <= 0:
        return np.nan
    cond = float(np.sum(yy * xx)) / denom
    base = float(np.nanmean(E))
    return float(cond - base)


@register_operator(
    name="ts_extremogram",
    category="quantile_dynamics",
    business_category="quantile_dynamics",
    canonical="ts_extremogram",
    source="advanced_quantile_dynamics",
)
class TsExtremogram(SeriesOperator):
    """极值自相关 excess dependence（大跌之后是否继续 cluster）。

    ``P(E_{t+lag}=1 | E_t=1) - P(E)``：给定 t 时处于极值状态，t+lag 时再次
    处于极值状态的超额概率（减掉基线超阈值概率）。正 = 极端状态聚簇；
    负 = 极端后回归；0 = 随机。与"尾部有多肥"不同——这是"极端事件出现后
    是否持续"。P1。
    """

    metadata = _metadata(
        "ts_extremogram",
        "极值 excess dependence P(E_{t+lag}|E_t) - P(E)。",
        ["x", "window", "quantile", "lag", "side", "fixed_threshold"],
        domain="price_volume",
        unit="probability",
        cost=4,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 120,
        quantile: float = 0.1,
        lag: int = 1,
        side: str = "lower",
        fixed_threshold: bool = False,
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        q = _check_q(quantile)
        lg = int(lag)
        _check_side(side)
        if lg < 1:
            raise ValueError("ts_extremogram requires lag >= 1")
        if w < lg + 2:
            raise ValueError("ts_extremogram requires window >= lag + 2")
        return frame_like(
            x,
            map_rolling(
                x.to_numpy(dtype=float),
                w,
                lambda c: _extremogram_excess_chunk(c, q, lg, side, fixed_threshold),
            ),
        )


def _cross_extremogram_chunk(
    tc: np.ndarray,
    sc: np.ndarray,
    tq: float,
    sq: float,
    lag: int,
    ts: str,
    ss: str,
    fixed_threshold: bool = False,
) -> float:
    Et = _hit_series(tc, tq, ts, demean=False, fixed_threshold=fixed_threshold)
    Es = _hit_series(sc, sq, ss, demean=False, fixed_threshold=fixed_threshold)
    m = Et.shape[0]
    if m <= lag + 1:
        return np.nan
    s_past = Es[: m - lag]  # source extreme at t-lag
    t_fut = Et[lag:]  # target extreme at t
    ok = np.isfinite(s_past) & np.isfinite(t_fut)
    if int(ok.sum()) < 3:
        return np.nan
    sp = s_past[ok]
    tf = t_fut[ok]
    denom = float(sp.sum())
    if denom <= 0:
        return np.nan
    cond = float(np.sum(sp * tf)) / denom
    base = float(np.nanmean(Et))
    return float(cond - base)


@register_operator(
    name="ts_cross_extremogram",
    category="quantile_dynamics",
    business_category="quantile_dynamics",
    canonical="ts_cross_extremogram",
    source="advanced_quantile_dynamics",
)
class TsCrossExtremogram(SeriesOperator):
    """跨序列极值溢出 excess dependence（行业极值后个股极值概率）。

    ``P(E_target_t=1 | E_source_{t-lag}=1) - P(E_target)``。例：行业进入
    极端下跌后，个股随后进入 extreme downside 的超额概率；或 turnover 极值 →
    volatility 极值、market 极值 → stock 极值。P1。
    """

    metadata = _metadata(
        "ts_cross_extremogram",
        "跨序列极值溢出 P(E_target_t|E_source_{t-lag}) - P(E_target)。",
        ["target", "source", "window", "target_q", "source_q", "lag", "target_side", "source_side", "fixed_threshold"],
        domain="price_volume",
        unit="probability",
        cost=5,
    )

    def _calculate_series(
        self,
        target: pd.DataFrame,
        source: pd.DataFrame,
        window: int = 120,
        target_q: float = 0.1,
        source_q: float = 0.1,
        lag: int = 1,
        target_side: str = "lower",
        source_side: str = "lower",
        fixed_threshold: bool = False,
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        tq = _check_q(target_q)
        sq = _check_q(source_q)
        lg = int(lag)
        _check_side(target_side)
        _check_side(source_side)
        if lg < 1:
            raise ValueError("ts_cross_extremogram requires lag >= 1")
        if w < lg + 2:
            raise ValueError("ts_cross_extremogram requires window >= lag + 2")
        return frame_like(
            target,
            map_pair_rolling(
                target.to_numpy(dtype=float),
                source.to_numpy(dtype=float),
                w,
                lambda a, b: _cross_extremogram_chunk(
                    a, b, tq, sq, lg, target_side, source_side, fixed_threshold
                ),
            ),
        )


def _extremal_decay_chunk(
    chunk: np.ndarray, q: float, side: str, max_lag: int, fixed_threshold: bool = False
) -> float:
    E = _hit_series(chunk, q, side, demean=False, fixed_threshold=fixed_threshold)
    m = E.shape[0]
    H = int(max_lag)
    if m <= H + 3:
        return np.nan
    base = float(np.nanmean(E))
    hs: list[float] = []
    es: list[float] = []
    for h in range(1, H + 1):
        y = E[: m - h]
        x = E[h:]
        ok = np.isfinite(y) & np.isfinite(x)
        if int(ok.sum()) < 3:
            continue
        yy = y[ok]
        xx = x[ok]
        denom = float(yy.sum())
        if denom <= 0:
            continue
        ex = float(np.sum(yy * xx)) / denom - base
        # R5 P1-40(c): SELECTION BIAS — only *positive* excess lags enter the
        # log-linear fit, so tau is estimated on a censored subset of lags and
        # the fitted decay is biased (overstates decay).  Kept as a Research-only
        # heuristic (the surface classification excludes it from production).
        if ex > 0.0:
            hs.append(float(h))
            es.append(float(np.log(ex + _EPS)))
    if len(hs) < 2:
        return np.nan
    hh = np.asarray(hs, dtype=float)
    ee = np.asarray(es, dtype=float)
    dh = hh - hh.mean()
    denom = float(np.sum(dh * dh))
    if denom <= _EPS:
        return np.nan
    slope = float(np.sum(dh * (ee - ee.mean()))) / denom
    if slope >= 0.0:
        return np.nan  # no decay (persistent or growing) -> tau undefined
    tau = -1.0 / slope
    return float(min(tau, 1000.0))


@register_operator(
    name="ts_extremal_dependence_decay",
    category="quantile_dynamics",
    business_category="quantile_dynamics",
    canonical="ts_extremal_dependence_decay",
    source="advanced_quantile_dynamics",
    status="experimental",
)
class TsExtremalDependenceDecay(SeriesOperator):
    """极值依赖的记忆长度（extremogram 指数衰减时间常数 tau）。

    对 ``h=1..max_lag`` 的 extremogram excess 序列拟合 ``Excess(h) ~ a·e^{-h/tau}``
    （对正 excess 做 log-linear 回归），返回 tau。大 tau = 极端状态持续很久；
    小 tau = 快速回归。与普通波动率持久性不同。R5 P1-40(c)：只有正 excess 的
    lag 进入 log 拟合，存在 selection bias（高估衰减/有偏 tau），故仅供
    Research（surface 分类已排除生产）。P2 / Research。
    """

    metadata = _metadata(
        "ts_extremal_dependence_decay",
        "极值依赖指数衰减时间常数 tau（extremogram decay；仅正 excess 拟合，有 selection bias）。",
        ["x", "window", "quantile", "side", "max_lag", "fixed_threshold"],
        domain="price_volume",
        unit="time",
        cost=5,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 120,
        quantile: float = 0.1,
        side: str = "lower",
        max_lag: int = 5,
        fixed_threshold: bool = False,
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        q = _check_q(quantile)
        _check_side(side)
        H = int(max_lag)
        if H < 2:
            raise ValueError("ts_extremal_dependence_decay requires max_lag >= 2")
        if w < H + 4:
            raise ValueError("ts_extremal_dependence_decay requires window >= max_lag + 4")
        return frame_like(
            x,
            map_rolling(
                x.to_numpy(dtype=float),
                w,
                lambda c: _extremal_decay_chunk(c, q, side, H, fixed_threshold),
            ),
        )


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({
            "ts_quantilogram",
            "ts_cross_quantilogram",
            "ts_extremogram",
            "ts_cross_extremogram",
        })
    _surface.RESEARCH_ONLY_CANONICALS = frozenset(
        set(_surface.RESEARCH_ONLY_CANONICALS)
        | {
            "ts_quantile_crossing_spectral_concentration",
            "ts_extremal_dependence_decay",
        }
    )
    for _canon in (
        "ts_quantilogram",
        "ts_cross_quantilogram",
        "ts_quantile_crossing_spectral_concentration",
        "ts_extremogram",
        "ts_cross_extremogram",
        "ts_extremal_dependence_decay",
    ):
        register_polars_udf(_canon)


_register_surface()
