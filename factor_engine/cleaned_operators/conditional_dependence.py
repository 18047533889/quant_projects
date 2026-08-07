# -*- coding: utf-8 -*-
"""Conditional / band dependence operators (2026-08 market language, P2).

* ``ts_conditional_transfer_entropy`` — transfer entropy conditioned on a third
  series: ``I(T_{s+lag}; S_s | T_s, C_s)`` in nats.  Where plain TE says "S
  predicts T", CTE asks whether that predictive power survives *given* the
  conditioning state (regime, market, valuation...).
* ``ts_modwt_band_corr`` — correlation between the level-``band`` Haar MODWT
  detail components of two series over a window (band-limited co-movement,
  e.g. does turnover co-move with returns only on the weekly band?).

Both are deterministic and prefix-causal.  The MODWT uses a causal boundary
(``a[-1] = 0``), so no future sample leaks into the coefficients.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like, map_pair_rolling, register_polars_udf

_ALPHA = 0.5  # Jeffreys smoothing, matching the plain-TE kernel.
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
        category="conditional_dependence",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "conditional_dependence", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic", *extra_tags,
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.array([-np.inf, np.inf], dtype=float)
    if n_bins < 2:
        n_bins = 2
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size == 1:
        edges = np.array([edges[0] - 1.0, edges[0] + 1.0])
    elif edges.size < n_bins + 1:
        lo, hi = float(edges[0]), float(edges[-1])
        if hi - lo <= _EPS:
            lo, hi = lo - 1.0, hi + 1.0
        edges = np.linspace(lo, hi, n_bins + 1)
        edges[0] = -np.inf
        edges[-1] = np.inf
    return edges


def _conditional_te_window(
    tw: np.ndarray, sw: np.ndarray, cw: np.ndarray, bins: int, lag: int, min_transitions: int
) -> float:
    """Conditional transfer entropy I(T_{s+lag}; S_s | T_s, C_s) for one window."""
    valid = np.isfinite(tw) & np.isfinite(sw) & np.isfinite(cw)
    idx = np.flatnonzero(valid)
    if idx.size < max(lag + 2, min_transitions):
        return np.nan
    xs = tw[idx[:-lag]]  # t_s
    ys = sw[idx[:-lag]]  # s_s
    cs = cw[idx[:-lag]]  # c_s
    xn = tw[idx[lag:]]  # t_{s+lag}
    if np.unique(xs).size < 2 or np.unique(ys).size < 2 or np.unique(cs).size < 2:
        return np.nan
    n = xs.shape[0]
    x_edges = _quantile_edges(xs, bins)
    y_edges = _quantile_edges(ys, bins)
    c_edges = _quantile_edges(cs, bins)
    xb = np.clip(np.digitize(xs, x_edges) - 1, 0, bins - 1).astype(np.int64)
    xnb = np.clip(np.digitize(xn, x_edges) - 1, 0, bins - 1).astype(np.int64)
    yb = np.clip(np.digitize(ys, y_edges) - 1, 0, bins - 1).astype(np.int64)
    cb = np.clip(np.digitize(cs, c_edges) - 1, 0, bins - 1).astype(np.int64)
    joint = np.zeros((bins, bins, bins, bins), dtype=np.float64)  # [t', t, s, c]
    for k in range(n):
        joint[xnb[k], xb[k], yb[k], cb[k]] += 1.0
    joint += _ALPHA
    joint /= joint.sum()
    # Marginals (axes: 0=t', 1=t, 2=s, 3=c).
    p4 = joint
    ptc = joint.sum(axis=(0, 2))  # [t, c]
    ptptc = joint.sum(axis=2)  # [t', t, c]
    ptsc = joint.sum(axis=0)  # [t, s, c]
    te = 0.0
    for i in range(bins):  # t'
        for j in range(bins):  # t
            for k in range(bins):  # s
                for l in range(bins):  # c
                    p = p4[i, j, k, l]
                    if p <= _EPS:
                        continue
                    p_denom = ptptc[i, j, l] * ptsc[j, k, l]
                    if p_denom <= _EPS:
                        continue
                    te += p * (np.log(p) + np.log(ptc[j, l]) - np.log(p_denom))
    return float(max(te, 0.0))


def _tri_rolling(a: np.ndarray, b: np.ndarray, c: np.ndarray, w: int, fn) -> np.ndarray:
    rows, cols = a.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for col in range(cols):
        for r in range(rows):
            lo = max(0, r - w + 1)
            out[r, col] = fn(a[lo : r + 1, col], b[lo : r + 1, col], c[lo : r + 1, col])
    return out


@register_operator(
    name="ts_conditional_transfer_entropy",
    category="conditional_dependence",
    business_category="conditional_dependence",
    canonical="ts_conditional_transfer_entropy",
    source="conditional_dependence",
    status="experimental",
)
class TsConditionalTransferEntropy(SeriesOperator):
    """条件传递熵 I(T_{s+lag}; S_s | T_s, C_s)（nats）。

    在给定条件序列 C（regime/market/valuation...）状态的前提下，S 对 T 的方向
    信息传递。用于检验预测关系是否在控制状态变量后依然存在（如 turnover→return
    的关系在估值高/低状态下是否显著不同）。分位数分箱 + Jeffreys 平滑，
    常量/退化状态 fail-closed。P2 / Research。
    """

    metadata = _metadata(
        "ts_conditional_transfer_entropy",
        "条件传递熵 I(T_{s+lag}; S_s | T_s, C_s) nats。",
        ["target", "source", "condition", "window", "bins", "lag", "min_transitions"],
        domain="price_volume",
        unit="nats",
        cost=7,
    )

    def _calculate_series(
        self,
        target: pd.DataFrame,
        source: pd.DataFrame,
        condition: pd.DataFrame,
        window: int = 60,
        bins: int = 3,
        lag: int = 1,
        min_transitions: Any = None,
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        nb = int(bins)
        lg = int(lag)
        if not (2 <= nb <= 5):
            raise ValueError("ts_conditional_transfer_entropy requires 2 <= bins <= 5")
        if lg < 1:
            raise ValueError("ts_conditional_transfer_entropy requires lag >= 1")
        if w < lg + 2:
            raise ValueError("ts_conditional_transfer_entropy requires window >= lag + 2")
        if min_transitions is None:
            mt = max(80, 6 * nb * nb * nb)
        else:
            mt = max(lg + 2, int(min_transitions))
        return frame_like(
            target,
            _tri_rolling(
                target.to_numpy(dtype=float),
                source.to_numpy(dtype=float),
                condition.to_numpy(dtype=float),
                w,
                lambda a, b, c: _conditional_te_window(a, b, c, nb, lg, mt),
            ),
        )


def _haar_detail(v: np.ndarray, level: int, band: int) -> np.ndarray:
    """Causal Haar MODWT detail coefficients at the requested band."""
    a = np.asarray(v, dtype=float)
    for j in range(1, level + 1):
        prev = np.concatenate([[0.0], a[:-1]])  # causal boundary (no lookahead)
        detail = (prev - a) / np.sqrt(2.0)
        smooth = (prev + a) / np.sqrt(2.0)
        a = smooth
        if j == band:
            return detail
    return a


def _modwt_band_corr_chunk(xc: np.ndarray, yc: np.ndarray, level: int, band: int) -> float:
    if xc.shape[0] < level + 4:
        return np.nan
    dx = _haar_detail(xc, level, band)
    dy = _haar_detail(yc, level, band)
    ok = np.isfinite(dx) & np.isfinite(dy)
    if int(ok.sum()) < level + 3:
        return np.nan
    xx = dx[ok]
    yy = dy[ok]
    vx = float(np.var(xx))
    vy = float(np.var(yy))
    if vx <= _EPS or vy <= _EPS:
        return np.nan
    return float(np.corrcoef(xx, yy)[0, 1])


@register_operator(
    name="ts_modwt_band_corr",
    category="conditional_dependence",
    business_category="conditional_dependence",
    canonical="ts_modwt_band_corr",
    source="conditional_dependence",
    status="experimental",
)
class TsModwtBandCorr(SeriesOperator):
    """Haar MODWT 频带相关（level-band 细节系数窗口相关）。

    用因果边界 Haar MODWT 把 x、y 分解到多分辨率频带，取第 ``band`` 层的细节
    系数做窗口相关。回答"两序列是否只在某个频带（如周线/月线）协同"。PIT 安全
    （因果边界不引入未来）。P2 / Research。
    """

    metadata = _metadata(
        "ts_modwt_band_corr",
        "Haar MODWT 频带细节系数窗口相关 Corr(D_band x, D_band y)。",
        ["x", "y", "window", "level", "band"],
        domain="price_volume",
        unit="corr",
        cost=6,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        window: int = 120,
        level: int = 3,
        band: int = 1,
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        lv = int(level)
        bd = int(band)
        if lv < 1:
            raise ValueError("ts_modwt_band_corr requires level >= 1")
        if not (1 <= bd <= lv):
            raise ValueError("ts_modwt_band_corr requires 1 <= band <= level")
        if w < lv + 8:
            raise ValueError("ts_modwt_band_corr requires window >= level + 8")
        return frame_like(
            x,
            map_pair_rolling(
                x.to_numpy(dtype=float),
                y.to_numpy(dtype=float),
                w,
                lambda a, b: _modwt_band_corr_chunk(a, b, lv, bd),
            ),
        )


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.RESEARCH_ONLY_CANONICALS = frozenset(
        set(_surface.RESEARCH_ONLY_CANONICALS)
        | {"ts_conditional_transfer_entropy", "ts_modwt_band_corr"}
    )
    for _canon in ("ts_conditional_transfer_entropy", "ts_modwt_band_corr"):
        register_polars_udf(_canon)


_register_surface()
