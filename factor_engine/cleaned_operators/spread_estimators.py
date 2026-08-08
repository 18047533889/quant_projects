# -*- coding: utf-8 -*-
"""Bid-ask spread estimators from OHLC (2026-08 market language, P1/P2).

No L2 queue data is assumed — only daily OHLC.  Two classic estimators:

* ``ohlc_corwin_schultz_spread`` — Corwin-Schultz two-day high/low estimator
  that separates the volatility contribution from the spread contribution
  (P1).  Negative raw spreads are set to 0 (as in the original paper), then
  smoothed over ``smooth_window``.
* ``ts_roll_effective_spread``  — Roll's effective spread from the first-order
  covariance of log-price changes: ``S = 2 sqrt(max(-Cov(dx_t, dx_{t-1}), 0))``
  (P2; note the covariance is frequently non-negative, which yields 0).

Both are prefix-causal (a spread at ``t`` uses rows ``<= t``), deterministic
and fail-closed to NaN when the required pairs are invalid.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like, register_polars_udf

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
        category="spread_estimator",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "spread_estimator", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic", *extra_tags,
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _cs_pair_spread(h1: float, h2: float, l1: float, l2: float) -> float:
    """Corwin-Schultz spread (fraction of price) for the pair (t-1, t).

    Per the CS (2000) definition:
        beta  = E[ (ln H_t/L_t)^2 + (ln H_{t-1}/L_{t-1})^2 ]   # SUM of single-day ranges
        gamma = E[ (ln H_{t,t-1}/L_{t,t-1})^2 ]                # aggregate two-day range
        alpha = (sqrt(2 beta) - sqrt(beta)) / c - sqrt(gamma / c),  c = 3 - 2 sqrt(2)
    The earlier implementation had beta/gamma reversed (two-day range as beta,
    single-day sum as gamma), which is not the standard estimator — under a
    pure-spread construction it returns 0 instead of the spread.
    """
    if h1 <= l1 or h2 <= l2:
        return np.nan
    H = max(h1, h2)
    L = min(l1, l2)
    if H <= L:
        return np.nan
    c = 3.0 - 2.0 * np.sqrt(2.0)  # ~0.171573
    beta = np.log(h1 / l1) ** 2.0 + np.log(h2 / l2) ** 2.0
    gamma = np.log(H / L) ** 2.0
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / c - np.sqrt(gamma / c)
    if alpha <= 0.0:
        return 0.0  # negative alpha -> zero spread (as in the paper)
    return float(2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha)))


def _rolling_mean_trailing(v: np.ndarray, w: int) -> np.ndarray:
    rows = v.shape[0]
    out = np.full(rows, np.nan, dtype=float)
    cum = np.cumsum(np.where(np.isfinite(v), v, 0.0))
    cnt = np.cumsum(np.isfinite(v).astype(float))
    for r in range(rows):
        lo = max(0, r - w + 1)
        c = cnt[r] - (cnt[lo - 1] if lo > 0 else 0.0)
        if c < 1:
            continue
        s = cum[r] - (cum[lo - 1] if lo > 0 else 0.0)
        out[r] = s / c
    return out


def _cs_spread_series(hv: np.ndarray, lv: np.ndarray, smooth: int) -> np.ndarray:
    rows, cols = hv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        raw = np.full(rows, np.nan, dtype=float)
        for r in range(1, rows):
            h1, h2 = float(hv[r - 1, c]), float(hv[r, c])
            l1, l2 = float(lv[r - 1, c]), float(lv[r, c])
            if not (np.isfinite(h1) and np.isfinite(h2) and np.isfinite(l1) and np.isfinite(l2)):
                continue
            raw[r] = _cs_pair_spread(h1, h2, l1, l2)
        out[:, c] = _rolling_mean_trailing(raw, smooth)
    return out


@register_operator(
    name="ohlc_corwin_schultz_spread",
    category="spread_estimator",
    business_category="spread_estimator",
    canonical="ohlc_corwin_schultz_spread",
    source="spread_estimators",
)
class OhlcCorwinSchultzSpread(SeriesOperator):
    """Corwin-Schultz 两日高低价 bid-ask spread 估计（无 L2 数据）。

    用单日 high-low range 与两日聚合 high-low range 分离波动率与 spread 贡献，
    输出 spread 占价格比例，再按 ``smooth_window`` 平滑。负 alpha → 0（论文
    原处理）。A 股无 L2 队列数据时的流动性代理。PIT 安全。P1。
    """

    metadata = _metadata(
        "ohlc_corwin_schultz_spread",
        "Corwin-Schultz 两日高低价 spread 估计（平滑后）。",
        ["high", "low", "smooth_window"],
        domain="price_volume",
        unit="fraction",
        cost=4,
    )

    def _calculate_series(
        self, high: pd.DataFrame, low: pd.DataFrame, smooth_window: int = 5, **_: Any
    ) -> pd.DataFrame:
        sm = int(smooth_window)
        if sm < 1:
            raise ValueError("ohlc_corwin_schultz_spread requires smooth_window >= 1")
        return frame_like(
            high,
            _cs_spread_series(high.to_numpy(dtype=float), low.to_numpy(dtype=float), sm),
        )


def _roll_spread_series(xv: np.ndarray, window: int) -> np.ndarray:
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        x = xv[:, c]
        with np.errstate(invalid="ignore", divide="ignore"):
            dx = np.diff(np.log(x), prepend=np.nan)
        for r in range(rows):
            lo = max(1, r - window + 1)
            seg = dx[lo : r + 1]
            if seg.size < 4:
                continue
            a = seg[1:]
            b = seg[:-1]
            ok = np.isfinite(a) & np.isfinite(b)
            if int(ok.sum()) < 4:
                continue
            cov = float(np.cov(a[ok], b[ok], ddof=1)[0, 1])
            if not np.isfinite(cov):
                continue
            out[r, c] = 2.0 * np.sqrt(max(-cov, 0.0))
    return out


@register_operator(
    name="ts_roll_effective_spread",
    category="spread_estimator",
    business_category="spread_estimator",
    canonical="ts_roll_effective_spread",
    source="spread_estimators",
    status="experimental",
)
class TsRollEffectiveSpread(SeriesOperator):
    """Roll 有效 spread（log 价格差分一阶协方差）。

    ``S = 2 sqrt(max(-Cov(dx_t, dx_{t-1}), 0))``，x 视为价格序列（内部取 log
    差分）。负协方差条件使其经常为 0（流动性极好或协方差为正），这是 Roll
    估计器的固有特性。A 股无 L2 时的 cheap liquidity transform。P2 / Research。
    """

    metadata = _metadata(
        "ts_roll_effective_spread",
        "Roll 有效 spread 2 sqrt(max(-Cov(dx_t, dx_{t-1}), 0))。",
        ["x", "window"],
        domain="price_volume",
        unit="fraction",
        cost=4,
    )

    def _calculate_series(self, x: pd.DataFrame, window: int = 20, **_: Any) -> pd.DataFrame:
        w = int(window)
        if w < 5:
            raise ValueError("ts_roll_effective_spread requires window >= 5")
        return frame_like(x, _roll_spread_series(x.to_numpy(dtype=float), w))


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | {"ohlc_corwin_schultz_spread"}
    )
    _surface.RESEARCH_ONLY_CANONICALS = frozenset(
        set(_surface.RESEARCH_ONLY_CANONICALS) | {"ts_roll_effective_spread"}
    )
    for _canon in ("ohlc_corwin_schultz_spread", "ts_roll_effective_spread"):
        register_polars_udf(_canon)


_register_surface()
