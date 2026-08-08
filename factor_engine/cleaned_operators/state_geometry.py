# -*- coding: utf-8 -*-
"""Local state-geometry and ordinal time-asymmetry operators (2026-08 V2/V3).

* ``ts_state_density``                  — Epanechnikov density of the historical
  state space around the current value (price/variation congestion, not holder
  cost mass).  (P1)
* ``ts_ordinal_irreversibility``        — JS divergence between forward and
  backward ordinal-pattern distributions (time arrow strength).  (P1)
* ``ts_multiscale_permutation_entropy_slope`` — slope of normalized permutation
  entropy vs log-scale after coarse graining (P2 research).

Ordinal ties are *dropped* (never jittered) so limit-locked flat patches cannot
fabricate a spurious ordinal pattern.  All kernels are deterministic and
prefix-causal.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12
_LN2 = float(np.log(2.0))


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="state_geometry",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "state_geometry", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:price_volume",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return frame_like(template, values)


def _column_map(xv: np.ndarray, fn) -> np.ndarray:
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        out[:, c] = fn(xv[:, c])
    return out


# ---------------------------------------------------------------------------
# ts_state_density
# ---------------------------------------------------------------------------

def _state_density_series(series: np.ndarray, window: int, bandwidth: float, min_periods: int) -> np.ndarray:
    n = series.shape[0]
    w = max(2, int(window))
    bw = max(1e-6, float(bandwidth))
    mp = max(1, int(min_periods))
    out = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - w)
        past = series[lo:t]                       # strictly past [t-W, t-1]
        cur = series[t]
        finite = past[np.isfinite(past)]
        if finite.size < mp or not np.isfinite(cur):
            continue
        med = float(np.median(finite))
        mad = 1.4826 * float(np.median(np.abs(finite - med)))
        scale = mad if mad > _EPS else float(np.std(finite))
        if scale <= _EPS:
            # Degenerate history: a point mass has no kernel density — the
            # previous arbitrary 1.0 was discontinuous with the kernel's peak
            # (P1-05).  Fail closed to NaN instead.
            out[t] = np.nan
            continue
        h = bw * scale
        u = (finite - cur) / (h + _EPS)
        kern = 0.75 * (1.0 - u * u) * (np.abs(u) <= 1.0)
        # Proper kernel-density normalisation: f̂(x) = (1/n) Σ K(u)/h, NOT the
        # raw kernel mass mean(K) — otherwise this is a "local proximity" score,
        # not a density (P1-05).
        out[t] = float(np.mean(kern)) / (h + _EPS)
    return out


@register_operator(
    name="ts_state_density",
    category="state_geometry",
    business_category="state_geometry",
    canonical="ts_state_density",
    source="state_geometry",
)
class TsStateDensity(SeriesOperator):
    """当前值附近历史状态空间的 Epanechnikov 密度。

    严格过去窗口 ``R_t = {x_{t-W},...,x_{t-1}}`` 提供状态空间，当前 ``x_t`` 仅作
    query。尺度用 ``s = 1.4826*MAD(R_t)``、带宽 ``h = bandwidth*s``，输出
    ``(1/N) Σ 0.75(1-u²) I(|u|≤1)``。高 = 当前处于历史拥挤区；低 = 历史状态真空。
    与筹码 ``near_cost_mass``（持仓成本空间）不同：这里测任意变量自己的历史状态空间。
    """

    metadata = _metadata(
        "ts_state_density",
        "当前状态的历史 Epanechnikov 密度（拥挤度，MAD 尺度）。",
        ["x", "window", "bandwidth", "min_periods"],
        unit="density",
        cost=3,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, bandwidth: float = 1.0, min_periods: int = 5, **_: Any
    ) -> pd.DataFrame:
        return _frame_like(
            x,
            _column_map(x.to_numpy(dtype=float), lambda s: _state_density_series(s, window, bandwidth, min_periods)),
        )


# ---------------------------------------------------------------------------
# Ordinal patterns (ties dropped) shared by irreversibility + multiscale slope.
# ---------------------------------------------------------------------------

def _permutation_index(perm: np.ndarray) -> int:
    """Lehmer code of a permutation of 0..n-1 -> index in [0, n!)."""
    n = perm.shape[0]
    index = 0
    for i in range(n):
        smaller = int(np.sum(perm[i + 1 :] < perm[i]))
        index += smaller * math.factorial(n - 1 - i)
    return index


def _ordinal_patterns(values: np.ndarray, order: int, delay: int) -> list[int]:
    """Ordinal permutation indices; embeddings with ties/NaN are dropped."""
    codes: list[int] = []
    embed_len = (order - 1) * delay + 1
    n = len(values)
    for i in range(n - embed_len + 1):
        idx = [i + d * delay for d in range(order)]
        vals = values[idx]
        if not np.isfinite(vals).all():
            continue
        if np.unique(vals).size < order:
            continue  # exact tie -> drop, never jitter
        perm = np.argsort(vals, kind="stable")
        pattern = np.argsort(perm, kind="stable")
        codes.append(_permutation_index(pattern))
    return codes


def _distribution(codes: list[int], n_patterns: int) -> np.ndarray:
    counts = np.bincount(codes, minlength=n_patterns).astype(np.float64)
    total = float(counts.sum())
    if total <= _EPS:
        return counts
    return counts / total


def _permutation_entropy(values: np.ndarray, order: int, delay: int) -> float:
    codes = _ordinal_patterns(values, order, delay)
    if len(codes) < 2:
        return np.nan
    n_patterns = math.factorial(order)
    p = _distribution(codes, n_patterns)
    p = p[p > 0.0]
    h = -float(np.sum(p * np.log(p)))
    return float(h / math.log(n_patterns))


# ---------------------------------------------------------------------------
# ts_ordinal_irreversibility
# ---------------------------------------------------------------------------

def _kl(p: np.ndarray, q: np.ndarray) -> float:
    ok = (p > 0.0) & (q > 0.0)
    if not ok.any():
        return 0.0
    pp = p[ok]
    qq = q[ok]
    return float(np.sum(pp * np.log(pp / qq)))


def _irreversibility_series(series: np.ndarray, window: int, order: int, delay: int, min_patterns: int) -> np.ndarray:
    n = series.shape[0]
    w = max(2, int(window))
    mp = max(2, int(min_patterns))
    n_patterns = math.factorial(order)
    out = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - w + 1)
        chunk = series[lo : t + 1]
        if chunk.shape[0] < w:
            continue
        fwd = _ordinal_patterns(chunk, order, delay)
        if len(fwd) < mp:
            continue
        rvs = _ordinal_patterns(chunk[::-1], order, delay)
        if len(rvs) < mp:
            continue
        pf = _distribution(fwd, n_patterns)
        pr = _distribution(rvs, n_patterns)
        m = 0.5 * (pf + pr)
        js = 0.5 * _kl(pf, m) + 0.5 * _kl(pr, m)
        js = max(js, 0.0)
        out[t] = float(np.sqrt(js / _LN2))
    return out


@register_operator(
    name="ts_ordinal_irreversibility",
    category="state_geometry",
    business_category="state_geometry",
    canonical="ts_ordinal_irreversibility",
    source="state_geometry",
)
class TsOrdinalIrreversibility(SeriesOperator):
    """序数模式时间不可逆性 ``IR = sqrt(JS(P_f || P_r)/ln2)`` ∈ [0,1]。

    窗口正向与反向的序数排列模式分布之间的 JS 距离。0 = 正反向统计结构几乎
    相同；高 = 强时间箭头 / 非线性方向动力学。与 ``ts_permutation_entropy``
    （路径复杂度）互补。相等值（一字涨跌停）embedding 一律丢弃，不做 jitter。
    """

    metadata = _metadata(
        "ts_ordinal_irreversibility",
        "序数模式时间不可逆性 sqrt(JS/ln2)（[0,1]，正反向动力学差异）。",
        ["x", "window", "order", "delay", "min_patterns"],
        unit="ratio",
        cost=5,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 60, order: int = 3, delay: int = 1, min_patterns: int = 5, **_: Any
    ) -> pd.DataFrame:
        ord_ = int(order)
        if not 2 <= ord_ <= 6:
            raise ValueError("ts_ordinal_irreversibility requires order in [2, 6]")
        dl = int(delay)
        if dl < 1:
            raise ValueError("ts_ordinal_irreversibility requires delay >= 1")
        return _frame_like(
            x,
            _column_map(
                x.to_numpy(dtype=float),
                lambda s: _irreversibility_series(s, window, ord_, dl, min_patterns),
            ),
        )


# ---------------------------------------------------------------------------
# ts_multiscale_permutation_entropy_slope (P2 research)
# ---------------------------------------------------------------------------

_SCALES = (1, 2, 4, 8)


def _multiscale_slope_series(series: np.ndarray, window: int, order: int, min_patterns: int) -> np.ndarray:
    n = series.shape[0]
    w = max(2, int(window))
    mp = max(2, int(min_patterns))
    out = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - w + 1)
        chunk = series[lo : t + 1]
        if chunk.shape[0] < w:
            continue
        pts: list[tuple[float, float]] = []
        for s in _SCALES:
            n_seg = int(len(chunk) // s)
            if n_seg < order + 1:
                continue
            coarse = chunk[: n_seg * s].reshape(n_seg, s).mean(axis=1)
            codes = _ordinal_patterns(coarse, order, 1)
            if len(codes) < mp:
                continue
            n_patterns = math.factorial(order)
            p = _distribution(codes, n_patterns)
            p = p[p > 0.0]
            h = -float(np.sum(p * np.log(p))) / math.log(n_patterns)
            pts.append((math.log(s), h))
        if len(pts) < 3:
            continue
        xs = np.array([x for x, _ in pts])
        ys = np.array([y for _, y in pts])
        var_x = float(np.sum((xs - xs.mean()) ** 2))
        if var_x <= _EPS:
            continue
        slope = float(np.sum((xs - xs.mean()) * (ys - ys.mean())) / var_x)
        out[t] = slope
    return out


@register_operator(
    name="ts_multiscale_permutation_entropy_slope",
    category="state_geometry",
    business_category="state_geometry",
    canonical="ts_multiscale_permutation_entropy_slope",
    source="state_geometry",
    status="experimental",
)
class TsMultiscalePermutationEntropySlope(SeriesOperator):
    """多尺度排列熵斜率：非重叠粗粒化后 PE(s) 对 log(s) 回归的斜率。

    固定尺度 ``[1,2,4,8]``。正 = 短期结构强、尺度拉长后变随机；负 = 短周期
    噪声大、长周期反而出现结构。P2 / Research。
    """

    metadata = _metadata(
        "ts_multiscale_permutation_entropy_slope",
        "多尺度排列熵斜率 b（PE vs log-scale 回归）。",
        ["x", "window", "order", "min_patterns"],
        unit="ratio",
        cost=7,
    )

    def _calculate_series(
        self, x: pd.DataFrame, window: int = 120, order: int = 3, min_patterns: int = 5, **_: Any
    ) -> pd.DataFrame:
        ord_ = int(order)
        if not 2 <= ord_ <= 5:
            raise ValueError("ts_multiscale_permutation_entropy_slope requires order in [2, 5]")
        return _frame_like(
            x,
            _column_map(
                x.to_numpy(dtype=float),
                lambda s: _multiscale_slope_series(s, window, ord_, min_patterns),
            ),
        )


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS)
        | {"ts_state_density", "ts_ordinal_irreversibility"}
    )
    _surface.RESEARCH_ONLY_CANONICALS = frozenset(
        set(_surface.RESEARCH_ONLY_CANONICALS)
        | {"ts_multiscale_permutation_entropy_slope"}
    )


_register_surface()
