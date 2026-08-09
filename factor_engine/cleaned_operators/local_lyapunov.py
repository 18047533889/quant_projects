# -*- coding: utf-8 -*-
"""Local nonlinear divergence (2026-08 V3, P2 research).

``ts_local_lyapunov_exponent`` measures how fast nearby phase-space trajectories
diverge via a Takens embedding.  It is strictly prefix-causal (anchors and their
neighbours stay ``<= t``), deterministic (nearest neighbour by stable argmin,
Theiler window avoids temporal neighbours) and should be read as a *nonlinear
local-divergence feature* — never as proof that markets are deterministic chaos.

R14 P1/P2:
* UNIT: the operator estimates a per-forward-step exponent — the slope of
  ``L(k) = mean_i[log d_i(k) - log d_i(0)]`` against the step index ``k`` — so
  the declared unit is ``rate`` (per step), not ``ratio``.
* NaN policy: interior NaN rows are dropped before embedding so a sparse gap
  does not poison the whole window, but the *current* row must be finite (a NaN
  today never emits a stale value from an older window), and Theiler exclusion
  uses the ORIGINAL physical time positions of the surviving points (never the
  compressed ordinal index after dropping rows).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="nonlinear_dynamics",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "nonlinear_dynamics", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:price_volume",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _embedding_matrix(series: np.ndarray, tau: int, dim: int) -> np.ndarray:
    span = (dim - 1) * tau
    n = series.shape[0]
    if n <= span:
        return np.empty((0, dim))
    X = np.stack([series[span - d * tau : n - d * tau] for d in range(dim - 1, -1, -1)], axis=1)
    return X


def _lyapunov_series(series: np.ndarray, window: int, tau: int, dim: int, horizon: int, min_anchors: int) -> np.ndarray:
    n = series.shape[0]
    w = max(2, int(window))
    th = max(1, dim * tau)
    H = max(1, int(horizon))
    ma = max(1, int(min_anchors))
    out = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - w + 1)
        chunk = series[lo : t + 1]
        if chunk.shape[0] < w:
            continue
        # R14 P1/P2 (current-row fail-closed): the trailing window requires the
        # CURRENT observation.  A NaN at row ``t`` must not be silently dropped
        # and replaced by a value computed on the older window — that is a stale
        # "yesterday" factor leaking onto today.
        if not np.isfinite(series[t]):
            continue
        finite = np.isfinite(chunk)
        f = chunk[finite]
        # phys[k] = the ORIGINAL (within-window) time position of the k-th
        # surviving finite point.  Used below for real-time Theiler exclusion.
        phys = np.nonzero(finite)[0]
        X = _embedding_matrix(f, tau, dim)
        if X.shape[0] < 2:
            continue
        # Robust-normalise each embedding dimension over the finite window.
        med = np.median(X, axis=0)
        mad = 1.4826 * np.median(np.abs(X - med), axis=0)
        scale = np.where(mad > _EPS, mad, np.std(X, axis=0))
        if np.any(~np.isfinite(scale)) or np.any(scale <= _EPS):
            continue
        Z = (X - med) / scale
        z_len = Z.shape[0]
        m_f = int(f.shape[0])
        # Anchors must fit [i, i+H] inside the embedded finite window end.
        last_anchor = m_f - 1 - H
        if last_anchor < 1:
            continue
        if z_len < last_anchor + 1:
            continue
        logs: list[np.ndarray] = []
        n_anchors = 0
        for i in range(last_anchor + 1):
            d0 = np.sqrt(np.sum((Z - Z[i]) ** 2, axis=1))
            d0[i] = np.inf
            # R14 P1/P2 (Theiler in PHYSICAL time): after NaN rows are dropped,
            # the compressed ordinal ``abs(j - i)`` no longer measures real time.
            # Two points adjacent in the compressed index can be far apart in
            # physical time (and vice versa).  Use the original positions of the
            # surviving points so a NaN gap neither re-pairs temporal neighbours
            # nor splits a physically-distant pair.
            theiler = np.abs(phys[:z_len] - phys[i]) <= th
            d0[theiler] = np.inf
            j = int(np.argmin(d0))
            if not np.isfinite(d0[j]) or d0[j] <= _EPS:
                continue
            d0j = float(d0[j])
            if i + H >= z_len or j + H >= z_len:
                continue
            div = np.zeros(H + 1)
            for k in range(H + 1):
                dk = float(np.sqrt(np.sum((Z[i + k] - Z[j + k]) ** 2)))
                div[k] = np.log(dk + _EPS) - np.log(d0j + _EPS)
            logs.append(div)
            n_anchors += 1
        if n_anchors < ma:
            continue
        L = np.mean(np.stack(logs), axis=0)
        ks = np.arange(H + 1, dtype=float)
        var_k = float(np.sum((ks - ks.mean()) ** 2))
        if var_k <= _EPS:
            continue
        lam = float(np.sum((ks - ks.mean()) * (L - L.mean())) / var_k)
        out[t] = lam
    return out


@register_operator(
    name="ts_local_lyapunov_exponent",
    category="nonlinear_dynamics",
    business_category="nonlinear_dynamics",
    canonical="ts_local_lyapunov_exponent",
    source="local_lyapunov",
    status="experimental",
)
class TsLocalLyapunovExponent(SeriesOperator):
    """Takens 嵌入下相邻轨迹的局部发散速率 λ（Research）。

    对每个 anchor 找 Theiler 窗口之外的最近历史邻居，测未来 k 步对数距离增长
    ``L(k)=mean_i[log d_i(k)-log d_i(0)]`` 的斜率 λ。λ>0 = 附近状态快速发散
    （local predictability 低）。单位 ``rate``（每步指数，非无量纲 ratio）。
    金融数据噪声大/样本短/非平稳，勿解读为"确定性混沌"，仅作非线性局部发散
    特征。P2 / Research。
    R14 P1/P2: 当前行 NaN -> NaN（绝不输出陈旧值）；内部 NaN 行剔除后 Theiler
    排除使用原始物理时间距离。
    """

    metadata = _metadata(
        "ts_local_lyapunov_exponent",
        "局部 Lyapunov 指数 λ（Takens 嵌入最近邻居发散斜率，per-step rate）。",
        ["x", "window", "tau", "embedding_dim", "horizon", "min_anchors"],
        unit="rate",
        cost=8,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 120,
        tau: int = 1,
        embedding_dim: int = 3,
        horizon: int = 5,
        min_anchors: int = 3,
        **_: Any,
    ) -> pd.DataFrame:
        tau_i = max(1, int(tau))
        dim = int(embedding_dim)
        if not 2 <= dim <= 6:
            raise ValueError("ts_local_lyapunov_exponent requires embedding_dim in [2, 6]")
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            out[:, c] = _lyapunov_series(xv[:, c], window, tau_i, dim, horizon, min_anchors)
        return frame_like(x, out)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_research_only({"ts_local_lyapunov_exponent"})


_register_surface()
