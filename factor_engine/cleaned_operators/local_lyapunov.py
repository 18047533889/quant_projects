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

M-150 (divergence horizon): the DEFAULT path measures divergence over COMPRESSED
ordinal steps — ``Z[i+k]`` / ``Z[j+k]`` advance the compressed index, so after an
interior NaN drop ``k`` no longer spans ``k`` physical bars.  ``physical_time=True``
switches the divergence horizon to the PHYSICAL clock: the k-th successor of each
trajectory is the surviving embedded point whose anchor sits exactly ``k`` physical
bars later (the physical index advances even over dropped rows) and the least-squares
slope is taken against physical elapsed bars.  The physical clock is the honest
interpretation of the module contract; the compressed path is preserved as the
default for backward compatibility.

M-151 (strict params): ``window``/``tau``/``embedding_dim``/``horizon``/
``min_anchors`` are validated strictly (never ``int()``-truncated or clamped);
a fractional, non-finite, bool or out-of-range value RAISES.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, ParamRole, ParamSpec, SeriesOperator, register_operator
from cleaned_operators.closure.strict_scalar import strict_bool, strict_int
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


def _physical_divergence(
    Z: np.ndarray,
    phys: np.ndarray,
    z_len: int,
    i: int,
    j: int,
    horizon: int,
    d0j: float,
) -> np.ndarray | None:
    """Divergence curve ``log d(k) - log d(0)`` against PHYSICAL elapsed bars.

    M-150: the compressed-ordinal path (``Z[i+k]`` / ``Z[j+k]``) redefines the
    step index when interior NaN rows are dropped — ``k`` compressed steps can
    span more than ``k`` physical bars, and the two trajectories' indices no
    longer advance in lockstep.  This path scans ``k`` PHYSICAL bars forward
    from each trajectory's anchor position, so the k-th successor of anchor
    ``i`` (physical position ``phys[i]``) and neighbour ``j`` (physical position
    ``phys[j]``) is the surviving embedded point whose anchor sits exactly
    ``k`` physical bars later — the physical index advances even across dropped
    rows.  A physical bar with no surviving point yields NaN for that ``k``.
    Returns None when no ``k`` in ``1..horizon`` has a finite successor on both
    trajectories (nothing to regress).
    """
    H = int(horizon)
    div = np.full(H + 1, np.nan)
    div[0] = 0.0
    # Map physical position -> compressed embedded index.  Only rows ``m < z_len``
    # are valid anchors/successors (the trailing ``span`` physical positions have
    # no complete future window).
    pos_to_idx = {int(phys[m]): m for m in range(z_len)}
    p = int(phys[i])
    q = int(phys[j])
    found = 0
    for k in range(1, H + 1):
        ip = pos_to_idx.get(p + k)
        iq = pos_to_idx.get(q + k)
        if ip is None or iq is None:
            continue
        dk = float(np.sqrt(np.sum((Z[ip] - Z[iq]) ** 2)))
        if not np.isfinite(dk) or dk <= _EPS:
            continue
        div[k] = np.log(dk + _EPS) - np.log(d0j + _EPS)
        found += 1
    if found == 0:
        return None
    return div


def _lyapunov_series(
    series: np.ndarray,
    window: int,
    tau: int,
    dim: int,
    horizon: int,
    min_anchors: int,
    physical_time: bool = False,
) -> np.ndarray:
    n = series.shape[0]
    # M-151: strict scalar contract — a fractional/non-finite/bool/out-of-range
    # value is a contract violation, never a value to silently coerce.  ``dim``
    # is kept in [2, 6] (the documented embedding contract).
    w = strict_int(window, "window", lower=2)
    th = dim * strict_int(tau, "tau", lower=1)
    H = strict_int(horizon, "horizon", lower=1)
    ma = strict_int(min_anchors, "min_anchors", lower=1)
    dim = strict_int(dim, "embedding_dim", lower=2, upper=6)
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
            if physical_time:
                # M-150: physical-clock divergence horizon.  The k-th successor
                # is found by scanning k PHYSICAL bars forward from each
                # trajectory's anchor position; a NaN row advances the physical
                # index but contributes no successor point (NaN for that k).
                div = _physical_divergence(Z, phys, z_len, i, j, H, d0j)
                if div is None:
                    continue
            else:
                # Default (compressed-ordinal) divergence horizon.  DOCUMENTED
                # M-150: ``k`` counts COMPRESSED steps, not physical bars — after
                # an interior NaN drop ``k`` can span more physical bars.  Use
                # ``physical_time=True`` for the physical-clock interpretation.
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
        Lstack = np.stack(logs)
        if physical_time:
            # M-150: physical-time path — some k steps may be unavailable for a
            # given anchor (no surviving point at that physical offset), so the
            # mean curve and the least-squares slope use only the k's with a
            # finite mean over anchors.  The regressor IS physical elapsed bars:
            # each k = exactly k physical bars forward from the anchor.
            L = np.nanmean(Lstack, axis=0)
            ks = np.arange(H + 1, dtype=float)
            ok = np.isfinite(L)
            if int(ok.sum() < 2:
                continue
            kk = ks[ok]
            LL = L[ok]
            var_k = float(np.sum((kk - kk.mean()) ** 2))
            if var_k <= _EPS:
                continue
            lam = float(np.sum((kk - kk.mean()) * (LL - LL.mean())) / var_k)
        else:
            L = np.mean(Lstack, axis=0)
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
    M-150: 默认发散步长按 COMPRESSED ordinal 推进（k 步可能跨更多物理 bar）；
    ``physical_time=True`` 改为按物理时钟扫 k 个物理 bar，回归对物理流逝时间。
    M-151: 参数严格校验（window/tau/horizon/min_anchors 非法值抛错，不裁剪）。
    """

    metadata = _metadata(
        "ts_local_lyapunov_exponent",
        "局部 Lyapunov 指数 λ（Takens 嵌入最近邻居发散斜率，per-step rate）。",
        ["x", "window", "tau", "embedding_dim", "horizon", "min_anchors", "physical_time"],
        unit="rate",
        cost=8,
    )
    # M-151/M-150: strict int/bool contracts + role-aware search classification.
    # ``window``/``horizon`` are the economic look-ahead dimensions; ``tau``/
    # ``embedding_dim`` are estimator resolution; ``min_anchors`` is a support
    # floor and ``physical_time`` is a governance switch — neither is searched.
    metadata.param_specs = {
        "window": ParamSpec(dtype=int, min=2, param_role=ParamRole.HORIZON),
        "tau": ParamSpec(dtype=int, min=1, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        "embedding_dim": ParamSpec(dtype=int, min=2, max=6, param_role=ParamRole.ESTIMATOR_RESOLUTION),
        "horizon": ParamSpec(dtype=int, min=1, param_role=ParamRole.HORIZON),
        "min_anchors": ParamSpec(dtype=int, min=1, searchable=False, param_role=ParamRole.SUPPORT_POLICY),
        "physical_time": ParamSpec(dtype=bool, default=False, searchable=False, param_role=ParamRole.POLICY),
    }

    def _calculate_series(
        self,
        x: pd.DataFrame,
        window: int = 120,
        tau: int = 1,
        embedding_dim: int = 3,
        horizon: int = 5,
        min_anchors: int = 3,
        physical_time: bool = False,
        **_: Any,
    ) -> pd.DataFrame:
        # M-151: strict integer/boolean contract — a fractional window, a bool
        # tau, a NaN horizon or an out-of-range value raises instead of being
        # silently clamped (Master Spec A-4/5).
        w = strict_int(window, "window", lower=2)
        tau_i = strict_int(tau, "tau", lower=1)
        dim = strict_int(embedding_dim, "embedding_dim", lower=2, upper=6)
        H = strict_int(horizon, "horizon", lower=1)
        ma = strict_int(min_anchors, "min_anchors", lower=1)
        pt = strict_bool(physical_time, "physical_time")
        xv = x.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            out[:, c] = _lyapunov_series(xv[:, c], w, tau_i, dim, H, ma, pt)
        return frame_like(x, out)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_research_only({"ts_local_lyapunov_exponent"})


_register_surface()
