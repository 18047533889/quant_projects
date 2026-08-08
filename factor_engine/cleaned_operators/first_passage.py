# -*- coding: utf-8 -*-
"""First-passage / barrier-crossing dynamics (2026-08 V2, P1).

``ts_first_passage_bias`` learns, from many historical anchors, which side of a
``± barrier*scale`` band is touched first and how quickly — direction,
probability and speed in one path-dependent stopping-time statistic.

* ``x``       — the series to test (recommended: ``log(close)``).
* ``scale``   — a per-row volatility-like series aligned with ``x`` (e.g.
  rolling daily return volatility), so ``barrier*scale_s`` shares its units.
  The metadata declares ``scale_horizon`` (default 1 bar) — the horizon the
  scale is aggregated over — because 1-day vs 20-day volatility are different
  barriers even though both are "return volatility".
* anchors ``s`` with ``s + horizon <= t`` only: no future information reaches
  ``t``.  Unreached anchors contribute ``w_s = 0``; the output is the mean of
  ``d_s * w_s`` over anchors in ``[t-W, t-horizon]``.

Positive output: from similar historical states the upper barrier is usually
touched earlier/easier; negative: the lower side dominates.  Deterministic and
prefix-causal.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like


def _metadata(
    name: str, description: str, params: list[str], *, unit: str, cost: int, scale_horizon: int = 1
) -> OperatorMetadata:
    # R4-85: ``barrier * scale`` is added to ``x``, so ``scale`` must share the
    # units of ``x``.  x = price with scale = return-volatility violates the
    # contract (adding return-vol to a price level); x = log(price) with
    # scale = return-volatility is consistent.  The typed grammar enforces this;
    # ``input_units`` documents the contract for the operator surface.
    #
    # P1 (round 7): ``scale_horizon`` makes the scale's aggregation horizon
    # explicit — a 1-day vs 20-day return volatility are DIFFERENT barriers even
    # though both are "return volatility".  The metadata/type now declares which
    # horizon the ``scale`` input is assumed to be, so recipes cannot silently
    # mix horizons.
    return OperatorMetadata(
        name=name,
        category="first_passage",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "first_passage", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:price_volume",
            f"unit:{unit}", f"cost:{cost}",
            f"scale_horizon:{scale_horizon}",
        ],
        input_units={
            "x": "price_or_log_price",
            "scale": "volatility_with_same_unit_as_x",
        },
        compatible_units={"scale": ("log_price_volatility", "return_volatility")},
    )


def _check_barrier(barrier: Any) -> float:
    """Validate ``barrier > 0``.

    R4-86: ``b = max(EPS, float(barrier))`` silently collapsed ``-1 / 0 /
    1e-20`` into the same value, manufacturing duplicate ASTs out of different
    parameters.  Non-positive barriers are rejected loudly.
    """
    b = float(barrier)
    if not np.isfinite(b) or b <= 0.0:
        raise ValueError("barrier must be a finite positive number")
    return b


def _first_passage_series(
    x: np.ndarray,
    scale: np.ndarray,
    window: int,
    barrier: float,
    horizon: int,
    min_anchors: int,
) -> np.ndarray:
    n = x.shape[0]
    w = max(2, int(window))
    H = max(1, int(horizon))
    b = _check_barrier(barrier)
    ma = max(1, int(min_anchors))
    out = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - w)
        first_anchor = max(lo, 0)
        last_anchor = t - H              # need s + H <= t
        if last_anchor < first_anchor:
            continue
        anchor_range = range(first_anchor, last_anchor + 1)
        signs: list[float] = []
        for s in anchor_range:
            xs = x[s]
            sc = scale[s]
            if not np.isfinite(xs) or not np.isfinite(sc) or sc <= 0.0:
                continue
            up = xs + b * sc
            down = xs - b * sc
            tau = 0
            d = 0.0
            fully_observed = True
            for h in range(1, H + 1):
                val = x[s + h]
                if not np.isfinite(val):
                    fully_observed = False
                    break
                if val >= up:
                    tau = h
                    d = 1.0
                    break
                if val <= down:
                    tau = h
                    d = -1.0
                    break
            if not fully_observed:
                continue  # no information: exclude the anchor entirely
            if tau > 0:
                wgt = (H + 1 - tau) / H
                signs.append(d * wgt)
            else:
                # Full-observed non-hit anchor contributes 0, so the mean is the
                # direction × speed × hit-probability (audit Q02, option A) —
                # not a conditional-on-hit statistic.
                signs.append(0.0)
        if len(signs) < ma:
            continue
        out[t] = float(np.mean(signs))
    return out


@register_operator(
    name="ts_first_passage_bias",
    category="first_passage",
    business_category="first_passage",
    canonical="ts_first_passage_bias",
    source="first_passage",
)
class TsFirstPassageBias(SeriesOperator):
    """首达偏向：历史锚点中哪一侧 barrier 更早/更容易被触及。

    对每个满足 ``s+H ≤ t`` 的历史锚点 s，定义上/下 barrier
    ``x_s ± barrier*scale_s``，在 ``[s+1, s+H]`` 内找首达（取先到的 barrier），
    输出 ``mean(d_s * w_s)``（d=+1 上 / -1 下 / 0 未触及；w=(H+1-τ)/H）。
    接近 +1 = 类似状态通常更快触上沿；接近 -1 = 下沿主导。确定性、PIT 安全。
    """

    metadata = _metadata(
        "ts_first_passage_bias",
        "首达偏向 mean(d_s*w_s)（[-1,1]，方向×速度×概率）。",
        ["x", "scale", "window", "barrier", "horizon", "min_anchors"],
        unit="ratio",
        cost=6,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        scale: pd.DataFrame,
        window: int = 120,
        barrier: float = 1.0,
        horizon: int = 10,
        min_anchors: int = 3,
        **_: Any,
    ) -> pd.DataFrame:
        _check_barrier(barrier)
        xv = x.to_numpy(dtype=float)
        sv = scale.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            out[:, c] = _first_passage_series(xv[:, c], sv[:, c], window, barrier, horizon, min_anchors)
        return frame_like(x, out)


def _fp_stats_series(
    x: np.ndarray,
    scale: np.ndarray,
    window: int,
    barrier: float,
    horizon: int,
    min_anchors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-row first-passage statistics (PIT-safe: anchors with ``s+H <= t``).

    For every anchor ``s`` with a *fully observed* horizon path ``[s+1, s+H]``
    (all values finite), record which barrier is touched first and at which
    lag ``τ``.  Returns ``(up_frac, dn_frac, up_ct, dn_ct)``:
      up_frac / dn_frac — fraction of fully-observed anchors that touch the
        upper / lower barrier within the horizon (hit *probability*, includes
        non-hit anchors in the denominator).
      up_ct / dn_ct — mean ``τ/H`` over anchors that touched that side
        (conditional time, normalized to [0, 1]).
    """
    n = x.shape[0]
    w = max(2, int(window))
    H = max(1, int(horizon))
    b = _check_barrier(barrier)
    ma = max(1, int(min_anchors))
    up_frac = np.full(n, np.nan)
    dn_frac = np.full(n, np.nan)
    up_ct = np.full(n, np.nan)
    dn_ct = np.full(n, np.nan)
    for t in range(n):
        lo = max(0, t - w)
        first_anchor = max(lo, 0)
        last_anchor = t - H
        if last_anchor < first_anchor:
            continue
        n_obs = 0
        n_up = 0
        n_dn = 0
        tau_up: list[float] = []
        tau_dn: list[float] = []
        for s in range(first_anchor, last_anchor + 1):
            xs = x[s]
            sc = scale[s]
            if not np.isfinite(xs) or not np.isfinite(sc) or sc <= 0.0:
                continue
            # fully observed horizon path required for a determinate outcome
            seg = x[s + 1 : s + H + 1]
            if not np.all(np.isfinite(seg)):
                continue
            n_obs += 1
            up = xs + b * sc
            down = xs - b * sc
            for h in range(H):
                val = seg[h]
                if val >= up:
                    n_up += 1
                    tau_up.append((h + 1) / H)
                    break
                if val <= down:
                    n_dn += 1
                    tau_dn.append((h + 1) / H)
                    break
        if n_obs < ma:
            continue
        up_frac[t] = n_up / n_obs
        dn_frac[t] = n_dn / n_obs
        if len(tau_up) >= ma:
            up_ct[t] = float(np.mean(tau_up))
        if len(tau_dn) >= ma:
            dn_ct[t] = float(np.mean(tau_dn))
    return up_frac, dn_frac, up_ct, dn_ct


@register_operator(
    name="ts_first_passage_hit_probability",
    category="first_passage",
    business_category="first_passage",
    canonical="ts_first_passage_hit_probability",
    source="first_passage",
)
class TsFirstPassageHitProbability(SeriesOperator):
    """历史首达命中概率 ``P(τ^± ≤ H)``。

    对严格过去的锚点（``s+H ≤ t``，且 ``[s+1,s+H]`` 全程可观测），统计上/下
    barrier 在 H 内被首达的比例。与 ``ts_first_passage_bias`` 不同：bias 只对
    已命中锚点求方向×速度均值，命中概率把"没发生的锚点"也算进分母，回答"到
    底会不会发生"。两个不同股票 bias≈0（一个几乎必破、一个几乎不破）在此可区分。
    """

    metadata = _metadata(
        "ts_first_passage_hit_probability",
        "历史首达命中概率 P(τ≤H)（含未命中锚点作分母）。",
        ["x", "scale", "window", "barrier", "horizon", "min_anchors", "side"],
        unit="probability",
        cost=6,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        scale: pd.DataFrame,
        window: int = 120,
        barrier: float = 1.0,
        horizon: int = 10,
        min_anchors: int = 3,
        side: str = "upper",
        **_: Any,
    ) -> pd.DataFrame:
        _check_barrier(barrier)
        xv = x.to_numpy(dtype=float)
        sv = scale.to_numpy(dtype=float)
        rows, cols = xv.shape
        side = str(side).lower()
        if side not in ("upper", "lower"):
            raise ValueError(f"unknown side {side!r}; expected upper/lower")
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            up, dn, _, _ = _fp_stats_series(xv[:, c], sv[:, c], window, barrier, horizon, min_anchors)
            out[:, c] = up if side == "upper" else dn
        return frame_like(x, out)


@register_operator(
    name="ts_first_passage_conditional_time",
    category="first_passage",
    business_category="first_passage",
    canonical="ts_first_passage_conditional_time",
    source="first_passage",
)
class TsFirstPassageConditionalTime(SeriesOperator):
    """历史命中条件下的平均首达时间 ``E[τ | τ ≤ H] / H``。

    对过去已触及指定 barrier 的锚点（``s+H ≤ t``，全程可观测），求首达时滞
    τ 相对 H 的均值（[0,1]）。与 hit_probability 互补：probability×speed 构成
    "会不会去 × 去得有多快"的 recipe。bias 已含速度加权，这里是纯时间尺度。
    """

    metadata = _metadata(
        "ts_first_passage_conditional_time",
        "命中条件下的平均首达时间 E[τ|τ≤H]/H（[0,1]）。",
        ["x", "scale", "window", "barrier", "horizon", "min_anchors", "side"],
        unit="ratio",
        cost=6,
    )

    def _calculate_series(
        self,
        x: pd.DataFrame,
        scale: pd.DataFrame,
        window: int = 120,
        barrier: float = 1.0,
        horizon: int = 10,
        min_anchors: int = 3,
        side: str = "upper",
        **_: Any,
    ) -> pd.DataFrame:
        _check_barrier(barrier)
        xv = x.to_numpy(dtype=float)
        sv = scale.to_numpy(dtype=float)
        rows, cols = xv.shape
        side = str(side).lower()
        if side not in ("upper", "lower"):
            raise ValueError(f"unknown side {side!r}; expected upper/lower")
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            _, _, up_ct, dn_ct = _fp_stats_series(xv[:, c], sv[:, c], window, barrier, horizon, min_anchors)
            out[:, c] = up_ct if side == "upper" else dn_ct
        return frame_like(x, out)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({
            "ts_first_passage_bias",
            "ts_first_passage_hit_probability",
            "ts_first_passage_conditional_time",
        })


_register_surface()
