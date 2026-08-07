# -*- coding: utf-8 -*-
"""First-passage / barrier-crossing dynamics (2026-08 V2, P1).

``ts_first_passage_bias`` learns, from many historical anchors, which side of a
``± barrier*scale`` band is touched first and how quickly — direction,
probability and speed in one path-dependent stopping-time statistic.

* ``x``       — the series to test (recommended: ``log(close)``).
* ``scale``   — a per-row volatility-like series aligned with ``x`` (e.g.
  rolling daily return volatility), so ``barrier*scale_s`` shares its units.
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

_EPS = 1e-12


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
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
        ],
    )


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
    b = max(_EPS, float(barrier))
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
            for h in range(1, H + 1):
                val = x[s + h]
                if not np.isfinite(val):
                    break
                if val >= up:
                    tau = h
                    d = 1.0
                    break
                if val <= down:
                    tau = h
                    d = -1.0
                    break
            if tau > 0:
                wgt = (H + 1 - tau) / H
                signs.append(d * wgt)
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
        xv = x.to_numpy(dtype=float)
        sv = scale.to_numpy(dtype=float)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            out[:, c] = _first_passage_series(xv[:, c], sv[:, c], window, barrier, horizon, min_anchors)
        return frame_like(x, out)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | {"ts_first_passage_bias"}
    )


_register_surface()
