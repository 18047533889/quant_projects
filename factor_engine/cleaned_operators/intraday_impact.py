# -*- coding: utf-8 -*-
"""Intraday liquidity resilience (2026-08-08 Gemini V2 round).

``intraday_impact_decay_rate`` — one scalar per (date, symbol): detect
shock minutes (``|ret|`` above the day's ``shock_quantile`` with positive
amount), then fit the price-impact decay ``|I_h| ≈ A·e^{-κh}`` over
``h = 1..horizon`` after each shock (log-linear fit on the impact magnitude).

* ``κ > 0``  — impact decays (liquidity resilient);
* ``κ ≈ 0``  — impact persists;
* ``κ < 0``  — impact amplifies (fragile).

Session semantics: shocks and their horizons never cross a lunch/close
boundary — the day's minute grid is split into contiguous session blocks and
each block is processed independently (the session minute axis is never
compressed).  Minute-source input → daily scalar per symbol.  No L2/order-flow
data is assumed (minute OHLCV only), so this is an impact-decay *proxy*, not a
full market-impact propagator.  Strict-PIT within the day, deterministic,
NaN fail-closed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like

_EPS = 1e-12


def _metadata(name: str, description: str, params: list[str]) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="intraday_microstructure",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "intraday", "minute", "daily", "pit_safe", "causal", "deterministic",
            f"signature:{','.join(params)}->series", "domain:intraday",
            "unit:level", "cost:6",
        ],
    )


def _impact_decay_day(
    rets: np.ndarray,
    amounts: np.ndarray,
    minutes: pd.DatetimeIndex,
    horizon: int,
    shock_quantile: float,
) -> float:
    n = rets.shape[0]
    if n < horizon + 3:
        return np.nan
    if not np.all(np.isfinite(rets)) or not np.all(np.isfinite(amounts)):
        return np.nan
    if np.any(amounts < 0.0):
        return np.nan

    # contiguous session blocks (never bridge a lunch/close gap)
    blocks: list[list[int]] = []
    cur = [0]
    for i in range(1, n):
        gap = (minutes[i] - minutes[i - 1]) > pd.Timedelta(minutes=2)
        if gap:
            blocks.append(cur)
            cur = [i]
        else:
            cur.append(i)
    blocks.append(cur)

    kappas: list[float] = []
    for blk in blocks:
        idx = np.asarray(blk, dtype=int)
        if idx.size < horizon + 2:
            continue
        r = rets[idx]
        a = amounts[idx]
        absr = np.abs(r)
        thr = float(np.quantile(absr, float(shock_quantile)))
        logp = np.log(np.maximum(np.cumprod(1.0 + r), 1e-12))
        for e in range(idx.size - horizon):
            if not (absr[e] > thr and absr[e] > _EPS and a[e] > 0.0):
                continue
            sign_e = 1.0 if r[e] > 0.0 else -1.0
            impacts = sign_e * (logp[e + 1 : e + 1 + horizon] - logp[e])
            valid = np.abs(impacts) > 1e-9
            if int(valid.sum()) < 2:
                continue
            hvals = np.arange(1, horizon + 1, dtype=float)[valid]
            logi = np.log(np.abs(impacts[valid]))
            slope = np.polyfit(hvals, logi, 1)[0]
            kappa = float(-slope)
            if np.isfinite(kappa):
                kappas.append(kappa)
    if not kappas:
        return np.nan
    return float(np.median(kappas))


@register_operator(
    name="intraday_impact_decay_rate",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_impact_decay_rate",
    source="intraday_impact",
    status="implemented",
)
class IntradayImpactDecayRate(SeriesOperator):
    """分钟冲击后价格冲击衰减率 κ（log|I| 对 h 的线性斜率取负）。"""

    metadata = _metadata(
        "intraday_impact_decay_rate",
        "冲击后价格冲击幅度的衰减率 κ（>0 衰减 / <0 放大）。",
        ["ret", "amount", "volume", "horizon", "shock_quantile"],
    )

    def _calculate_series(
        self,
        ret: pd.DataFrame,
        amount: pd.DataFrame,
        volume: pd.DataFrame,
        horizon: int = 10,
        shock_quantile: float = 0.9,
        **_: Any,
    ) -> pd.DataFrame:
        h = max(2, int(horizon))
        sq = float(shock_quantile)
        if not 0.0 < sq < 1.0:
            raise ValueError("intraday_impact_decay_rate requires 0 < shock_quantile < 1")
        idx = ret.index
        r_arr = ret.to_numpy(dtype=float)
        a_arr = amount.to_numpy(dtype=float)
        out: dict[str, pd.Series] = {}
        for inst in ret.columns:
            col_idx = inst
            day_map: dict[pd.Timestamp, float] = {}
            pos = np.arange(r_arr.shape[0])
            for day, g in pd.Series(pos, index=idx).groupby(idx.normalize()):
                positions = np.asarray(g, dtype=int)
                day_map[day] = _impact_decay_day(
                    r_arr[positions], a_arr[positions], idx[positions], h, sq
                )
            out[col_idx] = pd.Series(day_map, dtype=float)
        if not out:
            return pd.DataFrame(dtype=float)
        return pd.DataFrame(out).sort_index()


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.EXTENDED_ONLY_CANONICALS = frozenset(
        set(_surface.EXTENDED_ONLY_CANONICALS) | {"intraday_impact_decay_rate"}
    )
    from cleaned_operators.rolling_pack import register_polars_udf

    register_polars_udf("intraday_impact_decay_rate")


_register_surface()
