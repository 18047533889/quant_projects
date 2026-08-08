# -*- coding: utf-8 -*-
"""Directional-Change / intrinsic-time operators (2026-08 market language, P1).

Directional Change (DC) is an event-based clock: a DC event confirms when price
moves ``theta`` away from the previous extreme; the subsequent same-direction
run is the *overshoot*.  The four operators measure the CTA-relevant anatomy of
that clock:

* ``ts_dc_overshoot_ratio``  — overshoot / theta of the last completed DC event
  (trend maturity: 0.2 = barely continued, 2.0 = ran two thresholds further).
* ``ts_dc_event_rate``       — DC events per day (intrinsic-time speed).
* ``ts_dc_duration_asymmetry`` / ``ts_dc_overshoot_asymmetry`` — do up vs down
  legs last longer / extend further (slow-rise-fast-crash etc.).

``theta`` is not a fixed percentage.  ``threshold_mode`` selects one of two
explicit threshold semantics (R5 P1-41(c)):

* ``adaptive`` (default) — the confirmation threshold at bar ``i`` is
  ``theta * scale_i`` where ``scale_i`` is the historical scale value available
  *at that bar* (ATR / realised vol / MAD).  The clock is online: it walks the
  series bar-by-bar, confirms a leg the moment the move exceeds the per-bar
  threshold, freezes the event (confirm-then-freeze) and only then looks for the
  opposite confirmation.  It never rescans a historical window under today's
  end-of-window scale, so historical DC events are NOT repainted by today's
  volatility.
* ``fixed_absolute`` — one absolute threshold ``theta * scale_anchor`` where
  ``scale_anchor`` is the first available scale of the trailing window; the
  whole past is scanned with that single fixed threshold.

Missing prices censor/break the clock (R5 P1-41(a)): a NaN bar resets the
ongoing leg so no DC event can straddle a data gap, and durations / event rates
are measured on the same *observed-time* clock (finite bars).
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
        category="directional_change",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "directional_change", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic", *extra_tags,
            f"signature:{','.join(params)}->series", f"domain:{domain}",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _dc_events(
    x: np.ndarray, scale: np.ndarray, theta: float, fixed_scale: float | None = None
) -> list[tuple[int, str]]:
    """Online intrinsic-time DC confirmations ``(index, kind)``, kind in {up, down}.

    Walks the series bar-by-bar.  In the default *adaptive* mode the
    confirmation threshold at bar ``i`` is ``theta * scale[i]`` — the historical
    scale available *at that bar*, not the current end-of-window scale, so
    historical DC events are never rescanned under today's volatility.  In
    *fixed_absolute* mode ``fixed_scale`` pins one absolute threshold
    ``theta * fixed_scale`` for the entire window.

    A DC confirms when price moves ``theta*scale_i`` away from the tracked
    extreme; during the overshoot the extreme is updated to the most extreme
    price, and the next confirmation is the opposite kind (confirm-then-freeze).

    R5 P1-41(a): a NaN price bar does NOT skip the clock — it censors/breaks the
    ongoing leg (direction reset, extreme invalidated) so the next finite bar
    restarts fresh tracking; a gap must not silently count as "no price move".
    NaN/non-positive scale bars admit no confirmation.
    """
    n = x.shape[0]
    events: list[tuple[int, str]] = []
    if n < 3 or not np.isfinite(theta) or theta <= 0:
        return events
    direction = 0  # +1 up, -1 down, 0 undecided
    extreme = float(x[0]) if np.isfinite(x[0]) else np.nan
    for i in range(1, n):
        p = float(x[i])
        if not np.isfinite(p):
            # a gap breaks the clock: the current leg is invalidated and the
            # next finite bar restarts fresh (no event can straddle the gap).
            direction = 0
            extreme = np.nan
            continue
        if not np.isfinite(extreme):
            extreme = p
            continue
        if fixed_scale is not None:
            if not np.isfinite(fixed_scale) or fixed_scale <= 0:
                continue  # fixed threshold undefined -> no confirmation
            thr = theta * fixed_scale
        else:
            s = float(scale[i]) if i < scale.shape[0] else np.nan
            if not np.isfinite(s) or s <= 0:
                continue
            thr = theta * s
        if direction != -1:
            if p > extreme:
                extreme = p
            if extreme - p >= thr:
                events.append((i, "down"))
                direction = -1
                extreme = p
        if direction != 1:
            if p < extreme:
                extreme = p
            if p - extreme >= thr:
                events.append((i, "up"))
                direction = 1
                extreme = p
    return events


def _dc_completed_overshoots(
    x: np.ndarray, scale: np.ndarray, theta: float, fixed_scale: float | None = None
) -> tuple[list[float], list[float]]:
    """Per-event overshoot magnitudes for up and down completed legs."""
    events = _dc_events(x, scale, theta, fixed_scale)
    up_o: list[float] = []
    dn_o: list[float] = []
    if len(events) < 2:
        return up_o, dn_o
    for (i_m, kind), (i_n, _) in zip(events[:-1], events[1:]):
        seg = x[i_m:i_n]
        seg = seg[np.isfinite(seg)]
        if seg.size < 1:
            continue
        if kind == "up":
            up_o.append(float(np.max(seg) - x[i_m]))
        else:
            dn_o.append(float(x[i_m] - np.min(seg)))
    return up_o, dn_o


def _dc_durations(
    x: np.ndarray, scale: np.ndarray, theta: float, fixed_scale: float | None = None
) -> tuple[list[float], list[float]]:
    events = _dc_events(x, scale, theta, fixed_scale)
    up_d: list[float] = []
    dn_d: list[float] = []
    if len(events) < 2:
        return up_d, dn_d
    for (i_m, kind), (i_n, _) in zip(events[:-1], events[1:]):
        # R5 P1-41(b): duration uses the *observed-time* clock (count of finite
        # bars between the two confirmations), matching the event-rate
        # denominator — raw row distance would over-count gaps/suspended days.
        seg = x[i_m:i_n]
        dur = float(np.isfinite(seg).sum())
        if dur < 1.0:
            continue
        (up_d if kind == "up" else dn_d).append(dur)
    return up_d, dn_d


def _median_asym(up: list[float], dn: list[float]) -> float:
    if not up or not dn:
        return np.nan
    mu = float(np.median(up))
    md = float(np.median(dn))
    return float((mu - md) / (mu + md + _EPS))


def _dc_rolling(
    xv: np.ndarray, sv: np.ndarray, w: int, threshold: float, fn,
    fixed_scale_per_window: bool = False,
) -> np.ndarray:
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            s = sv[r, c]
            if not np.isfinite(s) or s <= 0:
                continue
            lo = max(0, r - w + 1)
            if fixed_scale_per_window:
                # fixed_absolute threshold mode: anchor one absolute threshold at
                # the FIRST available historical scale of the window, so the whole
                # past is scanned with a single threshold (no repaint from later
                # bars' volatility).
                win_s = sv[lo : r + 1, c]
                fin_s = win_s[np.isfinite(win_s)]
                fixed = float(fin_s[0]) if fin_s.size else np.nan
            else:
                fixed = None
            out[r, c] = fn(xv[lo : r + 1, c], sv[lo : r + 1, c], threshold, fixed)
    return out


def _overshoot_ratio_chunk(
    chunk: np.ndarray, scale_chunk: np.ndarray, theta: float, fixed_scale: float | None = None
) -> float:
    events = _dc_events(chunk, scale_chunk, theta, fixed_scale)
    if len(events) < 2:
        return np.nan
    i_m, kind = events[-2]  # last *completed* leg (overshoot ended at events[-1])
    i_n, _ = events[-1]
    seg = chunk[i_m:i_n]
    seg = seg[np.isfinite(seg)]
    if seg.size < 1:
        return np.nan
    if kind == "up":
        os = float(np.max(seg) - chunk[i_m])
    else:
        os = float(chunk[i_m] - np.min(seg))
    if fixed_scale is not None:
        thr = fixed_scale
    else:
        thr = float(scale_chunk[i_n]) if i_n < scale_chunk.shape[0] else np.nan
    if not np.isfinite(thr) or thr <= 0:
        return np.nan
    return float(os / (theta * thr))


def _event_rate_chunk(
    chunk: np.ndarray, scale_chunk: np.ndarray, theta: float, fixed_scale: float | None = None
) -> float:
    events = _dc_events(chunk, scale_chunk, theta, fixed_scale)
    # R5 P1-41(b): observed-time clock — finite (traded) bars are the
    # denominator, matching the observed-time durations.
    n_fin = int(np.isfinite(chunk).sum())
    if n_fin < 2:
        return np.nan
    return float(len(events) / n_fin)


def _duration_asym_chunk(
    chunk: np.ndarray, scale_chunk: np.ndarray, theta: float, fixed_scale: float | None = None
) -> float:
    up_d, dn_d = _dc_durations(chunk, scale_chunk, theta, fixed_scale)
    return _median_asym(up_d, dn_d)


def _overshoot_asym_chunk(
    chunk: np.ndarray, scale_chunk: np.ndarray, theta: float, fixed_scale: float | None = None
) -> float:
    up_o, dn_o = _dc_completed_overshoots(chunk, scale_chunk, theta, fixed_scale)
    return _median_asym(up_o, dn_o)


def _make_dc_op(canonical: str, description: str, unit: str, cost: int, fn) -> SeriesOperator:
    def _calculate_series(
        self,
        x: pd.DataFrame,
        scale: pd.DataFrame,
        threshold: float = 1.0,
        window: int = 120,
        threshold_mode: str = "adaptive",
        **_: Any,
    ) -> pd.DataFrame:
        w = int(window)
        thr = float(threshold)
        if thr <= 0:
            raise ValueError(f"{canonical} requires threshold > 0")
        if w < 3:
            raise ValueError(f"{canonical} requires window >= 3")
        # R5 P1-41(c): two explicit threshold semantics.
        #  * adaptive        — per-bar historical threshold ``theta * scale[i]``
        #    (no repaint of history under today's volatility; default).
        #  * fixed_absolute  — one absolute threshold anchored at the first
        #    scale of the window, applied to the whole past.
        if threshold_mode not in ("adaptive", "fixed_absolute"):
            raise ValueError(
                f"{canonical} threshold_mode must be 'adaptive' or 'fixed_absolute'"
            )
        return frame_like(
            x,
            _dc_rolling(
                x.to_numpy(dtype=float),
                scale.to_numpy(dtype=float),
                w,
                thr,
                fn,
                fixed_scale_per_window=(threshold_mode == "fixed_absolute"),
            ),
        )

    metadata = _metadata(
        canonical,
        description,
        ["x", "scale", "threshold", "window", "threshold_mode"],
        domain="price_volume",
        unit=unit,
        cost=cost,
    )
    return register_operator(
        name=canonical,
        category="directional_change",
        business_category="directional_change",
        canonical=canonical,
        source="directional_change",
    )(
        type(
            canonical.replace("_", " ").title().replace(" ", "") + "Op",
            (SeriesOperator,),
            {"metadata": metadata, "_calculate_series": _calculate_series, "__module__": __name__},
        )
    )


TsDcOvershootRatio = _make_dc_op(
    "ts_dc_overshoot_ratio",
    "最后一个已完成 DC 事件的 overshoot/theta（趋势成熟度）。",
    "ratio",
    5,
    _overshoot_ratio_chunk,
)
TsDcEventRate = _make_dc_op(
    "ts_dc_event_rate",
    "DC 事件频率（每 bar 事件数，内在时间速度）。",
    "rate",
    4,
    _event_rate_chunk,
)
TsDcDurationAsymmetry = _make_dc_op(
    "ts_dc_duration_asymmetry",
    "上/下 DC 腿时长中位数不对称性（慢涨快跌等）。",
    "ratio",
    5,
    _duration_asym_chunk,
)
TsDcOvershootAsymmetry = _make_dc_op(
    "ts_dc_overshoot_asymmetry",
    "上/下 DC 腿 overshoot 中位数不对称性（趋势延伸方向差异）。",
    "ratio",
    5,
    _overshoot_asym_chunk,
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({
            "ts_dc_overshoot_ratio",
            "ts_dc_event_rate",
            "ts_dc_duration_asymmetry",
            "ts_dc_overshoot_asymmetry",
        })
    for _canon in (
        "ts_dc_overshoot_ratio",
        "ts_dc_event_rate",
        "ts_dc_duration_asymmetry",
        "ts_dc_overshoot_asymmetry",
    ):
        register_polars_udf(_canon)


_register_surface()
