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
  ``scale_anchor`` is pinned ONCE at stream init (the first finite scale) and
  carried in the streaming checkpoint, so it never drifts as the rolling
  window slides (review #32).  The whole past is scanned with that single
  fixed threshold.

The DC event process is *streamed forward once* (recursive): the state
(direction, extreme, current leg's own start threshold) lives in a checkpoint
and is never rebuilt from a rolling window's left boundary, so a historical
bar belongs to exactly one event sequence regardless of the window (review
#31).  Each completed leg's overshoot is normalised by the threshold in force
when THAT leg started, not by the opposite-side threshold at the leg's end
(review #33).

Missing prices censor/break the clock (R5 P1-41(a)); a missing / non-positive
scale bar BREAKS the ongoing episode rather than skipping it (review #34) — a
DC event must never straddle a scale gap.  Durations / event rates are
measured on the same *observed-time* clock (finite bars).
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


def _median_asym(up: list[float], dn: list[float]) -> float:
    if not up or not dn:
        return np.nan
    mu = float(np.median(up))
    md = float(np.median(dn))
    return float((mu - md) / (mu + md + _EPS))


def _dc_column(
    x: np.ndarray, scale: np.ndarray, theta: float, w: int, threshold_mode: str
) -> dict[str, np.ndarray]:
    """Forward-walk (recursive) DC event process for one instrument column.

    The event state (direction, extreme, current leg's own start threshold) is
    streamed bar-by-bar ONCE from the first finite bar and carried in a
    checkpoint, so a historical bar belongs to exactly ONE event sequence
    regardless of where the trailing-window left boundary sits (review #31).
    The trailing window only *selects* which completed legs / events feed each
    row's aggregate.

    Threshold semantics (R5 P1-41(c)):
    * ``adaptive`` — confirmation threshold at bar ``i`` is ``theta*scale[i]``,
      the historical scale available *at that bar* (no repaint of history).
    * ``fixed_absolute`` — one absolute threshold ``theta*anchor`` where the
      anchor is pinned ONCE at stream init (first finite scale) and enters the
      checkpoint, so it never drifts as the window slides (review #32).

    Missing policies:
    * a NaN price bar censors/breaks the clock (R5 P1-41(a));
    * a missing / non-positive scale bar BREAKS the episode instead of skipping
      it — a DC event must never straddle a scale gap (review #34);
    * every completed leg's overshoot is normalised by the threshold that was
      in force when THAT leg started (``leg_start_threshold``), never by the
      opposite-side threshold at the leg's end (review #33).
    """
    n = x.shape[0]
    osr = np.full(n, np.nan, dtype=float)
    evr = np.full(n, np.nan, dtype=float)
    duram = np.full(n, np.nan, dtype=float)
    osam = np.full(n, np.nan, dtype=float)

    fin_pref = np.zeros(n + 1, dtype=float)
    ev_pref = np.zeros(n + 1, dtype=float)

    legs_start: list[int] = []
    legs_end: list[int] = []
    legs_kind: list[str] = []
    legs_dur: list[float] = []
    legs_os: list[float] = []
    legs_ratio: list[float] = []

    # --- streaming checkpoint ---
    direction = 0  # +1 up, -1 down, 0 undecided
    extreme = np.nan
    fixed_anchor = np.nan  # fixed_absolute: pinned once at stream init (#32)
    leg_kind: str | None = None
    leg_start = -1
    leg_thr = np.nan

    for i in range(n):
        fin_pref[i + 1] = fin_pref[i] + (1.0 if np.isfinite(x[i]) else 0.0)
        ev_pref[i + 1] = ev_pref[i]

        p = float(x[i])
        if not np.isfinite(p):
            # missing price censors/breaks the clock.
            direction = 0
            extreme = np.nan
            leg_kind = None
            leg_start = -1
            leg_thr = np.nan
            continue

        if threshold_mode == "fixed_absolute":
            if not np.isfinite(fixed_anchor):
                s = float(scale[i]) if i < scale.shape[0] else np.nan
                if np.isfinite(s) and s > 0.0:
                    fixed_anchor = s
                else:
                    # no anchor yet and scale unavailable -> break.
                    direction = 0
                    extreme = np.nan
                    leg_kind = None
                    leg_start = -1
                    leg_thr = np.nan
                    continue
            thr = theta * fixed_anchor
        else:  # adaptive
            s = float(scale[i]) if i < scale.shape[0] else np.nan
            if not np.isfinite(s) or s <= 0.0:
                # #34: scale-missing BREAKS the episode (not a skip).
                direction = 0
                extreme = np.nan
                leg_kind = None
                leg_start = -1
                leg_thr = np.nan
                continue
            thr = theta * s

        if not np.isfinite(extreme):
            extreme = p
            continue

        confirmed: str | None = None
        if direction != -1:
            if p > extreme:
                extreme = p
            if extreme - p >= thr:
                confirmed = "down"
        if direction != 1 and confirmed is None:
            if p < extreme:
                extreme = p
            if p - extreme >= thr:
                confirmed = "up"

        if confirmed is not None:
            # complete the previous leg (if any) with its OWN start threshold.
            if leg_kind is not None and leg_start >= 0:
                if leg_kind == "up":
                    os_m = float(np.max(x[leg_start:i])) - float(x[leg_start])
                else:
                    os_m = float(x[leg_start]) - float(np.min(x[leg_start:i]))
                dur = float(np.isfinite(x[leg_start:i]).sum())
                ratio = (os_m / leg_thr) if np.isfinite(leg_thr) and leg_thr > 0.0 else np.nan
                legs_start.append(leg_start)
                legs_end.append(i)
                legs_kind.append(leg_kind)
                legs_dur.append(dur)
                legs_os.append(os_m)
                legs_ratio.append(ratio)
            ev_pref[i + 1] += 1.0
            direction = 1 if confirmed == "up" else -1
            extreme = p
            leg_kind = confirmed
            leg_start = i
            leg_thr = thr

        lo = max(0, i - w + 1)
        n_fin = fin_pref[i + 1] - fin_pref[lo]
        n_ev = ev_pref[i + 1] - ev_pref[lo]
        if n_fin >= 2:
            evr[i] = n_ev / n_fin

    # Aggregate over the completed legs whose endpoints lie inside each row's
    # trailing window (legs are streamed once and are boundary-invariant).
    for i in range(n):
        lo = max(0, i - w + 1)
        idxs = [j for j in range(len(legs_start)) if legs_start[j] >= lo and legs_end[j] <= i]
        if not idxs:
            continue
        last = idxs[-1]
        osr[i] = legs_ratio[last]
        up_os = [legs_os[j] for j in idxs if legs_kind[j] == "up"]
        dn_os = [legs_os[j] for j in idxs if legs_kind[j] == "down"]
        up_d = [legs_dur[j] for j in idxs if legs_kind[j] == "up"]
        dn_d = [legs_dur[j] for j in idxs if legs_kind[j] == "down"]
        duram[i] = _median_asym(up_d, dn_d)
        osam[i] = _median_asym(up_os, dn_os)

    return {
        "overshoot_ratio": osr,
        "event_rate": evr,
        "duration_asymmetry": duram,
        "overshoot_asymmetry": osam,
    }


def _make_dc_op(canonical: str, description: str, unit: str, cost: int, out_key: str) -> SeriesOperator:
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
        #  * fixed_absolute  — one absolute threshold anchored once at stream
        #    init and carried in the checkpoint (never re-anchored per window).
        if threshold_mode not in ("adaptive", "fixed_absolute"):
            raise ValueError(
                f"{canonical} threshold_mode must be 'adaptive' or 'fixed_absolute'"
            )
        xv = x.to_numpy(dtype=float)
        sv = scale.to_numpy(dtype=float)
        rows, cols = xv.shape
        cols_out = []
        for c in range(cols):
            cols_out.append(_dc_column(xv[:, c], sv[:, c], thr, w, threshold_mode)[out_key])
        return frame_like(x, np.column_stack(cols_out) if cols else np.empty((rows, 0)))

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
    "overshoot_ratio",
)
TsDcEventRate = _make_dc_op(
    "ts_dc_event_rate",
    "DC 事件频率（每 bar 事件数，内在时间速度）。",
    "rate",
    4,
    "event_rate",
)
TsDcDurationAsymmetry = _make_dc_op(
    "ts_dc_duration_asymmetry",
    "上/下 DC 腿时长中位数不对称性（慢涨快跌等）。",
    "ratio",
    5,
    "duration_asymmetry",
)
TsDcOvershootAsymmetry = _make_dc_op(
    "ts_dc_overshoot_asymmetry",
    "上/下 DC 腿 overshoot 中位数不对称性（趋势延伸方向差异）。",
    "ratio",
    5,
    "overshoot_asymmetry",
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
