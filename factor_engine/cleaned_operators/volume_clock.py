# -*- coding: utf-8 -*-
"""Volume-clock intraday path geometry (2026-08 V3, P1, minute → daily).

Natural-clock path statistics (``ts_path_efficiency`` / ``ts_roughness`` /
``intra_path_efficiency``) measure price motion per *minute of wall clock*.
Volume-clock geometry re-prices the day along cumulative *trading activity*:
two stocks can have identical natural-time paths yet very different volume-clock
paths when one moves its price mostly on thin volume.

* ``intraday_volume_clock_path_efficiency`` — ``|p(1)-p(0)| / Σ|Δp_b|`` along the
  equal-activity grid (P1).
* ``intraday_volume_clock_roughness`` — ``Σ(Δ²p_b)² / (Σ(Δp_b)² + eps)`` on the
  same resampled log-price path (P1).

``activity`` is a user-supplied per-minute intensity (volume / amount / trades);
the clock is built on its cumulative sum, so the operator stays activity-agnostic.
Only the day's own bars enter the computation — prefix-causal and deterministic.

P0-O #80/#81/#82: a zero-activity bar must carry the SAME price as the previous
observable bar (else the day fails closed — deleting the bar would reconnect two
different prices); the Q=0 grid point is anchored at the day's SESSION OPEN
(optional ``open`` panel; default = first observed bar's price) instead of an
interpolation artifact; and ``buckets`` is ESTIMATOR resolution on the fixed
grid {8, 16, 32}, not an economic search dimension.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, ParamRole, ParamSpec, SeriesOperator, register_operator
from cleaned_operators.microstructure.intraday_agg import _as_panel

_EPS = 1e-12


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="intraday_microstructure",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "intraday", "daily_agg", "minute", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:price_volume",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _volume_clock_log_path(
    price: np.ndarray,
    activity: np.ndarray,
    buckets: int,
    open_px: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Resampled log-price path on the equal-activity grid -> (q_grid, log_p).

    R6-141: price must be strictly positive (``log`` domain) — a non-positive
    price is data-invalid, not a valid log-price.  R6-142: negative activity is
    a data error, not "no activity" — it fails the whole path closed rather
    than being filtered out and silently reconnecting the grid around it.

    P0-O #80: zero-activity bars are dropped from the equal-activity grid, so a
    zero-activity bar whose price differs from the previous OBSERVABLE
    (positive-activity) price would silently reconnect two different prices
    across the deleted bar.  When ``activity == 0`` the price must equal the
    previous observable price, else the day's data is inconsistent and the path
    fails closed (``None`` -> NaN).

    P0-O #81: the Q=0 point is the SESSION OPEN, prepended EXPLICITLY (never
    left to ``np.interp`` to clamp/extrapolate the first observation back to
    q=0).  When an ``open_px`` panel is supplied its first finite value is the
    session open; otherwise the first observed bar's price is the honest
    session-open proxy.
    """
    # R6-142: negative activity invalidates the entire path (a negative
    # contribution to cumulative activity is not a valid clock).  Zero activity
    # is a valid "no trade" minute.  NaN in either series fails the path.
    if np.any(~np.isfinite(price)) or np.any(~np.isfinite(activity)) or np.any(activity < 0.0):
        return None
    # R6-141: non-positive price is invalid for log-price (fail closed, never
    # a silently-produced -inf that then gets filtered by the finite mask).
    if np.any(price <= 0.0):
        return None
    valid = activity > 0.0
    if int(valid.sum()) < 3:
        return None
    # P0-O #80: a zero-activity bar must carry the SAME price as the previous
    # observable bar, else deleting it reconnects two different prices.
    if np.any(~valid):
        obs_idx = np.flatnonzero(valid)
        zero_idx = np.flatnonzero(~valid)
        pos = np.searchsorted(obs_idx, zero_idx, side="left") - 1
        for i, z in enumerate(zero_idx):
            if pos[i] < 0:
                continue  # leading zero-activity bars: nothing to reconnect
            prev = obs_idx[pos[i]]
            if float(price[z]) != float(price[prev]):
                return None
    act = activity[valid].astype(float)
    logp = np.log(price[valid].astype(float))
    Q = np.cumsum(act)
    Q = Q / Q[-1]
    # Deduplicate ties (zero-activity bars) so np.interp's xp is strictly rising.
    keep = np.concatenate(([True], np.diff(Q) > 0.0))
    Q = Q[keep]
    logp = logp[keep]
    if Q.shape[0] < 2 or Q[0] != Q[0]:
        return None
    # P0-O #81: explicit session-open anchor at Q=0.
    if open_px is not None:
        fin = np.isfinite(open_px)
        session_open = float(open_px[fin][0]) if np.any(fin) else float(price[valid][0])
    else:
        session_open = float(price[valid][0])
    if not np.isfinite(session_open) or session_open <= 0.0:
        return None
    Q0 = np.concatenate(([0.0], Q))
    logp0 = np.concatenate(([np.log(session_open)], logp))
    B = max(4, int(buckets))
    # R6-144: a smooth B-point path cannot be built from fewer distinct activity
    # points than B+1 — the interpolation would fabricate a path between
    # missing observations (roughness/curvature becomes an interpolation
    # artifact).  Fail closed instead of manufacturing smoothness.
    if Q0.shape[0] < B + 1:
        return None
    grid = np.linspace(0.0, 1.0, B + 1)
    path = np.interp(grid, Q0, logp0)
    return grid, path


def _volume_clock_efficiency(
    price: np.ndarray, activity: np.ndarray, buckets: int, open_px: np.ndarray | None = None
) -> float:
    res = _volume_clock_log_path(price, activity, buckets, open_px)
    if res is None:
        return np.nan
    _, path = res
    total_path = float(np.sum(np.abs(np.diff(path))))
    if total_path <= _EPS:
        return 0.0
    return float(abs(path[-1] - path[0]) / total_path)


def _volume_clock_roughness(
    price: np.ndarray, activity: np.ndarray, buckets: int, open_px: np.ndarray | None = None
) -> float:
    res = _volume_clock_log_path(price, activity, buckets, open_px)
    if res is None:
        return np.nan
    _, path = res
    delta = np.diff(path)
    delta2 = np.diff(delta)
    denom = float(np.sum(delta * delta))
    if denom <= _EPS:
        return 0.0
    return float(np.sum(delta2 * delta2) / denom)


def _volume_clock_daily_agg(
    price: pd.DataFrame,
    activity: pd.DataFrame,
    open_px: pd.DataFrame | None,
    fn: Callable[[np.ndarray, np.ndarray, np.ndarray | None], float],
) -> pd.DataFrame:
    """Per-(instrument, calendar-day) volume-clock aggregation.

    Replicates the PAIRED-MISSING policy of ``_daily_agg_two`` (a usable bar
    requires BOTH ``price`` and ``activity`` finite; a bar with exactly one
    finite member is dropped from both — never zero-filled), and attaches the
    minute-level ``open`` panel as a third series so the kernel can anchor the
    Q=0 point at the day's session open (P0-O #81).
    """
    pa = _as_panel(price)
    act = _as_panel(activity)
    opn = _as_panel(open_px) if open_px is not None else None
    out: dict[str, pd.Series] = {}
    for inst in pa.columns:
        a, b = pa[inst], act[inst]
        # R15-INC-186: NO ``dropna`` — dropping the missing bars would compress
        # the physical session axis and silently connect gaps into the volume-
        # clock path.  NaN positions are preserved and a required field missing
        # ANYWHERE inside the day fails the whole day (paired-censor with time
        # position retained, never time-axis compression).
        joined = pd.concat([a, b], axis=1, keys=["p", "a"])
        if opn is not None and inst in opn.columns:
            joined["o"] = opn[inst]
        else:
            joined["o"] = np.nan
        joined["day"] = joined.index.normalize()
        per_day: dict[pd.Timestamp, float] = {}
        for day, group in joined.groupby("day"):
            vals_p = np.asarray(group["p"], dtype=float)
            vals_a = np.asarray(group["a"], dtype=float)
            vals_o = np.asarray(group["o"], dtype=float)
            if not np.all(np.isfinite(vals_p)) or not np.all(np.isfinite(vals_a)):
                per_day[day] = np.nan
                continue
            try:
                per_day[day] = float(fn(vals_p, vals_a, vals_o))
            except (ValueError, ZeroDivisionError, OverflowError):
                per_day[day] = np.nan
        out[inst] = pd.Series(per_day, dtype=float)
    if not out:
        return pd.DataFrame(dtype=float)
    return pd.DataFrame(out).sort_index()


@register_operator(
    name="intraday_volume_clock_path_efficiency",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_volume_clock_path_efficiency",
    source="volume_clock",
)
class IntradayVolumeClockPathEfficiency(SeriesOperator):
    """成交量时钟路径效率 ``|p(1)-p(0)| / Σ|Δp_b|``（等 activity 网格）。

    按累计 activity 等分 ``buckets`` 档，在每档处对 log-price 线性插值；效率
    接近 1 = 一单位真实成交对应稳定单向位移；接近 0 = 大量成交却在来回震荡。
    与自然时钟 ``intra_path_efficiency`` 信息不同。P1。
    """

    metadata = _metadata(
        "intraday_volume_clock_path_efficiency",
        "成交量时钟路径效率（等 activity 网格，log-price）。",
        ["price", "activity", "buckets", "open"],
        unit="ratio",
        cost=4,
    )
    # R6-141/142: price must be > 0 (log domain), activity must be >= 0
    # (negative activity is a data error, not "no trade").  R6-144: buckets
    # must not exceed the distinct positive-activity points - 1, else the path
    # is interpolated smoothness, not observation.  P0-O #82: ``buckets`` is
    # ESTIMATOR resolution, not an economic dimension — a fixed small grid
    # {8, 16, 32}, non-searchable.
    metadata.param_specs = {
        "buckets": ParamSpec(
            dtype=int,
            choices=(8, 16, 32),
            default=16,
            searchable=False,
            param_role=ParamRole.ESTIMATOR_RESOLUTION,
        ),
    }

    def _calculate_series(
        self,
        price: pd.DataFrame,
        activity: pd.DataFrame,
        buckets: int = 16,
        open: pd.DataFrame | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        b = max(4, int(buckets))
        return _volume_clock_daily_agg(
            price, activity, open, lambda p, a, o: _volume_clock_efficiency(p, a, b, o)
        )


@register_operator(
    name="intraday_volume_clock_roughness",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="intraday_volume_clock_roughness",
    source="volume_clock",
)
class IntradayVolumeClockRoughness(SeriesOperator):
    """成交量时钟粗糙度 ``Σ(Δ²p_b)² / (Σ(Δp_b)² + eps)``。

    高 = 同一单位真实成交量下价格不断来回跳动；低 = 成交量推动价格稳定单向
    移动。与自然时钟 ``ts_roughness`` 不重复。P1。
    """

    metadata = _metadata(
        "intraday_volume_clock_roughness",
        "成交量时钟粗糙度（等 activity 网格二阶/一阶差平方比）。",
        ["price", "activity", "buckets", "open"],
        unit="ratio",
        cost=4,
    )
    # P0-O #82: ``buckets`` is estimator resolution, not an economic dimension.
    metadata.param_specs = {
        "buckets": ParamSpec(
            dtype=int,
            choices=(8, 16, 32),
            default=16,
            searchable=False,
            param_role=ParamRole.ESTIMATOR_RESOLUTION,
        ),
    }

    def _calculate_series(
        self,
        price: pd.DataFrame,
        activity: pd.DataFrame,
        buckets: int = 16,
        open: pd.DataFrame | None = None,
        **_: Any,
    ) -> pd.DataFrame:
        b = max(4, int(buckets))
        return _volume_clock_daily_agg(
            price, activity, open, lambda p, a, o: _volume_clock_roughness(p, a, b, o)
        )


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({
            "intraday_volume_clock_path_efficiency",
            "intraday_volume_clock_roughness",
        })


_register_surface()
