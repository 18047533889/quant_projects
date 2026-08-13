# -*- coding: utf-8 -*-
"""Intraday shock-recovery / resilience (2026-08 V2, P1, minute → daily).

``session_event_recovery_score`` measures how fast the day's event shocks get
absorbed.  For each event minute ``s`` the base ``B_s = x_{s-1}`` and the shock
``A_s = |x_s - B_s|``; the first ``k ∈ [1, horizon]`` with
``|x_{s+k} - B_s| ≤ residual_fraction * A_s`` gives ``τ_s`` (else ``H+1``).
The daily score is ``median_s τ_s / (H+1)``.

Low = shocks absorbed quickly (liquidity resilient); high = shock impact
persists (fragile).  ``event`` is a user-supplied EventBool (0/1) minute
indicator (return shock / volume shock / limit approach / imbalance shock); the
operator does not hard-code what an event is.

Round-11 contract (findings #75-#79)
-------------------------------------
* ``event`` must be EventBool — any finite non-binary value makes the day fail
  closed (R11 #75).
* ``residual_fraction`` is a recovery threshold: ``0 < residual_fraction <= 1``.
  Arbitrary large values are rejected (R11 #76).
* Session grouping uses the market's session wall-clock, never bare
  ``index.normalize()`` on a tz-aware index (R11 #77).
* A missing minute inside a recovery horizon makes the exact recovery time
  unidentifiable: the day emits NaN, never a false precise recovery at a later
  observed minute (R11 #78).
* Overlapping shocks do not reuse the same future recovery path as separate
  events: a ``refractory``-minute quiet window suppresses re-firing (R11 #79).

Round-3 audit (findings P0-87/P0-88/P0-89)
------------------------------------------
* The session timezone is NEVER guessed: a bare UTC / unknown tz-aware index
  must not silently default to Asia/Shanghai (a future US minute-data deployment
  would be misaligned).  An explicit ``session_tz`` or a ``calendar`` whose
  market maps to a session zone is required; otherwise the operator FAILS CLOSED
  (raises) (P0-87).
* A single shock is statistically unstable: the daily median over ``min_events``
  effective events is required, otherwise the day emits NaN (P0-88).
* ``residual_fraction`` is ``0 < f <= 1`` at BOTH compile time and runtime
  (a ``RelationalParamSpec`` makes 0 compile-invalid, matching the runtime) so
  the contract never admits a value that the kernel rejects (P0-89).

This is an EOD daily factor: it may use the whole day's minute data, so its
policy carries ``available_at = session_close`` and it must never be used as a
mid-session real-time minute signal.  Deterministic and prefix-causal.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import (
    OperatorMetadata,
    ParamRole,
    ParamSpec,
    RelationalParamSpec,
    SeriesOperator,
    register_operator,
)
from cleaned_operators.microstructure.intraday_agg import _as_panel
from runtime.session_panel import build_session_panel, default_ashare_calendar

_EPS = 1e-12
_SESSION_TZ = "Asia/Shanghai"
_US_TZ = "America/New_York"
_KNOWN_SESSION_ZONES = {
    "America/New_York", "US/Eastern", "Asia/Shanghai", "Asia/Hong_Kong",
    "Asia/Chongqing", "Asia/Urumqi",
}


def _market_session_tz(market: str | None) -> str:
    """Session wall-clock zone for an explicit calendar market; "" if unknown.

    P0-87: this is a DECLARED mapping from an explicit ``SessionCalendar``
    market — never a guess from a bare UTC index.
    """
    m = str(market or "").upper()
    if m in {"CN", "ASHARE", "A_SHARE"}:
        return _SESSION_TZ
    if m in {"HK", "HONG_KONG"}:
        return "Asia/Hong_Kong"
    if m in {"US"}:
        return _US_TZ
    return ""


def _resolve_session_tz(
    index: pd.DatetimeIndex,
    session_tz: str | None = None,
    calendar: Any = None,
) -> str:
    """Resolve the session wall-clock zone for a minute index (P0-87).

    Resolution order:
      1. explicit ``session_tz`` parameter;
      2. an explicit ``calendar`` whose ``market`` maps to a known session zone;
      3. a KNOWN session zone already stored on the index (conversion is then a
         no-op rename);
    otherwise FAIL CLOSED (raise) — the zone is never guessed.
    """
    tz = str(session_tz or "").strip()
    if not tz and calendar is not None:
        tz = _market_session_tz(getattr(calendar, "market", ""))
    if tz:
        return tz
    stored = str(getattr(index, "tz", None) or "")
    if stored and stored in _KNOWN_SESSION_ZONES:
        return stored
    if not stored:
        # tz-naive index: already in session wall-clock, no conversion needed.
        return ""
    raise ValueError(
        "session_event_recovery_score cannot resolve the session timezone for a "
        f"tz-aware index stored in {stored!r}: pass an explicit `session_tz` (or "
        "a `calendar` with a known market) — the session timezone is never "
        "guessed (P0-87)"
    )


def _session_local_frame(
    frame: pd.DataFrame,
    session_tz: str | None = None,
    calendar: Any = None,
) -> pd.DataFrame:
    """Convert a tz-aware minute panel to session wall-clock (naive) (R11 #77)."""
    idx = frame.index
    if isinstance(idx, pd.DatetimeIndex) and getattr(idx, "tz", None) is not None:
        tz = _resolve_session_tz(idx, session_tz, calendar)
        out = frame.tz_convert(tz)
        out.index = out.index.tz_localize(None)
        return out
    return frame


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="intraday_microstructure",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "intraday", "daily_agg", "minute", "pit_safe", "causal", "typed_v2",
            "deterministic", "session_close",
            f"signature:{','.join(params)}->series", "domain:price_volume",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _recovery_day(
    x: np.ndarray,
    event: np.ndarray,
    horizon: int,
    residual_fraction: float,
    refractory: int = 1,
    min_events: int = 3,
) -> float:
    """Recovery median over grid-aligned arrays.

    R26-050: ``x`` / ``event`` must be aligned to the official session grid, so
    ``s + k`` is ``k`` REAL official slots — never ``k`` observed rows.  The
    caller (the operator) builds grid-aligned panels; a missing timestamp is an
    explicit NaN slot, and the existing missing-price censor already fails
    closed across it.
    """
    H = max(1, int(horizon))
    if not (0.0 < float(residual_fraction) <= 1.0):
        raise ValueError("session_event_recovery_score requires 0 < residual_fraction <= 1")
    c = float(residual_fraction)
    rf = max(0, int(refractory))
    me = max(1, int(min_events))
    n = x.shape[0]
    # R11 #75: event must be EventBool (0/1).  Any finite non-binary value is an
    # invalid event panel -> the whole day fails closed.
    fin = np.isfinite(event)
    if np.any(fin & ~np.isin(event[fin], (0.0, 1.0))):
        return np.nan
    taus: list[float] = []
    suppress_until = -1
    for s in range(1, n):
        if event[s] == 0.0:
            continue
        if not np.isfinite(event[s]):
            # R26-049 EventMissingPolicy.BREAK/CENSOR: an unknown event slot must
            # never silently read as "no event".  It does not trigger a recovery
            # path, and the refractory context must not be assumed continuous
            # across it — leave the day fail-closed only when a *used* path is
            # ambiguous (handled in the horizon loop below).
            continue
        if s <= suppress_until:
            # R11 #79: refractory — an overlapping shock inside the quiet window
            # must not reuse the same future recovery path as a separate event.
            continue
        # Right censoring: an event whose full recovery horizon extends past the
        # last observed minute is *unobserved*, not a failure.  Excluding it
        # entirely is the honest survival-analysis treatment — a late-day event
        # must not drag the daily median toward "no recovery" (P0-04).
        if s + H >= n:
            continue
        b = x[s - 1]
        cur = x[s]
        if not np.isfinite(b) or not np.isfinite(cur):
            continue
        a = abs(cur - b)
        if a <= _EPS:
            continue
        tau = H + 1
        censored = False
        for k in range(1, H + 1):
            v = x[s + k]
            if not np.isfinite(v):
                # R11 #78: a missing minute inside the recovery horizon makes the
                # exact recovery time only interval-identifiable — the true
                # recovery lies in (s+k-1, s+k].  Emit NaN (fail closed) rather
                # than a false precise recovery at a later observed minute.
                censored = True
                break
            if not np.isfinite(event[s + k]):
                # R26-049: an UNKNOWN event slot inside the horizon censors the
                # path — the episode is not seamlessly connected across it.
                censored = True
                break
            if abs(v - b) <= c * a:
                tau = k
                break
        if censored:
            return np.nan
        taus.append(float(tau))
        suppress_until = s + rf
    if len(taus) < me:
        # P0-88 / R26-047/048: fewer than ``min_events`` effective events
        # (after right-censoring / refractory suppression / unknown-event
        # censoring) makes the median statistically unstable — fail closed.
        # The DEFAULT (``min_events=3``) is the reviewed floor; the doc no
        # longer claims single-shock output while the default allows it.
        return np.nan
    return float(np.median(taus) / (H + 1))


@register_operator(
    name="session_event_recovery_score",
    category="intraday_microstructure",
    business_category="intraday_microstructure",
    canonical="session_event_recovery_score",
    source="session_recovery",
)
class SessionEventRecoveryScore(SeriesOperator):
    """日内事件冲击恢复得分（EOD，``available_at=session_close``）。

    对每个事件分钟 s，测价格回到冲击前基线的首个 horizon 内位置；输出
    ``median_s τ_s/(H+1)`` ∈ (0,1]。低 = 冲击快速被吸收；高 = 影响持续。
    PIT 安全仅限日终因子，禁止盘中实时信号。
    """

    metadata = _metadata(
        "session_event_recovery_score",
        "日内事件冲击恢复得分 median τ/(H+1)（EOD 因子）。",
        ["x", "event", "horizon", "residual_fraction", "refractory", "session_tz", "min_events", "calendar"],
        unit="ratio",
        cost=4,
    )
    metadata.param_specs = {
        "horizon": ParamSpec(dtype=int, min=1),
        # P0-89: the strict lower bound lives in relational_specs below so that
        # 0 is compile-INVALID exactly like the runtime, not just runtime-invalid.
        "residual_fraction": ParamSpec(dtype=float, min=0.0, max=1.0),
        "refractory": ParamSpec(dtype=int, min=0, searchable=False, param_role=ParamRole.POLICY),
        # R26-048: default + ParamSpec floor = the reviewed min-effective-sample
        # (>=3 effective events before a daily median is meaningful).
        "min_events": ParamSpec(dtype=int, min=3, searchable=False, param_role=ParamRole.ESTIMATOR_RESOLUTION),  # P0-88 / R26-048
        "session_tz": ParamSpec(dtype=str, searchable=False, param_role=ParamRole.POLICY),  # R11 #77 / P0-87
    }
    metadata.relational_specs = [
        RelationalParamSpec(
            expression="residual_fraction > 0",
            message="residual_fraction must be > 0 (compile-time and runtime agree, P0-89)",
        )
    ]

    def _calculate_series(
        self,
        x: pd.DataFrame,
        event: pd.DataFrame,
        horizon: int = 10,
        residual_fraction: float = 0.25,
        refractory: int = 1,
        session_tz: str | None = None,
        min_events: int = 3,
        calendar: Any = None,
        **_: Any,
    ) -> pd.DataFrame:
        # R26-047/048: min_events default = the reviewed floor (3 effective
        # events) — a single shock must NOT produce a daily score by default.
        if not (0.0 < float(residual_fraction) <= 1.0):
            raise ValueError("session_event_recovery_score requires 0 < residual_fraction <= 1")
        rf = max(0, int(refractory))
        me = max(1, int(min_events))
        x = _session_local_frame(_as_panel(x), session_tz, calendar)
        event = _session_local_frame(_as_panel(event), session_tz, calendar)
        cal = calendar if calendar is not None else default_ashare_calendar(bar_freq="1min")
        tz = session_tz or _SESSION_TZ
        # R30 §30 (P0-015): the session builder's market must come from the
        # calendar it is given — a calendar that claims a market drives the
        # panel build; a bare A-share default stays A-share.  Never pretend a
        # US/HK timezone is supported while the session builder is still
        # hardcoded to the A-share calendar.
        cal_market = str(getattr(cal, "market", "") or "")
        if not cal_market:
            raise ValueError(
                "session_event_recovery_score requires a calendar with an explicit "
                "market (R30 §30 — no half-generic session builder)"
            )
        _market = cal_market.lower()
        out: dict[str, pd.Series] = {}
        for inst in x.columns:
            col = x[inst]
            ev = event[inst] if inst in event.columns else None
            per_day: dict[pd.Timestamp, float] = {}
            for day, group in col.groupby(col.index.normalize()):
                vals = np.asarray(group, dtype=float)
                times = np.asarray(group.index, dtype="datetime64[ns]")
                if not np.any(np.isfinite(vals)):
                    per_day[day] = np.nan
                    continue
                try:
                    px = build_session_panel(
                        times, vals, cal, market=_market, session_timezone=tz,
                        source_timezone=None, trade_date=pd.Timestamp(day),
                    )
                    if ev is None:
                        per_day[day] = np.nan
                        continue
                    ev_vals = np.asarray(ev.reindex(group.index), dtype=float)
                    pe = build_session_panel(
                        times, ev_vals, cal, market=_market, session_timezone=tz,
                        source_timezone=None, trade_date=pd.Timestamp(day),
                    )
                    # R26-051: an EOD recovery factor requires proof the session
                    # actually reached its close.  A truncated session (data ends
                    # at 14:00) must not emit a full-session factor — the close
                    # slot must carry a valid bar.
                    if not px.is_valid_bar[px.n_slots - 1]:
                        per_day[day] = np.nan
                        continue
                    per_day[day] = _recovery_day(px.values, pe.values, horizon, float(residual_fraction), rf, me)
                except (ValueError, ZeroDivisionError, OverflowError):
                    per_day[day] = np.nan
            out[inst] = pd.Series(per_day, dtype=float)
        if not out:
            return pd.DataFrame(dtype=float)
        return pd.DataFrame(out).sort_index()


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only({"session_event_recovery_score"})


_register_surface()
