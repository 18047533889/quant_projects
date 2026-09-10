# -*- coding: utf-8 -*-
"""Shared kernels for next-stage intraday operators.

Self-contained helpers (adapted from the reviewed ``microstructure`` contract):
every operator accepts minute-frequency panels (row index is a minute
timestamp, columns are instruments) and returns a daily-frequency panel
(row index is the calendar date, columns are instruments).

Contract
--------
* One scalar per (TradeDate, Symbol).  Never a row per minute.
* All kernels are causal: only the day's own minute data plus that day's daily
  limit / weight panels, never future bars or future days.
* Empty / all-NaN windows yield NaN, never Inf or a fabricated zero.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, SeriesOperator

_EPS = 1e-12
# P0-09: minimum number of finite within-session returns required before a
# realized skewness / kurtosis estimate is emitted.  30 returns == at least 31
# closes; below that the sample is too degenerate to identify a third/fourth
# moment (n=1 trivially yields +/-1 skewness and 1.0 kurtosis, n=2 is noise).
_REALIZED_MIN_RETURNS = 30
_SESSION_TZ = "Asia/Shanghai"
_MORNING = (570, 690)   # 09:30 .. 11:30 minute-of-day
_AFTERNOON = (780, 900)  # 13:00 .. 15:00 minute-of-day
_SEGMENT_RANGES = {"morning": _MORNING, "afternoon": _AFTERNOON}


class DataDegeneracy(ValueError):
    """A kernel hit genuinely degenerate *data* (not a programming error).

    P1-18: the daily-aggregation wrappers map only this exception (plus
    ``ZeroDivisionError`` / ``OverflowError``, which are by definition data
    degeneracies) to ``NaN``.  A plain ``ValueError`` from a bad parameter or a
    kernel bug must propagate so it is never silently swallowed as a missing
    estimate.
    """


class SessionCoveragePolicy(str, Enum):
    """Coverage policy for accepting a session's partial-bar mask."""

    STRICT_FULL_SESSION = "strict_full_session"
    MIN_COVERAGE_095 = "min_coverage_095"
    PAIRWISE_ALLOWED = "pairwise_allowed"
    EVENT_SPARSE = "event_sparse"


@dataclass
class SessionGrid:
    """Session structure of a minute panel (P0-08).

    Describes the per-trade-date minute-slot grid so multi-input minute panels
    can be checked for alignment *before* a ``pd.concat(...).dropna(...)`` step
    silently compresses a mismatched axis into a shorter (fabricated) grid.

    Fields
    ------
    market:          market identifier (``A-share`` when the index is tz-aware).
    trade_date:      sorted tuple of unique trade dates (normalized).
    expected_slots:  number of minute slots in a full session (modal day width).
    session_slot:    per-row minute-of-day (int).
    present_mask:    per-row bool; True when the row lies on a full-width session
                     day (all expected slots observed).
    """

    market: str
    trade_date: tuple
    expected_slots: int
    session_slot: np.ndarray
    present_mask: np.ndarray
    day_slots: tuple = ()  # ((day, (minute-of-day slots...)), ...)

    def slot_structure(self) -> tuple:
        """Canonical (day -> minute-slot tuple) key for grid-equality checks.

        Two grids are equal only when every trade date maps to the SAME set of
        minute-of-day slots — comparing just the date list or the slot *count*
        would miss a panel whose bars sit at different minutes of the session.
        """
        return self.day_slots


def build_session_grid(index: Any, market: str | None = None) -> SessionGrid:
    """Build the session grid of a minute ``DatetimeIndex`` (P0-08).

    ``present_mask`` marks bars on days whose observed slot count equals the
    modal full-session width; partial days are still represented in
    ``trade_date`` but are not ``present``.  The grid is used by
    ``require_same_session_grid`` to fail closed on mismatched multi-input
    panels.
    """
    idx = pd.DatetimeIndex(index)
    if market is None:
        market = "A-share" if getattr(idx, "tz", None) is not None else "unknown"
    minutes = minute_of_day(idx.to_numpy(dtype="datetime64[ns]"))
    days = idx.normalize()
    day_slots: dict[pd.Timestamp, set[int]] = {}
    for day, m in zip(days, minutes):
        day_slots.setdefault(pd.Timestamp(day), set()).add(int(m))
    sorted_days = tuple(sorted(day_slots.keys()))
    slots_per_day = tuple((d, tuple(sorted(day_slots[d]))) for d in sorted_days)
    expected_slots = int(max((len(s) for _, s in slots_per_day), default=0))
    full_days = {d for d, s in slots_per_day if len(s) == expected_slots}
    present_mask = np.asarray([pd.Timestamp(day) in full_days for day in days], dtype=bool)
    return SessionGrid(
        market=market,
        trade_date=sorted_days,
        expected_slots=expected_slots,
        session_slot=minutes.astype(np.int64),
        present_mask=present_mask,
        day_slots=slots_per_day,
    )


def require_same_session_grid(*frames: Any) -> None:
    """Fail closed when minute panels do NOT share the same session grid (P0-08).

    A grid mismatch — different trade dates or different per-day slot
    resolution — would otherwise be silently compressed by the
    ``pd.concat(...).dropna(subset=[...])`` sites in the daily-aggregation
    kernels, producing a day estimate on a partial axis.  Raise instead.
    """
    if len(frames) < 2:
        return
    grids = [build_session_grid(f.index) for f in frames]
    first = grids[0]
    for g in grids[1:]:
        if g.slot_structure() != first.slot_structure():
            raise ValueError(
                "intraday panels do not share the same session grid: "
                f"{first.market} trade_dates={len(first.trade_date)} "
                f"expected_slots={first.expected_slots} vs "
                f"{g.market} trade_dates={len(g.trade_date)} "
                f"expected_slots={g.expected_slots} — refusing to splice a "
                "partial axis into a day estimate"
            )


def session_coverage_ok(present_mask: np.ndarray | None, policy: SessionCoveragePolicy) -> bool:
    """Return whether a session's present-bar mask satisfies ``policy`` (P0-09)."""
    if present_mask is None or len(present_mask) == 0:
        return False
    mask = np.asarray(present_mask, dtype=bool)
    n_present = int(np.sum(mask))
    n_total = int(len(mask))
    coverage = n_present / n_total if n_total else 0.0
    if policy is SessionCoveragePolicy.STRICT_FULL_SESSION:
        return n_present == n_total
    if policy is SessionCoveragePolicy.MIN_COVERAGE_095:
        return coverage >= 0.95
    if policy is SessionCoveragePolicy.PAIRWISE_ALLOWED:
        return n_present >= 2
    if policy is SessionCoveragePolicy.EVENT_SPARSE:
        return n_present >= 1
    return False


class SessionAggregationOperator(SeriesOperator):
    """Base class for intraday minute -> daily aggregation operators (P0-07).

    Every operator here genuinely changes time frequency: it consumes a
    minute-frequency panel and emits one scalar per (TradeDate, Symbol).  The
    grain contract is declared on the class and mirrored into
    ``OperatorMetadata.input_grain`` / ``output_grain`` by ``_core.metadata()``.
    ``calculate`` runs the shared framework validation gate then delegates to
    ``_calculate_series`` exactly like ``SeriesOperator`` — no kernel math is
    changed, only the frequency contract is made explicit.
    """

    input_grain: str = "minute"
    output_grain: str = "daily"

    # R5-02: ``calculate`` below routes through ``_prepare_call`` ->
    # ``validate_operator_call`` (identical to the framework ``SeriesOperator``
    # pattern), so this custom method must be declared contract-handling.
    _HANDLES_CALL_CONTRACT = True

    def calculate(self, *args, **kwargs) -> pd.DataFrame:
        processed_args, processed_kwargs = self._prepare_call(args, kwargs)
        return self._calculate_series(*processed_args, **processed_kwargs)


def metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    unit: str,
    domain: str = "intraday",
    cost: int = 6,
    extra_tags: list[str] | None = None,
    available_at: str | None = None,
    same_session_usable: bool | None = None,
) -> OperatorMetadata:
    """Standard metadata for intraday -> daily aggregation operators."""
    tags = [
        "intraday", "daily_agg", "minute", "pit_safe", "causal",
        "typed_v2", "source_blocked", "grain_minute_to_daily",
        f"signature:{','.join(params)}->series", f"unit:{unit}",
        f"cost:{cost}", f"domain:{domain}",
    ]
    if extra_tags:
        tags.extend(extra_tags)
    return OperatorMetadata(
        name=name,
        category="intraday_microstructure",
        description=description,
        param_names=params,
        return_type="series",
        tags=tags,
        input_grain="minute",
        output_grain="daily",
        available_at=available_at,
        same_session_usable=same_session_usable,
    )


def as_panel(x: Any) -> pd.DataFrame:
    if isinstance(x, pd.Series):
        return x.to_frame(getattr(x, "name", None) or "value")
    return x


def minute_of_day(times: np.ndarray) -> np.ndarray:
    """Timestamp array -> minute-of-day (integer)."""
    seconds = times.astype("datetime64[s]").astype("int64") % 86400
    return seconds // 60


def log_returns(vals: np.ndarray) -> np.ndarray:
    out = np.full(len(vals), np.nan)
    if len(vals) > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[1:] = np.log(vals[1:] / vals[:-1])
    return out


def session_local(frame: pd.DataFrame, tz: str | None = None) -> pd.DataFrame:
    """Convert a tz-aware index to session wall-clock (naive) for minute math.

    A-share COS minute data is stored in UTC; session segments are defined in
    Asia/Shanghai wall-clock.  Naive indexes are assumed to already be local.
    """
    frame = as_panel(frame)
    if isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None:
        tz = tz or _SESSION_TZ
        frame = frame.tz_convert(tz)
        frame.index = frame.index.tz_localize(None)
    return frame


def seg_mask(times: np.ndarray, segment: str) -> np.ndarray:
    lo, hi = _SEGMENT_RANGES[str(segment)]
    minutes = minute_of_day(times)
    return (minutes >= lo) & (minutes <= hi)


def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.isfinite(den) & (np.abs(den) > _EPS), num / den, np.nan)
    return out


def daily_agg(
    frame: pd.DataFrame,
    fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    min_finite: int = 2,
    _grid: tuple | None = None,
) -> pd.DataFrame:
    """Apply per-(instrument, calendar-day) aggregation fn(vals, times).

    ``min_finite`` (P0-09) is the smallest number of finite minute values a day
    must have before a day estimate is produced; a single finite value can no
    longer manufacture one.  Only ``DataDegeneracy`` (P1-18) — plus genuine
    ``ZeroDivisionError`` / ``OverflowError`` — maps to ``NaN``; a plain
    ``ValueError`` from a parameter/kernel bug propagates.

    ``_grid`` (optional) is a pre-built ``_grid3`` tuple so the vectorized fast
    path can reuse a grid materialized once by ``IntradayFeatureCompiler``
    (P0#5 single-scan).  When omitted the fast path builds its own grid.
    """
    frame = as_panel(frame)
    _fast = _vec_daily_agg(frame, fn, min_finite=min_finite, _grid=_grid)
    if _fast is not None:
        return _fast
    import factor_engine.cleaned_operators.intraday.perf_vec_telemetry as _pvt
    _pvt.increment_scalar_fallback()
    out: dict[str, pd.Series] = {}
    for inst in frame.columns:
        col = frame[inst]
        per_day: dict[pd.Timestamp, float] = {}
        for day, group in col.groupby(col.index.normalize()):
            vals = np.asarray(group, dtype=float)
            times = np.asarray(group.index, dtype="datetime64[ns]")
            if int(np.sum(np.isfinite(vals))) < int(min_finite):
                per_day[day] = np.nan
                continue
            try:
                per_day[day] = float(fn(vals, times))
            except (DataDegeneracy, ZeroDivisionError, OverflowError):
                per_day[day] = np.nan
        out[inst] = pd.Series(per_day, dtype=float)
    if not out:
        return pd.DataFrame(dtype=float)
    return pd.DataFrame(out).sort_index()


def daily_agg_two(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    min_finite: int = 2,
    _grid: tuple | None = None,
) -> pd.DataFrame:
    """Apply fn(a_vals, b_vals) per (instrument, day).

    The two frames must share the same session grid (P0-08). Rows remain on
    that physical clock so a missing primary value breaks adjacent returns.

    PERF-2 dispatch (100k GO §6.2): ``fn`` is the KERNEL ITSELF (e.g.
    ``tg._price_delay_kernel``), which carries the ``__vec__`` attribute; the
    legacy lambda-wrapped call sites (``lambda v, vol: fn(v, vol, None)``)
    cannot expose ``__vec__``.  Scalar kernels take ``(a, b, times)`` with
    ``times=None`` tolerated — the unwrap below binds the third argument so
    both call conventions work.

    ``_grid`` (optional) is a pre-built ``_grid3`` tuple so the vectorized fast
    path can reuse a grid materialized once by ``IntradayFeatureCompiler``
    (P0#5 single-scan).  When omitted the fast path builds its own grid.
    """
    frame_a, frame_b = as_panel(frame_a), as_panel(frame_b)
    require_same_session_grid(frame_a, frame_b)
    _fast = _vec_daily_agg_two(frame_a, frame_b, fn, min_finite=min_finite, _grid=_grid)
    if _fast is not None:
        return _fast
    import factor_engine.cleaned_operators.intraday.perf_vec_telemetry as _pvt
    _pvt.increment_scalar_fallback()
    fn2 = fn
    import inspect as _inspect

    try:
        _params = _inspect.signature(fn).parameters
        if len(_params) >= 3:
            fn2 = (lambda f: lambda va, vb: f(va, vb, None))(fn)
    except (TypeError, ValueError):
        fn2 = fn
    out: dict[str, pd.Series] = {}
    for inst in frame_a.columns:
        a, b = frame_a[inst], frame_b[inst]
        joined = pd.concat([a, b], axis=1, keys=["a", "b"])
        joined["day"] = joined.index.normalize()
        per_day: dict[pd.Timestamp, float] = {}
        for day, group in joined.groupby("day"):
            vals_a = np.asarray(group["a"], dtype=float)
            vals_b = np.asarray(group["b"], dtype=float)
            if int(np.sum(np.isfinite(vals_a))) < int(min_finite):
                per_day[day] = np.nan
                continue
            try:
                per_day[day] = float(fn2(vals_a, vals_b))
            except (DataDegeneracy, ZeroDivisionError, OverflowError):
                per_day[day] = np.nan
        out[inst] = pd.Series(per_day, dtype=float)
    if not out:
        return pd.DataFrame(dtype=float)
    return pd.DataFrame(out).sort_index()


def daily_agg_three(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    frame_c: pd.DataFrame,
    fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    *,
    min_finite: int = 2,
    _grid: tuple | None = None,
) -> pd.DataFrame:
    """Apply fn(a_vals, b_vals, c_vals) per (instrument, day).

    Same session-grid guard (P0-08) and ``DataDegeneracy`` catch (P1-18) as
    ``daily_agg_two``.

    ``_grid`` (optional) is a pre-built ``_grid3`` tuple so the vectorized fast
    path can reuse a grid materialized once by ``IntradayFeatureCompiler``
    (P0#5 single-scan).  When omitted the fast path builds its own grid.
    """
    frame_a, frame_b, frame_c = as_panel(frame_a), as_panel(frame_b), as_panel(frame_c)
    require_same_session_grid(frame_a, frame_b, frame_c)
    _fast = _vec_daily_agg_three(frame_a, frame_b, frame_c, fn, min_finite=min_finite, _grid=_grid)
    if _fast is not None:
        return _fast
    import factor_engine.cleaned_operators.intraday.perf_vec_telemetry as _pvt
    _pvt.increment_scalar_fallback()
    out: dict[str, pd.Series] = {}
    for inst in frame_a.columns:
        joined = pd.concat(
            [frame_a[inst], frame_b[inst], frame_c[inst]], axis=1, keys=["a", "b", "c"]
        ).dropna(subset=["a"])
        joined["day"] = joined.index.normalize()
        per_day: dict[pd.Timestamp, float] = {}
        for day, group in joined.groupby("day"):
            vals_a = np.asarray(group["a"], dtype=float)
            vals_b = np.asarray(group["b"], dtype=float)
            vals_c = np.asarray(group["c"], dtype=float)
            if int(np.sum(np.isfinite(vals_a))) < int(min_finite):
                per_day[day] = np.nan
                continue
            try:
                per_day[day] = float(fn(vals_a, vals_b, vals_c))
            except (DataDegeneracy, ZeroDivisionError, OverflowError):
                per_day[day] = np.nan
        out[inst] = pd.Series(per_day, dtype=float)
    if not out:
        return pd.DataFrame(dtype=float)
    return pd.DataFrame(out).sort_index()


def broadcast_daily_panel(close_frame: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Broadcast a daily panel (index=date) onto a minute panel by date."""
    daily = as_panel(daily)
    out: dict[str, pd.Series] = {}
    for inst in close_frame.columns:
        if inst not in daily.columns:
            continue
        close_col = close_frame[inst]
        daily_col = daily[inst]
        days = close_col.index.normalize()
        mapped = daily_col.reindex(pd.DatetimeIndex(days.unique()))
        out[inst] = pd.Series(
            mapped.reindex(pd.DatetimeIndex(days)).to_numpy(),
            index=close_col.index,
        )
    return pd.DataFrame(out)


def register_surface(canonicals: list[str]) -> None:
    """Append new canonicals to the reviewed extended surface.

    Union is order-independent so concurrent modules may each add their own
    names without clobbering one another.
    """
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(canonicals))


def register_research_surface(canonicals: list[str]) -> None:
    """Append new canonicals to the research surface (experimental operators)."""
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_research_only(set(canonicals))


# Silent-numpy context for statistic kernels that may overflow intermediate
# products on extreme minute moves.
_np_err = {"divide": "ignore", "invalid": "ignore", "over": "ignore"}


def np_errstate():
    return np.errstate(**_np_err)


def mu_p(p: float) -> float:
    """E|Z|^p for a standard normal Z."""
    return float(2.0 ** (p / 2.0) * math.gamma((p + 1.0) / 2.0) / math.gamma(0.5))


def tripower_scale() -> float:
    """mu_{4/3}^{-3} constant for tripower quarticity."""
    return float(mu_p(4.0 / 3.0) ** -3)


# ---------------------------------------------------------------------------
# PERF-2: (day, bar, inst) vectorized daily_agg fast paths.
#
# The scalar ``daily_agg{,_two,_three}`` loops ``by (instrument, day)`` grouped
# frames and calls the python kernel ~(D * C) times.  For minute panels
# (C ~ up to 4000 instruments, D ~ trade days, 240 bars/day) the Python +
# groupby overhead dwarfs the kernel math.  These fast paths re-materialize
# the grid as a 3-D ndarray (day, bar_max, inst) and run a small set of
# algebraically-equivalent kernels in bulk with numpy ops.
#
# CONSTRAINTS
# * One scalar per (TradeDate, Symbol); never a row per minute.
# * Byte-identical only for kernels already proven equivalent by the
#   PERF-2 harness (rtol/atol 1e-12).  Never fake PASS: an unproven kernel
#   is NOT whitelisted and silently falls back to the scalar kernel path.
# * min_finite / DataDegeneracy mapping are preserved exactly.
# ---------------------------------------------------------------------------

_DAILY_AGG_VEC_REGISTRY: dict[str, tuple[int, Any]] = {}


def register_daily_agg_vec(rid: str, n: int, fn) -> None:
    """Declare a byte-equivalent vectorized kernel (PERF-2 whitelist).

    ``rid`` unambiguously identifies the (inserter-module, operator, params)
    tuple; ``n`` is the number of input panels it consumes; ``fn`` as
    documented on the daily_agg dispatch points.
    """
    _DAILY_AGG_VEC_REGISTRY[rid] = (int(n), fn)


# ---- (day, bar, inst) grid materialization ---------------------------------


def _grid3(
    a: pd.DataFrame,
    b: pd.DataFrame | None = None,
    c: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, list[pd.DataFrame]]:
    """Materialize (day, bar, inst) grids for panels a, [b, c].

    Returns
    -------
    grids : list of (D, B, C) float64 arrays (a always first).
    codes : (B,) int day codes for each bar row.
    uniq_days : pd.DatetimeIndex of unique normalized days (sorted).
    active : (D,) int count of a-finite rows per day.
    frames : the reindexed-to-union panels (a, b, c).
    """
    frames = [as_panel(a)]
    for f in (b, c):
        if f is not None:
            frames.append(as_panel(f))
    n_panels = len(frames)
    joined = pd.concat(frames, axis=1, join="outer", keys=[f"p{i}" for i in range(n_panels)])
    idx = joined.index
    days = pd.DatetimeIndex(idx).normalize()
    codes, uniques = pd.factorize(days, sort=True)
    D = int(len(uniques))
    C = int(frames[0].shape[1])
    filled = [f.reindex(idx).to_numpy(dtype=float) for f in frames]
    lengths = np.bincount(codes, minlength=D) if D else np.zeros(0, dtype=np.int64)
    slots = int(lengths.max()) if lengths.size else 0
    grids = [np.full((D, slots, C), np.nan, dtype=np.float64) for _ in filled]
    active = np.zeros(D, dtype=np.int64)
    for d in range(D):
        rows = codes == d
        nrows = int(lengths[d])
        for grid, values in zip(grids, filled):
            grid[d, :nrows, :] = values[rows]
        active[d] = int(np.sum(np.isfinite(filled[0][rows])))
    uniq_days = pd.DatetimeIndex(uniques)
    return grids[0], grids[1] if len(grids) > 1 else None, (
        grids[2] if len(grids) > 2 else None
    ), codes, uniq_days, active, filled


def _mask_finite(x: np.ndarray) -> np.ndarray:
    """(D,B,C) finite mask (used by kernels)."""
    return np.isfinite(x)


def _to_panel(result: np.ndarray, uniq_days: pd.DatetimeIndex, columns) -> pd.DataFrame:
    """Wrap a (D, C) result array as a daily panel with float dtype."""
    return pd.DataFrame(result, index=uniq_days, columns=columns, dtype=float).sort_index()


def _count_valid(grid: np.ndarray, min_finite: int) -> np.ndarray:
    """(D, C) count of finite values per (day, inst)."""
    return np.sum(np.isfinite(grid), axis=1)


def _vec_out(mask: np.ndarray, result: np.ndarray) -> np.ndarray:
    """NaN-out result cells where mask is False (day invalid)."""
    out = result.copy()
    out[~mask] = np.nan
    return out


# ---- dispatch hooks (attached to daily_agg*) -------------------------------


def _vec_daily_agg(
    a: pd.DataFrame,
    fn,
    *,
    min_finite: int = 2,
    _grid: tuple | None = None,
) -> pd.DataFrame | None:
    fnv = getattr(fn, "__vec__", None)
    if fnv is None:
        return None
    return fnv(a, min_finite=min_finite, _grid=_grid)


def _vec_daily_agg_two(
    a: pd.DataFrame,
    b: pd.DataFrame,
    fn,
    *,
    min_finite: int = 2,
    _grid: tuple | None = None,
) -> pd.DataFrame | None:
    fnv = getattr(fn, "__vec__", None)
    if fnv is None:
        return None
    return fnv(a, b, min_finite=min_finite, _grid=_grid)


def _vec_daily_agg_three(
    a: pd.DataFrame,
    b: pd.DataFrame,
    c: pd.DataFrame,
    fn,
    *,
    min_finite: int = 2,
    _grid: tuple | None = None,
) -> pd.DataFrame | None:
    fnv = getattr(fn, "__vec__", None)
    if fnv is None:
        return None
    return fnv(a, b, c, min_finite=min_finite, _grid=_grid)


# ---- shared vectorized helpers (used by whitelisted kernels below) ---------


def _vec_valid_counter(grid: np.ndarray, min_finite: int) -> np.ndarray:
    """(D, C) count of finite cells per (day, inst)."""
    return np.sum(np.isfinite(grid), axis=1)


def _vec_result_df(result: np.ndarray, uniq_days: pd.DatetimeIndex, columns) -> pd.DataFrame:
    return pd.DataFrame(result, index=uniq_days, columns=columns, dtype=float).sort_index()


def _vec_autocorr1(x: np.ndarray) -> np.ndarray:
    """(D,C) lag-1 Pearson autocorrelation of finite-value-only slices.

    Matches np.corrcoef on the same non-NaN slice (r = Sxy/(Sx*Sy)).
    """
    n = np.sum(np.isfinite(x), axis=1)
    xr = np.where(np.isfinite(x), x, 0.0)
    xo = np.roll(xr, shift=-1, axis=1)
    fin = np.isfinite(x)
    f1 = np.roll(fin, shift=-1, axis=1) & fin
    f1[:, -1] = False
    cnt = np.sum(f1, axis=1)
    zero = cnt < 2
    xa = np.where(f1, x, 0.0)
    xb = np.where(f1, np.roll(x, shift=-1, axis=1), 0.0)
    sx = np.sum(xa, axis=1)
    sy = np.sum(xb, axis=1)
    sxx = np.sum(xa * xa, axis=1)
    syy = np.sum(xb * xb, axis=1)
    sxy = np.sum(xa * xb, axis=1)
    denom = np.sqrt((cnt * sxx - sx * sx) * (cnt * syy - sy * sy))
    r = np.full(cnt.shape, np.nan)
    ok = (denom > 0) & np.isfinite(denom)
    r[ok] = (cnt * sxy - sx * sy)[ok] / denom[ok]
    r[zero] = np.nan
    return r


# ---------------------------------------------------------------------------
# PERF-2 whitelisted kernels (registered below).  These mirror the scalar
# kernels in true_gap_batch3/vwap_path/smart_money EXACTLY; equivalence is
# enforced by the PERF-2 harness (factor_engine/tests/perf_intra_vec_equiv.py),
# which compares vectored vs scalar over random fixtures incl. NaN gaps and
# all-NaN days with rtol/atol 1e-12.  Only harness-proven kernels stay here.
# ---------------------------------------------------------------------------

def _vec_session_mean_reversion(
    a: pd.DataFrame, *, min_finite: int = 2, _grid: tuple | None = None
) -> pd.DataFrame:
    """Vector intra_session_mean_reversion (equal-weighted session mean reversion).

    Scalar kernel (``true_gap_batch3._session_mean_reversion_kernel``): drop
    NaN bars, take the mean of the finite values, compute deviations, then the
    lag-1 autocorrelation over CONSECUTIVE FINITE values (``dev[:-1]`` vs
    ``dev[1:]``) and negate.  The drop-then-pair semantic is what the scalar
    ``np.corrcoef(dev[:-1], dev[1:])`` produces — pairing consecutive finite
    bars, NOT consecutive grid positions (a NaN gap must not split the pair).

    Vectorization: per (day, inst) column the finite values are packed to the
    front with a stable argsort (preserving intra-session order), so the
    packed prefix ``[:cnt]`` is exactly the scalar ``finite`` slice; the
    autocorrelation then runs on the packed prefix with the same algebra.

    ``_grid`` (optional) is a pre-built ``_grid3`` tuple so ``IntradayFeatureCompiler``
    can materialize the (day, bar, inst) grid ONCE and dispatch several kernels
    on the same grid (P0#5 single-scan).  When omitted the kernel builds its own
    grid (standalone backward-compatible path).
    """
    import numpy as np
    if _grid is None:
        g, _, _, codes, uniq, active, filled = _grid3(a)
    else:
        g, _, _, codes, uniq, active, filled = _grid
    n, B, C = g.shape
    fin = np.isfinite(g)
    cnts = np.sum(fin, axis=1)  # (D, C) finite count per (day, inst)

    # Pack finite values to the front per (day, inst) column, order-preserving
    # (stable sort on the negated mask puts finite rows first, keeping their
    # original bar order — exactly the scalar's ``vals[np.isfinite(vals)]``).
    neg_fin = ~fin
    order = np.argsort(neg_fin, axis=1, kind="stable")  # (D, B, C)
    packed = np.take_along_axis(g, order, axis=1)
    packed[~np.isfinite(packed)] = 0.0

    max_cnt = int(cnts.max()) if cnts.size and cnts.max() > 0 else 1
    # (D, max_cnt, C) packed finite values; rows >= cnt are zero-padded and
    # masked out below.
    ar = np.arange(max_cnt)[None, :, None]  # (1, max_cnt, 1)
    mask = ar < cnts[:, None, :]  # (D, max_cnt, C) finite membership
    pf = np.where(mask, packed[:, :max_cnt, :], 0.0)

    psum = np.sum(pf, axis=1)  # (D, C)
    pmean = (psum / np.maximum(cnts, 1))[:, None, :]  # (1-ish, C) broadcast
    pdev = np.where(mask, pf - pmean, 0.0)  # deviations, zero outside mask

    # lag-1 pairs over the packed prefix: pair j with j+1 where both < cnt.
    m2 = mask[:, :-1, :] & mask[:, 1:, :]  # (D, max_cnt-1, C)
    xa = np.where(m2, pdev[:, :-1, :], 0.0)
    xb = np.where(m2, pdev[:, 1:, :], 0.0)
    cnt2 = np.sum(m2, axis=1)  # (D, C)
    sx = np.sum(xa, axis=1); sy = np.sum(xb, axis=1)
    sxx = np.sum(xa * xa, axis=1); syy = np.sum(xb * xb, axis=1)
    sxy = np.sum(xa * xb, axis=1)
    denom = np.sqrt((cnt2 * sxx - sx * sx) * (cnt2 * syy - sy * sy))
    r = np.full((n, C), np.nan)
    ok = (denom > 0) & np.isfinite(denom)
    r[ok] = -(cnt2 * sxy - sx * sy)[ok] / denom[ok]

    # min_finite gate and constant-deviation degeneracy -> NaN (scalar parity).
    valid = cnts >= min_finite
    r[~valid] = np.nan
    with np.errstate(invalid="ignore"):
        std_ok = np.nanstd(np.where(fin, g, np.nan), axis=1) > 1e-12
    r[~std_ok] = np.nan
    return _vec_result_df(r, uniq, a.columns)


def _vec_price_delay(
    a: pd.DataFrame, b: pd.DataFrame, *, min_finite: int = 2, _grid: tuple | None = None
) -> pd.DataFrame:
    """Vector intra_price_delay (volume-weighted lag-1 return autocorrelation)."""
    if _grid is None:
        g, gb, _, codes, uniq, active, filled = _grid3(a, b)
    else:
        g, gb, _, codes, uniq, active, filled = _grid
    n = g.shape[0]
    fin = np.isfinite(g) & np.isfinite(gb) & (gb > 0)
    cnt = np.sum(fin, axis=1)
    prices = np.where(fin, g, np.nan)
    vols = np.where(fin, gb, 0.0)
    # --- pack finite (price, vol>0) bars to the front per (day, inst) ---
    # The scalar kernel computes log returns between CONSECUTIVE FINITE bars
    # (``prices[1:] / prices[:-1]`` after the NaN drop), so a NaN gap must not
    # split a return pair.  A stable argsort on the negated mask packs the
    # finite bars to the prefix in original bar order (same as the scalar's
    # ``vals[np.isfinite(vals)]``), then pair j with j+1 inside the prefix.
    order = np.argsort(~fin, axis=1, kind="stable")  # (D, B, C)
    p_px = np.take_along_axis(prices, order, axis=1)
    p_vol = np.take_along_axis(vols, order, axis=1)
    p_px[~np.isfinite(p_px)] = 0.0
    p_vol[~np.isfinite(p_vol)] = 0.0

    max_cnt = int(cnt.max()) if cnt.size and cnt.max() > 0 else 1
    ar = np.arange(max_cnt)[None, :, None]  # (1, max_cnt, 1)
    mask = ar < cnt[:, None, :]  # (D, max_cnt, C) finite membership
    pf = np.where(mask, p_px[:, :max_cnt, :], 0.0)
    vf = np.where(mask, p_vol[:, :max_cnt, :], 0.0)

    # log returns on the packed prefix: pair j with j+1 where both < cnt
    m2 = mask[:, :-1, :] & mask[:, 1:, :]  # (D, max_cnt-1, C)
    ra = np.log(np.where(m2, pf[:, 1:, :], 1.0) / np.where(m2, pf[:, :-1, :], 1.0))
    ra = np.where(m2, ra, 0.0)
    vb = np.where(m2, vf[:, 1:, :], 0.0)  # volume at the "next" bar

    sw = np.sum(vb, axis=1)  # (D, C) total volume of finite bars 1..
    norm = np.where(sw[:, None, :] > 1e-10, vb / np.maximum(sw[:, None, :], 1e-12), 0.0)
    weighted = ra * norm
    # Scalar pairs consecutive weighted returns: corrcoef(wr[:-1], wr[1:]) —
    # the observation unit is the per-pair weighted return, so the lag-1
    # correlation over the cnt-1 pair entries uses cnt-2 pair-of-pair obs.
    xm2 = m2[:, :-1, :] & m2[:, 1:, :]  # (D, max_cnt-2, C) valid lag-1 pairs
    xa = np.where(xm2, weighted[:, :-1, :], 0.0)
    xb = np.where(xm2, weighted[:, 1:, :], 0.0)
    sx = np.sum(xa, axis=1); sy = np.sum(xb, axis=1)
    sxx = np.sum(xa * xa, axis=1); syy = np.sum(xb * xb, axis=1); sxy = np.sum(xa * xb, axis=1)
    cnt2 = np.sum(xm2, axis=1)
    denom = np.sqrt((cnt2 * sxx - sx * sx) * (cnt2 * syy - sy * sy))
    r = np.full((n, g.shape[2]), np.nan)
    ok = (denom > 0) & np.isfinite(denom)
    r[ok] = (cnt2 * sxy - sx * sy)[ok] / denom[ok]
    valid = cnt >= min_finite
    r[~valid] = np.nan
    r[cnt2 < 2] = np.nan
    return _vec_result_df(r, uniq, a.columns)


def _vec_autocorr_pairs(x: np.ndarray) -> np.ndarray:
    """lag-1 correlation over the masked (finite, twice-next-finite) pairs."""
    fin = np.isfinite(x)
    f1 = np.roll(fin, shift=-1, axis=1) & fin
    f1[:, -1] = False
    cnt = np.sum(f1, axis=1)
    xa = np.where(f1, x, 0.0)
    xb = np.where(f1, np.roll(x, shift=-1, axis=1), 0.0)
    sx = np.sum(xa, axis=1); sy = np.sum(xb, axis=1)
    sxx = np.sum(xa*xa, axis=1); syy = np.sum(xb*xb, axis=1); sxy = np.sum(xa*xb, axis=1)
    denom = np.sqrt((cnt*sxx - sx*sx) * (cnt*syy - sy*sy))
    out = np.full(cnt.shape, np.nan)
    ok = (denom > 0) & np.isfinite(denom)
    out[ok] = (cnt*sxy - sx*sy)[ok] / denom[ok]
    out[cnt < 2] = np.nan
    return out


def _vec_volume_imbalance(
    a: pd.DataFrame, b: pd.DataFrame, *, min_finite: int = 2, _grid: tuple | None = None
) -> pd.DataFrame:
    """Vector intra_volume_imbalance."""
    import numpy as np
    if _grid is None:
        g, gb, _, codes, uniq, active, filled = _grid3(a, b)
    else:
        g, gb, _, codes, uniq, active, filled = _grid
    n = g.shape[0]
    fin = np.isfinite(g) & np.isfinite(gb) & (gb > 0)
    cnt = np.sum(fin, axis=1)
    prices = np.where(fin, g, 0.0)
    vols = np.where(fin, gb, 0.0)
    total_vol = np.sum(vols, axis=1)
    vwap = np.where(total_vol > 0, np.sum(prices*vols, axis=1) / np.maximum(total_vol, 1e-12), np.nan)
    above = (prices > vwap[:, None, :]) & fin
    below = (prices < vwap[:, None, :]) & fin
    ab_vol = np.where(above, vols, 0.0)
    bl_vol = np.where(below, vols, 0.0)
    vol_above = np.sum(ab_vol, axis=1)
    vol_below = np.sum(bl_vol, axis=1)
    r = np.full((n, g.shape[2]), np.nan)
    ok = total_vol > 1e-10
    r[ok] = (vol_above[ok] - vol_below[ok]) / total_vol[ok]
    valid = cnt >= min_finite
    r[~valid] = np.nan
    return _vec_result_df(r, uniq, a.columns)


def _vec_stock_graph_features(
    a: pd.DataFrame, *, min_finite: int = 2, _grid: tuple | None = None
) -> pd.DataFrame:
    """Vector intra_dynamic_stock_graph_features (correlation of log returns).

    Scalar kernel (``smart_money._stock_graph_features``): drop NaN bars, log-
    returns over CONSECUTIVE FINITE bars (``diff(log(finite))`` so a NaN gap
    does not split a return pair), then lag-1 autocorrelation over the returns
    ``corrcoef(ret[:-1], ret[1:])``.  Returns the lag-1 autocorrelation (NOT
    negated — the graph proxy is positive co-movement).

    Vectorization: pack finite bars to the prefix with a stable argsort
    (order-preserving, same as the scalar's ``finite`` slice), derive returns
    on packed pairs, then the pair-of-pair lag-1 autocorrelation on the returns.
    """
    import numpy as np
    if _grid is None:
        g, _, _, codes, uniq, active, filled = _grid3(a)
    else:
        g, _, _, codes, uniq, active, filled = _grid
    n = g.shape[0]
    fin = np.isfinite(g)
    cnt = np.sum(fin, axis=1)  # (D, C)

    order = np.argsort(~fin, axis=1, kind="stable")  # (D, B, C)
    packed = np.take_along_axis(g, order, axis=1)
    packed[~np.isfinite(packed)] = 0.0

    max_cnt = int(cnt.max()) if cnt.size and cnt.max() > 0 else 1
    ar = np.arange(max_cnt)[None, :, None]  # (1, max_cnt, 1)
    mask = ar < cnt[:, None, :]  # (D, max_cnt, C)
    pf = np.where(mask, packed[:, :max_cnt, :], 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        logpf = np.where(mask, np.log(pf), 0.0)
    # log returns on consecutive packed pairs (both < cnt == a valid pair).
    m2 = mask[:, :-1, :] & mask[:, 1:, :]  # (D, max_cnt-1, C)
    ra = np.where(m2, logpf[:, 1:, :] - logpf[:, :-1, :], 0.0)

    # lag-1 autocorrelation over the return pairs (pair-of-pair).
    xm2 = m2[:, :-1, :] & m2[:, 1:, :]  # (D, max_cnt-2, C)
    xa = np.where(xm2, ra[:, :-1, :], 0.0)
    xb = np.where(xm2, ra[:, 1:, :], 0.0)
    sx = np.sum(xa, axis=1); sy = np.sum(xb, axis=1)
    sxx = np.sum(xa * xa, axis=1); syy = np.sum(xb * xb, axis=1)
    sxy = np.sum(xa * xb, axis=1)
    cnt2 = np.sum(xm2, axis=1)  # (D, C)
    denom = np.sqrt((cnt2 * sxx - sx * sx) * (cnt2 * syy - sy * sy))
    r = np.full((n, g.shape[2]), np.nan)
    ok = (denom > 0) & np.isfinite(denom)
    r[ok] = (cnt2 * sxy - sx * sy)[ok] / denom[ok]

    valid = cnt >= min_finite
    r[~valid] = np.nan
    # constant-returns degeneracy (std(ret) <= _EPS) -> NaN (scalar parity).
    with np.errstate(invalid="ignore"):
        rstd = np.nanstd(np.where(m2, ra, np.nan), axis=1)
    r[rstd <= _EPS] = np.nan
    r[cnt2 < 2] = np.nan
    return _vec_result_df(r, uniq, a.columns)


def _vec_common_trading_intensity(
    a: pd.DataFrame, b: pd.DataFrame, *, min_finite: int = 2, _grid: tuple | None = None
) -> pd.DataFrame:
    """Vector intra_common_trading_intensity (peak/mean volume x concentration).

    Scalar kernel (``smart_money._common_trading_intensity``): drop non-finite
    / non-positive-volume bars, then ``(max_vol / mean_vol) * sum((v/total)^2)``.
    No sequential state and no cross-bar correlation → the packed-order is not
    needed for the algebra, only for the finite mask (volume > 0 filter).
    """
    import numpy as np
    if _grid is None:
        g, gb, _, codes, uniq, active, filled = _grid3(a, b)
    else:
        g, gb, _, codes, uniq, active, filled = _grid
    n = g.shape[0]
    fin = np.isfinite(g) & np.isfinite(gb) & (g > 0)
    cnt = np.sum(fin, axis=1)

    vols = np.where(fin, g, 0.0)
    total_vol = np.sum(vols, axis=1)  # (D, C) (vols>0 on valid bars)
    mean_vol = total_vol / np.maximum(cnt, 1)
    max_vol = np.max(np.where(fin, vols, -np.inf), axis=1)  # masked max
    share = vols / np.maximum(total_vol[:, None, :], 1e-12)
    conc = np.sum(np.where(fin, share * share, 0.0), axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        intensity = (max_vol / np.maximum(mean_vol, 1e-12)) * conc
    r = np.full((n, g.shape[2]), np.nan)
    valid = (cnt >= min_finite) & np.isfinite(intensity)
    r[valid] = intensity[valid]
    return _vec_result_df(r, uniq, a.columns)


def _vec_time_above_vwap(
    a: pd.DataFrame,
    b: pd.DataFrame,
    c: pd.DataFrame,
    *,
    min_finite: int = 2,
    _grid: tuple | None = None,
) -> pd.DataFrame:
    """Vector intra_time_above_vwap (fraction of minutes price > cum-VWAP).

    Scalar kernel (``vwap_path._time_above_vwap``): drop non-finite /
    non-positive-volume bars preserving ORDER, build cumulative VWAP
    (cumsum(amount) / cumsum(volume)) over the finite bars, then
    ``mean(close > cum_vwap)``.  Sequential cumsum is fine for numpy — only
    the packed-order-preserving prefix must be identical to the scalar's
    filtered bars.
    """
    import numpy as np
    if _grid is None:
        g, gb, gc, codes, uniq, active, filled = _grid3(a, b, c)
    else:
        g, gb, gc, codes, uniq, active, filled = _grid
    n = g.shape[0]
    fin = np.isfinite(g) & np.isfinite(gb) & np.isfinite(gc) & (gc > 0)
    cnt = np.sum(fin, axis=1)

    px = np.where(fin, g, 0.0)
    am = np.where(fin, gb, 0.0)
    vo = np.where(fin, gc, 0.0)
    order = np.argsort(~fin, axis=1, kind="stable")  # order-preserving pack
    pp = np.take_along_axis(px, order, axis=1)
    pa = np.take_along_axis(am, order, axis=1)
    pv = np.take_along_axis(vo, order, axis=1)

    max_cnt = int(cnt.max()) if cnt.size and cnt.max() > 0 else 1
    ar = np.arange(max_cnt)[None, :, None]  # (1, max_cnt, 1)
    mask = ar < cnt[:, None, :]
    pp = pp[:, :max_cnt, :]; pa = pa[:, :max_cnt, :]; pv = pv[:, :max_cnt, :]

    # cumulative sums only over the valid prefix (zero-padded tail is neutral).
    cm = np.cumsum(np.where(mask, pv, 0.0), axis=1)
    ca = np.cumsum(np.where(mask, pa, 0.0), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cum_vwap = np.where(cm > _EPS, ca / np.maximum(cm, _EPS), np.nan)

    above = np.where(mask, pp, np.nan) > cum_vwap
    above_cnt = np.sum(np.where(mask & np.isfinite(above), above, 0.0), axis=1)
    frac = above_cnt / np.maximum(cnt, 1)

    r = np.full((n, g.shape[2]), np.nan)
    valid = cnt >= min_finite
    r[valid] = frac[valid]
    return _vec_result_df(r, uniq, a.columns)


def _param_vec(impl, **params):
    """Wrap a parameterized 3-D vector kernel into a dispatch callable.

    ``daily_agg{,_two,_three}`` dispatch hooks call ``fnv(*frames, min_finite=..,
    _grid=..)``.  Parameterized kernels (VWAP-path slope/curvature, excursion,
    streak, drawdown side/metric) share one vector implementation in ``_core``
    distinguished by extra parameters (degree / coeff_idx / pct / side / metric).
    This factory closes over those parameters and produces a callable with the
    exact dispatch signature so it can serve as ``__vec__`` on a scalar kernel.
    """
    def vecfn(*args, min_finite: int = 2, _grid: tuple | None = None):
        return impl(*args, min_finite=min_finite, _grid=_grid, **params)

    vecfn.__name__ = f"_vec_param({impl.__name__})"
    return vecfn


def _vec_cum_vwap(g, gb, gc):
    """(D,B,C) order-preserving packed cum-VWAP over the valid finite prefix.

    Mirrors ``vwap_path._vwap_valid`` + ``_cum_vwap``: the three series are
    masked by a *common* finite & volume>0 mask (so price / amount / volume stay
    aligned), finite bars are packed to the front in original bar order, and the
    cumulative price-weighted amount over volume is assembled over that packed
    prefix only (the zero-padded tail is neutral for cumsums).
    """
    fin = np.isfinite(g) & np.isfinite(gb) & np.isfinite(gc) & (gc > 0)
    cnt = np.sum(fin, axis=1)  # (D, C) finite count per (day, inst)
    px = np.where(fin, g, 0.0)
    am = np.where(fin, gb, 0.0)
    vo = np.where(fin, gc, 0.0)
    order = np.argsort(~fin, axis=1, kind="stable")  # order-preserving pack
    pp = np.take_along_axis(px, order, axis=1)
    pa = np.take_along_axis(am, order, axis=1)
    pv = np.take_along_axis(vo, order, axis=1)
    max_cnt = int(cnt.max()) if cnt.size and cnt.max() > 0 else 1
    ar = np.arange(max_cnt)[None, :, None]
    mask = ar < cnt[:, None, :]  # (D, max_cnt, C) finite membership
    pp = pp[:, :max_cnt, :]; pa = pa[:, :max_cnt, :]; pv = pv[:, :max_cnt, :]
    cm = np.cumsum(np.where(mask, pv, 0.0), axis=1)
    ca = np.cumsum(np.where(mask, pa, 0.0), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cum_vwap = np.where(cm > _EPS, ca / np.maximum(cm, _EPS), np.nan)
    return fin, cnt, mask, pp, cum_vwap


def _vec_vwap_path(
    a, b, c, *, min_finite: int = 2, _grid: tuple | None = None,
    degree: int = 1, coeff_idx: int = 1, pct: bool = False,
) -> pd.DataFrame:
    """Vector intra_vwap_path_slope / _curvature [and _pct] (cum-VWAP path fit).

    Scalar kernel (``vwap_path._vwap_path_common`` / ``_pct_common``): pack the
    common finite/vol>0 mask, build cumulative VWAP over the packed prefix, and
    least-squares fit the (degree)-order polynomial in normalized time
    ``t=linspace(0,1,n)``; return the ``coeff_idx``-th coefficient.  ``n <
    degree+2 -> NaN``.  The ``_pct`` family fits ``cum_vwap/first_price - 1``
    and gates on ``first <= _EPS``.

    Vectorization: per (day,inst) the packed cum-VWAP prefix is exactly the
    scalar's ``y``.  Because ``t`` depends only on the count ``n``, cells sharing
    the same count share one Vandermonde; the normal-equations solve per count
    group reproduces scalar ``np.linalg.lstsq(rcond=None)`` to ~1e-15.
    """
    if _grid is None:
        g, gb, gc, codes, uniq, active, filled = _grid3(a, b, c)
    else:
        g, gb, gc, codes, uniq, active, filled = _grid
    n = g.shape[0]; S = g.shape[2]
    fin, cnt, mask, pp, cum_vwap = _vec_cum_vwap(g, gb, gc)
    needed = degree + 2  # scalar: ``n < degree + 2 -> NaN``
    res = np.full((n, S), np.nan)
    sel_base = cnt >= needed
    if pct:
        first = pp[:, 0, :]  # first packed close == scalar ``c[0]``
        sel_base = sel_base & np.isfinite(first) & (first > _EPS)
    prefix = cum_vwap[:, :, :]
    if pct:
        with np.errstate(divide="ignore", invalid="ignore"):
            prefix = np.where(mask, prefix / np.maximum(first[:, None, :], _EPS) - 1.0, np.nan)
    p = degree + 1
    for gcnt in np.unique(cnt[sel_base]):
        gcnt = int(gcnt)
        sel = sel_base & (cnt == gcnt)
        d_i, c_i = np.where(sel)
        if d_i.size == 0:
            continue
        t = np.linspace(0.0, 1.0, gcnt)
        Pd = np.column_stack([t ** k for k in range(p)])  # (gcnt, p)
        # Stable least squares via QR; matches np.linalg.lstsq (SVD/QR-based)
        # far more closely than the normal equations when the Vandermonde is
        # ill-conditioned (gcnt near degree+1, tiny curvature coefficients).
        Qd, Rd = np.linalg.qr(Pd)
        Y = prefix[:, :gcnt, :][d_i, :, c_i]  # (Nsel, gcnt)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            beta = np.linalg.solve(Rd, (Qd.T @ Y.T))  # (p, Nsel)
        res[d_i, c_i] = beta[coeff_idx, :d_i.size]
    res[cnt < min_finite] = np.nan
    return _vec_result_df(res, uniq, a.columns)


def _vec_vwap_excursion(
    a, b, c, *, min_finite: int = 2, _grid: tuple | None = None, side: str = "max",
) -> pd.DataFrame:
    """Vector intra_price_vwap_max_positive/negative_excursion.

    Scalar kernel (``vwap_path._vwap_excursion``): dev = close/cum_vwap - 1 over
    the (finite & cum_vwap > _EPS) bars; return max (side="max") or min (side
    "min"); all-invalid -> NaN.
    """
    if _grid is None:
        g, gb, gc, codes, uniq, active, filled = _grid3(a, b, c)
    else:
        g, gb, gc, codes, uniq, active, filled = _grid
    n = g.shape[0]; S = g.shape[2]
    fin, cnt, mask, pp, cum_vwap = _vec_cum_vwap(g, gb, gc)
    ok = mask & (cum_vwap > _EPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = np.where(ok, np.divide(pp, np.where(ok, cum_vwap, 1)) - 1.0, np.nan)
    ok_any = ok.any(axis=1)
    res = np.full((n, S), np.nan)
    if side == "max":
        m = np.where(ok, dev, -np.inf).max(axis=1)
    else:
        m = np.where(ok, dev, np.inf).min(axis=1)
    res[ok_any] = m[ok_any]
    res[cnt < min_finite] = np.nan
    return _vec_result_df(res, uniq, a.columns)


def _vec_longest_streak(
    a, b, c, *, min_finite: int = 2, _grid: tuple | None = None, side: str = "above",
) -> pd.DataFrame:
    """Vector intra_longest_above/below_vwap_streak (longest run of bars vs cum-VWAP).

    Scalar kernel (``vwap_path._longest_streak``): over the packed bars, flag
    ``close > cum_vwap`` (or its negation for below), then the longest run of
    True flags.  The run length counts *bars* (minutes), measured in finite bars.
    """
    if _grid is None:
        g, gb, gc, codes, uniq, active, filled = _grid3(a, b, c)
    else:
        g, gb, gc, codes, uniq, active, filled = _grid
    n = g.shape[0]; S = g.shape[2]
    fin, cnt, mask, pp, cum_vwap = _vec_cum_vwap(g, gb, gc)
    above = (np.where(mask, pp, np.nan) > cum_vwap) & mask
    if side == "below":
        above = mask & ~above
    # longest run of True per row (run[i] = current streak ending at i; resets
    # at each False via cumulative-max of the reset point).
    s = np.cumsum(above, axis=1)
    reset = np.where(above, 0.0, s)
    last_reset = np.maximum.accumulate(reset, axis=1)
    run = s - last_reset
    best = np.max(run, axis=1)
    res = np.full((n, S), np.nan)
    res[cnt >= 1] = best[cnt >= 1]
    res[cnt < min_finite] = np.nan
    return _vec_result_df(res, uniq, a.columns)


def _vec_vwap_reversion_speed(
    a, b, c, *, min_finite: int = 2, _grid: tuple | None = None,
) -> pd.DataFrame:
    """Vector intra_vwap_reversion_speed (AR(1) coefficient of VWAP deviation).

    Scalar kernel (``vwap_path._vwap_reversion_speed``): dev = close/cum_vwap-1
    over finite bars; regress ``dev[1:]`` on ``dev[:-1]`` via
    ``cov(d, dprev)[0,1] / var(dprev)`` — NOTE ``np.cov`` uses ddof=1 while
    ``np.var`` uses ddof=0, so beta = Sxy*n/(Sxx*(n-1)) with scatter sums over
    the n pairs.  Gates: valid.sum()<5 -> NaN, len(d)<3 -> NaN,
    std(dprev)<=_EPS -> NaN.
    """
    if _grid is None:
        g, gb, gc, codes, uniq, active, filled = _grid3(a, b, c)
    else:
        g, gb, gc, codes, uniq, active, filled = _grid
    n = g.shape[0]; S = g.shape[2]
    fin, cnt, mask, pp, cum_vwap = _vec_cum_vwap(g, gb, gc)
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = np.where(mask, np.divide(pp, np.where(mask, cum_vwap, 1)) - 1.0, np.nan)
    m2 = mask[:, :-1, :] & mask[:, 1:, :]  # consecutive finite pairs
    dd = np.where(m2, dev[:, 1:, :], 0.0)
    dp = np.where(m2, dev[:, :-1, :], 0.0)
    n2 = np.sum(m2, axis=1)  # n pairs = cnt - 1
    sx = np.sum(dd, axis=1); sy = np.sum(dp, axis=1)
    sxy = np.sum(dd * dp, axis=1); sxx = np.sum(dp * dp, axis=1)
    mx = sx / np.maximum(n2, 1); my = sy / np.maximum(n2, 1)
    Sxy = sxy - n2 * mx * my
    Sxx = sxx - n2 * my * my
    with np.errstate(divide="ignore", invalid="ignore"):
        b = np.where((Sxx > 0) & (n2 > 1), Sxy * n2 / (np.maximum(Sxx, 1e-300) * (n2 - 1)), np.nan)
    # scalar gates: len(d)>=3 -> n2>=3 ; std(dprev)=sqrt(Sxx/n) > _EPS
    popvar = np.where(n2 > 0, Sxx / np.maximum(n2, 1), np.inf)
    std_dprev = np.sqrt(np.maximum(popvar, 0.0))
    good = (n2 >= 3) & (std_dprev > _EPS) & np.isfinite(b)
    res = np.full((n, S), np.nan)
    res[good] = b[good]
    res[cnt < 5] = np.nan  # scalar: valid.sum() < 5 -> NaN
    res[cnt < min_finite] = np.nan
    return _vec_result_df(res, uniq, a.columns)


def _vec_max_drawdown(
    a, *, min_finite: int = 2, _grid: tuple | None = None, side: str = "down",
) -> pd.DataFrame:
    """Vector intra_max_drawdown / intra_max_drawup (single close panel).

    Scalar kernel (``vwap_path._max_drawdown``): drop non-finite prices
    preserving order; ``down`` uses the running maximum (path = p/running_max-1,
    take min), ``up`` uses the running minimum (path = p/running_min-1, take
    max).  Fewer than 2 finite prices -> NaN.
    """
    if _grid is None:
        g, _, _, codes, uniq, active, filled = _grid3(a)
    else:
        g, _, _, codes, uniq, active, filled = _grid
    n = g.shape[0]; S = g.shape[2]
    fin = np.isfinite(g)
    cnt = np.sum(fin, axis=1)
    order = np.argsort(~fin, axis=1, kind="stable")
    pf = np.take_along_axis(g, order, axis=1)
    pf[~np.isfinite(pf)] = 0.0
    max_cnt = int(cnt.max()) if cnt.size and cnt.max() > 0 else 1
    ar = np.arange(max_cnt)[None, :, None]
    mask = ar < cnt[:, None, :]
    pf = pf[:, :max_cnt, :]
    if side == "down":
        running = np.maximum.accumulate(np.where(mask, pf, 0.0), axis=1)
    else:
        running = np.minimum.accumulate(np.where(mask, pf, np.inf), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        path = np.where(mask, pf / running - 1.0, np.nan)
    has = mask.any(axis=1)
    res = np.full((n, S), np.nan)
    if side == "down":
        # scalar: return np.min(path) — the most deeply negative drawdown.
        v = np.where(mask, path, np.inf).min(axis=1)
    else:
        # scalar: return np.max(path) — the largest gain over running min.
        v = np.where(mask, path, -np.inf).max(axis=1)
    res[has] = v[has]
    res[cnt < min_finite] = np.nan
    return _vec_result_df(res, uniq, a.columns)


def _vec_drawdown_metrics(
    a, *, min_finite: int = 2, _grid: tuple | None = None, metric: str = "depth",
) -> pd.DataFrame:
    """Vector intra_drawdown_depth / _duration / _recovery_half_life.

    Scalar kernels (``vwap_path._drawdown_locate`` + ``_depth/_duration/
    _recovery_half_life``): locate the true max drawdown trough as argmin of
    ``p/running_max - 1`` (NOT the global-peak-then-min), the peak as the running
    maximum up to the trough.  Indexing is in the *finite-preserving packed*
    (bars) coordinate space, which is what recovery-time / duration semantics
    require.
    """
    if _grid is None:
        g, _, _, codes, uniq, active, filled = _grid3(a)
    else:
        g, _, _, codes, uniq, active, filled = _grid
    n = g.shape[0]; S = g.shape[2]
    fin = np.isfinite(g)
    cnt = np.sum(fin, axis=1)
    order = np.argsort(~fin, axis=1, kind="stable")
    pf = np.take_along_axis(g, order, axis=1)
    pf[~np.isfinite(pf)] = 0.0
    max_cnt = int(cnt.max()) if cnt.size and cnt.max() > 0 else 1
    ar = np.arange(max_cnt)[None, :, None]
    mask = ar < cnt[:, None, :]
    pf = pf[:, :max_cnt, :]
    running = np.maximum.accumulate(np.where(mask, pf, 0.0), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(mask, pf / running - 1.0, np.inf)
    trough = np.argmin(dd, axis=1)  # (D, C) first min within prefix
    kk = ar  # (1, max_cnt, 1)
    in_to_trough = kk <= trough[:, None, :]
    pfseg = np.where(in_to_trough & mask, pf, -np.inf)
    peak = np.argmax(pfseg, axis=1)  # running max position up to trough
    pv = np.take_along_axis(pf, trough[:, None, :], axis=1)[:, 0, :]
    pk = np.take_along_axis(pf, peak[:, None, :], axis=1)[:, 0, :]
    trough_i = trough.astype(float)
    peak_i = peak.astype(float)
    res = np.full((n, S), np.nan)
    valid = cnt >= min_finite
    if metric == "depth":
        # scalar: finite[peak] <= _EPS -> NaN ; depth = trough/peak - 1
        ok = valid & (pk > _EPS)
        with np.errstate(divide="ignore", invalid="ignore"):
            res[ok] = np.where(ok, pv / pk - 1.0, np.nan)[ok]
    elif metric == "duration":
        ok = valid
        res[ok] = (trough_i - peak_i)[ok]
    else:  # recovery
        # scalar: trough<=_EPS or peak<=trough -> NaN
        ok = valid & (pv > _EPS) & (pk > pv)
        halfway = pv + 0.5 * (pk - pv)
        ge = (kk >= trough[:, None, :]) & (kk < cnt[:, None, :]) & (pf >= halfway[:, None, :]) & mask
        isrec = ge.any(axis=1) & ok
        first = np.argmax(ge, axis=1)
        res[isrec] = (first - trough)[isrec]
    return _vec_result_df(res, uniq, a.columns)


def _vec_logret_grid(g):
    """(D,B,C) log returns on the *original minute-slot* grid.

    Mirrors ``_core.log_returns``: ``r[t+1] = log(c[t+1]/c[t])``, NaN anywhere
    the previous close is missing/non-finite (divide-by-zero ignored).  The grid
    is day-padded so the (B-bar) row axis below is the union of all days' bars;
    the day boundaries keep their leading NaN row (a single-bar day has zero
    returns), which replicates the per-day scalar ``log_returns`` exactly.
    """
    n, B, C = g.shape
    r = np.full((n, B, C), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        r[:, 1:, :] = np.log(
            np.where(np.isfinite(g[:, 1:, :]), g[:, 1:, :], np.nan)
            / np.where(np.isfinite(g[:, :-1, :]), g[:, :-1, :], np.nan)
        )
    return r


def _vec_realized_moments(
    a, *, min_finite: int = 2, _grid: tuple | None = None, moment: str = "skew",
) -> pd.DataFrame:
    """Vector intra_realized_skewness _/_kurtosis/_quarticity (higher_moments).

    Scalar kernels (``higher_moments._realized_skewness/_kurtosis/_quarticity``):
    log-returns over the ORIGINAL minute grid (``_core.log_returns``), then:

    * skewness:  n>=30 = ``_REALIZED_MIN_RETURNS``; ``r2=sum(r^2)>_EPS``;
                 ``sqrt(n)*sum(r^3)/r2^1.5``.
    * kurtosis:  same n>=30 and r2>_EPS gates; ``n*sum(r^4)/r2^2``.
    * quarticity: n>=2; ``n/3*sum(r^4)``.

    ``n`` is the FINITE RETURN count, gated against ``_REALIZED_MIN_RETURNS``,
    while ``daily_agg``'s own ``min_finite`` gate is separately on the CLOSE
    finite count (raw vals) — both are reproduced.  ``r2`` uses ``sum`` over the
    finite mask (zero-padded ``nansum``); degenerate ``r2<=_EPS`` -> NaN.
    """
    if _grid is None:
        g, _, _, codes, uniq, active, filled = _grid3(a)
    else:
        g, _, _, codes, uniq, active, filled = _grid
    n = g.shape[0]
    r = _vec_logret_grid(g)
    fin = np.isfinite(r)
    cnt = np.sum(fin, axis=1)  # (D, C) finite RETURN count
    sq = np.sum(np.where(fin, r * r, 0.0), axis=1)  # (D, C) sum(r^2)
    close_cnt = np.sum(np.isfinite(g), axis=1)
    valid = close_cnt >= min_finite  # daily_agg raw-CLOSE gate (scalar parity)

    out = np.full((n, g.shape[2]), np.nan)
    if moment == "skew":
        ok = valid & (cnt >= _REALIZED_MIN_RETURNS) & (sq > _EPS) & np.isfinite(sq)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[ok] = (
                np.sqrt(cnt) * np.sum(np.where(fin, r ** 3, 0.0), axis=1) / sq ** 1.5
            )[ok]
    elif moment == "kurt":
        ok = valid & (cnt >= _REALIZED_MIN_RETURNS) & (sq > _EPS) & np.isfinite(sq)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[ok] = (
                cnt * np.sum(np.where(fin, r ** 4, 0.0), axis=1) / (sq * sq)
            )[ok]
    else:  # quarticity
        ok = valid & (cnt >= 2)
        out[ok] = (cnt / 3.0 * np.sum(np.where(fin, r ** 4, 0.0), axis=1))[ok]
    return _vec_result_df(out, uniq, a.columns)


def _vec_tripower_quarticity(
    a, *, min_finite: int = 2, _grid: tuple | None = None,
) -> pd.DataFrame:
    """Vector intra_tripower_quarticity (higher_moments).

    Scalar kernel (``higher_moments._tripower_quarticity``): triple absolute
    moments ``|r_t|^(4/3)|r_{t-1}|^(4/3)|r_{t-2}|^(4/3)`` over ADJACENT grid
    slots, all three finite (P1-100: never bridge a missing minute).  The result
    is ``tripower_scale() * n_finite * sum(prod)``.  Scalar parity requires a
    ``n_finite>=4`` gate AND an ``any(valid triple)`` gate — with >=4 finite
    returns but no three consecutive ones the scalar returns NaN, never 0.
    """
    if _grid is None:
        g, _, _, codes, uniq, active, filled = _grid3(a)
    else:
        g, _, _, codes, uniq, active, filled = _grid
    n = g.shape[0]
    r = _vec_logret_grid(g)
    fin = np.isfinite(r)
    cnt = np.sum(fin, axis=1)
    tri = fin[:, 2:, :] & fin[:, 1:-1, :] & fin[:, :-2, :]  # (n, B-2, C)
    any_tri = np.any(tri, axis=1)  # scalar ``np.any(valid)`` (P1-100)
    with np.errstate(invalid="ignore", over="ignore"):
        prod = np.where(
            tri,
            np.abs(r[:, 2:, :]) ** (4.0 / 3.0)
            * np.abs(r[:, 1:-1, :]) ** (4.0 / 3.0)
            * np.abs(r[:, :-2, :]) ** (4.0 / 3.0),
            0.0,
        )
    total = np.sum(prod, axis=1)
    close_cnt = np.sum(np.isfinite(g), axis=1)
    out = np.full((n, g.shape[2]), np.nan)
    ok = (close_cnt >= min_finite) & (cnt >= 4) & any_tri
    out[ok] = tripower_scale() * cnt[ok] * total[ok]
    return _vec_result_df(out, uniq, a.columns)


def _vec_bipower_and_jump(
    a, *, min_finite: int = 2, _grid: tuple | None = None, mode: str = "bipower",
) -> pd.DataFrame:
    """Vector intra_bipower_variation / _continuous_variance / _jump_variation.

    Scalar kernels (``higher_moments._bipower_var`` / ``_continuous_and_jump``):
    log-returns on the ORIGINAL minute grid, then:

    * bipower:   ``(pi/2) * sum(|r_t||r_{t-1}|)`` over ADJACENT slots, both
                 finite.  Scalar returns NaN when cnt<2 OR no adjacent pair
                 (P1-100 — never a cross-product over a missing minute).  A day
                 with two finite but non-adjacent returns yields NaN, not 0.
    * continuous: ``min(RV, BV)``; ``_continuous_and_jump`` returns NaN when rv
                 or bv non-finite or rv<=EPS.
    * jump:      ``max(RV-BV, 0)`` under the same gate.

    ``RV = sum(r^2)`` over finite returns; ``BV`` is the slot-adjacent bipower
    (NaN when no valid pair).  ``min_finite`` (raw-CLOSE count) and the bipower
    finite-count / finite-adjacency gates are both reproduced for scalar parity.
    """
    if _grid is None:
        g, _, _, codes, uniq, active, filled = _grid3(a)
    else:
        g, _, _, codes, uniq, active, filled = _grid
    n = g.shape[0]
    r = _vec_logret_grid(g)
    fin = np.isfinite(r)
    cnt = np.sum(fin, axis=1)
    both = fin[:, 1:, :] & fin[:, :-1, :]  # (n, B-1, C) adjacent finite pair
    any_pair = np.any(both, axis=1)  # scalar ``np.any(valid)`` (P1-100)
    with np.errstate(invalid="ignore", over="ignore"):
        prod = np.where(both, np.abs(r[:, 1:, :]) * np.abs(r[:, :-1, :]), 0.0)
    bp = (np.pi / 2.0) * np.sum(prod, axis=1)  # (D, C) bipower (slot-adjacent)

    close_cnt = np.sum(np.isfinite(g), axis=1)
    out = np.full((n, g.shape[2]), np.nan)
    if mode == "bipower":
        ok = (close_cnt >= min_finite) & (cnt >= 2) & any_pair
        out[ok] = bp[ok]
        return _vec_result_df(out, uniq, a.columns)

    # continuous / jump
    sq = np.sum(np.where(fin, r * r, 0.0), axis=1)  # RV
    ok = (
        np.isfinite(sq) & (sq > _EPS) & np.isfinite(bp)
        & (close_cnt >= min_finite) & (cnt >= 2) & any_pair
    )
    if mode == "continuous":
        out[ok] = np.minimum(sq, bp)[ok]
    else:  # jump
        out[ok] = np.maximum(sq - bp, 0.0)[ok]
    return _vec_result_df(out, uniq, a.columns)


# ---------------------------------------------------------------------------
# R61-P1 #57: sufficient-statistics vector kernels.
#
# One shared per-frame bundle (``sufficient_stats.compute_sufficient_statistics``
# / ``two_panel_statistics`` / ``three_panel_statistics``, memoized per frame
# identity) materializes the (day, bar, inst) grid ONCE plus Σx Σx² Σx³ Σx⁴,
# Σv Σv² Σpv, Σa Σpa, max/min, first/last, argmax/argmin and the return
# moments Σr² Σr³ Σr⁴.  Every kernel below derives its per-(day, inst) output
# from the bundle:
#     sum / mean / variance / std / min / max / last / first / last_value /
#     argmax / argmin / realized_variance / vwap / amount-weighted mean
#   -> O(1) (direct statistic read / one arithmetic expression).
#     volume_weighted_return / realized_covariance
#   -> one O(n) weighted-moment pass per (day, inst) window (weighted moments
#      are not reducible from scalar aggregates), sharing the joint finite
#      mask and the grid materialization.
# Equivalence vs the scalar kernels in ``sufficient_stats_ops`` is enforced by
# the PERF-2 harness (rtol/atol 1e-12) over random fixtures incl. NaN gaps and
# all-NaN days; the fail-closed bind count covers them.
# ---------------------------------------------------------------------------

def _ss_bundle(frame: pd.DataFrame, grid: tuple | None = None, required: str | None = None) -> dict:
    """Memoized sufficient-statistics bundle for one frame (lazy import)."""
    from factor_engine.cleaned_operators.intraday import sufficient_stats as _ss
    return _ss.compute_sufficient_statistics(frame, grid=grid, required=required)


def _ss_pair(a: pd.DataFrame, b: pd.DataFrame, grid: tuple | None = None) -> dict:
    from factor_engine.cleaned_operators.intraday import sufficient_stats as _ss
    return _ss.two_panel_statistics(a, b, grid=grid)


def _ss_triple(a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame) -> dict:
    from factor_engine.cleaned_operators.intraday import sufficient_stats as _ss
    return _ss.three_panel_statistics(a, b, c)


def _vec_one_from_bundle(bundle: dict, kind: str, *, min_finite: int):
    """(D, C) result for a one-panel sufficient-statistics family member.

    ``kind`` in {sum, mean, variance, std, min, max, last, first, argmax,
    argmin, realized_variance}.  All direct-statistic reads (O(1)):
      sum        = Σx
      mean       = Σx / cnt (bundle mean)
      variance   = packed-prefix two-pass var (bit-identical to np.var)
      std        = sqrt(var)
      min/max    = direct masked extrema (NaN-aware)
      first/last = packed-prefix first/last value
      argmax/argmin = packed-prefix index of first extreme
      realized_variance = Σr² over finite returns
    """
    cnt = bundle["cnt"]
    if kind in ("sum", "mean", "variance", "std"):
        if kind == "sum":
            val = bundle["sum"].copy()
        elif kind == "mean":
            val = bundle["mean"].copy()
        else:
            # variance / std read the bundle's two-pass packed-prefix variance
            # (bit-identical to scalar ``np.var(compressed)``); std is
            # sqrt(var) elementwise.
            var = bundle["var"]
            val = var if kind == "variance" else np.sqrt(np.maximum(var, 0.0))
        out = np.full(val.shape, np.nan)
        ok = cnt >= min_finite
        out[ok] = val[ok]
        return out
    if kind in ("min", "max"):
        val = bundle["min" if kind == "min" else "max"]
        out = np.full(val.shape, np.nan)
        ok = np.isfinite(val) & (cnt >= min_finite)
        out[ok] = val[ok]
        return out
    if kind in ("first", "last"):
        val = bundle["first" if kind == "first" else "last"]
        out = np.full(val.shape, np.nan)
        ok = np.isfinite(val) & (cnt >= min_finite)
        out[ok] = val[ok]
        return out
    if kind in ("argmax", "argmin"):
        pos = bundle["argmax_pos" if kind == "argmax" else "argmin_pos"]
        out = np.full(pos.shape, np.nan)
        ok = cnt >= min_finite
        out[ok] = pos[ok].astype(float)
        return out
    if kind == "realized_variance":
        r2 = bundle["r2"]
        rcnt = bundle["ret_cnt"]
        out = np.full(r2.shape, np.nan)
        # scalar gate: realized variance NaN when fewer than 2 finite returns.
        ok = (cnt >= min_finite) & (rcnt >= 2)
        out[ok] = r2[ok]
        return out
    raise ValueError(f"unknown one-panel statistic kind {kind!r}")


def _vec_ts_sum(a, *, min_finite: int = 2, _grid: tuple | None = None) -> pd.DataFrame:
    b = _ss_bundle(a, _grid, "sum")
    return _vec_result_df(_vec_one_from_bundle(b, "sum", min_finite=min_finite), b["uniq_days"], a.columns)


def _vec_ts_mean(a, *, min_finite: int = 2, _grid: tuple | None = None) -> pd.DataFrame:
    b = _ss_bundle(a, _grid, "mean")
    return _vec_result_df(_vec_one_from_bundle(b, "mean", min_finite=min_finite), b["uniq_days"], a.columns)


def _vec_ts_variance(a, *, min_finite: int = 2, _grid: tuple | None = None) -> pd.DataFrame:
    b = _ss_bundle(a, _grid)
    return _vec_result_df(_vec_one_from_bundle(b, "variance", min_finite=min_finite), b["uniq_days"], a.columns)


def _vec_ts_std(a, *, min_finite: int = 2, _grid: tuple | None = None) -> pd.DataFrame:
    b = _ss_bundle(a, _grid)
    return _vec_result_df(_vec_one_from_bundle(b, "std", min_finite=min_finite), b["uniq_days"], a.columns)


def _vec_ts_min(a, *, min_finite: int = 2, _grid: tuple | None = None) -> pd.DataFrame:
    b = _ss_bundle(a, _grid)
    return _vec_result_df(_vec_one_from_bundle(b, "min", min_finite=min_finite), b["uniq_days"], a.columns)


def _vec_ts_max(a, *, min_finite: int = 2, _grid: tuple | None = None) -> pd.DataFrame:
    b = _ss_bundle(a, _grid)
    return _vec_result_df(_vec_one_from_bundle(b, "max", min_finite=min_finite), b["uniq_days"], a.columns)


def _vec_ts_last(a, *, min_finite: int = 2, _grid: tuple | None = None) -> pd.DataFrame:
    b = _ss_bundle(a, _grid)
    return _vec_result_df(_vec_one_from_bundle(b, "last", min_finite=min_finite), b["uniq_days"], a.columns)


def _vec_ts_first(a, *, min_finite: int = 2, _grid: tuple | None = None) -> pd.DataFrame:
    b = _ss_bundle(a, _grid)
    return _vec_result_df(_vec_one_from_bundle(b, "first", min_finite=min_finite), b["uniq_days"], a.columns)


def _vec_ts_last_value(a, *, min_finite: int = 2, _grid: tuple | None = None) -> pd.DataFrame:
    b = _ss_bundle(a, _grid)
    return _vec_result_df(_vec_one_from_bundle(b, "last", min_finite=min_finite), b["uniq_days"], a.columns)


def _vec_ts_argmax(a, *, min_finite: int = 2, _grid: tuple | None = None) -> pd.DataFrame:
    b = _ss_bundle(a, _grid)
    return _vec_result_df(_vec_one_from_bundle(b, "argmax", min_finite=min_finite), b["uniq_days"], a.columns)


def _vec_ts_argmin(a, *, min_finite: int = 2, _grid: tuple | None = None) -> pd.DataFrame:
    b = _ss_bundle(a, _grid)
    return _vec_result_df(_vec_one_from_bundle(b, "argmin", min_finite=min_finite), b["uniq_days"], a.columns)


def _vec_ts_realized_variance(a, *, min_finite: int = 2, _grid: tuple | None = None) -> pd.DataFrame:
    b = _ss_bundle(a, _grid)
    return _vec_result_df(_vec_one_from_bundle(b, "realized_variance", min_finite=min_finite), b["uniq_days"], a.columns)


def _vec_ts_vwap(a, b, *, min_finite: int = 2, _grid: tuple | None = None) -> pd.DataFrame:
    """Vector intra_ts_vwap: Σ(p·v)/Σv over the joint finite (close, vol>0)."""
    sb = _ss_pair(a, b, _grid)
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(sb["vsum"] > _EPS, sb["pvsum"] / np.maximum(sb["vsum"], _EPS), np.nan)
    out = np.full(vwap.shape, np.nan)
    ok = np.isfinite(vwap) & (sb["cnt"] >= min_finite)
    out[ok] = vwap[ok]
    return _vec_result_df(out, sb["uniq_days"], a.columns)


def _vec_ts_volume_weighted_return(a, b, *, min_finite: int = 2, _grid: tuple | None = None) -> pd.DataFrame:
    """Vector intra_ts_volume_weighted_return: Σ(r·v)/Σv.

    Scalar contract: ``daily_agg_two`` drops NaN-CLOSE rows first, then the
    kernel computes ``log_returns`` over the CLOSE-COMPRESSED series (missing
    minutes do not split a return pair) and weights each return by the volume
    at the return's own bar with volume>0.  The bundle's packed prefix (pc,
    pv, rp) reproduces that compressed series exactly; one O(n) pass per
    (day, inst) over the shared packed prefix.
    """
    sb = _ss_pair(a, b, _grid)
    rp = sb["rp"]
    pv = sb["gb"]
    vmask = sb["pair_mask"]
    num = np.where(vmask, rp * pv, 0.0).sum(axis=1)
    den = np.where(vmask, pv, 0.0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = np.where(den > _EPS, num / np.maximum(den, _EPS), np.nan)
    out = np.full(val.shape, np.nan)
    # min_finite gate uses the CLOSE finite count (daily_agg_two's gate on the
    # A-panel; an all-NaN-returns day still passes if closes exist, then the
    # kernel returns NaN because den==0 -> val NaN).
    ok = np.isfinite(val) & (sb["close_cnt"] >= min_finite)
    out[ok] = val[ok]
    return _vec_result_df(out, sb["uniq_days"], a.columns)


def _vec_ts_realized_covariance(a, b, *, min_finite: int = 2, _grid: tuple | None = None) -> pd.DataFrame:
    """Vector intra_ts_realized_covariance: Σ(r²·v)/Σv over the packed prefix."""
    sb = _ss_pair(a, b, _grid)
    rp = sb["rp"]
    pv = sb["gb"]
    vmask = sb["pair_mask"]
    with np.errstate(divide="ignore", invalid="ignore"):
        rv = np.where(vmask, rp, 0.0)
        num = (rv * rv * np.where(vmask, pv, 0.0)).sum(axis=1)
        den = np.where(vmask, pv, 0.0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = np.where(den > _EPS, num / np.maximum(den, _EPS), np.nan)
    out = np.full(val.shape, np.nan)
    ok = np.isfinite(val) & (sb["close_cnt"] >= min_finite)
    out[ok] = val[ok]
    return _vec_result_df(out, sb["uniq_days"], a.columns)


def _vec_ts_amount_weighted_mean(a, b, *, min_finite: int = 2, _grid: tuple | None = None) -> pd.DataFrame:
    """Vector intra_ts_amount_weighted_mean: Σ(c·a)/Σa, amount>0 bars."""
    sb = _ss_pair(a, b, _grid)  # b treated as the amount panel (joint mask amount>0)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = np.where(sb["vsum"] > _EPS, sb["pvsum"] / np.maximum(sb["vsum"], _EPS), np.nan)
    out = np.full(val.shape, np.nan)
    ok = np.isfinite(val) & (sb["cnt"] >= min_finite)
    out[ok] = val[ok]
    return _vec_result_df(out, sb["uniq_days"], a.columns)


# Wrapper bodies — the scalar kernels from the consumer modules get their
# ``__vec__`` attribute attached lazily by ``_core`` (via the factories in the
# consumer modules / ``perf_vec_kernels.bind_whitelist``) so we avoid circular
# imports and keep _core a pure shared kernel.
