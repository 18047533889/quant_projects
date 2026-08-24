# -*- coding: utf-8 -*-
"""Intraday session-shape operators (2026-08 geometry/math expansion).

Minute panels (rows = minutes, columns = symbols) are segmented per symbol by an
integer ``session_id`` panel that tags every row with its trading session.  This
module compares the shape of the current session against the recent history of
*completed* sessions:

* ``intraday_session_shape_novelty`` — Euclidean distance of the current
  session's normalised price shape to the nearest historical session's shape.
* ``intraday_profile_pca_residual`` — residual of the current session's shape
  after projecting it on the principal components fitted on past sessions only.

Session semantics (Round-11 findings #68-#74)
---------------------------------------------
* A session is only a *completed* session when its last row timestamp is the
  official session close AND it covers the full expected minute grid.  A run
  whose data stops at 11:00 / 14:00 / close-minus-1min never emits a
  full-session factor (R11 #68).
* ``session_id`` is strict-typed: it must be integral.  A non-integer float
  (``1.9 -> 1``) or a boolean panel is rejected outright (R11 #69).
* A missing / censored ``session_id`` is NOT a second session — adjacent runs
  separated only by censored rows share the same ``sid`` and are merged into one
  session candidate; a gap then fails the coverage gate (R11 #70).
* ``history_days`` is a SESSION-count history, not a daily/minute-bar guess; the
  ``history_days`` parameter declares ``history_semantics="session_count"`` so
  the runtime never treats it as bar warmup (R11 #71).
* ``history_days`` counts COMPLETED sessions, not candidate runs: incomplete
  days are skipped, so the history window expands backward until ``history_days``
  completed sessions are collected (or the history is exhausted) (P0-85).
* Novelty needs a minimum number of completed historical sessions before its
  first output (``min_history_sessions``, R11 #72).
* Session PCA needs a numerical-rank gate — ``len(hist) >= n_components + 1``
  alone does not make the fit identifiable (R11 #73).
* History shapes are resampled onto a market-specific canonical grid (the
  official full-session width), never onto the *current* session's observation
  length, so half-days / gaps do not drift the factor definition (R11 #74).
* The novelty distance is RMSE-normalised (``sqrt(mean(d_i^2))``) so a 1-min
  240-node shape and a 5-min 48-node shape are comparable — raw Euclidean scales
  with node count (P0-86).
* Session completeness compares the OBSERVED minute-of-day slot set against the
  OFFICIAL set (``observed_slot_set == official_slot_set``), never just the
  counts — a missing 09:45 plus a stray 09:46:30 has the same length but a
  different set (P0-83).
* Bar width is a data-contract property: it comes from the calendar's declared
  ``bar_freq`` (SessionCalendar / DataContract / SourceMetadata), NEVER from a
  modal of observed minute deltas — a dataset that systematically drops every
  other bar must not get re-certified at 2-min resolution (P0-84).

Both operators are per-symbol and strictly prefix-causal / PIT-safe: a session's
shape is only known once the session has completed, so the value is emitted at
the official session-close minute (earlier minutes of that session are NaN).
Degenerate inputs (fewer than two finite points, zero session std, degenerate
PCA subspace) emit NaN.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base import OperatorMetadata, ParamSpec, SeriesOperator, register_operator
from factor_engine.cleaned_operators.rolling_pack import frame_like, register_polars_bridge

_EPS = 1e-12
_WARNED_NO_CALENDAR = False


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="intraday_session",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "intraday_session", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:intraday_session",
            f"unit:{unit}", f"cost:{cost}",
        ],
        # R6-196: EOD-realised availability contract — output at session close,
        # never usable in the same session (history-based session operators).
        available_at="session_close",
        same_session_usable=False,
    )


def _minute_of_day(times: np.ndarray) -> np.ndarray:
    """Timestamp array -> minute-of-day (integer)."""
    seconds = times.astype("datetime64[s]").astype("int64") % 86400
    return seconds // 60


def _numerical_rank(s: np.ndarray, tol: float = 1e-9) -> int:
    """Numerical rank of a singular-value spectrum (R11 #73)."""
    if s.size == 0:
        return 0
    smax = float(s[0])
    if smax <= _EPS:
        return 0
    return int(np.sum(s > tol * smax))


def _valid_sid(s: Any) -> bool:
    """Strict SessionID: integral (int or integral float), never bool (R11 #69)."""
    if isinstance(s, (bool, np.bool_)):
        return False
    if isinstance(s, (int, np.integer)):
        return True
    if isinstance(s, (float, np.floating)):
        return bool(np.isfinite(s)) and s == float(int(s))
    return False


def _validate_session_ids(sid_frame: pd.DataFrame) -> np.ndarray:
    """Return the session_id panel as a float array, rejecting non-integral ids.

    R11 #69: a SessionID is integral/categorical.  A bool panel or a non-integral
    finite float (``1.9 -> 1``) is rejected outright — silently truncating
    manufactures a session the caller never declared.
    """
    for c in sid_frame.columns:
        if pd.api.types.is_bool_dtype(sid_frame[c]):
            raise ValueError("session_id must be integral; got a boolean column")
    arr = sid_frame.to_numpy(dtype=float)
    fin = arr[np.isfinite(arr)]
    if np.any(np.abs(fin - np.round(fin)) > 1e-9):
        raise ValueError("session_id must be integral (a non-integer float is not a valid SessionID)")
    return arr


def _hhmm_minute(value: str) -> int:
    """Wall-clock ``"HH:MM"`` -> minute-of-day (``"15:00" -> 900``)."""
    hour, minute = (int(part) for part in str(value).split(":"))
    return hour * 60 + minute


def _calendar_bar_width_minutes(calendar: Any) -> int:
    """Structural bar width in minutes from the calendar's declared ``bar_freq``.

    P0-84: bar resolution is a data-contract property.  It comes from the
    calendar / DataContract / SourceMetadata, NEVER from a modal of observed
    minute deltas.  A dataset that systematically drops every other bar (modal
    delta 2) must not get its completeness silently re-certified at 2-min
    resolution.  An unresolvable resolution fails closed (raises) so the caller
    never emits on a guessed grid.
    """
    bar_freq = getattr(calendar, "bar_freq", None)
    if not bar_freq:
        raise ValueError(
            "calendar must declare `bar_freq` (bar resolution); the resolution "
            "is never inferred from observed minute deltas (P0-84)"
        )
    text = str(bar_freq).strip().lower()
    if text.endswith("min"):
        text = text[:-3]
    elif text.endswith("m"):
        text = text[:-1]
    if not text.isdigit():
        raise ValueError(
            f"calendar bar_freq must be a minute resolution, got {bar_freq!r}"
        )
    width = int(text)
    if width < 1:
        raise ValueError(f"calendar bar_freq must be >= 1 minute, got {bar_freq!r}")
    return width


def _official_grid(calendar: Any) -> tuple[int, int | None, set[int] | None]:
    """Official full-session grid derived from an EXPLICIT exchange calendar.

    P0-10 + P0-84: the official session grid is NEVER inferred from the observed
    minute data — neither the width nor the close.  The bar width comes from the
    calendar's declared ``bar_freq``, the minute-of-day slots from the calendar's
    ``segments`` + ``timestamp_convention``.  A dataset that systematically drops
    the final bar of every session (239 bars instead of 240) or every other bar
    must not be able to re-define "official".  The calendar is a
    ``runtime.session_calendar.SessionCalendar`` (duck-typed: ``segments`` of
    ``("HH:MM","HH:MM")``, ``timestamp_convention`` in {``bar_start``,
    ``bar_end``}, ``bar_freq`` in {``1min``, ``5min``, ...}).

    Returns ``(expected_slots, official_close_mod, official_slot_set)``:
    * ``expected_slots`` — the calendar's per-session bar count at its declared
      resolution;
    * ``official_close_mod`` — the minute-of-day of the final session bar's
      label (``stop - width`` under ``bar_start``, ``stop`` under ``bar_end``);
    * ``official_slot_set`` — the full ordered minute-of-day slot set of a
      complete session (P0-83: completeness compares the observed set against
      this set, not just the count).
    """
    segments = list(getattr(calendar, "segments", None) or ())
    if not segments:
        raise ValueError("calendar must define non-empty session segments")
    width = _calendar_bar_width_minutes(calendar)
    convention = str(getattr(calendar, "timestamp_convention", "bar_end")).lower()
    if convention not in {"bar_start", "bar_end"}:
        raise ValueError("calendar timestamp_convention must be bar_start or bar_end")
    slots: set[int] = set()
    for start_text, stop_text in segments:
        start = _hhmm_minute(start_text)
        stop = _hhmm_minute(stop_text)
        if stop <= start:
            raise ValueError("calendar segments must be increasing intervals")
        if convention == "bar_start":
            slots.update(range(start, stop, width))
        else:
            slots.update(range(start + width, stop + 1, width))
    if not slots:
        raise ValueError("calendar segments must span at least one bar")
    last_stop = _hhmm_minute(segments[-1][1])
    close_mod = last_stop - width if convention == "bar_start" else last_stop
    return len(slots), close_mod, slots


def _resolve_official_grid(calendar: Any) -> tuple[int, int | None, set[int] | None]:
    """Grid authority with fail-closed semantics (P0-10).

    With an explicit calendar the official grid comes from the calendar and
    NEVER from the observed data.  Without a calendar the official grid is
    UNKNOWN — no session can be judged ``completed`` (``expected_slots=0`` makes
    every candidate partial), so the operator emits NaN everywhere instead of
    self-certifying completeness from a modal grid.
    """
    global _WARNED_NO_CALENDAR
    if calendar is None:
        if not _WARNED_NO_CALENDAR:
            _WARNED_NO_CALENDAR = True
            warnings.warn(
                "intraday session operators require an explicit `calendar` "
                "(runtime.session_calendar.SessionCalendar).  Without one the "
                "official session grid is UNKNOWN and no session is judged "
                "completed (all NaN); the grid is never inferred from observed "
                "data (P0-10).",
                RuntimeWarning,
                stacklevel=2,
            )
        return 0, None, None
    return _official_grid(calendar)


def _session_runs(
    x_col: np.ndarray,
    sid_col: np.ndarray,
    dates: np.ndarray,
    mods: np.ndarray,
    expected_slots: int,
    official_close_mod: int | None,
    official_slot_set: set[int] | None = None,
    on_grid: np.ndarray | None = None,
) -> list[dict]:
    """Split one symbol's minute column into session candidates.

    Session identity is the COMPOSED key ``(trade_date, session_id)`` (P0-41): a
    caller that restarts ``session_id`` at 1 on every trading date must never
    merge blocks across days, and a session that straddles a date boundary is
    split — each day's rows form their own candidate.  Consecutive rows with the
    same integral ``session_id`` within one ``trade_date`` form a run; adjacent
    runs separated only by censored (NaN / non-integral) ids with the SAME
    ``(trade_date, sid)`` are merged into one session (R11 #70).  A candidate is
    ``completed`` only when it reaches the official session close AND covers the
    full expected minute grid (R11 #68).  ``vals`` are the x values at the
    valid-session rows only (censored rows belong to no known session).
    """
    n = len(x_col)
    blocks: list[tuple[int, int, int, int]] = []  # (date, sid, start, end)
    i = 0
    while i < n:
        if not _valid_sid(sid_col[i]):
            i += 1
            continue
        sid = int(sid_col[i])
        day = int(dates[i])
        j = i
        while (
            j < n
            and _valid_sid(sid_col[j])
            and int(sid_col[j]) == sid
            and int(dates[j]) == day
        ):
            j += 1
        blocks.append((day, sid, i, j - 1))
        i = j
    merged: list[dict] = []
    for day, sid, start, end in blocks:
        if merged and merged[-1]["sid"] == sid and merged[-1]["date"] == day:
            merged[-1]["blocks"].append((start, end))
        else:
            merged.append({"date": day, "sid": sid, "blocks": [(start, end)]})
    out: list[dict] = []
    for sess in merged:
        sid = sess["sid"]
        day = sess["date"]
        rows: list[int] = []
        for bs, be in sess["blocks"]:
            rows.extend(range(bs, be + 1))
        start, end = rows[0], rows[-1]
        vals = np.asarray([x_col[r] for r in rows], dtype=float)
        obs_slot_list = [int(mods[r]) for r in rows]
        obs_slots = set(obs_slot_list)
        # R15-INC-123/124: completeness is a MULTISET + grid-cleanliness check,
        # not a bare set equality.  A duplicate bar (two rows at 09:45) has the
        # same SET as a complete session but a different COUNT — require the
        # observed bar count to equal the official slot count so duplicates fail.
        # A timestamp that does not sit exactly on the calendar grid (a stray
        # 09:46:30 floor-mapped to 09:46) also must never certify a session.
        grid_clean = (
            on_grid is None
            or bool(np.all(on_grid[rows]))
        )
        completed = (
            expected_slots > 0
            and official_slot_set is not None
            and len(obs_slot_list) == expected_slots
            and obs_slots == official_slot_set
            and grid_clean
            and (official_close_mod is None or int(mods[end]) == official_close_mod)
        )
        out.append(
            {
                "date": day,
                "sid": sid,
                "vals": vals,
                "start": start,
                "end": end,
                "slots": len(obs_slots),
                "completed": completed,
            }
        )
    return out


def _normalized_shape(vals: np.ndarray) -> np.ndarray | None:
    """Standardised shape vector (x - mean)/std; None when degenerate."""
    v = vals.astype(float)
    if v.size < 2 or not np.all(np.isfinite(v)):
        return None
    sd = v.std()
    if sd <= _EPS:
        return None
    return (v - v.mean()) / sd


def _resample_shape(shape: np.ndarray, target_len: int) -> np.ndarray:
    """Linear-interpolate a shape onto a grid of length ``target_len``."""
    n = shape.size
    if target_len == n:
        return shape.astype(float)
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, target_len)
    return np.interp(dst, src, shape)


def _canonical_shape(vals: np.ndarray, n_nodes: int) -> np.ndarray | None:
    """Normalised shape resampled onto the CANONICAL full-session grid (R11 #74).

    Every session — half-day, gap, full day — is mapped to the same 0..1
    official-session grid, so the factor definition never drifts with the current
    observation length.
    """
    shape = _normalized_shape(vals)
    if shape is None:
        return None
    return _resample_shape(shape, n_nodes)


def _shape_novelty_series(
    x2d: np.ndarray,
    sid2d: np.ndarray,
    dates: np.ndarray,
    mods: np.ndarray,
    history_days: int,
    min_history_sessions: int,
    expected_slots: int,
    official_close_mod: int | None,
    official_slot_set: set[int] | None = None,
    on_grid: np.ndarray | None = None,
) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    hd = int(history_days)
    mhs = max(1, int(min_history_sessions))
    for c in range(cols):
        n_nodes = max(2, expected_slots)
        runs = _session_runs(x2d[:, c], sid2d[:, c], dates, mods, expected_slots, official_close_mod, official_slot_set, on_grid)
        for i, run in enumerate(runs):
            if not run["completed"]:
                continue  # partial session never emits a full-session factor (R11 #68)
            shape = _canonical_shape(run["vals"], n_nodes)
            if shape is None:
                continue
            best = np.inf
            seen = 0
            # P0-85: history_days counts COMPLETED sessions, not candidate runs.
            # Scan backward from the previous run, skipping incomplete candidates,
            # until ``history_days`` completed sessions are collected (or the
            # history is exhausted).  P0-86: the distance is RMSE-normalised so a
            # 1-min 240-node shape and a 5-min 48-node shape are comparable.
            for j in range(i - 1, -1, -1):
                hrun = runs[j]
                if not hrun["completed"]:
                    continue
                hshape = _canonical_shape(hrun["vals"], n_nodes)
                if hshape is None:
                    continue
                d = float(np.linalg.norm(hshape - shape) / np.sqrt(n_nodes))
                seen += 1
                if d < best:
                    best = d
                if seen >= hd:
                    break
            if seen < mhs:
                continue  # too few historical sessions -> too weak an estimate (R11 #72)
            out[run["end"], c] = float(best)
    return out


def _pca_residual_series(
    x2d: np.ndarray,
    sid2d: np.ndarray,
    dates: np.ndarray,
    mods: np.ndarray,
    history_days: int,
    n_components: int,
    min_history_sessions: int,
    expected_slots: int,
    official_close_mod: int | None,
    official_slot_set: set[int] | None = None,
    on_grid: np.ndarray | None = None,
) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    hd = int(history_days)
    nc = int(n_components)
    mhs = max(1, int(min_history_sessions))
    for c in range(cols):
        n_nodes = max(2, expected_slots)
        runs = _session_runs(x2d[:, c], sid2d[:, c], dates, mods, expected_slots, official_close_mod, official_slot_set, on_grid)
        for i, run in enumerate(runs):
            if not run["completed"]:
                continue
            shape = _canonical_shape(run["vals"], n_nodes)
            if shape is None:
                continue
            hist: list[np.ndarray] = []
            # P0-85: history_days counts COMPLETED sessions, not candidate runs —
            # scan backward skipping incomplete candidates until ``history_days``
            # completed sessions are collected (or history is exhausted).
            for j in range(i - 1, -1, -1):
                hrun = runs[j]
                if not hrun["completed"]:
                    continue
                hshape = _canonical_shape(hrun["vals"], n_nodes)
                if hshape is None:
                    continue
                hist.append(hshape)
                if len(hist) >= hd:
                    break
            if len(hist) < max(mhs, nc + 1):
                continue  # too few past sessions for an nc-component fit (R11 #72/#73)
            M = np.stack(hist, axis=0)  # (n_past, n_nodes)
            mean_m = M.mean(axis=0)
            Mc = M - mean_m
            _U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
            if _numerical_rank(S) < nc + 1:
                continue  # R11 #73: len(hist) >= nc+1 is NOT a rank gate
            if float(S[0]) <= _EPS:
                continue  # degenerate history (all identical) -> subspace undefined
            comps = Vt[:nc]  # (nc, n_nodes)
            coeff = (shape - mean_m) @ comps.T  # (nc,)
            z_hat = mean_m + coeff @ comps
            num = float(np.linalg.norm(shape - z_hat))
            den = float(np.linalg.norm(shape)) + _EPS
            out[run["end"], c] = num / den
    return out


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------
@register_operator(
    name="intraday_session_shape_novelty",
    category="intraday_session",
    business_category="intraday_session",
    canonical="intraday_session_shape_novelty",
    source="intraday_session",
)
class IntradaySessionShapeNovelty(SeriesOperator):
    """日内分时形态新颖度：当前 session 分时形态相对最近历史 session 形态的最近距离。

    每个 session 的形态先标准化 ``(x-mean)/std``，再重采样到官方全场 session 网格
    （``expected_slots`` 个节点），最后求 **RMSE** 距离（``sqrt(mean(d_i^2))``，
    消除节点数对欧氏距离的尺度依赖）取最小值。只有 *completed* session（末行即官方
    收盘且覆盖完整网格）才在收盘分钟输出；少于 ``min_history_sessions`` 个 completed
    历史 session -> NaN。P1。
    """

    metadata = _metadata(
        "intraday_session_shape_novelty",
        "当前 session 标准化分时形态到最近历史 session 形态的 RMSE 距离（节点数无关）。",
        ["x", "session_id", "history_days", "min_history_sessions", "calendar"],
        unit="ratio",
        cost=7,
    )
    metadata.param_specs = {
        # R11 #71: history_days counts SESSIONS, not trading bars.
        "history_days": ParamSpec(dtype=int, min=1, history_semantics="session_count"),
        "min_history_sessions": ParamSpec(dtype=int, min=1, searchable=False),
    }

    def _calculate_series(
        self,
        x: pd.DataFrame,
        session_id: pd.DataFrame,
        history_days: int = 20,
        min_history_sessions: int = 5,
        calendar: Any = None,
        **_: Any,
    ) -> pd.DataFrame:
        sid_arr = _validate_session_ids(session_id)
        # P0-41: session identity is the composed (trade_date, session_id) key —
        # a caller that restarts ``sid`` at 1 each day must never collapse blocks
        # across days.  ``dates`` is the per-row day key derived from the index.
        index_ns = x.index.to_numpy(dtype="datetime64[ns]")
        dates = index_ns.astype("datetime64[D]").astype("int64")
        mods = _minute_of_day(index_ns)
        # R15-INC-124: a timestamp that does not sit exactly on a minute grid
        # (stray seconds/microseconds) is an invalid slot — it must never certify
        # a session as complete.
        on_grid = (
            index_ns.astype("datetime64[ns]").astype("int64") % 60_000_000_000 == 0
        )
        # P0-10/P0-84: official grid from the explicit exchange calendar (bar
        # width included), never from a modal of the observed minutes.
        expected, close_mod, slot_set = _resolve_official_grid(calendar)
        arr = _shape_novelty_series(
            x.to_numpy(dtype=float), sid_arr, dates, mods, history_days, min_history_sessions,
            expected, close_mod, slot_set, on_grid,
        )
        return frame_like(x, arr)


@register_operator(
    name="intraday_profile_pca_residual",
    category="intraday_session",
    business_category="intraday_session",
    canonical="intraday_profile_pca_residual",
    source="intraday_session",
)
class IntradayProfilePcaResidual(SeriesOperator):
    """日内分时 PCA 残差：当前 session 形态相对仅用历史 session 拟合的主成分重建的残差。

    把最近 history_days 个 completed 历史 session 的标准化形态重采样到官方全场
    网格后组成矩阵，只在这些过去 session 上拟合 PCA；残差 =
    ``|Z - Ẑ| / (|Z|+eps)``。历史 session 少于 ``max(min_history_sessions,
    n_components+1)`` 个或数值秩不足 ``n_components+1`` -> NaN。在收盘分钟输出。P1。
    """

    metadata = _metadata(
        "intraday_profile_pca_residual",
        "当前 session 形态相对历史 PCA 主成分重建的归一化残差。",
        ["x", "session_id", "history_days", "n_components", "min_history_sessions", "calendar"],
        unit="ratio",
        cost=8,
    )
    metadata.param_specs = {
        "history_days": ParamSpec(dtype=int, min=1, history_semantics="session_count"),  # R11 #71
        "n_components": ParamSpec(dtype=int, min=1),
        "min_history_sessions": ParamSpec(dtype=int, min=1, searchable=False),
    }

    def _calculate_series(
        self,
        x: pd.DataFrame,
        session_id: pd.DataFrame,
        history_days: int = 20,
        n_components: int = 3,
        min_history_sessions: int = 5,
        calendar: Any = None,
        **_: Any,
    ) -> pd.DataFrame:
        sid_arr = _validate_session_ids(session_id)
        # P0-41: session identity is the composed (trade_date, session_id) key —
        # a caller that restarts ``sid`` at 1 each day must never collapse blocks
        # across days.  ``dates`` is the per-row day key derived from the index.
        index_ns = x.index.to_numpy(dtype="datetime64[ns]")
        dates = index_ns.astype("datetime64[D]").astype("int64")
        mods = _minute_of_day(index_ns)
        # R15-INC-124: a timestamp that does not sit exactly on a minute grid
        # (stray seconds/microseconds) is an invalid slot — it must never certify
        # a session as complete.
        on_grid = (
            index_ns.astype("datetime64[ns]").astype("int64") % 60_000_000_000 == 0
        )
        # P0-10/P0-84: official grid from the explicit exchange calendar (bar
        # width included), never from a modal of the observed minutes.
        expected, close_mod, slot_set = _resolve_official_grid(calendar)
        arr = _pca_residual_series(
            x.to_numpy(dtype=float), sid_arr, dates, mods,
            history_days, n_components, min_history_sessions,
            expected, close_mod, slot_set, on_grid,
        )
        return frame_like(x, arr)


_NEW_CANONICALS = (
    "intraday_session_shape_novelty",
    "intraday_profile_pca_residual",
)


def _register_surface() -> None:
    import factor_engine.cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


def _declare_stateful_contracts() -> None:
    """R11 #71: ``history_days`` is a SESSION-count history, not a bar warmup.

    Declare the session-clock history contract so ``history_requirement()``
    reports ``HistoryRequirement(kind='session_count')`` instead of guessing a
    daily/minute-bar warmup from the parameter name.
    """
    from factor_engine.runtime.execution_contract import declare_stateful

    for _canon in _NEW_CANONICALS:
        declare_stateful(
            _canon,
            state_model="session_state",
            chunking="required_full_history",
            minimum_history=5,
            history_kind="session_count",
            history_count=20,
        )


_register_surface()
_declare_stateful_contracts()
