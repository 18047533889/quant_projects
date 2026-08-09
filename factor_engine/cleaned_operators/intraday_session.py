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
* Novelty needs a minimum number of completed historical sessions before its
  first output (``min_history_sessions``, R11 #72).
* Session PCA needs a numerical-rank gate — ``len(hist) >= n_components + 1``
  alone does not make the fit identifiable (R11 #73).
* History shapes are resampled onto a market-specific canonical grid (the
  official full-session width), never onto the *current* session's observation
  length, so half-days / gaps do not drift the factor definition (R11 #74).

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

from cleaned_operators.base import OperatorMetadata, ParamSpec, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like, register_polars_bridge

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


def _bar_width_minutes(mods: np.ndarray) -> int:
    """Structural bar width in minutes (modal positive minute delta).

    This is a data-RESOLUTION property (1-min vs 5-min bars), never a
    completeness signal.  It is used only to convert the calendar's total
    session minutes into the expected per-session BAR count.
    """
    diffs = np.diff(mods)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1
    vals, counts = np.unique(diffs, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _official_grid(calendar: Any, mods: np.ndarray) -> tuple[int, int | None]:
    """Official full-session grid derived from an EXPLICIT exchange calendar.

    P0-10: the official session width and the official session close are NEVER
    inferred from the observed minute data.  A dataset that systematically drops
    the final bar of every session (239 bars instead of 240) must not be able to
    re-define "official" — the modal-grid self-certification is removed.  The
    calendar is a ``runtime.session_calendar.SessionCalendar`` (duck-typed:
    ``segments`` of ``("HH:MM","HH:MM")``, ``timestamp_convention`` in
    {``bar_start``, ``bar_end``}).

    Returns ``(expected_slots, official_close_mod)``:
    * ``expected_slots`` — the calendar's per-session bar count at the data's
      structural bar width;
    * ``official_close_mod`` — the minute-of-day of the final session bar's
      label (``stop - 1`` under ``bar_start``, ``stop`` under ``bar_end``).
    """
    segments = list(getattr(calendar, "segments", None) or ())
    if not segments:
        raise ValueError("calendar must define non-empty session segments")
    total_minutes = 0
    for start_text, stop_text in segments:
        total_minutes += _hhmm_minute(stop_text) - _hhmm_minute(start_text)
    if total_minutes <= 0:
        raise ValueError("calendar segments must be increasing intervals")
    convention = str(getattr(calendar, "timestamp_convention", "bar_end")).lower()
    if convention not in {"bar_start", "bar_end"}:
        raise ValueError("calendar timestamp_convention must be bar_start or bar_end")
    width = _bar_width_minutes(mods)
    expected_slots = int(np.ceil(total_minutes / max(1, width)))
    stop = _hhmm_minute(segments[-1][1])
    close_mod = stop - 1 if convention == "bar_start" else stop
    return expected_slots, close_mod


def _resolve_official_grid(calendar: Any, mods: np.ndarray) -> tuple[int, int | None]:
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
        return 0, None
    return _official_grid(calendar, mods)


def _session_runs(
    x_col: np.ndarray,
    sid_col: np.ndarray,
    mods: np.ndarray,
    expected_slots: int,
    official_close_mod: int | None,
) -> list[dict]:
    """Split one symbol's minute column into session candidates.

    Consecutive rows with the same integral ``session_id`` form a run; adjacent
    runs separated only by censored (NaN / non-integral) ids with the SAME
    ``sid`` are merged into one session (R11 #70).  A candidate is ``completed``
    only when it reaches the official session close AND covers the full expected
    minute grid (R11 #68).  ``vals`` are the x values at the valid-session rows
    only (censored rows belong to no known session).
    """
    n = len(x_col)
    blocks: list[tuple[int, int, int]] = []  # (sid, start, end)
    i = 0
    while i < n:
        if not _valid_sid(sid_col[i]):
            i += 1
            continue
        sid = int(sid_col[i])
        j = i
        while j < n and _valid_sid(sid_col[j]) and int(sid_col[j]) == sid:
            j += 1
        blocks.append((sid, i, j - 1))
        i = j
    merged: list[dict] = []
    for sid, start, end in blocks:
        if merged and merged[-1]["sid"] == sid:
            merged[-1]["blocks"].append((start, end))
        else:
            merged.append({"sid": sid, "blocks": [(start, end)]})
    out: list[dict] = []
    for sess in merged:
        sid = sess["sid"]
        rows: list[int] = []
        for bs, be in sess["blocks"]:
            rows.extend(range(bs, be + 1))
        start, end = rows[0], rows[-1]
        vals = np.asarray([x_col[r] for r in rows], dtype=float)
        obs_slots = {int(mods[r]) for r in rows}
        completed = (
            expected_slots > 0
            and len(obs_slots) == expected_slots
            and (official_close_mod is None or int(mods[end]) == official_close_mod)
        )
        out.append(
            {
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
    mods: np.ndarray,
    history_days: int,
    min_history_sessions: int,
    expected_slots: int,
    official_close_mod: int | None,
) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    hd = int(history_days)
    mhs = max(1, int(min_history_sessions))
    for c in range(cols):
        n_nodes = max(2, expected_slots)
        runs = _session_runs(x2d[:, c], sid2d[:, c], mods, expected_slots, official_close_mod)
        for i, run in enumerate(runs):
            if not run["completed"]:
                continue  # partial session never emits a full-session factor (R11 #68)
            shape = _canonical_shape(run["vals"], n_nodes)
            if shape is None:
                continue
            best = np.inf
            seen = 0
            for j in range(max(0, i - hd), i):  # completed sessions ending before current only
                hrun = runs[j]
                if not hrun["completed"]:
                    continue
                hshape = _canonical_shape(hrun["vals"], n_nodes)
                if hshape is None:
                    continue
                d = float(np.linalg.norm(hshape - shape))
                seen += 1
                if d < best:
                    best = d
            if seen < mhs:
                continue  # too few historical sessions -> too weak an estimate (R11 #72)
            out[run["end"], c] = float(best)
    return out


def _pca_residual_series(
    x2d: np.ndarray,
    sid2d: np.ndarray,
    mods: np.ndarray,
    history_days: int,
    n_components: int,
    min_history_sessions: int,
    expected_slots: int,
    official_close_mod: int | None,
) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    hd = int(history_days)
    nc = int(n_components)
    mhs = max(1, int(min_history_sessions))
    for c in range(cols):
        n_nodes = max(2, expected_slots)
        runs = _session_runs(x2d[:, c], sid2d[:, c], mods, expected_slots, official_close_mod)
        for i, run in enumerate(runs):
            if not run["completed"]:
                continue
            shape = _canonical_shape(run["vals"], n_nodes)
            if shape is None:
                continue
            hist: list[np.ndarray] = []
            for j in range(max(0, i - hd), i):  # past completed sessions only
                hrun = runs[j]
                if not hrun["completed"]:
                    continue
                hshape = _canonical_shape(hrun["vals"], n_nodes)
                if hshape is None:
                    continue
                hist.append(hshape)
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
    （``expected_slots`` 个节点），最后求欧氏距离取最小值。只有 *completed* session
    （末行即官方收盘且覆盖完整网格）才在收盘分钟输出；少于 ``min_history_sessions``
    个 completed 历史 session -> NaN。P1。
    """

    metadata = _metadata(
        "intraday_session_shape_novelty",
        "当前 session 标准化分时形态到最近历史 session 形态的欧氏距离。",
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
        mods = _minute_of_day(x.index.to_numpy(dtype="datetime64[ns]"))
        # P0-10: official grid from the explicit exchange calendar, never modal.
        expected, close_mod = _resolve_official_grid(calendar, mods)
        arr = _shape_novelty_series(
            x.to_numpy(dtype=float), sid_arr, mods, history_days, min_history_sessions,
            expected, close_mod,
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
        mods = _minute_of_day(x.index.to_numpy(dtype="datetime64[ns]"))
        # P0-10: official grid from the explicit exchange calendar, never modal.
        expected, close_mod = _resolve_official_grid(calendar, mods)
        arr = _pca_residual_series(
            x.to_numpy(dtype=float), sid_arr, mods,
            history_days, n_components, min_history_sessions,
            expected, close_mod,
        )
        return frame_like(x, arr)


_NEW_CANONICALS = (
    "intraday_session_shape_novelty",
    "intraday_profile_pca_residual",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


def _declare_stateful_contracts() -> None:
    """R11 #71: ``history_days`` is a SESSION-count history, not a bar warmup.

    Declare the session-clock history contract so ``history_requirement()``
    reports ``HistoryRequirement(kind='session_count')`` instead of guessing a
    daily/minute-bar warmup from the parameter name.
    """
    from runtime.execution_contract import declare_stateful

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
