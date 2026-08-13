# -*- coding: utf-8 -*-
"""Event-interval statistics operators (2026-08 geometry/math expansion).

An *event panel* is a ``TradeDate x Symbol`` frame with explicit **EventBool**
semantics (reviews R4-54 / R4-96): ``1`` (or any finite nonzero) marks an
event, ``0`` marks a *confirmed* no-event row, and ``NaN`` marks an *unknown*
row.  For each instrument column the distance between consecutive event rows
defines the inter-event intervals ``τ_i`` inside the trailing window; when a
pre-window event is known the gap from it into the window is included as the
first interval (so the statistics are never starved at the window boundary).
An interval that crosses an unknown (NaN) row is **censored** — its distance is
not defined and it invalidates the window's statistic (``NaN``) rather than
silently spanning the unknown region.  The family characterises how
regular / memory-laden / bursty the event process is:

* ``event_interval_memory``   — Pearson correlation of consecutive intervals
  (short/long interval persistence).
* ``event_local_variation``   — local variation coefficient of the intervals
  (regular vs. irregular spacing).
* ``event_fano_factor``       — block-count dispersion (variance/mean) of the
  event process over blocks of the window (burstiness).

All operators are trailing-window, prefix-causal (row ``r`` uses rows ``<= r``
only) and deterministic.  Windows with too few intervals emit ``NaN``; invalid
parameters raise ``ValueError``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.rolling_pack import frame_like, register_polars_bridge

_EPS = 1e-12


def _metadata(name: str, description: str, params: list[str], *, unit: str, cost: int) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="event_interval",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "event_interval", "daily", "pit_safe", "causal", "typed_v2",
            "deterministic",
            f"signature:{','.join(params)}->series", "domain:event_process",
            f"unit:{unit}", f"cost:{cost}",
        ],
    )


def _event_mask(col: np.ndarray) -> np.ndarray:
    """Strict EventBool mask (Master Spec Part M-63/64).

    Only ``1`` is an event, only ``0`` is a *confirmed* no-event row and only
    ``NaN`` is an unknown row.  Any other finite value (``-1``, ``0.2``, ``2``)
    is out-of-domain and raises ``ValueError`` instead of being silently
    interpreted — a mis-labelled mark must fail loudly, never invent an event.
    """
    valid_bool = np.isnan(col) | (col == 0.0) | (col == 1.0)
    if not bool(np.all(valid_bool | ~np.isfinite(col))):
        bad = col[np.isfinite(col) & ~valid_bool]
        raise ValueError(
            "event panel must be strict EventBool {0, 1, NaN}; got "
            f"out-of-domain value(s) {np.unique(bad)[:5]!r}"
        )
    return col == 1.0


def _window_taus(
    ev_pos: np.ndarray,
    lo_idx: int,
    hi_idx: int,
    i0: int,
    unknown_mask: np.ndarray,
    max_pre_window_age: int,
) -> np.ndarray | None:
    """Inter-event intervals ``τ`` for events in ``[i0, r]``.

    The gap from the last known pre-window event (``ev_pos[lo_idx-1] < i0``,
    when it exists) into the first in-window event is prepended — but only when
    that pre-window event lies within ``max_pre_window_age`` rows of the window
    start.  Round-7 P0: without this bound the operator silently reaches back an
    *arbitrary* distance for a sparse event process (window=240 could read a
    pre-window event from 1000 rows earlier), so ``event_interval_memory``'s
    effective history is unbounded and ``chunk != full`` / incremental restart
    parity breaks.  Bounding the reach to ``max_pre_window_age`` makes the
    operator's history exactly ``window + max_pre_window_age`` rows.

    Returns ``None`` when the window contains no event rows.

    Review R4-54: an interval whose open range crosses an *unknown* (NaN) row
    is censored — the inter-event distance is undefined across an unknown
    region, so the whole window's statistic is NaN rather than spanning it.
    """
    if hi_idx <= lo_idx:
        return None
    pos = ev_pos[lo_idx:hi_idx]
    if lo_idx > 0:
        prev = ev_pos[lo_idx - 1]
        if i0 - max_pre_window_age <= prev < i0:
            pos = np.concatenate([[prev], pos])
    if pos.size < 2:
        return None
    taus = np.diff(pos).astype(float)
    for idx in range(taus.size):
        a, b = int(pos[idx]), int(pos[idx + 1])
        if np.any(unknown_mask[a + 1 : b]):
            return None  # interval crosses unknown -> censored
    return taus


def _interval_memory_series(event2d: np.ndarray, window: int, max_pre_window_age: int) -> np.ndarray:
    rows, cols = event2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        col = event2d[:, c]
        unknown = ~np.isfinite(col)
        ev_pos = np.flatnonzero(_event_mask(col))
        for r in range(rows):
            i0 = max(0, r - w + 1)
            lo = np.searchsorted(ev_pos, i0, side="left")
            hi = np.searchsorted(ev_pos, r, side="right")
            taus = _window_taus(ev_pos, lo, hi, i0, unknown, max_pre_window_age)
            # Master Spec N-69: 3-4 intervals is not production support for a
            # correlation — the effective sample is the interval COUNT.  Six
            # intervals minimum.
            if taus is None or taus.size < 6:
                continue
            corr = np.corrcoef(taus[:-1], taus[1:])
            val = corr[0, 1]
            if np.isfinite(val):
                out[r, c] = float(val)
    return out


def _local_variation_series(event2d: np.ndarray, window: int, max_pre_window_age: int) -> np.ndarray:
    rows, cols = event2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        col = event2d[:, c]
        unknown = ~np.isfinite(col)
        ev_pos = np.flatnonzero(_event_mask(col))
        for r in range(rows):
            i0 = max(0, r - w + 1)
            lo = np.searchsorted(ev_pos, i0, side="left")
            hi = np.searchsorted(ev_pos, r, side="right")
            taus = _window_taus(ev_pos, lo, hi, i0, unknown, max_pre_window_age)
            # Master Spec N-69: minimum 6 intervals for production support.
            if taus is None or taus.size < 6:
                continue
            n = taus.size
            d = taus[1:] - taus[:-1]
            s = taus[1:] + taus[:-1]
            lv = float(np.sum((d / (s + _EPS)) ** 2))
            out[r, c] = np.where((n - 1.0)) * lv != 0, (3.0 / (n - 1.0)) * lv, np.nan)
    return out


def _fano_factor_series(
    event2d: np.ndarray,
    window: int,
    block: int,
    *,
    min_valid_blocks: int = 5,
    sample_variance: bool = True,
) -> np.ndarray:
    rows, cols = event2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w, b = int(window), int(block)
    for c in range(cols):
        ev = event2d[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            chunk = ev[i0 : r + 1]
            length = chunk.shape[0]
            # Master Spec N-66: the blocks are RIGHT-aligned to the newest bar —
            # window 45 / block 20 must use the latest 40 rows (20/20), not the
            # oldest 40, so the current observation is never dropped.  A trailing
            # partial block on the OLD side is discarded (R4-55 stays: partial
            # blocks never enter the distribution).
            n_full = length // b
            if n_full < min_valid_blocks:
                continue
            start = r + 1 - n_full * b
            counts = np.empty(n_full, dtype=float)
            valid = np.zeros(n_full, dtype=bool)
            for k in range(n_full):
                seg = chunk[start - i0 + k * b : start - i0 + (k + 1) * b]
                if np.any(~np.isfinite(seg)):
                    continue  # unknown minute -> block count is not defined
                counts[k] = float(np.count_nonzero(_event_mask(seg)))
                valid[k] = True
            if int(valid.sum()) < min_valid_blocks:
                continue
            v = counts[valid]
            if float(v.sum()) <= 0.0:  # degenerate: no events in the window
                continue
            mean = float(v.mean())
            # N-67: sample variance (ddof=1); with a small number of blocks the
            # population variance mechanically understates the Poisson baseline.
            var = float(v.var(ddof=1) if sample_variance and len(v) > 1 else v.var())
            out[r, c] = var / (mean + _EPS)
    return out


def _check_event_params(window: int, block: int | None = None) -> tuple[int, int | None]:
    """Strict-int param gate (Master Spec A-4): a float ``20.2`` must not
    silently truncate to 20 — it is a contract error and raises."""
    if isinstance(window, (bool, np.bool_)):
        raise ValueError("window must be an integer, not bool")
    if not isinstance(window, (int, np.integer)):
        raise ValueError(f"window must be an integer, got {window!r}")
    w = int(window)
    if w < 2:
        raise ValueError("window must be >= 2")
    if block is not None:
        if isinstance(block, (bool, np.bool_)):
            raise ValueError("block must be an integer, not bool")
        if not isinstance(block, (int, np.integer)):
            raise ValueError(f"block must be an integer, got {block!r}")
        b = int(block)
        if b < 1:
            raise ValueError("block must be >= 1")
        return w, b
    return w, None


@register_operator(
    name="event_interval_memory",
    category="event_interval",
    business_category="event_interval",
    canonical="event_interval_memory",
    source="event_interval",
)
class EventIntervalMemory(SeriesOperator):
    """事件间隔记忆：连续事件间隔对 ``(τ_i, τ_{i+1})`` 的 Pearson 相关。

    正 → 间隔长短持续（聚集/惯性）；负 → 长短交替；≈0 → 间隔近似独立。
    R26-090/091：需要窗口内至少 6 个有效间隔，否则 NaN（与 kernel 一致）。
    EventBool 语义（R26-092）：严格 ``{1, 0, NaN}`` —— 1 = 事件，0 = 确认
    非事件，NaN = unknown；其他有限值直接抛错；跨越 unknown 的间隔被
    censored → 窗口 NaN（R4-54/96）。
    """

    metadata = _metadata(
        "event_interval_memory",
        "连续事件间隔的 Pearson 相关（间隔记忆 / 聚集性）。",
        ["event", "window", "max_pre_window_age"],
        unit="corr",
        cost=3,
    )

    def _calculate_series(self, event: pd.DataFrame, window: int = 240, max_pre_window_age: Any = None, **_: Any) -> pd.DataFrame:
        w, _ = _check_event_params(window)
        # Round-7 P0: the pre-window event may only reach ``max_pre_window_age``
        # rows before the window (default = window), so the effective history is
        # bounded to ``2 * window`` and full/chunk/incremental parity holds.
        pre = int(max_pre_window_age) if max_pre_window_age is not None else w
        if pre < 1:
            raise ValueError("max_pre_window_age must be >= 1")
        return frame_like(event, _interval_memory_series(event.to_numpy(dtype=float), w, pre))


@register_operator(
    name="event_local_variation",
    category="event_interval",
    business_category="event_interval",
    canonical="event_local_variation",
    source="event_interval",
)
class EventLocalVariation(SeriesOperator):
    """事件间隔局部变异：``LV = (3/(n-1))·Σ ((τ_{i+1}-τ_i)/(τ_{i+1}+τ_i))²``。

    规则事件序列 → LV≈0；间隔不规则 / 间歇性 → 大值。R26-090/091：需要至少
    6 个有效间隔（与 kernel 一致）。EventBool：严格 ``{1, 0, NaN}``，NaN =
    unknown，跨越 unknown 的间隔被 censored → 窗口 NaN。
    """

    metadata = _metadata(
        "event_local_variation",
        "事件间隔的局部变异系数（规则 vs 不规则）。",
        ["event", "window", "max_pre_window_age"],
        unit="ratio",
        cost=3,
    )

    def _calculate_series(self, event: pd.DataFrame, window: int = 240, max_pre_window_age: Any = None, **_: Any) -> pd.DataFrame:
        w, _ = _check_event_params(window)
        pre = int(max_pre_window_age) if max_pre_window_age is not None else w
        if pre < 1:
            raise ValueError("max_pre_window_age must be >= 1")
        return frame_like(event, _local_variation_series(event.to_numpy(dtype=float), w, pre))


@register_operator(
    name="event_fano_factor",
    category="event_interval",
    business_category="event_interval",
    canonical="event_fano_factor",
    source="event_interval",
)
class EventFanoFactor(SeriesOperator):
    """事件 Fano 因子：把窗口切成 ``block`` 行的完整块，``N_k`` = 每块事件数，
    ``F = Var(N_k)/Mean(N_k)``（分母加 eps 防除零）。

    F≈1 → 泊松型随机过程；F>1 → 聚集/爆发；F<1 → 更规则。仅统计完整块
    （R4-55，尾部 partial block 不入分布）；含 unknown(NaN) 的块不计数。
    R26-090/091：少于 ``min_valid_blocks``（默认 5）个有效完整块 → NaN（与
    kernel/ParamSpec 一致）。EventBool：严格 ``{1, 0, NaN}``。
    """

    metadata = _metadata(
        "event_fano_factor",
        "块内事件数方均比（burstiness / 聚集性）。",
        ["event", "window", "block"],
        unit="ratio",
        cost=3,
    )

    def _calculate_series(self, event: pd.DataFrame, window: int = 240, block: int = 20, **_: Any) -> pd.DataFrame:
        w, b = _check_event_params(window, block)
        return frame_like(event, _fano_factor_series(event.to_numpy(dtype=float), w, b))


@register_operator(
    name="event_fano_excess",
    category="event_interval",
    business_category="event_interval",
    canonical="event_fano_excess",
    source="event_interval",
)
class EventFanoExcess(SeriesOperator):
    """事件 Fano 超额：``F - 1``。

    ``event_fano_factor`` 的 Poisson-null 校准版（Master Spec N-68）：对泊松
    过程 F≈1，所以 ``excess = F - 1`` 是零中心的 burstiness 度量，比 raw Fano
    更适合自动搜索（正 → 聚集/爆发，负 → 更规则，≈0 → 泊松）。继承 Fano 的
    右对齐完整块、sample variance (ddof=1)、``min_valid_blocks >= 5`` 支持门。
    EventBool 严格 {0,1,NaN}，NaN 块不计数，跨 unknown 的间隔被 censored。
    """

    metadata = _metadata(
        "event_fano_excess",
        "事件 Fano 因子相对泊松基线（=1）的超额：F-1（零中心 burstiness）。",
        ["event", "window", "block"],
        unit="ratio",
        cost=3,
    )

    def _calculate_series(self, event: pd.DataFrame, window: int = 240, block: int = 20, **_: Any) -> pd.DataFrame:
        w, b = _check_event_params(window, block)
        fano = _fano_factor_series(event.to_numpy(dtype=float), w, b)
        out = fano - 1.0
        out[~np.isfinite(fano)] = np.nan
        return frame_like(event, out)


_NEW_CANONICALS = (
    "event_interval_memory",
    "event_local_variation",
    "event_fano_factor",
    "event_fano_excess",
)


def _register_surface() -> None:
    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_NEW_CANONICALS))
    for _canon in _NEW_CANONICALS:
        register_polars_bridge(_canon)


_register_surface()


# ---------------------------------------------------------------------------
# Semantic Closure declarations (Master Spec P0)
# ---------------------------------------------------------------------------

def _declare_closure_contracts() -> None:
    from cleaned_operators.closure import (
        MissingPolicy,
        WindowSemantics,
        declare_missing_policy,
        declare_window_semantics,
    )

    # R26-088/089: the kernels use a TRAILING BAR window (``i0 = r - window + 1``,
    # the last ``window`` ROWS), not the last ``window`` EVENTS.  The declared
    # semantics must match the implementation clock exactly — an event-count
    # declaration would mis-drive planner prefetch / warmup / cache identity /
    # miner parameter meaning.  MIN_SUPPORT_WINDOW: a partial bar window is
    # legitimate when the interval support floor (>=6 intervals) is met; an
    # interval crossing an unknown (NaN) mark is censored (BREAK), never
    # silently spanned.
    for _canon in ("event_interval_memory", "event_local_variation"):
        declare_missing_policy(_canon, MissingPolicy.BREAK)
        declare_window_semantics(_canon, WindowSemantics.MIN_SUPPORT_WINDOW)
    for _canon in ("event_fano_factor", "event_fano_excess"):
        declare_missing_policy(_canon, MissingPolicy.BREAK)
        declare_window_semantics(_canon, WindowSemantics.CONTIGUOUS_FULL_WINDOW)


_declare_closure_contracts()
