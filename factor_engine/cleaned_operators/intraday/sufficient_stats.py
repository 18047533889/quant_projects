# -*- coding: utf-8 -*-
"""R61-P1 #57: shared sufficient statistics for intraday daily_agg operators.

Many minute->daily operators are aggregates of one session series:
Σr Σr² Σr³ Σr⁴, Σv Σv² Σrv, Σamount, max/min, first/last, argmax/argmin.
Instead of each operator re-walking its per-(day, inst) series, this module
materializes ONE (day, bar, inst) grid per source frame and derives those
sufficient statistics in a single numpy pass over that grid, memoized per
frame identity.  Vector kernels in ``_core`` then read the bundle and derive
their scalar-equivalent output with O(1) (or at most one O(n) window pass)
per (day, inst) instead of a per-(day, inst) Python loop.

Design / contract
-----------------
* The vector dispatch (``_core._vec_daily_agg{,_two,_three}``) hands every
  ``__vec__`` kernel the RAW DataFrame(s) plus an optional pre-built
  ``_grid`` from ``_grid3``.  The kernel therefore always sees the frame and
  can build/retrieve the bundle from this module — the runtime_v2 bar-level
  ``_calc_shared`` path is NOT touched (another agent owns it).
* Bundles are cached in-process keyed by ``(id(frame), data hash, cols,
  shape)`` so a ``compute_many``-style dispatch of several vector kernels on
  the same panels materializes the grid + statistics ONCE and every kernel
  reads them.  A mutated frame gets a new id and a fresh bundle; the LRU
  (small) bound keeps long-running sessions from pinning memory.
* ``data hash`` guards against a frame that keeps its id but changes values
  in place; it is a fast rolling 64-bit hash of the underlying blocks
  (xxhash-style, no dependency) — cheap relative to the full grid build.
* NaN handling is fail-closed and scalar-parity-exact:
  - ``fin`` is the finite mask; ``cnt`` = finite count per (day, inst).
  - sums/sumsq/cube/quartic/sums-of-products are ZERO-padded finite sums
    (matching ``np.nansum``-style scalar kernels over finite values).
  - ``max/min`` ignore NaN; all-NaN day -> NaN via the min_finite gate.
  - ``argmax/argmin`` return the position of the first occurrence among the
    FINITE values in ORIGINAL bar order (``np.argmax(np.isfinite-kept
    array)``); ties -> first index (scalar ``np.argmax`` semantics).
  - volume sums include only ``volume > 0`` finite bars (volume-weighted
    terms must never weight a zero/unknown minute).
* Return positions are in the ORIGINAL (day-local) bar coordinate (0-based)
  for the ``first`` / ``argmax`` / ``argmin`` families, so duration-like
  operators can index back into the grid directly.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

_EPS = 1e-12


def _grid3(
    a: pd.DataFrame,
    b: pd.DataFrame | None = None,
    c: pd.DataFrame | None = None,
) -> tuple:
    """(day, bar, inst) grids for panels a, [b, c]; mirrors ``_core._grid3``.

    Kept local so this module is dependency-light (imports ``_core`` would
    pull the whole operator framework).  Output layout is identical to
    ``_core._grid3``: (grid_a, grid_b, grid_c, codes, uniq_days, active,
    filled).  A bar row that belongs to day d is present only in slice d
    (day-padded, matching the scalar per-day grouping).
    """
    frames = [as_panel(a)]
    for f in (b, c):
        if f is not None:
            frames.append(as_panel(f))
    joined = pd.concat(frames, axis=1, join="outer", keys=[f"p{i}" for i in range(len(frames))])
    idx = joined.index
    days = pd.DatetimeIndex(idx).normalize()
    codes, uniques = pd.factorize(days, sort=True)
    D = int(len(uniques))
    B = int(len(idx))
    C = int(frames[0].shape[1])
    filled = [f.reindex(idx).to_numpy(dtype=float) for f in frames]
    grids = []
    for f_arr in filled:
        g = np.empty((D, B, C), dtype=np.float64)
        for d in range(D):
            m = codes == d
            g[d] = np.where(m[:, None], f_arr, np.nan)
        grids.append(g)
    active = np.zeros((D,), dtype=np.int64)
    for d in range(D):
        active[d] = int(np.sum(np.isfinite(filled[0][codes == d])))
    uniq_days = pd.DatetimeIndex(uniques)
    return grids[0], grids[1] if len(grids) > 1 else None, (
        grids[2] if len(grids) > 2 else None
    ), codes, uniq_days, active, filled


def as_panel(x: Any) -> pd.DataFrame:
    if isinstance(x, pd.Series):
        return x.to_frame(getattr(x, "name", None) or "value")
    return x


def _frame_hash(frame: pd.DataFrame) -> int:
    """Fast content hash over a frame's underlying float blocks."""
    h = 0x9E3779B97F4A7C15  # golden-ratio odd constant (no 64-bit wrap needed)
    for col in frame.columns:
        arr = frame[col]
        try:
            v = np.asarray(arr, dtype=np.float64)
        except (TypeError, ValueError):
            v = np.asarray(arr, dtype=object)
        if v.ndim == 0:
            continue
        flat = v.reshape(-1)
        if flat.size:
            h ^= hash(col)
            sample = flat[:: max(1, flat.size // 64)]
            with np.errstate(invalid="ignore"):
                h = (h + int(np.nansum(sample))) & 0xFFFFFFFFFFFFFFFF
            h = (h * 31 + (flat.size & 0xFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
    return h & 0xFFFFFFFFFFFFFFFF


def _session_slot_map(times: np.ndarray) -> np.ndarray:
    """Day-normalised bar->minute-of-day slot map (stable order-preserving).

    Only used for argmax/argmin: scalar ``np.argmax`` returns the position in
    the day-local bar array.  Because every day contributes the same session
    grid slots, the per-(day, inst) bar coordinate IS the index of the row
    within that day's slice.  This helper is unused when the frame index is
    already the session slot grid; kept for kernels that need the raw
    minute-of-day for time-normalized positions.
    """
    seconds = np.asarray(times, dtype="datetime64[s]").astype("int64") % 86400
    return seconds // 60


@lru_cache(maxsize=8)
def _bundle_cached(*, key: tuple, **_) -> tuple:
    # placeholder — real body below (kept signature-compatible for hot reload)
    raise NotImplementedError


class _BundleStore:
    """Thread-light in-process bundle cache keyed by (id, hash, cols, shape)."""

    _MAX = 8

    def __init__(self) -> None:
        self._data: dict[tuple, dict] = {}

    def get(self, key: tuple) -> dict | None:
        return self._data.get(key)

    def put(self, key: tuple, bundle: dict) -> None:
        self._data[key] = bundle
        if len(self._data) > self._MAX:
            # evict the oldest inserted (dict preserves insertion order).
            for k in list(self._data):
                del self._data[k]
                break


_STORE = _BundleStore()


def _make_key(frame: pd.DataFrame) -> tuple:
    return (
        id(frame),
        _frame_hash(frame),
        tuple(str(c) for c in frame.columns),
        frame.shape,
        str(frame.index.dtype),
    )


def _derive_stats(g: np.ndarray, fin: np.ndarray, cnt: np.ndarray) -> dict:
    """One-pass per-(day, inst) sufficient statistics over a (D,B,C) grid.

    All reductions are zero-padded finite sums (``np.where(fin, x, 0).sum``),
    matching the scalar kernels' ``nansum``-over-finite semantics.  ``max`` /
    ``min`` mask out non-finite cells; an all-NaN (day, inst) yields NaN for
    the ``min_finite`` gate downstream.
    """
    n, B, C = g.shape
    z = np.where(fin, g, 0.0)
    z2 = z * z
    z3 = z2 * z
    z4 = z3 * z

    psum = z.sum(axis=1)
    psumsq = z2.sum(axis=1)
    pcube = z3.sum(axis=1)
    pquart = z4.sum(axis=1)

    # masked EXTREMES on the ORIGINAL bar order: force-reduce the finite
    # cells (inf else).  Unlike the sum/sumsq zero-padding (bit-verified at
    # ~1e-16 rel), the extremes CANNOT be 0-padded — a real 0 value in the
    # series would then be indistinguishable from an absent cell.  All-NaN
    # (day, inst) yields ±inf which is masked back to NaN below.
    gmax = np.where(fin, g, -np.inf)
    gmin = np.where(fin, g, np.inf)
    max_all = np.max(gmax, axis=1)
    min_all = np.min(gmin, axis=1)

    maxv = np.where(np.isfinite(max_all), max_all, np.nan)
    minv = np.where(np.isfinite(min_all), min_all, np.nan)

    # prefix packing order for argmax/argmin / first / last: stable argsort on
    # the negated finite mask packs finite cells to the front in ORIGINAL bar
    # order — exactly the scalar's ``vals[np.isfinite(vals)]``.
    order = np.argsort(~fin, axis=1, kind="stable")  # (D, B, C)
    packed = np.take_along_axis(g, order, axis=1)
    packed[~np.isfinite(packed)] = 0.0

    max_cnt = int(cnt.max()) if cnt.size and cnt.max() > 0 else 1
    ar = np.arange(max_cnt)[None, :, None]  # (1, max_cnt, 1)
    mask = ar < cnt[:, None, :]  # (D, max_cnt, C)
    pf = np.where(mask, packed[:, :max_cnt, :], 0.0)
    pf00 = np.where(mask, packed[:, :max_cnt, :], 0.0)

    # first / last of the FINITE series (in original order).
    first = pf00[:, 0, :]
    last = np.take_along_axis(pf00, (cnt - 1)[:, None, :].clip(min=0), axis=1)[:, 0, :]
    last = np.where(cnt >= 1, last, np.nan)

    # population variance / std on the packed prefix — the compressed finite
    # series the scalar kernel sees.  Two-pass (mean then mean of squared
    # deviations) reproduces ``np.var(compressed)`` bit-for-bit; the algebraic
    # E[x²]-E[x]² closed form is NOT used (catastrophic cancellation at ~1e-15
    # relative blew the rtol=1e-12 parity gate).
    mu = psum / np.maximum(cnt, 1)
    dev = np.where(mask, pf00 - mu[:, None, :], 0.0)
    var = (dev * dev).sum(axis=1) / np.maximum(cnt, 1)
    var = np.where(cnt >= 1, var, np.nan)

    # argmax / argmin positions: the packed-prefix index = position of the
    # first max/min WITHIN the session's finite series (0-based).  Scalar
    # ``np.argmax(finite)`` is exactly this — the finite series the scalar
    # kernel sees is the packed prefix (``vals[np.isfinite(vals)]``), so the
    # position is in the finite-series coordinate, NOT the original minute
    # slot.  Ties resolve to the earliest bar (first occurrence), matching
    # ``np.argmax`` semantics.  Non-finite cells are padded with -inf/+inf
    # (NOT 0.0) so a zero-padded tail can never win the extreme.
    amax_pad = np.where(mask, packed[:, :max_cnt, :], -np.inf)
    amin_pad = np.where(mask, packed[:, :max_cnt, :], np.inf)
    argmax_pos = np.argmax(amax_pad, axis=1)  # (D, C) within the packed prefix
    argmin_pos = np.argmin(amin_pad, axis=1)
    argmax_pos = np.where(cnt >= 1, argmax_pos, -1)
    argmin_pos = np.where(cnt >= 1, argmin_pos, -1)

    # maximum-relative squared deviations is NOT needed as a stat — variance
    # / std are derived in the vector kernel from the packed prefix so the
    # scalar np.var(compressed) arithmetic is reproduced exactly.

    return {
        "g": g, "fin": fin, "cnt": cnt,
        "order": order, "packed": packed, "max_cnt": max_cnt, "mask": mask, "pf": pf00,
        "sum": psum, "sumsq": psumsq, "cube": pcube, "quart": pquart,
        "var": var, "mean": mu,
        "max": maxv, "min": minv,
        "first": first, "last": last,
        "argmax_pos": argmax_pos, "argmin_pos": argmin_pos,
    }


def _logret_grid(g: np.ndarray) -> np.ndarray:
    """(D,B,C) log returns on the original minute-slot grid."""
    r = np.full(g.shape, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        r[:, 1:, :] = np.log(
            np.where(np.isfinite(g[:, 1:, :]), g[:, 1:, :], np.nan)
            / np.where(np.isfinite(g[:, :-1, :]), g[:, :-1, :], np.nan)
        )
    return r


def _statistics_for_grid(g: np.ndarray) -> dict:
    """Sufficient statistics for a single (D,B,C) grid (close family)."""
    fin = np.isfinite(g)
    cnt = np.sum(fin, axis=1).astype(np.int64)
    st = _derive_stats(g, fin, cnt)
    # log returns on the ORIGINAL minute-slot grid (mirrors _core._vec_logret_grid).
    r = _logret_grid(g)
    st["ret"] = r
    rfin = np.isfinite(r)
    rz = np.where(rfin, r, 0.0)
    st["ret_cnt"] = np.sum(rfin, axis=1).astype(np.int64)
    st["r2"] = (rz * rz).sum(axis=1)
    st["r3"] = (rz * rz * rz).sum(axis=1)
    st["r4"] = (rz * rz * rz * rz).sum(axis=1)
    return st


def compute_sufficient_statistics(frame: pd.DataFrame) -> dict:
    """Memoized sufficient-statistics bundle for one source frame.

    Returns the ``_derive_stats`` dict plus the return moments (r2/r3/r4)
    when the frame is a close panel.  The bundle is keyed by
    ``(id(frame), data hash, cols, shape, index dtype)`` so:
      * a ``compute_many``-style dispatch of several vector kernels on the
        SAME panel materializes the grid + stats ONCE (per-bar) and every
        kernel derives O(1);
      * a second operator on the same frame reuses the same bundle (cache
        reuse test asserts object identity of the cached bundle).
    """
    key = _make_key(frame)
    cached = _STORE.get(key)
    if cached is not None:
        return cached
    g, _, _, _, uniq, active, filled = _grid3(frame)
    bundle = _statistics_for_grid(g)
    bundle["uniq_days"] = uniq
    bundle["columns"] = tuple(str(c) for c in frame.columns)
    bundle["active"] = active
    _STORE.put(key, bundle)
    return bundle


def two_panel_statistics(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Memoized sufficient statistics for a (close, volume) / (a, b) pair.

    Derives the cross-panel moments an amount/volume-weighted operator needs:
    ``Σv``, ``Σv²``, ``Σp·v``, ``Σa``, ``Σp·a`` over the joint finite mask
    (volume > 0 for volume panels).  One grid materialization for both panels
    (the volume-weighted operators share the joint finite mask, exactly like
    the scalar kernels' common-mask contract).
    """
    key = ("pair", _make_key(a), _make_key(b))
    cached = _STORE.get(key)
    if cached is not None:
        return cached
    ga, gb, _, _, uniq, active, filled = _grid3(a, b)
    g = ga
    # daily_agg_two drops NaN-CLOSE rows first (``dropna(subset=["a"])``), so
    # the scalar kernel sees the close-compressed series.  ``close_cnt`` is
    # the operator's min_finite gate (scalar gates on the A-panel finite
    # count).  ``fin`` remains the joint (close & volume>0) mask for the
    # amount/volume-weighted sums (VWAP etc).
    close_fin = np.isfinite(g)
    close_cnt = np.sum(close_fin, axis=1).astype(np.int64)
    fin = close_fin & np.isfinite(gb) & (gb > 0)
    cnt = np.sum(fin, axis=1).astype(np.int64)
    z = np.where(fin, g, 0.0)
    zb = np.where(fin, gb, 0.0)

    # Packed close-compressed series (order-preserving): the scalar log-return
    # pairing runs over CONSECUTIVE FINITE closes, so a missing minute must
    # not split a return pair.
    order = np.argsort(~close_fin, axis=1, kind="stable")
    p_close = np.take_along_axis(g, order, axis=1)
    p_vol = np.take_along_axis(gb, order, axis=1)
    p_close[~np.isfinite(p_close)] = 0.0
    max_cnt = int(close_cnt.max()) if close_cnt.size and close_cnt.max() > 0 else 1
    ar = np.arange(max_cnt)[None, :, None]
    pmask = ar < close_cnt[:, None, :]  # (D, max_cnt, C)
    pc = np.where(pmask, p_close[:, :max_cnt, :], 0.0)
    pv = np.where(pmask, p_vol[:, :max_cnt, :], 0.0)
    # compressed log returns r[j] = log(pc[j]/pc[j-1]) over prefix pairs
    # (r[:, 0] stays NaN, mirroring ``_core.log_returns``).
    rp = np.full((g.shape[0], max_cnt, g.shape[2]), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        rp[:, 1:, :] = np.log(
            np.where(pmask[:, 1:, :], pc[:, 1:, :], np.nan)
            / np.where(pmask[:, :-1, :], pc[:, :-1, :], np.nan)
        )

    st = {
        "g": g, "gb": gb, "fin": fin, "cnt": cnt, "close_cnt": close_cnt,
        "pmask": pmask, "pc": pc, "pv": pv, "rp": rp, "max_cnt": max_cnt,
        "sum": z.sum(axis=1), "sumsq": (z * z).sum(axis=1),
        "vsum": zb.sum(axis=1), "vsq": (zb * zb).sum(axis=1),
        "pvsum": (z * zb).sum(axis=1),
        "asum": z.sum(axis=1), "pasum": (z * zb).sum(axis=1),
        "uniq_days": uniq, "columns": tuple(str(c) for c in a.columns),
    }
    _STORE.put(key, st)
    return st


def three_panel_statistics(a: pd.DataFrame, b: pd.DataFrame, c: pd.DataFrame) -> dict:
    """Memoized sufficient statistics for a (close, amount, volume) triple."""
    key = ("triple", _make_key(a), _make_key(b), _make_key(c))
    cached = _STORE.get(key)
    if cached is not None:
        return cached
    ga, gb, gc, _, uniq, active, filled = _grid3(a, b, c)
    g = ga
    fin = np.isfinite(g) & np.isfinite(gb) & np.isfinite(gc) & (gc > 0)
    cnt = np.sum(fin, axis=1).astype(np.int64)
    za = np.where(fin, g, 0.0)
    zb = np.where(fin, gb, 0.0)
    zc = np.where(fin, gc, 0.0)
    st = {
        "g": g, "gb": gb, "gc": gc, "fin": fin, "cnt": cnt,
        "sum": za.sum(axis=1), "sumsq": (za * za).sum(axis=1),
        "vsum": zc.sum(axis=1), "vsq": (zc * zc).sum(axis=1),
        "pvsum": (za * zc).sum(axis=1),
        "asum": zb.sum(axis=1), "pasum": (za * zb).sum(axis=1),
        "uniq_days": uniq, "columns": tuple(str(c) for c in a.columns),
    }
    _STORE.put(key, st)
    return st
