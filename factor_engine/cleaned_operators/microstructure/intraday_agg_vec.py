# -*- coding: utf-8 -*-
"""Batched official-grid vectorization for ``microstructure.intraday_agg``.

R63: the R26 SessionPanel path builds one :class:`SessionPanel` per
(instrument, day) in Python (~1.3 ms/day, dominated by
``MinuteGrid.from_calendar`` rebuilding the 240-slot grid plus a per-day
``pd.to_datetime``).  On a 300-day single-column panel that is ~385 ms per
operator call — the per-(instrument, day) loop class R63 eliminates.

This module materializes the aligned official-grid values ONCE as a
``(D, S, C)`` matrix with present/duplicate masks replicating
``build_session_panel`` slot-for-slot, then derives every kernel from
vector reductions along the slot axis.

Semantics contract (bit-for-bit vs the scalar loop):

* ``aligned[d, s, c]`` keeps the FIRST observed value at official slot ``s``;
  later bars at the same slot mark the slot duplicate (the value stays the
  first, and the slot is DQ-invalid for coverage/validity);
* ``is_valid_bar = present & ~duplicate`` — coverage counts these whether or
  not the value is finite;
* log-returns bridge NOTHING: slot-adjacent pairs with BOTH prices finite
  and > 0 (``SessionPanel.log_returns`` contract);
* a day whose column has no finite observed value at all short-circuits to
  NaN in the scalar loop (``_daily_agg`` early exit) — replicated by the
  raw-observation ``has_finite`` gate, not by the aligned matrix;
* coverage gate = valid slots / n_slots >= ``_COVERAGE_FLOOR``;
* ``mode="production"`` never takes this path (SessionPanel DQ hard-fail
  semantics stay on the scalar loop);
* reductions keep the scalar op order per element; sums over compressed
  finite slices become masked sums over the slot axis, so only float
  summation-order rounding (<= ~1e-15 relative) may differ.  Count-based
  outputs (limit duration/reopen, argmax positions, max/min) are exact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from factor_engine.runtime.session_panel import MinuteGrid

_CHUNK = 256  # columns per batch chunk (bounds temp memory)


class _Batch:
    """Aligned official-grid batch for one column chunk."""

    __slots__ = (
        "M", "valid", "finite", "EM", "S", "D", "C",
        "uniq_days", "columns", "has_finite", "cov_ok", "_R",
    )

    def __init__(self, M, present, dup, EM, uniq_days, columns, has_finite,
                 coverage_floor):
        self.M = M
        self.EM = EM
        self.S = int(EM.size)
        self.D, _, self.C = M.shape
        self.uniq_days = uniq_days
        self.columns = columns
        self.valid = present & ~dup
        self.finite = np.isfinite(M)
        self.has_finite = has_finite                      # (D, C) raw-observation gate
        cov = self.valid.sum(axis=1).astype(float) / float(self.S)
        self.cov_ok = cov >= coverage_floor               # (D, C)
        self._R = None

    @property
    def log_returns(self):
        if self._R is None:
            M = self.M
            R = np.full(M.shape, np.nan, dtype=float)
            a = M[:, 1:, :]
            b = M[:, :-1, :]
            ok = np.isfinite(a) & np.isfinite(b) & (a > 0.0) & (b > 0.0)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                ratio = a / b
                R[:, 1:, :] = np.where(ok, np.log(ratio), np.nan)
            self._R = R
        return self._R


def _build_batch(frame, cal, coverage_floor):
    """Build the (D, S, C) aligned batch; ``None`` -> caller must fall back."""
    idx = frame.index
    if not isinstance(idx, pd.DatetimeIndex) or idx.tz is not None:
        return None
    if frame.shape[1] == 0 or not frame.columns.is_unique:
        return None
    # expected_minutes is independent of trade_date; one anchor call also
    # validates the calendar exactly like the scalar path does.
    grid = MinuteGrid.from_calendar(cal, pd.Timestamp("2000-01-01"))
    EM = np.asarray(grid.expected_minutes, dtype=np.int64)
    S = int(EM.size)
    if S == 0:
        return None
    from factor_engine.cleaned_operators.microstructure.intraday_agg import (
        _COVERAGE_FLOOR,
        _minute_of_day,
    )

    if coverage_floor is None:
        coverage_floor = _COVERAGE_FLOOR
    vals = frame.to_numpy(dtype=float)                    # (N, C)
    N, C = vals.shape
    day_codes, uniq = pd.factorize(idx.normalize(), sort=True)
    D = int(len(uniq))
    mods = _minute_of_day(idx.to_numpy(dtype="datetime64[ns]"))
    slot = np.searchsorted(EM, mods)
    slot_c = np.minimum(slot, S - 1)
    on_grid = (slot < S) & (EM[slot_c] == mods)
    base = (day_codes.astype(np.int64) * S + slot_c) * C
    keys = (base[:, None] + np.arange(C, dtype=np.int64)[None, :]).ravel()
    v = vals.ravel()
    og = np.repeat(on_grid, C)
    keys = keys[og]
    v = v[og]
    M = np.full(D * S * C, np.nan, dtype=float)
    if keys.size:
        order = np.argsort(keys, kind="stable")
        ks = keys[order]
        first = np.empty(ks.size, dtype=bool)
        first[0] = True
        if ks.size > 1:
            np.not_equal(ks[1:], ks[:-1], out=first[1:])
        M[ks[first]] = v[order][first]
    if keys.size:
        counts = np.bincount(keys, minlength=D * S * C)
    else:
        counts = np.zeros(D * S * C, dtype=np.int64)
    counts = counts.reshape(D, S, C)
    present = counts >= 1
    dup = counts > 1
    M = M.reshape(D, S, C)
    # raw-observation has_finite per (day, inst) — replicates the scalar
    # `_daily_agg` early exit which runs on the RAW column, before alignment.
    fin_raw = np.isfinite(vals)
    if fin_raw.any():
        raw_keys = (day_codes.astype(np.int64)[:, None] * C
                    + np.arange(C, dtype=np.int64)[None, :]).ravel()
        has_finite = np.bincount(
            raw_keys[fin_raw.ravel()], minlength=D * C,
        ).reshape(D, C) > 0
    else:
        has_finite = np.zeros((D, C), dtype=bool)
    uniq_days = pd.DatetimeIndex(np.asarray(uniq, dtype="datetime64[ns]"))
    return _Batch(M, present, dup, EM, uniq_days, frame.columns, has_finite,
                  coverage_floor)


def batch_daily_agg(frame, cal, tz, vec, *, coverage_floor=None):
    """Vector fast path for ``_daily_agg``; ``None`` -> scalar fallback.

    ``frame`` must already be session-local (the caller applies
    ``_session_local_frame`` exactly as the scalar loop does).
    """
    del tz  # the frame is already session-local; kept for signature symmetry
    if not isinstance(frame.index, pd.DatetimeIndex):
        return None
    if frame.shape[1] == 0:
        return pd.DataFrame(dtype=float)
    outs = []
    B = None
    for c0 in range(0, frame.shape[1], _CHUNK):
        sub = frame.iloc[:, c0:c0 + _CHUNK]
        B = _build_batch(sub, cal, coverage_floor)
        if B is None:
            return None
        r = np.asarray(vec(B), dtype=float)
        if r.shape != (B.D, B.C):
            raise ValueError(
                f"panel-vec kernel returned {r.shape}, expected {(B.D, B.C)}"
            )
        outs.append(r)
    res = np.concatenate(outs, axis=1)
    return pd.DataFrame(res, index=B.uniq_days, columns=frame.columns,
                        dtype=float).sort_index()


def _align_day_axis(B, days):
    """Scatter a batch onto ``days`` (a DatetimeIndex); missing days -> empty.

    Pair/triple aggregation iterates the day axis of the FIRST frame (the
    scalar loop only ever emits rows for days observed in frame_a), so the
    secondary batches are reindexed onto that day axis: days absent from the
    secondary frame get all-NaN values, no valid slots and ``has_finite=False``
    — replicating the scalar ``gb[gb.index.normalize() == day]`` empty-group
    early exit.
    """
    days = pd.DatetimeIndex(days)
    if B.uniq_days.equals(days):
        return B
    pos = B.uniq_days.get_indexer(days)
    hit = pos >= 0
    Dn, S, C = len(days), B.S, B.C
    M = np.full((Dn, S, C), np.nan)
    present = np.zeros((Dn, S, C), dtype=bool)
    M[hit] = B.M[pos[hit]]
    present[hit] = B.valid[pos[hit]]
    has_finite = np.zeros((Dn, C), dtype=bool)
    has_finite[hit] = B.has_finite[pos[hit]]
    from factor_engine.cleaned_operators.microstructure.intraday_agg import (
        _COVERAGE_FLOOR,
    )

    return _Batch(M, present, np.zeros_like(present), B.EM, days, B.columns,
                  has_finite, _COVERAGE_FLOOR)


def _batch_out(res, days, columns):
    return pd.DataFrame(np.asarray(res, dtype=float), index=days,
                        columns=columns, dtype=float).sort_index()


def batch_pair_agg(frame_a, frame_b, cal, tz, vec, *, coverage_floor=None):
    """Vector fast path for ``_pair_agg``; ``None`` -> scalar fallback.

    Both frames must already be session-local (the caller applies
    ``_session_local_frame`` exactly as the scalar loop does).  The output day
    axis is frame_a's (the scalar loop only iterates frame_a's days); frame_b
    is aligned onto it.
    """
    del tz  # frames already session-local; kept for signature symmetry
    if not isinstance(frame_a.index, pd.DatetimeIndex) \
            or not isinstance(frame_b.index, pd.DatetimeIndex):
        return None
    if frame_a.shape[1] == 0:
        return pd.DataFrame(dtype=float)
    if not frame_a.columns.equals(frame_b.columns):
        return None
    outs = []
    days = None
    for c0 in range(0, frame_a.shape[1], _CHUNK):
        Ba = _build_batch(frame_a.iloc[:, c0:c0 + _CHUNK], cal, coverage_floor)
        Bb = _build_batch(frame_b.iloc[:, c0:c0 + _CHUNK], cal, coverage_floor)
        if Ba is None or Bb is None:
            return None
        Bb = _align_day_axis(Bb, Ba.uniq_days)
        r = np.asarray(vec(Ba, Bb), dtype=float)
        if r.shape != (Ba.D, Ba.C):
            raise ValueError(
                f"pair-vec kernel returned {r.shape}, expected {(Ba.D, Ba.C)}"
            )
        outs.append(r)
        days = Ba.uniq_days
    return _batch_out(np.concatenate(outs, axis=1), days, frame_a.columns)


def batch_triple_agg(frame_a, frame_b, frame_c, cal, tz, vec, *,
                     coverage_floor=None):
    """Vector fast path for ``_triple_agg``; ``None`` -> scalar fallback."""
    del tz
    if not isinstance(frame_a.index, pd.DatetimeIndex) \
            or not isinstance(frame_b.index, pd.DatetimeIndex) \
            or not isinstance(frame_c.index, pd.DatetimeIndex):
        return None
    if frame_a.shape[1] == 0:
        return pd.DataFrame(dtype=float)
    if not (frame_a.columns.equals(frame_b.columns)
            and frame_a.columns.equals(frame_c.columns)):
        return None
    outs = []
    days = None
    for c0 in range(0, frame_a.shape[1], _CHUNK):
        Ba = _build_batch(frame_a.iloc[:, c0:c0 + _CHUNK], cal, coverage_floor)
        Bb = _build_batch(frame_b.iloc[:, c0:c0 + _CHUNK], cal, coverage_floor)
        Bc = _build_batch(frame_c.iloc[:, c0:c0 + _CHUNK], cal, coverage_floor)
        if Ba is None or Bb is None or Bc is None:
            return None
        Bb = _align_day_axis(Bb, Ba.uniq_days)
        Bc = _align_day_axis(Bc, Ba.uniq_days)
        r = np.asarray(vec(Ba, Bb, Bc), dtype=float)
        if r.shape != (Ba.D, Ba.C):
            raise ValueError(
                f"triple-vec kernel returned {r.shape}, expected {(Ba.D, Ba.C)}"
            )
        outs.append(r)
        days = Ba.uniq_days
    return _batch_out(np.concatenate(outs, axis=1), days, frame_a.columns)


# ---------------------------------------------------------------------------
# shared gates
# ---------------------------------------------------------------------------

def _gated(B, res, *, coverage=True):
    out = np.asarray(res, dtype=float)
    ok = B.has_finite
    if coverage:
        ok = ok & B.cov_ok
    return np.where(ok, out, np.nan)


def _seg_slot_mask(B, segment):
    from factor_engine.cleaned_operators.microstructure.intraday_agg import (
        _SEGMENT_RANGES,
    )

    lo, hi = _SEGMENT_RANGES[str(segment)]
    return (B.EM >= lo) & (B.EM <= hi)


# ---------------------------------------------------------------------------
# kernels — each mirrors one scalar kernel in intraday_agg.py
# ---------------------------------------------------------------------------

def vec_rv(B):
    """``_rv``: coverage gate + nansum(r^2)."""
    R = B.log_returns
    with np.errstate(invalid="ignore", over="ignore"):
        s = np.nansum(R * R, axis=1)
    return _gated(B, s)


def vec_semivariance(B, side):
    """``_semivariance``: sign-masked nansum(r^2)."""
    R = B.log_returns
    with np.errstate(invalid="ignore", over="ignore"):
        m = (R < 0.0) if side == "down" else (R > 0.0)
        r = np.where(m, R, 0.0)
        s = np.nansum(r * r, axis=1)
    return _gated(B, s)


def vec_bipower(B):
    """``_bipower``: (pi/2) * nansum(|r_t||r_{t-1}|) over slot-adjacent pairs."""
    R = B.log_returns
    with np.errstate(invalid="ignore", over="ignore"):
        s = np.nansum(np.abs(R[:, 1:, :]) * np.abs(R[:, :-1, :]), axis=1)
    return _gated(B, (np.pi / 2.0) * s)


def vec_jump_ratio(B):
    """``_jump_ratio``: max(rv - bv, 0) / rv with the rv/bv finite gates."""
    rv = vec_rv(B)
    bv = vec_bipower(B)
    ok = np.isfinite(rv) & np.isfinite(bv) & (rv > _eps())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok, np.maximum(rv - bv, 0.0) / np.where(ok, rv, 1.0),
                        np.nan)


def vec_path_efficiency(B):
    """``_path_efficiency``: |net| / arc length over the trailing contiguous
    complete run.  No coverage gate (the scalar kernel has none)."""
    f = B.valid & B.finite                                  # (D, S, C)
    S = B.S
    idx = np.arange(S, dtype=np.int64).reshape(1, S, 1)
    any_f = f.any(axis=1)                                   # (D, C)
    last = (S - 1) - np.argmax(f[:, ::-1, :], axis=1)       # (D, C) last valid slot
    brk = np.where(f, np.int64(-1), idx)
    last_break = np.maximum.accumulate(brk, axis=1)
    lb = np.take_along_axis(last_break, last[:, None, :], axis=1)[:, 0, :]
    lb = np.minimum(lb, last)                               # safety (f[last] true)
    rl = last - lb                                          # trailing run length
    M = B.M
    with np.errstate(invalid="ignore"):
        cd = np.abs(M[:, 1:, :] - M[:, :-1, :])             # (D, S-1, C)
    cdz = np.where(np.isfinite(cd), cd, 0.0)
    P = np.cumsum(cdz, axis=1)                              # P[j] = sum cdz[0..j]
    # clamp gather indices: rows with no valid&finite slot have last=S-1 and
    # lb=S-1 (garbage) — the ok gate masks them, but the gathers must stay
    # inside P's (S-1)-slot axis.
    lm1 = np.clip(last - 1, 0, S - 2)
    lb_safe = np.clip(lb, 0, S - 2)
    Pl = np.take_along_axis(P, lm1[:, None, :], axis=1)[:, 0, :]
    Pl0 = np.where(lb >= 0,
                   np.take_along_axis(P, lb_safe[:, None, :], axis=1)[:, 0, :],
                   0.0)
    length = Pl - Pl0                                       # sum over run diffs
    first_slot = np.minimum(lb + 1, last)                   # in [0, S-1]
    fv = np.take_along_axis(M, first_slot[:, None, :], axis=1)[:, 0, :]
    lv = np.take_along_axis(M, last[:, None, :], axis=1)[:, 0, :]
    net = np.abs(lv - fv)
    ok = any_f & (rl >= 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        res = np.where(length <= _eps(), 0.0,
                       net / np.where(length > 0.0, length, 1.0))
    return np.where(ok, res, np.nan)


def vec_position_of(B, *, low):
    """``_position_of``: official-slot ordinal of the extreme / n_slots."""
    sel = B.valid & B.finite
    any_sel = sel.any(axis=1)
    fill = np.inf if low else -np.inf
    x = np.where(sel, B.M, fill)
    t = np.argmin(x, axis=1) if low else np.argmax(x, axis=1)
    res = t.astype(float) / float(B.S)
    return np.where(any_sel, res, np.nan)


def _neg_total(B):
    """Shared concentration/entropy pre-gates: negative-value gate + total."""
    neg = ((B.M < 0.0) & B.valid).any(axis=1)
    fin = B.valid & B.finite
    total = np.sum(np.where(fin, B.M, 0.0), axis=1)         # (D, C)
    return fin, neg, total


def vec_concentration(B):
    """``_concentration``: sum((v/total)^2) over valid finite values."""
    fin, neg, total = _neg_total(B)
    ok = B.has_finite & ~neg & (total > _eps())
    w = np.where(fin, B.M, 0.0) / np.where(ok, total, 1.0)[:, None, :]
    res = np.sum(w * w, axis=1)
    return np.where(ok, res, np.nan)


def vec_entropy(B, normalize):
    """``_entropy``: -sum(w log w) over positive weights, optional /log(n)."""
    fin, neg, total = _neg_total(B)
    ok = B.has_finite & ~neg & (total > _eps())
    w = np.where(fin, B.M, 0.0) / np.where(ok, total, 1.0)[:, None, :]
    pos = w > 0.0
    wl = np.where(pos, w, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.sum(wl * np.log(wl), axis=1)
    n = pos.sum(axis=1)
    if normalize:
        res = np.where(n == 1, 0.0, H / np.log(np.maximum(n, 2).astype(float)))
    else:
        res = H
    return np.where(ok, res, np.nan)


def vec_extreme_bar_return(B, side):
    """``_extreme_bar_return``: max/min finite grid log-return."""
    R = B.log_returns
    cnt = np.sum(np.isfinite(R), axis=1)
    fill = -np.inf if side == "max" else np.inf
    x = np.where(np.isfinite(R), R, fill)
    res = np.max(x, axis=1) if side == "max" else np.min(x, axis=1)
    return np.where(cnt >= 1, res, np.nan)


def vec_seg_return(B, segment, endpoint_policy):
    """``_seg_return``: exact official endpoints (default) or recent-valid."""
    if endpoint_policy == "exact":
        from factor_engine.cleaned_operators.microstructure.intraday_agg import (
            _SEGMENT_END_MOD,
            _SEGMENT_START_MOD,
        )

        def _at(minute):
            s = int(np.searchsorted(B.EM, int(minute)))
            if s >= B.S or int(B.EM[s]) != int(minute):
                return np.full((B.D, B.C), np.nan)
            v = B.M[:, s, :]
            return np.where(B.valid[:, s, :], v, np.nan)

        start = _at(_SEGMENT_START_MOD[str(segment)])
        end = _at(_SEGMENT_END_MOD[str(segment)])
        ok = (B.has_finite & np.isfinite(start) & np.isfinite(end)
              & (start > _eps()))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(ok, end / start - 1.0, np.nan)
    # explicit recent_valid policy
    segm = _seg_slot_mask(B, segment)
    seg_idx = np.flatnonzero(segm)
    if seg_idx.size == 0:
        return np.full((B.D, B.C), np.nan)
    i0, i1 = int(seg_idx[0]), int(seg_idx[-1])
    sel = (B.valid & B.finite & segm[None, :, None])[:, i0:i1 + 1, :]
    cnt = sel.sum(axis=1)
    first_local = np.argmax(sel, axis=1)
    last_local = (i1 - i0) - np.argmax(sel[:, ::-1, :], axis=1)
    sub = B.M[:, i0:i1 + 1, :]
    fv = np.take_along_axis(sub, first_local[:, None, :], axis=1)[:, 0, :]
    lv = np.take_along_axis(sub, last_local[:, None, :], axis=1)[:, 0, :]
    ok = (B.has_finite & (cnt >= 2) & np.isfinite(fv) & (fv > _eps())
          & np.isfinite(lv))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok, lv / fv - 1.0, np.nan)


def vec_seg_volume_share(B, segment):
    """``_seg_volume_share``: segment total / full-day total (coverage gated)."""
    segm = _seg_slot_mask(B, segment)
    finv = B.valid & B.finite
    total = np.sum(np.where(finv, B.M, 0.0), axis=1)
    seg_total = np.sum(np.where(finv & segm[None, :, None], B.M, 0.0), axis=1)
    ok = B.has_finite & B.cov_ok & (total > _eps())
    return np.where(ok, seg_total / np.where(ok, total, 1.0), np.nan)


def vec_seg_realized_vol(B, segment):
    """``_seg_realized_vol``: sqrt(nansum(segment r^2)) (coverage gated)."""
    R = B.log_returns
    r = np.where(_seg_slot_mask(B, segment)[None, :, None], R, np.nan)
    with np.errstate(invalid="ignore", over="ignore"):
        s = np.nansum(r * r, axis=1)
    return _gated(B, np.sqrt(np.maximum(s, 0.0)))


# ---------------------------------------------------------------------------
# pair/triple kernels — each mirrors one scalar kernel in intraday_agg.py
# ---------------------------------------------------------------------------

def _slot_at(EM, minute):
    """Official slot ordinal of an exact minute-of-day; -1 when off-grid."""
    s = int(np.searchsorted(EM, int(minute)))
    return s if s < EM.size and int(EM[s]) == int(minute) else -1


def vec_seg_vwap_deviation(Bc, Ba, Bv, segment):
    """``_seg_vwap_deviation``: segment-end close vs cumulative segment VWAP."""
    from factor_engine.cleaned_operators.microstructure.intraday_agg import (
        _SEGMENT_END_MOD,
    )

    es = _slot_at(Bc.EM, _SEGMENT_END_MOD[str(segment)])
    if es < 0:
        return np.full((Bc.D, Bc.C), np.nan)
    end = np.where(Bc.valid[:, es, :], Bc.M[:, es, :], np.nan)
    segm = _seg_slot_mask(Bc, segment)
    valid = Bc.valid & Ba.valid & Bv.valid & segm[None, :, None]
    a = np.nansum(np.where(valid, Ba.M, np.nan), axis=1)
    v = np.nansum(np.where(valid, Bv.M, np.nan), axis=1)
    ok = Bc.cov_ok & np.isfinite(end) & (v > _eps())
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ok, end / (a / np.where(ok, v, 1.0)) - 1.0, np.nan)


def vec_vwap_above_ratio(Bc, Ba, Bv):
    """``_vwap_above_ratio``: fraction of usable bars closing above day VWAP."""
    valid = Bc.valid & Ba.valid & Bv.valid & (Bv.M > 0.0)
    a = np.where(valid, Ba.M, 0.0)
    v = np.where(valid, Bv.M, 0.0)
    tv = v.sum(axis=1)                                   # NaN propagates
    bad_tv = tv <= _eps()                                # False for NaN tv
    with np.errstate(invalid="ignore", divide="ignore"):
        dv = a.sum(axis=1) / np.where(bad_tv, 1.0, tv)
    day_vwap = dv[:, None, :]
    ok = valid & np.isfinite(Bc.M)
    cnt = ok.sum(axis=1)
    above = ok & (Bc.M > day_vwap)
    mean = np.where(cnt > 0, above.sum(axis=1) / np.maximum(cnt, 1), np.nan)
    res = np.where(bad_tv | (cnt == 0), np.nan, mean)
    return np.where(Bc.has_finite & Ba.has_finite & Bv.has_finite, res, np.nan)


def vec_vwap_cross_count(Bc, Ba, Bv):
    """``_vwap_cross_count``: sign changes of close vs cumulative VWAP.

    The scalar kernel compresses to usable slots and diffs consecutive signs;
    the vector kernel carries the last-usable sign forward across non-usable
    slots so every consecutive-usable pair is compared exactly once.
    """
    valid = Bc.valid & Ba.valid & Bv.valid
    vol = np.where(valid, Bv.M, np.nan)
    amt = np.where(valid, Ba.M, np.nan)
    cum_v = np.nancumsum(vol, axis=1)
    cum_a = np.nancumsum(amt, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cum_vwap = np.where(cum_v > _eps(), cum_a / cum_v, np.nan)
    ok = valid & np.isfinite(Bc.M) & np.isfinite(cum_vwap)
    cnt = ok.sum(axis=1)
    sign = np.sign(np.where(ok, Bc.M, np.nan) - cum_vwap)
    idx = np.arange(Bc.S, dtype=np.int64).reshape(1, Bc.S, 1)
    acc = np.maximum.accumulate(np.where(ok, idx, np.int64(-1)), axis=1)
    prev = np.full_like(acc, -1)
    prev[:, 1:, :] = acc[:, :-1, :]
    has_prev = prev >= 0
    sign_prev = np.take_along_axis(
        sign, np.clip(prev, 0, Bc.S - 1), axis=1)
    d = ok & has_prev & (sign != sign_prev)
    chg = d.sum(axis=1)
    res = np.where(cnt >= 2, chg.astype(float), np.nan)
    return np.where(Bc.has_finite & Ba.has_finite & Bv.has_finite, res, np.nan)


def vec_signed_imbalance_proxy(Bc, Bv):
    """``_signed_imbalance_proxy``: sum(sign(r)*value)/sum(value)."""
    R = Bc.log_returns
    valid = Bc.valid & Bv.valid
    value = np.where(valid, Bv.M, 0.0)
    total = value.sum(axis=1)
    signed = np.where(np.isfinite(R) & valid, np.sign(R), 0.0) * value
    num = signed.sum(axis=1)
    ok = total > _eps()
    with np.errstate(invalid="ignore"):
        res = np.where(ok, num / np.where(ok, total, 1.0), np.nan)
    return np.where(Bc.has_finite & Bv.has_finite, res, np.nan)


def vec_return_activity_corr(Bc, Ba, absolute_return):
    """``_return_activity_corr``: Pearson corr of (|r|) vs activity.

    Sufficient-statistic form of ``np.corrcoef`` (ddof cancels); rounding may
    differ by ~1e-15 from the two-pass scalar computation.
    """
    R = Bc.log_returns
    if absolute_return:
        R = np.abs(R)
    valid = Bc.valid & Ba.valid & np.isfinite(Ba.M) & np.isfinite(R)
    n = valid.sum(axis=1)
    nf = np.maximum(n, 1).astype(float)
    rf = np.where(valid, R, 0.0)
    af = np.where(valid, Ba.M, 0.0)
    sr = rf.sum(axis=1)
    sa = af.sum(axis=1)
    srr = (rf * rf).sum(axis=1)
    saa = (af * af).sum(axis=1)
    sra = (rf * af).sum(axis=1)
    var_r = srr / nf - (sr / nf) ** 2
    var_a = saa / nf - (sa / nf) ** 2
    ok = (n >= 2) & (np.sqrt(np.maximum(var_r, 0.0)) > _eps()) \
        & (np.sqrt(np.maximum(var_a, 0.0)) > _eps())
    num = nf * sra - sr * sa
    den = np.sqrt(np.maximum(nf * srr - sr * sr, 0.0)
                  * np.maximum(nf * saa - sa * sa, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = num / np.where(den > 0.0, den, 1.0)
    res = np.where(ok & (den > 0.0), corr, np.nan)
    return np.where(Bc.has_finite & Ba.has_finite, res, np.nan)


def vec_amihud(Bc, Ba, scale):
    """``_intra_amihud``: mean(|r| / max(amount, eps)) * scale."""
    R = np.abs(Bc.log_returns)
    valid = Bc.valid & Ba.valid
    amount = np.where(valid, Ba.M, np.nan)
    denom = np.maximum(amount, _eps())
    with np.errstate(invalid="ignore"):
        ratio = np.where(np.isfinite(R) & valid, R / denom, np.nan)
    cnt = np.isfinite(ratio).sum(axis=1)
    mean = np.nansum(ratio, axis=1) / np.maximum(cnt, 1)
    res = np.where(cnt > 0, mean * float(scale), np.nan)
    return np.where(Bc.has_finite & Ba.has_finite, res, np.nan)


def vec_kyle_lambda_proxy(Bc, Ba):
    """``_kyle_lambda_proxy``: cov(r, sign(r)*share, ddof=1) / var(share, ddof=0).

    Sufficient-statistic form; rounding may differ ~1e-15 from ``np.cov``.
    """
    R = Bc.log_returns
    valid = Bc.valid & Ba.valid
    amount = np.where(valid, Ba.M, 0.0)
    total = amount.sum(axis=1)
    ok_total = total > _eps()
    with np.errstate(invalid="ignore"):
        share = np.sign(R) * amount / np.where(ok_total, total, 1.0)[:, None, :]
    mask = valid & np.isfinite(R) & np.isfinite(share)
    n = mask.sum(axis=1)
    nf = np.maximum(n, 1).astype(float)
    rm = np.where(mask, R, 0.0)
    sm = np.where(mask, share, 0.0)
    sr = rm.sum(axis=1)
    ss = sm.sum(axis=1)
    srs = (rm * sm).sum(axis=1)
    ss2 = (sm * sm).sum(axis=1)
    var_s = ss2 / nf - (ss / nf) ** 2
    ok = (n >= 3) & ok_total & (np.sqrt(np.maximum(var_s, 0.0)) > _eps())
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = ((srs - sr * ss / nf) / np.maximum(nf - 1.0, 1.0)) / np.where(
            var_s > 0.0, var_s, 1.0)
    res = np.where(ok & (var_s > 0.0), beta, np.nan)
    return np.where(Bc.has_finite & Ba.has_finite, res, np.nan)


def vec_lunch_gap_return(Bc, Bo, morning_cutoff, afternoon_start,
                         endpoint_policy):
    """``_lunch_gap_return``: afternoon first Open / morning last Close - 1."""
    from factor_engine.cleaned_operators.microstructure.intraday_agg import (
        _SEGMENT_END_MOD,
        _SEGMENT_START_MOD,
        _minute_hm,
    )

    gate = Bc.has_finite & Bo.has_finite
    if endpoint_policy == "exact":
        es = _slot_at(Bc.EM, _SEGMENT_END_MOD["morning"])
        as_ = _slot_at(Bo.EM, _SEGMENT_START_MOD["afternoon"])
        if es < 0 or as_ < 0:
            return np.full((Bc.D, Bc.C), np.nan)
        mc = np.where(Bc.valid[:, es, :], Bc.M[:, es, :], np.nan)
        ao = np.where(Bo.valid[:, as_, :], Bo.M[:, as_, :], np.nan)
        ok = np.isfinite(mc) & np.isfinite(ao) & (mc > _eps())
        with np.errstate(divide="ignore", invalid="ignore"):
            res = np.where(ok, ao / np.where(ok, mc, 1.0) - 1.0, np.nan)
        return np.where(gate, res, np.nan)
    # explicit recent-valid policy: last/first finite valid within the window
    mm = (Bc.EM <= _minute_hm(morning_cutoff))[None, :, None]
    am = (Bo.EM >= _minute_hm(afternoon_start))[None, :, None]
    selm = mm & Bc.valid & np.isfinite(Bc.M)
    sela = am & Bo.valid & np.isfinite(Bo.M)
    cntm = selm.sum(axis=1)
    cnta = sela.sum(axis=1)
    lastm = (Bc.S - 1) - np.argmax(selm[:, ::-1, :], axis=1)
    firsta = np.argmax(sela, axis=1)
    mc = np.take_along_axis(Bc.M, np.clip(lastm, 0, Bc.S - 1)[:, None, :],
                            axis=1)[:, 0, :]
    ao = np.take_along_axis(Bo.M, np.clip(firsta, 0, Bo.S - 1)[:, None, :],
                            axis=1)[:, 0, :]
    ok = (cntm > 0) & (cnta > 0) & np.isfinite(mc) & np.isfinite(ao) \
        & (mc > _eps())
    with np.errstate(divide="ignore", invalid="ignore"):
        res = np.where(ok, ao / np.where(ok, mc, 1.0) - 1.0, np.nan)
    return np.where(gate, res, np.nan)


# ---------------------------------------------------------------------------
# limit family — batched `_daily_limit_agg`
# ---------------------------------------------------------------------------

def _limit_matrix(limit_frame, uniq_days, columns, series_limits=False):
    """(D, C) daily limit values; NaN where the day/instrument is unknown.

    ``series_limits``: the ORIGINAL limits input was a Series — the scalar
    path broadcasts ``_as_panel(limits).iloc[:, 0]`` to EVERY instrument
    regardless of column names, so the batch path must do the same.
    """
    if limit_frame is None:
        return np.full((len(uniq_days), len(columns)), np.nan)
    if series_limits:
        col = limit_frame.iloc[:, 0].reindex(uniq_days)
        return np.repeat(np.asarray(col.to_numpy(dtype=float))[:, None],
                         len(columns), axis=1)
    aligned = limit_frame.reindex(index=uniq_days, columns=columns)
    return aligned.to_numpy(dtype=float)


def batch_limit_agg(frame, limit_frame, kind, side, transition, *,
                    coverage_floor=None, series_limits=False):
    """Vector fast path for the three ``intra_limit_*`` operators.

    Replicates ``_daily_limit_agg`` (Series = common daily limit,
    DataFrame = per-instrument, missing day/instrument -> NaN limit ->
    day unknown -> NaN output).
    """
    from factor_engine.cleaned_operators.microstructure.intraday_agg import (
        _as_panel,
        _session_local_frame,
        _DEFAULT_SESSION_TZ,
    )

    if side not in {"up", "down"}:
        raise ValueError("side must be 'up' or 'down'")
    frame = _as_panel(frame)
    limit_frame_n = _as_panel(limit_frame) if limit_frame is not None else None
    if limit_frame_n is not None and isinstance(limit_frame_n.index,
                                                pd.DatetimeIndex) \
            and limit_frame_n.index.tz is not None:
        limit_frame_n = limit_frame_n.copy()
        limit_frame_n.index = (
            limit_frame_n.index.tz_convert(_DEFAULT_SESSION_TZ)
            .tz_localize(None).normalize()
        )
    if not frame.columns.is_unique or (limit_frame_n is not None and (
            not limit_frame_n.columns.is_unique
            or not limit_frame_n.index.is_unique)):
        raise ValueError("daily limits require unique instrument and date labels")
    if not isinstance(frame.index, pd.DatetimeIndex):
        return None
    try:
        limit_matrix_probe = _limit_matrix(
            limit_frame_n,
            pd.DatetimeIndex([pd.Timestamp("2000-01-01")]),
            frame.columns,
            series_limits=series_limits,
        )
        del limit_matrix_probe
    except (ValueError, TypeError):
        # scalar path converts per-day values via float(...) -> NaN; if the
        # limit frame cannot be coerced numerically at all, defer to scalar.
        return None
    frame = _session_local_frame(
        frame, session_tz=None, source_timezone=None, market=None,
    )
    cal = _default_calendar()
    outs = []
    B = None
    for c0 in range(0, frame.shape[1], _CHUNK):
        sub = frame.iloc[:, c0:c0 + _CHUNK]
        B = _build_batch(sub, cal, coverage_floor)
        if B is None:
            return None
        Lc = _limit_matrix(limit_frame_n, B.uniq_days, B.columns,
                           series_limits=series_limits)
        r = _limit_kernel(B, Lc, kind, side, transition)
        outs.append(np.asarray(r, dtype=float))
    res = np.concatenate(outs, axis=1)
    return pd.DataFrame(res, index=B.uniq_days, columns=frame.columns,
                        dtype=float).sort_index()


def _limit_kernel(B, L, kind, side, transition):
    from factor_engine.cleaned_operators.microstructure.intraday_agg import _EPS

    day_known = np.isfinite(L)                              # (D, C)
    Lb = L[:, None, :]
    usable = np.isfinite(Lb) & B.valid
    if side == "up":
        touch = usable & (B.M >= Lb - _EPS)
    else:
        touch = usable & (B.M <= Lb + _EPS)
    if kind == "first_hit":
        any_touch = touch.any(axis=1)
        t = np.argmax(touch, axis=1)
        res = t.astype(float) / float(B.S)
        return np.where(B.has_finite & day_known & any_touch, res, np.nan)
    if kind == "duration":
        n = usable.sum(axis=1)
        res = touch.sum(axis=1) / np.maximum(n, 1)
        return np.where(B.has_finite & day_known & (n > 0), res.astype(float),
                        np.nan)
    if kind == "reopen":
        if transition not in {"open", "reseal"}:
            raise ValueError(f"transition must be 'open' or 'reseal', got {transition!r}")
        pairs = usable[:, 1:, :] & usable[:, :-1, :]
        m0 = touch[:, :-1, :]
        m1 = touch[:, 1:, :]
        if transition == "open":
            cnt = (pairs & m0 & ~m1).sum(axis=1)
        else:
            cnt = (pairs & ~m0 & m1).sum(axis=1)
        return np.where(B.has_finite & day_known, cnt.astype(float), np.nan)
    raise ValueError(f"unknown limit kernel kind: {kind!r}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _eps():
    from factor_engine.cleaned_operators.microstructure.intraday_agg import _EPS
    return _EPS


def _default_calendar():
    from factor_engine.runtime.session_panel import default_ashare_calendar
    return default_ashare_calendar(bar_freq="1min")
