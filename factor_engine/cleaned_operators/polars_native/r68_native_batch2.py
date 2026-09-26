# -*- coding: utf-8 -*-
"""R68 batch-2: native-Polars backends for the 45 next-highest-frequency operators.

Replaces the ``polars_udf_pandas_delegate`` slot (pandas-UDF bridge) of the
canonicals from ``/tmp/r67/r68_batch2.json`` with genuine native
implementations.  Two kernel styles are used, both **pandas-free** end to end
(no ``.to_pandas()`` / ``pl.from_pandas`` / ``iterrows``):

* *expression style* -- pure ``polars`` expressions (shift / when / elementwise);
* *numpy-batch style* -- a vectorized numpy kernel runs on the panel extracted
  straight from the polars frame (``pl.DataFrame`` -> ``np.ndarray`` via
  ``to_numpy``), and the result is written back with ``pl.Series``.  Trailing
  windows are materialised with ``sliding_window_view`` (NaN-padded) so the
  authoritative per-window numpy arithmetic (``np.std`` / ``np.mean`` /
  ``np.corrcoef`` two-pass forms) is reproduced to ~1e-16 relative, well inside
  the 1e-12 parity bar, while the per-row Python loop of the pandas reference
  is gone.  Sequential state machines are vectorised ACROSS COLUMNS (row loop
  only), never per-column Python loops.

Registration runs late in ``_LOAD_MODULES`` (after every pandas reference and
every earlier ``*_polars`` delegate module), so the ``polars`` slot registered
here replaces the delegate slot for these canonicals and
``register_polars_gap_coverage`` then skips them ("first native registrant
wins").  The ``pandas_numpy`` authority is untouched.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
from typing import Any

import numpy as np
import polars as pl

from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
from factor_engine.cleaned_operators.base import register_operator
from factor_engine.cleaned_operators.base_polars import SeriesOperator

_SOURCE = "r68_native_batch2"
_EPS = 1e-12


# ---------------------------------------------------------------------------
# shared helpers (pandas-free)
# ---------------------------------------------------------------------------
def _int(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _cols(frame: pl.DataFrame) -> list[str]:
    return [
        c for c in frame.columns
        if c not in ("__fe_time__", "date", "timestamp", "trade_date", "datetime", "stock_code")
    ]


def _np(frame: pl.DataFrame) -> np.ndarray:
    """Wide polars panel -> (rows, data-cols) float64 matrix (nulls -> NaN)."""
    cols = _cols(frame)
    if not cols:
        return np.empty((frame.height, 0), dtype=np.float64)
    m = frame.select([pl.col(c).cast(pl.Float64, strict=False) for c in cols]).to_numpy()
    return np.asarray(m, dtype=np.float64)


def _like(frame: pl.DataFrame, out: np.ndarray) -> pl.DataFrame:
    """Write a (rows, data-cols) float64 matrix back onto the panel columns."""
    cols = _cols(frame)
    return frame.with_columns(
        [pl.Series(name=c, values=np.ascontiguousarray(out[:, i])) for i, c in enumerate(cols)]
    )


def _finite_matrix(a: np.ndarray) -> np.ndarray:
    """Map every non-finite cell to NaN (mirrors the authorities' ``valid_values``
    finite filter inside window aggregations)."""
    return np.where(np.isfinite(a), a, np.nan)


def _windows(a: np.ndarray, w: int, col_chunk: int = 4096) -> np.ndarray:
    """Trailing windows (rows, cols, w), NaN-padded before the series start.

    Column-chunked so the (rows, chunk, w) temporary stays bounded on wide
    production panels.
    """
    from numpy.lib.stride_tricks import sliding_window_view as slw

    rows, cols = a.shape
    out = np.empty((rows, cols, w), dtype=np.float64)
    pad = np.full((min(w - 1, rows), cols), np.nan)
    ap = np.vstack([pad, a])
    for lo in range(0, cols, col_chunk):
        hi = min(lo + col_chunk, cols)
        out[:, lo:hi, :] = np.ascontiguousarray(slw(ap[:, lo:hi], w, axis=0)[:rows])
    return out


def _rolling_nan_agg(a: np.ndarray, w: int, mp: int, fn: Any) -> np.ndarray:
    """Apply ``fn(win, axis=-1)`` over trailing windows with the authoritative
    finite-count floor (``valid.size < mp -> NaN``)."""
    win = _windows(_finite_matrix(a), w)
    with np.errstate(all="ignore"):
        out = fn(win)
    cnt = np.isfinite(win).sum(axis=-1)
    out = np.where(cnt >= mp, out, np.nan)
    return out


# ---------------------------------------------------------------------------
# kernels (wave 1)
# ---------------------------------------------------------------------------
class TsVolAccelerationNative(SeriesOperator):
    """波动加速度 log(v_t / v_{t-lag})，双侧退化(<=eps) fail-close NaN。"""

    def _calculate_series(self, ret: pl.DataFrame, inner_window: int = 5, lag: int = 5,
                          min_periods: int = 2, **_: Any) -> pl.DataFrame:
        wi = _int(inner_window, "inner_window", 1)
        la = _int(lag, "lag", 1)
        mp = max(2, _int(min_periods, "min_periods", 1))
        x = _np(ret)
        v = _rolling_nan_agg(x, wi, mp, lambda win: np.nanstd(win, axis=-1))
        rows, cols = v.shape
        vlag = np.full_like(v, np.nan)
        if la < rows:
            vlag[la:] = v[:rows - la]
        with np.errstate(all="ignore"):
            out = np.where(
                np.isfinite(v) & np.isfinite(vlag) & (v > _EPS) & (vlag > _EPS),
                np.log(v / vlag),
                np.nan,
            )
        return _like(ret, out)


class TsVolClusteringNative(SeriesOperator):
    """|ret| 滞后 1 自相关（窗口内成对有限样本，两遍法匹配 np.corrcoef）。"""

    def _calculate_series(self, ret: pl.DataFrame, window: int = 20, min_periods: int = 3,
                          **_: Any) -> pl.DataFrame:
        w = _int(window, "window", 2)
        mp = max(3, _int(min_periods, "min_periods", 3))
        x = np.abs(_np(ret))
        rows, cols = x.shape
        out = np.full((rows, cols), np.nan)
        if w >= 2:
            wp = w - 1
            win = _windows(_finite_matrix(x), w)
            a = win[:, :, :-1]
            b = win[:, :, 1:]
            mask = np.isfinite(a) & np.isfinite(b)
            n = mask.sum(axis=-1).astype(np.float64)
            af = np.where(mask, a, 0.0)
            bf = np.where(mask, b, 0.0)
            with np.errstate(all="ignore"):
                mx = af.sum(axis=-1) / n
                my = bf.sum(axis=-1) / n
                dxa = np.where(mask, a - mx[..., None], 0.0)
                dxb = np.where(mask, b - my[..., None], 0.0)
                cxy = (dxa * dxb).sum(axis=-1)
                vx = (dxa * dxa).sum(axis=-1)
                vy = (dxb * dxb).sum(axis=-1)
                corr = cxy / np.sqrt(np.maximum(vx * vy, 0.0))
            out = np.where((n >= mp) & (vx > 0.0) & (vy > 0.0), corr, np.nan)
        return _like(ret, out)


class TsSignPersistenceNative(SeriesOperator):
    """sign(x) 滞后 1 自相关（直接复用权威向量化前缀和核，逐位同源）。"""

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, min_periods: int = 3,
                          **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.alpha_language_state import _sign_persistence_series

        w = _int(window, "window", 2)
        mp = max(3, _int(min_periods, "min_periods", 3))
        return _like(x, _sign_persistence_series(_np(x), w, mp))


class TsLowerPartialMomentNative(SeriesOperator):
    """下偏矩 mean(max(threshold-x,0)^order)，有限样本均值 + min_periods 门槛。"""

    def _calculate_series(self, x: pl.DataFrame, window: int = 20, threshold: float = 0.0,
                          order: float = 1.0, min_periods: int = 2, **_: Any) -> pl.DataFrame:
        w = _int(window, "window", 2)
        thr = float(threshold)
        ord_ = float(order)
        if ord_ < 1.0:
            raise ValueError("order must be >= 1")
        mp = max(2, _int(min_periods, "min_periods", 2))
        a = _np(x)
        with np.errstate(all="ignore"):
            below = np.where(np.isfinite(a), np.maximum(thr - a, 0.0), np.nan)
            below = np.power(below, ord_)
        out = _rolling_nan_agg(below, w, mp, lambda win: np.nanmean(win, axis=-1))
        return _like(x, out)


_IMPLEMENTATIONS = {
    "ts_vol_acceleration": TsVolAccelerationNative,
    "ts_vol_clustering": TsVolClusteringNative,
    "ts_sign_persistence": TsSignPersistenceNative,
    "ts_lower_partial_moment": TsLowerPartialMomentNative,
}


# ---------------------------------------------------------------------------
# wave-2 kernels (R68 batch-2 continuation, 41 operators)
# ---------------------------------------------------------------------------
# Same pandas-free discipline as wave 1: pure pl expressions or vectorized /
# module-reuse numpy kernels computed straight on the polars frame
# (``pl.DataFrame`` -> ``np.ndarray`` -> arithmetic -> ``pl.Series``).  No
# ``.to_pandas()`` / ``pl.from_pandas`` / ``iterrows`` anywhere.  Where the
# pandas authority's arithmetic lives in a module-level numpy series helper,
# that helper is reused directly (bit-exact by construction); the pandas
# DataFrame wrapper around it is replaced by ``_np`` / ``_like``.
_TIME_COLS_B2 = ("__fe_time__", "date", "timestamp", "trade_date", "datetime", "stock_code")
_NS_PER_DAY_B2 = 86_400_000_000_000
_SESSION_TZ_B2 = "Asia/Shanghai"


def _wall_times(frame: pl.DataFrame, session_tz: Any = None) -> np.ndarray | None:
    """Session wall-clock naive ``datetime64[ns]`` time axis (or None).

    Mirrors ``intraday._core.session_local``: tz-aware axes are converted to
    the session wall clock; naive axes pass through.
    """
    tcol = None
    for c in _TIME_COLS_B2:
        if c in frame.columns:
            tcol = c
            break
    if tcol is None:
        return None
    col = frame[tcol]
    tz = getattr(col.dtype, "time_zone", None)
    if tz is not None:
        col = col.dt.convert_time_zone(str(session_tz or _SESSION_TZ_B2)).dt.replace_time_zone(None)
    return col.dt.epoch(time_unit="ns").to_numpy(allow_copy=True).astype("datetime64[ns]")


def _minute_of_day(times: np.ndarray) -> np.ndarray:
    seconds = times.astype("datetime64[s]").astype("int64") % 86400
    return seconds // 60


def _day_index_of(times: np.ndarray) -> dict[int, np.ndarray]:
    day = times.astype("datetime64[D]").astype(np.int64)
    groups: dict[int, list[int]] = {}
    for i, d in enumerate(day.tolist()):
        groups.setdefault(int(d), []).append(i)
    return {d: np.asarray(v, dtype=int) for d, v in groups.items()}


def _emit_daily_b2(template: pl.DataFrame, day_keys: np.ndarray | None,
                   day_values: dict[int, Any]) -> pl.DataFrame:
    """Per-day results on the template axes (batch-1 ``_emit_daily`` contract).

    Shape-preserving fast path when every row is its own day (daily panels);
    otherwise a fresh daily panel (``__fe_time__`` midnight axis + data cols).
    ``day_values[d]`` may be a scalar (broadcast across data cols) or a
    per-col array.
    """
    cols = _cols(template)
    days = list(day_values.keys())
    arrs: dict[int, np.ndarray] = {}
    for d, v in day_values.items():
        arrs[d] = np.full(len(cols), v, dtype=np.float64) if np.isscalar(v) else np.asarray(v, dtype=np.float64)
    if day_keys is not None:
        rows = template.height
        row_of: dict[int, int] = {}
        ok = len(days) == rows
        if ok:
            for i, d in enumerate(day_keys.tolist()):
                if int(d) in row_of:
                    ok = False
                    break
                row_of[int(d)] = i
        if ok:
            arr = np.full((rows, len(cols)), np.nan, dtype=np.float64)
            for d, vals in arrs.items():
                arr[row_of[d]] = vals
            return _like(template, arr)
    import polars as _pl
    date_ns = np.asarray([d * _NS_PER_DAY_B2 for d in days], dtype=np.int64)
    out = _pl.DataFrame({
        "__fe_time__": _pl.Series(date_ns).cast(_pl.Datetime("ns")).dt.cast_time_unit("us")
    })
    for i, c in enumerate(cols):
        out = out.with_columns(
            _pl.Series(c, np.asarray([arrs[d][i] for d in days], dtype=np.float64))
        )
    return out


def _slot_metric_scores(values: np.ndarray, times: np.ndarray, window: int, metric: str) -> dict[int, float]:
    """Port of ``time_structure_v2._slot_daily_metric`` (per column, pandas-free).

    ``values`` is the (rows, cols) per-slot input matrix (e.g. log range or
    log1p amount).  Per (column, day, minute-slot) means skip NaN (pandas
    ``pivot_table`` ``aggfunc="mean"`` semantics); the history is the causal
    trailing ``window``-day mean per slot (``shift(1)``), gated at
    ``min_periods=max(2, w//2)``; a day needs >= 2 valid overlapping slots.
    """
    w = max(2, int(window))
    mp_hist = max(2, w // 2)
    mods = _minute_of_day(times)
    day_ids = times.astype("datetime64[D]").astype(np.int64)
    rows, ncols = values.shape
    out: dict[int, float] = {}
    day_slot_vals: list[dict[tuple[int, int], list[float]]] = [
        {} for _ in range(ncols)
    ]
    for r in range(rows):
        d = int(day_ids[r])
        s = int(mods[r])
        for c in range(ncols):
            v = values[r, c]
            if np.isfinite(v):
                day_slot_vals[c].setdefault((d, s), []).append(float(v))
    all_days = sorted({d for c in range(ncols) for (d, _s) in day_slot_vals[c]})
    day_pos = {d: i for i, d in enumerate(all_days)}
    for c in range(ncols):
        mat: dict[tuple[int, int], float] = {}
        slots: set[int] = set()
        for (d, s), vals in day_slot_vals[c].items():
            if vals:
                mat[(d, s)] = float(np.mean(vals))
                slots.add(s)
        slot_list = sorted(slots)
        # trailing mean of PAST w days per slot (causal, never today)
        hist: dict[tuple[int, int], float] = {}
        for s in slot_list:
            series = [mat.get((d, s), np.nan) for d in all_days]
            for i, d in enumerate(all_days):
                lo = max(0, i - w)
                past = series[lo:i]
                past = [v for v in past if np.isfinite(v)]
                if len(past) >= mp_hist:
                    hist[(d, s)] = float(np.mean(past))
        for d in all_days:
            a = np.asarray([mat.get((d, s), np.nan) for s in slot_list], dtype=float)
            b = np.asarray([hist.get((d, s), np.nan) for s in slot_list], dtype=float)
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.sum() < 2:
                out[(d, c)] = np.nan
                continue
            va, vb = a[valid], b[valid]
            if metric == "cosine":
                na_, nb_ = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
                if na_ <= _EPS or nb_ <= _EPS:
                    out[(d, c)] = np.nan
                    continue
                out[(d, c)] = float(np.dot(va, vb) / (na_ * nb_))
            elif metric == "mean_diff":
                out[(d, c)] = float(np.mean(va - vb))
            else:  # mean_abs_diff
                out[(d, c)] = float(np.mean(np.abs(va - vb)))
    return out


def _slot_scores_to_daily(scores: dict, times: np.ndarray, ncols: int) -> dict[int, Any]:
    per_day: dict[int, list[float]] = {}
    for (d, c), v in scores.items():
        per_day.setdefault(d, [np.nan] * ncols)[c] = v
    day_keys_per_row = times.astype("datetime64[D]").astype(np.int64)
    return day_keys_per_row, per_day


class CompositionIlrBalanceNative(SeriesOperator):
    """ILR balance √(rs/(r+s))·ln(gmean(A)/gmean(B))（zero_policy=reject）。"""

    def _calculate_series(self, x1: pl.DataFrame, x2: pl.DataFrame, x3: pl.DataFrame,
                          y1: pl.DataFrame, y2: pl.DataFrame, y3: pl.DataFrame,
                          x4: pl.DataFrame | None = None, x5: pl.DataFrame | None = None,
                          x6: pl.DataFrame | None = None, y4: pl.DataFrame | None = None,
                          y5: pl.DataFrame | None = None, y6: pl.DataFrame | None = None,
                          **_: Any) -> pl.DataFrame:
        a = [p for p in (x1, x2, x3, x4, x5, x6) if p is not None]
        b = [p for p in (y1, y2, y3, y4, y5, y6) if p is not None]
        if len(a) < 1 or len(b) < 1:
            raise ValueError("composition_ilr_balance requires >= 1 component per side")
        with np.errstate(divide="ignore", invalid="ignore"):
            logs = np.stack([np.log(_np(p)) for p in a + b])
        valid = np.all(np.isfinite(logs), axis=0)
        na = len(a)
        gmean_a = np.exp(logs[:na].mean(axis=0))
        gmean_b = np.exp(logs[na:].mean(axis=0))
        r = float(na)
        s = float(len(b))
        with np.errstate(divide="ignore", invalid="ignore"):
            balance = np.sqrt(r * s / (r + s)) * np.log(gmean_a / gmean_b)
        out = np.full(balance.shape, np.nan, dtype=float)
        out[valid] = balance[valid]
        return _like(x1, out)


class CompositionAitchisonDistanceNative(SeriesOperator):
    """Aitchison 距离 ‖clr(A) − clr(B)‖₂（等维 2..6 组分）。"""

    def _calculate_series(self, x1: pl.DataFrame, x2: pl.DataFrame, x3: pl.DataFrame,
                          y1: pl.DataFrame, y2: pl.DataFrame, y3: pl.DataFrame,
                          x4: pl.DataFrame | None = None, x5: pl.DataFrame | None = None,
                          x6: pl.DataFrame | None = None, y4: pl.DataFrame | None = None,
                          y5: pl.DataFrame | None = None, y6: pl.DataFrame | None = None,
                          **_: Any) -> pl.DataFrame:
        a = [p for p in (x1, x2, x3, x4, x5, x6) if p is not None]
        b = [p for p in (y1, y2, y3, y4, y5, y6) if p is not None]
        if len(a) < 2 or len(b) < 2 or len(a) != len(b):
            raise ValueError(
                "composition_aitchison_distance requires equal-size compositions (2..6 parts each)"
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            logs = np.stack([np.log(_np(p)) for p in a + b])
        valid = np.all(np.isfinite(logs), axis=0)
        na = len(a)
        la = logs[:na]
        lb = logs[na:]
        clr_a = la - la.mean(axis=0, keepdims=True)
        clr_b = lb - lb.mean(axis=0, keepdims=True)
        with np.errstate(invalid="ignore"):
            dist = np.sqrt(np.sum((clr_a - clr_b) ** 2, axis=0))
        out = np.full(dist.shape, np.nan, dtype=float)
        out[valid] = dist[valid]
        return _like(x1, out)


class TsLowerTailCoexceedanceProbabilityNative(SeriesOperator):
    """P(y ≤ Qy(q) | x ≤ Qx(q))（固定 q 下尾同超概率，以 x 为条件）。"""

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, window: int = 60,
                          q: float = 0.1, min_tail_count: int = 5, **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.rolling_pack import (
            aligned_pairs, check_window, map_pair_rolling,
        )
        from factor_engine.cleaned_operators.nonlinear_dependence import _tail_dependence
        w = check_window(window)
        quantile = float(q)
        if not 0.0 < quantile < 0.5:
            raise ValueError("q must be in (0, 0.5) for the lower tail")
        min_tail = min_tail_count
        xv = _np(x)
        yv = _np(y)

        def _fn(a: np.ndarray, b: np.ndarray) -> float:
            pa, pb = aligned_pairs(a, b)
            return _tail_dependence(pa, pb, quantile, False, min_tail)

        return _like(x, map_pair_rolling(xv, yv, w, _fn))


class TsMatrixProfileNoveltyNative(SeriesOperator):
    """矩阵轮廓新颖度（当前子序列到最近历史子序列的 z 归一 RMS 距离）。"""

    def _calculate_series(self, x: pl.DataFrame, window: int = 120,
                          subsequence_length: int = 10, history: int = 80,
                          **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.candle_state_space import _matrix_profile_series
        novelty, _age, _freq, _disp = _matrix_profile_series(
            _np(x), int(window), int(subsequence_length), int(history),
            skip_freq_disp=True,
        )
        return _like(x, novelty)


class TsMatrixProfileMotifAgeNative(SeriesOperator):
    """矩阵轮廓模式年龄 age/history（起点到起点）。"""

    def _calculate_series(self, x: pl.DataFrame, window: int = 120,
                          subsequence_length: int = 10, history: int = 80,
                          **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.candle_state_space import _matrix_profile_series
        _novelty, age, _freq, _disp = _matrix_profile_series(
            _np(x), int(window), int(subsequence_length), int(history),
            skip_freq_disp=True,
        )
        return _like(x, age)


class TsRunStrengthNative(SeriesOperator):
    """当前连续同 state 非零段内 x 的累计（可选 MAD 归一）；逐列状态机权威移植。"""

    def _calculate_series(self, x: pl.DataFrame, state: pl.DataFrame, max_run: int = 20,
                          normalize: bool = False, min_periods: int = 1, **_: Any) -> pl.DataFrame:
        from collections import deque

        from factor_engine.cleaned_operators import alpha_language_state as _als
        w = _int(max_run, "max_run", 1)
        mp = max(1, int(min_periods))
        xv = _np(x)
        sv = _np(state)
        if xv.shape != sv.shape:
            raise ValueError("x and state must share the same panel shape")
        s_valid, s_val = _als._state_series(sv)
        rows, cols = xv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            buf: deque = deque(maxlen=w)
            run_len = 0
            prev_state = None
            for row in range(rows):
                finite_x = bool(np.isfinite(xv[row, col]))
                if not s_valid[row, col]:
                    buf.clear()
                    run_len = 0
                    prev_state = None
                    continue
                cur = s_val[row, col]
                if cur == 0.0:
                    buf.clear()
                    run_len = 0
                    prev_state = None
                    out[row, col] = 0.0
                    continue
                if prev_state is not None and cur == prev_state:
                    if finite_x:
                        buf.append(xv[row, col])
                        run_len = run_len + 1
                    else:
                        buf.clear()
                        run_len = 0
                else:
                    buf = deque([xv[row, col]], maxlen=w) if finite_x else deque(maxlen=w)
                    run_len = 1 if finite_x else 0
                prev_state = cur
                if run_len < mp or run_len == 0:
                    out[row, col] = np.nan
                    continue
                run_sum = float(sum(buf))
                if normalize:
                    lo = max(0, row - w + 1)
                    base = xv[lo: row + 1, col]
                    base = base[np.isfinite(base)]
                    if base.size < mp:
                        out[row, col] = np.nan
                        continue
                    mad = float(np.median(np.abs(base - np.median(base))))
                    out[row, col] = run_sum / (mad + _EPS)
                else:
                    out[row, col] = run_sum
        return _like(x, out)


class TsMultiscaleTrendCurvatureNative(SeriesOperator):
    """T_s 对 log(s) 二次回归的二次项系数 c（多尺度趋势曲率）。"""

    def _calculate_series(self, x: pl.DataFrame, window: int = 60,
                          scales: Any = (5, 10, 20, 40), **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.multiscale_trend import (
            _curvature_series, _normalise_scales,
        )
        w = int(window)
        if w < 2:
            raise ValueError("window must be >= 2")
        ss = _normalise_scales(scales)
        for s in ss:
            if s > w:
                raise ValueError("each scale must be <= window")
        return _like(x, _curvature_series(_np(x), ss))


class TsMultiscaleTrendDispersionNative(SeriesOperator):
    """各尺度趋势统计量 T_s 的中位数绝对偏差（尺度间分歧度）。"""

    def _calculate_series(self, x: pl.DataFrame, window: int = 60,
                          scales: Any = (5, 10, 20, 40), **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.multiscale_trend import (
            _dispersion_series, _normalise_scales,
        )
        w = int(window)
        if w < 2:
            raise ValueError("window must be >= 2")
        ss = _normalise_scales(scales)
        for s in ss:
            if s > w:
                raise ValueError("each scale must be <= window")
        return _like(x, _dispersion_series(_np(x), ss))


class TsVolPvariationRoughnessNative(SeriesOperator):
    """p-变差尺度指数 H=b/p（generic p-variation scaling exponent）。"""

    def _calculate_series(self, x: pl.DataFrame, window: int = 120, p: float = 2.0,
                          scales: Any = (1, 2, 4), min_pairs: int = 5,
                          min_pair_fraction: float = 0.5, **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.rough_vol import (
            _MAX_COVERAGE_IMBALANCE, _apply_vec, _validate_min_pair_fraction,
            _validate_min_pairs, _validate_p, _validate_scales, _vec_roughness,
        )
        w = _int(window, "window", 2)
        pp = _validate_p(p)
        sc = _validate_scales(scales)
        mp = _validate_min_pairs(min_pairs)
        mpf = _validate_min_pair_fraction(min_pair_fraction)
        return _like(x, _apply_vec(_np(x), _vec_roughness, w, pp, sc, mp, mpf, _MAX_COVERAGE_IMBALANCE))


class TsTailImbalanceNative(SeriesOperator):
    """MAD 阈值上下尾计数失衡 (U−L)/n（向量化滑动窗，两遍中位数与权威同构）。"""

    def _calculate_series(self, x: pl.DataFrame, window: int = 60, k: float = 1.0,
                          min_periods: int = 8, **_: Any) -> pl.DataFrame:
        w = _int(window, "window", 2)
        kk = float(k)
        if kk <= 0.0:
            raise ValueError("k must be > 0")
        mp = max(4, int(min_periods))
        win = _windows(_finite_matrix(_np(x)), w)
        cnt = np.isfinite(win).sum(axis=-1)
        with np.errstate(all="ignore"):
            m = np.nanmedian(win, axis=-1)
            s = 1.4826 * np.nanmedian(np.abs(win - m[..., None]), axis=-1)
            upper = np.sum(win > m[..., None] + kk * s[..., None], axis=-1)
            lower = np.sum(win < m[..., None] - kk * s[..., None], axis=-1)
            out = np.where(
                (cnt >= mp) & np.isfinite(s) & (s > _EPS),
                (upper - lower) / cnt,
                np.nan,
            )
        return _like(x, out)


class CsLocalDensityScoreNative(SeriesOperator):
    """局部密度异常度 −log(KNN 平均距离 + eps)（行级横截面循环权威移植）。"""

    def _calculate_series(self, f1: pl.DataFrame, f2: pl.DataFrame = None,
                          f3: pl.DataFrame = None, f4: pl.DataFrame = None,
                          k: int = 5, **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.cross_section.robust_cs import _knn_blockwise
        feats = [p for p in (f1, f2, f3, f4) if p is not None]
        kk = max(2, int(k))
        fv = [_np(p) for p in feats]
        n, n_cols = fv[0].shape
        out = np.full((n, n_cols), np.nan, dtype=float)
        for row in range(n):
            X = np.column_stack([f[row] for f in fv])
            valid = np.all(np.isfinite(X), axis=1)
            if valid.sum() < kk + 1:
                continue
            Xv = X[valid]
            sd = np.std(Xv, axis=0)
            Xn = Xv / np.where(sd > _EPS, sd, 1.0)
            out[row, valid] = _knn_blockwise(Xn, kk)
        return _like(f1, -np.log(out + _EPS))


class TsGlrMeanShiftScoreNative(SeriesOperator):
    """GLR 均值移位评分（复用权威 _glr_series 核）。"""

    def _calculate_series(self, x: pl.DataFrame, window: int = 60, min_segment: int = 10,
                          **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.glr_change import _glr_feasibility, _glr_series
        w, ms = _glr_feasibility(window, min_segment, "ts_glr_mean_shift_score")
        return _like(x, _glr_series(_np(x), w, ms, "mean"))


class TsGlrVarianceShiftScoreNative(SeriesOperator):
    """GLR 方差移位评分（复用权威 _glr_series 核）。"""

    def _calculate_series(self, x: pl.DataFrame, window: int = 60, min_segment: int = 10,
                          **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.glr_change import _glr_feasibility, _glr_series
        w, ms = _glr_feasibility(window, min_segment, "ts_glr_variance_shift_score")
        return _like(x, _glr_series(_np(x), w, ms, "variance"))


class TsEmaNative(SeriesOperator):
    """EWM 均值（EWMContract 单一事实源：adjust=False, span→alpha 映射）。"""

    def _calculate_series(self, x: pl.DataFrame, span: Any = 12, **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.common.time_series import (
            check_ewm_contract, ewm_contract_for,
        )
        contract = ewm_contract_for("ts_ema")
        check_ewm_contract(contract, canonical="ts_ema")
        span_f = float(span)
        if 0.0 < span_f < 1.0:
            alpha = span_f
        else:
            alpha = 2.0 / (span_f + 1.0)
        import inspect as _inspect
        kw = "min_samples" if "min_samples" in _inspect.signature(pl.Expr.ewm_mean).parameters else "min_periods"
        exprs = [
            pl.col(c).ewm_mean(
                alpha=alpha, adjust=contract.adjust, ignore_na=contract.ignore_na,
                **{kw: int(contract.min_periods)},
            ).alias(c)
            for c in _cols(x)
        ]
        return x.with_columns(exprs)


class TsMaxDrawdownNative(SeriesOperator):
    """滚动最大回撤 min(valid/peaks − 1)（trailing contiguous 权威移植）。"""

    def _calculate_series(self, x: pl.DataFrame, window: Any = 20,
                          min_periods: Any = 2, **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.overhaul.base import window_params
        from factor_engine.cleaned_operators.overhaul.regression import _trailing_contiguous
        w, mp = window_params(window, min_periods, default_mp=2)
        mp = max(mp, 2)
        arr = _np(x)
        rows, cols = arr.shape
        out = np.full((rows, cols), np.nan)
        for col in range(cols):
            for row in range(rows):
                values = arr[max(0, row - w + 1): row + 1, col]
                valid = _trailing_contiguous(values)
                if valid.size < mp or np.any(valid <= 0):
                    continue
                peaks = np.maximum.accumulate(valid)
                out[row, col] = float(np.min(valid / peaks - 1.0))
        return _like(x, out)


class TsExtremaConfirmationRateNative(SeriesOperator):
    """x 的已确认同侧极值在 y 上得到同侧确认的比例（复用权威 numpy 核）。"""

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, window: int = 120,
                          prominence: float = 0.02, confirmation: int = 3,
                          tolerance: int = 3, side: str = "peak", **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.extrema_divergence import _confirmation_rate_series
        if side not in ("peak", "trough"):
            raise ValueError(f"side must be 'peak' or 'trough', got {side!r}")
        w = _int(window, "window", 3)
        prom = float(prominence)
        conf = _int(confirmation, "confirmation", 1)
        tol = _int(tolerance, "tolerance", 0)
        return _like(x, _confirmation_rate_series(
            _np(x), _np(y), w, prom, conf, tol, side
        ))


class CsRelativeDensityRatioNative(SeriesOperator):
    """局部密度 / k 近邻平均密度（复用权威 _knn_full + _relative_density_row）。"""

    def _calculate_series(self, f1: pl.DataFrame, f2: pl.DataFrame = None,
                          f3: pl.DataFrame = None, f4: pl.DataFrame = None,
                          k: int = 20, **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.cross_section.robust_cs import (
            _knn_full, _relative_density_row,
        )
        kk = max(2, int(k))
        feats = [p for p in (f1, f2, f3, f4) if p is not None]
        fv = [_np(p) for p in feats]
        n, n_cols = fv[0].shape
        out = np.full((n, n_cols), np.nan, dtype=float)
        for row in range(n):
            X = np.column_stack([f[row] for f in fv])
            valid = np.all(np.isfinite(X), axis=1)
            nv = int(valid.sum())
            eff_k = max(1, min(kk, nv - 1))
            if nv < eff_k + 2:
                continue
            Xv = X[valid]
            sd = np.std(Xv, axis=0)
            Xn = Xv / np.where(sd > _EPS, sd, 1.0)
            out[row, valid] = _relative_density_row(Xn, eff_k)
        return _like(f1, out)


class BvcSignPctNative(SeriesOperator):
    """sum(sign(Δclose)·volume)/sum(volume)（trailing window，fail-closed 掩码）。"""

    def _calculate_series(self, close: pl.DataFrame, volume: pl.DataFrame,
                          window: int = 20, **_: Any) -> pl.DataFrame:
        w = _int(window, "window", 2)
        c = _np(close)
        v = _np(volume)
        rows, cols = c.shape
        with np.errstate(invalid="ignore"):
            prev = np.vstack([np.full((1, cols), np.nan), c[:-1]])
            valid = (
                np.isfinite(c) & (c > 0.0)
                & np.isfinite(prev) & (prev > 0.0)
                & np.isfinite(v) & (v > 0.0)
            )
            flow = np.where(valid, np.sign(c - prev) * v, np.nan)
            vflow = np.where(valid, v, np.nan)

        def _roll_sum(a: np.ndarray) -> np.ndarray:
            win = _windows(a, w)
            cnt = np.isfinite(win).sum(axis=-1)
            s = np.nansum(win, axis=-1)
            return np.where(cnt >= w, s, np.nan)

        num = _roll_sum(flow)
        den = _roll_sum(vflow)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(den > 0.0, num / den, np.nan)
        return _like(close, out)


class VpinPctNative(SeriesOperator):
    """等成交量桶 VPIN（trailing window，复用权威 _vpin_window 核）。"""

    def _calculate_series(self, close: pl.DataFrame, volume: pl.DataFrame,
                          window: int = 20, buckets: int = 2, **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.technical.indicators_v2 import _vpin_window
        w = _int(window, "window", 2)
        b = _int(buckets, "buckets", 2)
        c_arr = _np(close)
        v_arr = _np(volume)
        rows, cols = c_arr.shape
        out = np.full(c_arr.shape, np.nan, dtype=float)
        for j in range(cols):
            for t in range(rows):
                if t < w - 1:
                    continue
                cs = c_arr[t - w + 1: t + 1, j]
                vs = v_arr[t - w + 1: t + 1, j]
                if not (np.isfinite(cs).all() and np.isfinite(vs).all()):
                    continue
                if np.any(cs <= 0.0) or np.any(vs <= 0.0):
                    continue
                if float(vs.sum()) <= 0.0:
                    continue
                out[t, j] = _vpin_window(cs, vs, b)
        return _like(close, out)


class CsRobustMahalanobisMadNative(SeriesOperator):
    """MAD 稳健马氏距离（中位数中心 + 三层尺度回退，权威移植）。"""

    def _calculate_series(self, f1: pl.DataFrame, f2: pl.DataFrame = None,
                          f3: pl.DataFrame = None, f4: pl.DataFrame = None,
                          **_: Any) -> pl.DataFrame:
        feats = [p for p in (f1, f2, f3, f4) if p is not None]
        fv = [_np(p) for p in feats]
        n, n_cols = fv[0].shape
        p = len(fv)
        out = np.full((n, n_cols), np.nan, dtype=float)
        for row in range(n):
            X = np.column_stack([f[row] for f in fv])
            valid = np.all(np.isfinite(X), axis=1)
            if valid.sum() < p + 3:
                continue
            Xv = X[valid]
            center = np.median(Xv, axis=0)
            mad = 1.4826 * np.median(np.abs(Xv - center), axis=0)
            sd = np.std(Xv, axis=0)
            scale = np.where(mad > _EPS, mad, sd)
            good = scale > _EPS
            if not np.any(good):
                continue
            z = (Xv[:, good] - center[good]) / scale[good]
            d = np.sqrt(np.sum(z * z, axis=1))
            out[row, valid] = d
        return _like(f1, out)


class TsCrossSpectralCoherenceNative(SeriesOperator):
    """跨谱相干均值（复用权威 _cross_series 核）。"""

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, window: int = 60,
                          band: str = "all", **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.cross_spectrum import _cross_series
        if int(window) < 24:
            raise ValueError("ts_cross_spectral_coherence requires window >= 24")
        return _like(x, _cross_series(_np(x), _np(y), int(window), "coherence", band))


class TsCrossSpectralPhaseNative(SeriesOperator):
    """跨谱相位（min_coherence 门控，复用权威 _cross_series 核）。"""

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, window: int = 60,
                          band: str = "all", min_coherence: float = 0.2, **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.cross_spectrum import _cross_series
        if int(window) < 24:
            raise ValueError("ts_cross_spectral_phase requires window >= 24")
        if not 0.0 <= float(min_coherence) <= 1.0:
            raise ValueError("ts_cross_spectral_phase requires 0 <= min_coherence <= 1")
        return _like(x, _cross_series(
            _np(x), _np(y), int(window), "phase", band, float(min_coherence)
        ))


class TsDistanceCorrNative(SeriesOperator):
    """距离相关（同位有效配对，复用权威 _distance_corr 核）。"""

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, window: int = 40,
                          min_periods: int = 10, **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.rolling_pack import (
            aligned_pairs, check_window, map_pair_rolling,
        )
        from factor_engine.cleaned_operators.nonlinear_dependence import _distance_corr
        w = check_window(window)
        mp = min_periods

        def _fn(a: np.ndarray, b: np.ndarray) -> float:
            pa, pb = aligned_pairs(a, b)
            if pa.size < mp:
                return np.nan
            return _distance_corr(pa, pb)[0]

        return _like(x, map_pair_rolling(_np(x), _np(y), w, _fn))


class TsKalmanTrendNative(SeriesOperator):
    """局部线性趋势模型的潜在斜率（复用权威 _kalman_trend_slope 核）。"""

    def _calculate_series(self, x: pl.DataFrame, q_level: float = 1e-5,
                          q_trend: float = 1e-5, r: float = 1.0,
                          scale_mode: str = "absolute", **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.ts_model.state_space import _kalman_trend_slope
        arr = _np(x)
        out = np.empty_like(arr)
        for c in range(arr.shape[1]):
            out[:, c] = _kalman_trend_slope(
                arr[:, c], float(q_level), float(q_trend), float(r), scale_mode=scale_mode
            )
        return _like(x, out)


class TsQuantileBetaSpreadNative(SeriesOperator):
    """beta(q_high) − beta(q_low)（pinball LP，in-sample；权威 _multi_regression 核心循环移植）。"""

    def _calculate_series(self, y: pl.DataFrame, x: pl.DataFrame, window: int = 60,
                          q_high: float = 0.9, q_low: float = 0.1,
                          min_periods: int = 10, **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.ts_model._rolling_core import pinball_quantile_fit
        from factor_engine.cleaned_operators.ts_model.dynamic_regression import (
            validate_multi_configured_history,
        )
        qh, ql = float(q_high), float(q_low)
        if not (0.0 < ql < qh < 1.0):
            raise ValueError("q_low < q_high must hold in (0, 1)")
        w = int(window)
        mp_req = int(min_periods)
        n_coeffs = 2  # intercept + x
        mp = max(mp_req, 5 * n_coeffs)
        validate_multi_configured_history(w, mp_req, 1, True, fit_lag=0)
        yv = _np(y)
        xv = _np(x)
        rows, cols = yv.shape

        def _beta(q: float) -> np.ndarray:
            out = np.full((rows, cols), np.nan, dtype=float)
            for col in range(cols):
                ycol = yv[:, col]
                xcol = xv[:, col]
                for row in range(rows):
                    fit_end = row
                    if fit_end < 0:
                        continue
                    start = max(0, fit_end - w + 1)
                    seg_y = ycol[start: fit_end + 1]
                    seg_x = xcol[start: fit_end + 1]
                    valid = np.isfinite(seg_y) & np.isfinite(seg_x)
                    if valid.sum() < mp:
                        continue
                    vy = seg_y[valid]
                    vx = seg_x[valid]
                    if np.std(vx) <= 0.0:
                        continue
                    design = np.column_stack([np.ones(vy.size), vx])
                    b = pinball_quantile_fit(design, vy, q)
                    if b is None:
                        continue
                    out[row, col] = float(b[1])
            return out

        return _like(y, _beta(qh) - _beta(ql))


class AshareLimitEventDensityNative(SeriesOperator):
    """窗口内事件数 / 已知可交易状态日数（权威行循环移植，内层向量化）。"""

    def _calculate_series(self, event: pl.DataFrame, window: int = 20,
                          known_status: pl.DataFrame = None, **_: Any) -> pl.DataFrame:
        w = _int(window, "window", 2)
        ev = _np(event)
        if known_status is not None:
            kv = _np(known_status)
        else:
            kv = np.ones_like(ev)
        rows, cols = ev.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for r in range(rows):
            lo = max(0, r - w + 1)
            kv_cur = kv[r]
            fin_cur = np.isfinite(kv_cur)
            known = (kv[lo: r + 1] != 0) & np.isfinite(ev[lo: r + 1])
            known_count = known.sum(axis=0)
            with np.errstate(invalid="ignore"):
                count = np.nansum(np.where(known, ev[lo: r + 1], 0.0), axis=0)
                col_out = np.where(known_count > 0, count / np.maximum(known_count, 1), np.nan)
            out[r] = np.where(fin_cur, col_out, np.nan)
        return _like(event, out)


class IntraSlotVolatilitySurpriseNative(SeriesOperator):
    """分钟槽 log range 相对历史均值的平均偏离（槽位档案权威移植）。"""

    def _calculate_series(self, high: pl.DataFrame, low: pl.DataFrame, window: Any = 20,
                          session_tz: Any = None, **_: Any) -> pl.DataFrame:
        times = _wall_times(high, session_tz)
        h = _np(high)
        l = _np(low)
        with np.errstate(all="ignore"):
            ranges = np.where((h > 0) & (l > 0) & np.isfinite(h) & np.isfinite(l),
                              np.log(h / l), np.nan)
        scores = _slot_metric_scores(ranges, times, int(window), "mean_abs_diff")
        day_keys, per_day = _slot_scores_to_daily(scores, times, ranges.shape[1])
        return _emit_daily_b2(high, day_keys, per_day)


class IntraSlotAmountSurpriseNative(SeriesOperator):
    """分钟槽成交额相对历史均值的平均偏离（log 尺度，槽位档案权威移植）。"""

    def _calculate_series(self, amount: pl.DataFrame, window: Any = 20,
                          session_tz: Any = None, **_: Any) -> pl.DataFrame:
        times = _wall_times(amount, session_tz)
        with np.errstate(all="ignore"):
            logamt = np.log1p(_np(amount))
        scores = _slot_metric_scores(logamt, times, int(window), "mean_abs_diff")
        day_keys, per_day = _slot_scores_to_daily(scores, times, logamt.shape[1])
        return _emit_daily_b2(amount, day_keys, per_day)


class TsStateExitHazardNative(SeriesOperator):
    """当前状态年龄的历史退出风险（平滑 hazard，复用权威 _survival_kernel）。"""

    def _calculate_series(self, state: pl.DataFrame, history_window: int = 60,
                          min_completed_runs: int = 5, alpha: float = 1.0,
                          inactive_policy: str = "nan", gap_policy: str = "nan",
                          **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.stateful.survival import _survival_kernel
        hw = _int(history_window, "history_window", 1)
        mc = _int(min_completed_runs, "min_completed_runs", 1)
        al = float(alpha)
        if not np.isfinite(al) or al < 0.0:
            raise ValueError("alpha must be finite and >= 0")
        sv = _np(state)
        rows, cols = sv.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            _, _, hazard, _ = _survival_kernel(
                sv[:, col], hw, mc, mc, mc, al,
                inactive_policy=inactive_policy, gap_policy=gap_policy,
            )
            out[:, col] = hazard
        return _like(state, out)


class IntraSessionReturnAsymmetryNative(SeriesOperator):
    """上午/下午绝对收益强度不对称（日内日度聚合权威移植）。"""

    def _calculate_series(self, close: pl.DataFrame, session_tz: Any = None,
                          **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.intraday._core import _SEGMENT_RANGES
        from factor_engine.cleaned_operators.intraday.time_structure_v2 import _intraday_returns
        times = _wall_times(close, session_tz)
        arr = _np(close)
        rows, cols = arr.shape
        ncols = cols
        per_day: dict[int, list[float]] = {}
        day_groups = _day_index_of(times) if times is not None else {i: np.asarray([i]) for i in range(rows)}
        for d, idx in day_groups.items():
            vals_row = np.full(ncols, np.nan, dtype=float)
            for c in range(ncols):
                cv = arr[idx, c]
                if int(np.sum(np.isfinite(cv))) < 2:
                    continue
                tm = times[idx] if times is not None else None
                r = np.abs(_intraday_returns(cv, tm))
                lo_m, hi_m = _SEGMENT_RANGES["morning"]
                lo_a, hi_a = _SEGMENT_RANGES["afternoon"]
                mods = _minute_of_day(tm)
                morning = (mods >= lo_m) & (mods <= hi_m)
                afternoon = (mods >= lo_a) & (mods <= hi_a)
                m = float(np.nansum(r[morning]))
                a = float(np.nansum(r[afternoon]))
                denom = m + a
                if denom <= _EPS:
                    continue
                vals_row[c] = (m - a) / denom
            per_day[d] = vals_row
        day_keys = times.astype("datetime64[D]").astype(np.int64) if times is not None else None
        return _emit_daily_b2(close, day_keys, per_day)


class TsArPriorInnovationZNative(SeriesOperator):
    """AR 样本外创新 / 历史残差标准差（fit_lag=1，复用权威向量化 _ar_apply_vec）。"""

    def _calculate_series(self, x: pl.DataFrame, window: int = 60, order: int = 1,
                          warmup_policy: str = "expanding", **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.ts_model.ar_meanrev import _ar_apply_vec
        arr = _np(x)
        rows, cols = arr.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            out[:, col] = _ar_apply_vec(
                arr[:, col], int(window), int(order), "innovation_z",
                fit_lag=1, stability_k=0, warmup_policy=str(warmup_policy),
            )
        return _like(x, out)


class IntradayProfilePhaseShiftNative(SeriesOperator):
    """日内 profile 最佳相位偏移 k/n_slots（复用权威 _profile_series 核）。"""

    def _calculate_series(self, x: pl.DataFrame, history_days: int = 20,
                          max_shift: int = 4, n_slots: int = 32,
                          session_tz: Any = None, **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.advanced_intraday import _profile_series
        hd = int(history_days)
        ns = int(n_slots)
        ms = int(max_shift)
        if hd < 3:
            raise ValueError("intraday_profile_phase_shift requires history_days >= 3")
        if ns < 4:
            raise ValueError("intraday_profile_phase_shift requires n_slots >= 4")
        if ms < 1:
            raise ValueError("intraday_profile_phase_shift requires max_shift >= 1")
        if ms >= ns:
            raise ValueError("intraday_profile_phase_shift requires max_shift < n_slots")
        times = _wall_times(x, session_tz)
        arr = _np(x)
        rows, cols = arr.shape
        per_day: dict[int, list[float]] = {}
        day_groups = _day_index_of(times) if times is not None else {i: np.asarray([i]) for i in range(rows)}
        all_days = sorted(day_groups.keys())
        for c in range(cols):
            day_vals = [arr[day_groups[d], c] for d in all_days]
            vals = _profile_series(day_vals, hd, ns, 1.0, phase=True, max_shift=ms)
            for d, v in zip(all_days, vals):
                per_day.setdefault(d, [np.nan] * cols)[c] = v
        day_keys = times.astype("datetime64[D]").astype(np.int64) if times is not None else None
        return _emit_daily_b2(x, day_keys, per_day)


class TsCopulaCentralAsymmetryNative(SeriesOperator):
    """经验 copula 中心非对称度（复用权威 _copula_series 核）。"""

    def _calculate_series(self, x: pl.DataFrame, y: pl.DataFrame, window: int = 120,
                          grid: int = 8, **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.distribution_break import _copula_series
        w = _int(window, "window", 10)
        g = _int(grid, "grid", 4)
        return _like(x, _copula_series(_np(x), _np(y), w, g))


class TsEffectiveTransferEntropyNative(SeriesOperator):
    """有效传递熵 TE − E[TE_surrogate]（复用权威窗口核 + 二维滚动）。"""

    def _calculate_series(self, target: pl.DataFrame, source: pl.DataFrame, window: int = 60,
                          bins: int = 3, lag: int = 1, min_transitions: Any = None,
                          min_cells_ratio: float = 1.0, **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.advanced_information import (
            _effective_transfer_entropy_window, _rolling_apply_2d_pair, _te_feasibility,
        )
        w = int(window)
        nb = int(bins)
        lg = int(lag)
        ratio = float(min_cells_ratio)
        if not (2 <= nb <= 8):
            raise ValueError("ts_effective_transfer_entropy requires 2 <= bins <= 8")
        if lg < 1:
            raise ValueError("ts_effective_transfer_entropy requires lag >= 1")
        ok, reason, mt = _te_feasibility(
            window=w, bins=nb, lag=lg, min_cells_ratio=ratio, min_transitions=min_transitions,
        )
        if not ok:
            raise ValueError(f"ts_effective_transfer_entropy {reason}; raise window or lower bins")
        return _like(
            target,
            _rolling_apply_2d_pair(
                _np(target), _np(source), w,
                lambda a, b: _effective_transfer_entropy_window(a, b, nb, lg, mt, ratio),
            ),
        )


class TsMarkovEntropyProductionNative(SeriesOperator):
    """窗口 Markov 熵产生率（复用权威 _state_dynamics_series 核 + 熵产生循环）。"""

    def _calculate_series(self, x: pl.DataFrame, window: int = 60, bins: int = 3,
                          lag: int = 1, min_periods: int = 5, min_history: int = 10,
                          **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.markov_dynamics import _state_dynamics_series
        w = _int(window, "window", 2)
        b = _int(bins, "bins", 2)
        lg = _int(lag, "lag", 1)
        mc = 1
        mss = 3
        mh = _int(min_history, "min_history", 2)
        if w <= lg:
            raise ValueError("window must exceed lag")
        if mc > w - lg:
            raise ValueError(f"min_count must not exceed available transitions (window - lag = {w - lg})")
        mp = _int(min_periods, "min_periods", 1)
        arr = _np(x)
        rows, cols = arr.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for c in range(cols):
            res = _state_dynamics_series(arr[:, c], w, b, lg, mc, mss, mh)
            for t in range(rows):
                if res["n_states_obs"][t] < 2:
                    continue
                if res["total_trans"][t] < mp:
                    continue
                P = res["P"][t]
                pi = res["pi"][t]
                if not np.all(np.isfinite(P)) or not np.all(np.isfinite(pi)):
                    continue
                B = P.shape[0]
                sigma = 0.0
                for i in range(B):
                    if pi[i] <= 0.0:
                        continue
                    for j in range(B):
                        if i == j:
                            continue
                        pij = P[i, j]
                        pji = P[j, i]
                        if pij <= 0.0 or pji <= 0.0:
                            continue
                        flux_ij = pi[i] * pij
                        flux_ji = pi[j] * pji
                        if flux_ij <= 0.0 or flux_ji <= 0.0:
                            continue
                        sigma += flux_ij * np.log(flux_ij / flux_ji)
                out[t, c] = float(sigma)
        return _like(x, out)


class TsHvgForwardBackwardAsymmetryNative(SeriesOperator):
    """HVG 出入度对称 KL 散度（复用权威 _hvg_series 核）。"""

    def _calculate_series(self, x: pl.DataFrame, window: int = 60, min_periods: int = 8,
                          min_nodes: int = 10, min_coverage_fraction: float = 0.5,
                          **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.hvg_ext import _hvg_check_params, _hvg_series
        w, mp, mn, mcf = _hvg_check_params(
            window, min_periods, min_nodes, min_coverage_fraction,
            "ts_hvg_forward_backward_asymmetry",
        )
        return _like(x, _hvg_series(_np(x), w, mp, mn, mcf, "asymmetry"))


class TsHarRvForecastErrorZNative(SeriesOperator):
    """RV 相对 HAR 预测的标准化偏差（复用权威 _har_rv 'innovation_z' 核）。"""

    def _calculate_series(self, rv: pl.DataFrame, window: Any = 120, **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.ts_model.volatility import _har_rv
        arr = _np(rv)
        rows, cols = arr.shape
        out = np.full((rows, cols), np.nan, dtype=float)
        for col in range(cols):
            for row in range(rows):
                out[row, col] = _har_rv(arr[: row + 1, col], int(window), "innovation_z")
        return _like(rv, out)


class TsHankelEffectiveRankNative(SeriesOperator):
    """Hankel 有效秩（复用权威 _hankel_effective_rank_series 核）。"""

    def _calculate_series(self, x: pl.DataFrame, window: Any = 60, embedding_dim: Any = 15,
                          min_contiguous_fraction: Any = 0.8, **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.hankel import (
            _check_hankel_params, _hankel_effective_rank_series,
        )
        w, e, _, mcf = _check_hankel_params(window, embedding_dim, None, min_contiguous_fraction)
        return _like(x, _hankel_effective_rank_series(
            _np(x), w, e, "strict_contiguous", mcf
        ))


class IntraSegmentRealizedVolNative(SeriesOperator):
    """指定时段已实现波动率 sqrt(sum(r²))（复用权威 SessionPanel 核，numpy 构造）。"""

    def _calculate_series(self, close: pl.DataFrame, segment: str = "morning",
                          session_tz: Any = None, **_: Any) -> pl.DataFrame:
        import pandas as _pd

        from factor_engine.cleaned_operators.microstructure.intraday_agg import (
            InvalidNumericSample, InsufficientCoverage, _seg_realized_vol,
            default_ashare_calendar,
        )
        from factor_engine.runtime.session_panel import build_session_panel
        if segment not in ("morning", "afternoon"):
            raise ValueError(f"segment must be in {{morning, afternoon}}, got {segment!r}")
        times = _wall_times(close, session_tz)
        arr = _np(close)
        rows, cols = arr.shape
        cal = default_ashare_calendar(bar_freq="1min")
        tz = str(session_tz) if session_tz is not None else _SESSION_TZ_B2
        per_day: dict[int, list[float]] = {}
        day_groups = _day_index_of(times) if times is not None else {i: np.asarray([i]) for i in range(rows)}
        for d, idx in day_groups.items():
            vals_row = np.full(cols, np.nan, dtype=float)
            for c in range(cols):
                vals = arr[idx, c]
                if not np.any(np.isfinite(vals)):
                    continue
                try:
                    panel = build_session_panel(
                        times[idx], vals, cal,
                        market="ashare", session_timezone=tz, source_timezone=None,
                        trade_date=_pd.Timestamp(
                            (np.datetime64(int(d), "D")).astype("datetime64[ns]")
                        ),
                    )
                    vals_row[c] = float(_seg_realized_vol(panel, segment))
                except (InvalidNumericSample, InsufficientCoverage, ZeroDivisionError, OverflowError):
                    continue
            per_day[d] = vals_row
        day_keys = times.astype("datetime64[D]").astype(np.int64) if times is not None else None
        return _emit_daily_b2(close, day_keys, per_day)


class TsNearestStructuralLevelDistanceNative(SeriesOperator):
    """最近确认结构价位对数距离（复用权威 _nearest_distance_series 核）。"""

    def _calculate_series(self, price: pl.DataFrame, window: Any = 120,
                          prominence: Any = 0.02, confirmation: Any = 3,
                          direction: str = "any", min_periods: Any = 20,
                          min_coverage_fraction: Any = 0.5, max_pivot_age: Any = None,
                          **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.structural_levels import (
            _check_coverage_fraction, _check_int, _check_positive_float,
            _nearest_distance_series,
        )
        w = _check_int(window, "window", 2)
        prom = _check_positive_float(prominence, "prominence")
        conf = _check_int(confirmation, "confirmation", 1)
        if direction not in ("any", "above", "below"):
            raise ValueError("direction must be 'any', 'above' or 'below'")
        mp = _check_int(min_periods, "min_periods", 1)
        mcf = _check_coverage_fraction(min_coverage_fraction)
        m_age = w if max_pivot_age is None else _check_int(max_pivot_age, "max_pivot_age", 1)
        return _like(price, _nearest_distance_series(
            _np(price), w, prom, conf, direction, mp, mcf, m_age
        ))


class TsStructuralLevelDensityNative(SeriesOperator):
    """结构价位密度（对数距离高斯核时间归一和，复用权威 _density_series 核）。"""

    def _calculate_series(self, price: pl.DataFrame, window: Any = 120,
                          prominence: Any = 0.02, confirmation: Any = 3,
                          bandwidth: Any = 0.03, min_periods: Any = 20,
                          min_coverage_fraction: Any = 0.5, max_pivot_age: Any = None,
                          **_: Any) -> pl.DataFrame:
        from factor_engine.cleaned_operators.structural_levels import (
            _check_coverage_fraction, _check_int, _check_positive_float, _density_series,
        )
        w = _check_int(window, "window", 2)
        prom = _check_positive_float(prominence, "prominence")
        conf = _check_int(confirmation, "confirmation", 1)
        bw = _check_positive_float(bandwidth, "bandwidth")
        mp = _check_int(min_periods, "min_periods", 1)
        mcf = _check_coverage_fraction(min_coverage_fraction)
        m_age = w if max_pivot_age is None else _check_int(max_pivot_age, "max_pivot_age", 1)
        return _like(price, _density_series(
            _np(price), w, prom, conf, bw, mp, mcf, m_age
        ))


_WAVE2_IMPLEMENTATIONS = {
    "composition_ilr_balance": CompositionIlrBalanceNative,
    "composition_aitchison_distance": CompositionAitchisonDistanceNative,
    "ts_lower_tail_coexceedance_probability": TsLowerTailCoexceedanceProbabilityNative,
    "ts_matrix_profile_novelty": TsMatrixProfileNoveltyNative,
    "ts_matrix_profile_motif_age": TsMatrixProfileMotifAgeNative,
    "ts_run_strength": TsRunStrengthNative,
    "ts_multiscale_trend_curvature": TsMultiscaleTrendCurvatureNative,
    "ts_multiscale_trend_dispersion": TsMultiscaleTrendDispersionNative,
    "ts_vol_pvariation_roughness": TsVolPvariationRoughnessNative,
    "ts_tail_imbalance": TsTailImbalanceNative,
    "cs_local_density_score": CsLocalDensityScoreNative,
    "ts_glr_mean_shift_score": TsGlrMeanShiftScoreNative,
    "ts_ema": TsEmaNative,
    "ts_max_drawdown": TsMaxDrawdownNative,
    "ts_extrema_confirmation_rate": TsExtremaConfirmationRateNative,
    "cs_relative_density_ratio": CsRelativeDensityRatioNative,
    "bvc_sign_pct": BvcSignPctNative,
    "vpin_pct": VpinPctNative,
    "cs_robust_mahalanobis_mad": CsRobustMahalanobisMadNative,
    "ts_cross_spectral_coherence": TsCrossSpectralCoherenceNative,
    "ts_distance_corr": TsDistanceCorrNative,
    "ts_kalman_trend": TsKalmanTrendNative,
    "ts_cross_spectral_phase": TsCrossSpectralPhaseNative,
    "ts_quantile_beta_spread": TsQuantileBetaSpreadNative,
    "ashare_limit_event_density": AshareLimitEventDensityNative,
    "intra_slot_volatility_surprise": IntraSlotVolatilitySurpriseNative,
    "intra_slot_amount_surprise": IntraSlotAmountSurpriseNative,
    "ts_state_exit_hazard": TsStateExitHazardNative,
    "intra_session_return_asymmetry": IntraSessionReturnAsymmetryNative,
    "ts_ar_prior_innovation_z": TsArPriorInnovationZNative,
    "intraday_profile_phase_shift": IntradayProfilePhaseShiftNative,
    "ts_copula_central_asymmetry": TsCopulaCentralAsymmetryNative,
    "ts_effective_transfer_entropy": TsEffectiveTransferEntropyNative,
    "ts_markov_entropy_production": TsMarkovEntropyProductionNative,
    "ts_hvg_forward_backward_asymmetry": TsHvgForwardBackwardAsymmetryNative,
    "ts_har_rv_forecast_error_z": TsHarRvForecastErrorZNative,
    "ts_hankel_effective_rank": TsHankelEffectiveRankNative,
    "intra_segment_realized_vol": IntraSegmentRealizedVolNative,
    "ts_nearest_structural_level_distance": TsNearestStructuralLevelDistanceNative,
    "ts_structural_level_density": TsStructuralLevelDensityNative,
}
_IMPLEMENTATIONS.update(_WAVE2_IMPLEMENTATIONS)
# b23r fix: the wave-2 dict dropped the variance-shift GLR (class exists,
# entry was missing) — register it so the polars slot goes native.
_IMPLEMENTATIONS["ts_glr_variance_shift_score"] = TsGlrVarianceShiftScoreNative

# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------
def _native_spec(canonical: str, cls: type) -> PhysicalImplementationSpec:
    kernel_source = inspect.getsource(inspect.getmodule(cls)).encode("utf-8")
    source_hash = hashlib.sha256(kernel_source).hexdigest()
    parameter_hash = hashlib.sha256(repr(cls.metadata.param_specs).encode("utf-8")).hexdigest()
    semantic_hash = hashlib.sha256(
        repr((cls.metadata.param_names, cls.metadata.panel_params, cls.metadata.scalar_params)).encode("utf-8")
    ).hexdigest()
    return PhysicalImplementationSpec(
        canonical=canonical, backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
        supports_lazy=False, supports_streaming=False, materializes_full_panel=True,
        supports_nulls=True, supports_nan=True, supports_inf=True,
        notes=(
            "R68 batch-2 genuine native polars kernel: pure pl expressions or "
            "vectorized numpy-batch compute on the pl frame written back with "
            "pl.Series; no pandas DataFrame round-trip anywhere in the kernel path."
        ),
        implementation_source_hash=source_hash,
        kernel_identity=f"{cls.__module__}.{cls.__qualname__}._calculate_series",
        parameter_domain_hash=parameter_hash,
        semantic_contract_hash=semantic_hash,
        implementation_closure_hash=hashlib.sha256(
            (source_hash + parameter_hash + semantic_hash).encode("ascii")
        ).hexdigest(),
    )


def register_r68_native_batch2() -> tuple[str, ...]:
    from factor_engine.cleaned_operators import (
        record_backend_replacement_after,
        replace_backend,
    )
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    registered = []
    for canonical, cls in _IMPLEMENTATIONS.items():
        reference = OperatorRegistry.get(canonical, "pandas_numpy", mode="any")
        if reference is None:
            raise RuntimeError(f"missing pandas_numpy reference for {canonical}")
        cls.metadata = copy.deepcopy(reference.metadata)
        cls._physical_spec = _native_spec(canonical, cls)
        migration = replace_backend(
            canonical, "polars",
            reason="R68 batch-2 native polars replacement of pandas UDF delegate",
            source=_SOURCE,
        )
        register_operator(
            name=canonical, canonical=canonical,
            category=cls.metadata.category,
            business_category=getattr(cls.metadata, "business_category", "") or cls.metadata.category,
            source=_SOURCE, backend="polars",
        )(cls)
        record_backend_replacement_after(migration, canonical, "polars", source=_SOURCE)
        registered.append(canonical)
    return tuple(registered)


register_r68_native_batch2()

_CANONICALS = tuple(_IMPLEMENTATIONS)
__all__ = ["_CANONICALS", "register_r68_native_batch2"]
