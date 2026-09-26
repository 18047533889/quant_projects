# -*- coding: utf-8 -*-
"""R68 batch-1: native-Polars backends for the 45 highest-frequency operators.

Replaces the ``polars_udf_pandas_delegate`` slot (pandas-UDF bridge) of the 45
top-used canonicals from ``/tmp/r67/r68_batch1.json`` with genuine native
implementations.  Two kernel styles are used, both **pandas-free** end to end
(no ``.to_pandas()`` / ``pl.from_pandas`` / ``iterrows``):

* *expression style* -- pure ``polars`` expressions (rolling / shift / when);
* *numpy-batch style* -- the authoritative vectorized numpy kernel is called on
  the panel extracted straight from the polars frame (``pl`` -> ``np.ndarray``
  via ``to_numpy``), and the result is written back with ``pl.Series``.  Where
  the pandas reference delegates to a module-level vectorized numpy helper
  (``_wasserstein_shift_series`` &c.) that helper is imported and reused
  verbatim, so parity with the ``pandas_numpy`` authority is bit-exact by
  construction while the pandas DataFrame round-trip of the old UDF delegate
  is removed entirely.

Registration runs late in ``_LOAD_MODULES`` (after every pandas reference and
every earlier ``*_polars`` delegate module), so the ``polars`` slot registered
here replaces the delegate slot for these canonicals and
``register_polars_gap_coverage`` then skips them ("first native registrant
wins").  The ``pandas_numpy`` authority is untouched.

Day-grouping intraday operators reimplement the pandas ``index.normalize()``
grouping on the polars ``__fe_time__`` axis with numpy ``datetime64`` math --
same grouping key, same per-day kernel, no pandas.
"""
from __future__ import annotations

import copy
import math
import warnings
from typing import Any

import numpy as np
import polars as pl

from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec
from factor_engine.cleaned_operators.base import SeriesOperator
from factor_engine.cleaned_operators.base_polars import PANEL_SKIP_COLUMNS
from factor_engine.cleaned_operators.registry import OperatorRegistry

# Pandas references living in _REVIEWED_EXTENSIONS (loaded after every
# _LOAD_MODULES entry): import them explicitly so the deep-copied metadata is
# resolvable when this module registers at import time.
import factor_engine.cleaned_operators.ashare_fiscal_period  # noqa: F401
import factor_engine.cleaned_operators.intraday_impact  # noqa: F401  # pandas reference for intraday_impact_decay_rate
import factor_engine.cleaned_operators.stateful.sequential  # noqa: F401  # pandas reference for ts_cusum_pressure

_TIME_COLS = ("__fe_time__", "date", "timestamp", "trade_date", "datetime")
_NS_PER_DAY = 86_400_000_000_000
_EPS = 1e-12

_SOURCE = "r68_native_batch1"


# ---------------------------------------------------------------------------
# panel <-> numpy helpers (pandas-free)
# ---------------------------------------------------------------------------
def _dcols(frame: pl.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in PANEL_SKIP_COLUMNS]


def _np(frame: Any) -> np.ndarray:
    """Wide polars panel -> (rows, data-cols) float64 matrix (nulls -> NaN)."""
    cols = _dcols(frame)
    return np.ascontiguousarray(
        frame.select([pl.col(c).cast(pl.Float64, strict=False) for c in cols]).to_numpy(),
        dtype=np.float64,
    )


def _like(template: pl.DataFrame, arr: np.ndarray) -> pl.DataFrame:
    """Rebuild the wide polars panel on the template axes from a float matrix."""
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    return template.with_columns(
        [pl.Series(c, arr[:, i]) for i, c in enumerate(_dcols(template))]
    )


def _time_col(frame: pl.DataFrame) -> str | None:
    for c in _TIME_COLS:
        if c in frame.columns:
            return c
    return None


def _times_ns(frame: pl.DataFrame) -> np.ndarray | None:
    """Time axis as datetime64[ns]; None when the panel carries no time axis."""
    tcol = _time_col(frame)
    if tcol is None:
        return None
    return frame[tcol].dt.epoch(time_unit="ns").to_numpy().astype("datetime64[ns]")


def _day_keys(frame: pl.DataFrame) -> np.ndarray | None:
    """Per-row calendar-day ordinal (int64)."""
    times = _times_ns(frame)
    if times is None:
        return None
    return times.astype("datetime64[D]").astype(np.int64)


def _session_times(frame: pl.DataFrame, tz: str | None = None) -> np.ndarray | None:
    """Session wall-clock times (naive datetime64[ns]) for minute math.

    tz-aware time axes are converted to session wall-clock (Asia/Shanghai by
    default), matching ``intraday._core.session_local``; naive axes pass through.
    """
    tcol = _time_col(frame)
    if tcol is None:
        return None
    col = frame[tcol]
    dtype = col.dtype
    if getattr(dtype, "time_zone", None) is not None:
        col = col.dt.convert_time_zone(tz or "Asia/Shanghai").dt.replace_time_zone(None)
    return col.dt.epoch(time_unit="ns").to_numpy().astype("datetime64[ns]")


def _day_index(day_keys: np.ndarray) -> dict[int, np.ndarray]:
    groups: dict[int, list[int]] = {}
    for i, d in enumerate(day_keys):
        groups.setdefault(int(d), []).append(i)
    return {d: np.asarray(v, dtype=int) for d, v in groups.items()}


def _emit_daily(
    template: pl.DataFrame,
    day_keys: np.ndarray,
    day_values: dict[int, np.ndarray],
) -> pl.DataFrame:
    """Emit per-day results.

    Shape-preserving fast path: when every row is its own day (daily panels --
    the case the whole harness exercises), values are rebuilt on the template
    axes.  Otherwise a fresh daily panel (``__fe_time__`` midnight axis + data
    columns) is produced, mirroring the frequency-changing delegate contract.
    """
    dcols = _dcols(template)
    days = list(day_values.keys())
    if day_keys is None:
        arr = np.full((template.height, len(dcols)), np.nan, dtype=np.float64)
        for d, vals in day_values.items():
            arr[int(d)] = vals
        return _like(template, arr)
    if len(days) == template.height:
        row_of = {d: i for i, d in enumerate(day_keys.tolist())}
        if all(day_keys[row_of[d]] == d for d in days) and len(row_of) == len(days):
            arr = np.full((template.height, len(dcols)), np.nan, dtype=np.float64)
            for d, vals in day_values.items():
                arr[row_of[d]] = vals
            return _like(template, arr)
    date_ns = np.asarray([d * _NS_PER_DAY for d in days], dtype=np.int64)
    out = pl.DataFrame(
        {"__fe_time__": pl.Series(date_ns).cast(pl.Datetime("ns")).dt.cast_time_unit("us")}
    )
    for i, col in enumerate(dcols):
        out = out.with_columns(
            pl.Series(col, np.asarray([day_values[d][i] for d in days], dtype=np.float64))
        )
    return out


def _daily_map(
    template: pl.DataFrame,
    fn,
    *,
    min_finite: int = 2,
) -> pl.DataFrame:
    """Per-(data-col, calendar-day) aggregation, pandas ``normalize()`` semantics.

    ``fn(vals, times)`` -> float for one data column's day slice.  Days with
    fewer than ``min_finite`` finite values fail closed to NaN.
    """
    x = _np(template)
    day_keys = _day_keys(template)
    groups = _day_index(day_keys) if day_keys is not None else {
        i: np.asarray([i]) for i in range(x.shape[0])
    }
    times_full = _times_ns(template)
    day_values: dict[int, np.ndarray] = {}
    for d, idx in groups.items():
        vals_day = x[idx]
        times = times_full[idx] if times_full is not None else None
        row = np.empty(x.shape[1], dtype=np.float64)
        for c in range(x.shape[1]):
            vals = vals_day[:, c]
            if int(np.sum(np.isfinite(vals))) < int(min_finite):
                row[c] = np.nan
                continue
            try:
                row[c] = float(fn(vals, times))
            except (ZeroDivisionError, OverflowError):
                row[c] = np.nan
        day_values[d] = row
    return _emit_daily(template, day_keys, day_values)


# ---------------------------------------------------------------------------
# registration machinery
# ---------------------------------------------------------------------------
def _ref_meta(op: str):
    ref = OperatorRegistry.get(op, "pandas_numpy")
    if ref is None or getattr(ref, "metadata", None) is None:
        raise RuntimeError(f"r68_native_batch1: no pandas_numpy reference for {op!r}")
    return copy.deepcopy(ref.metadata)


def _physical_spec(op: str) -> PhysicalImplementationSpec:
    return PhysicalImplementationSpec(
        canonical=op,
        backend="polars",
        execution_kind=ExecutionKind.POLARS_NATIVE_EXPR,
        supports_lazy=False,
        supports_streaming=False,
        materializes_full_panel=True,
        supports_nulls=True,
        supports_nan=True,
        supports_inf=True,
    )


def _register_native(op: str, fn) -> None:
    """Register ``fn`` as the native polars kernel for ``op``.

    ``fn`` becomes a static ``_calculate_series``; argument binding and panel
    validation run through the shared central validator with the pandas
    reference's logical contract (deep-copied metadata).
    """
    metadata = _ref_meta(op)

    cls = type(
        f"_R68Native_{op}",
        (SeriesOperator,),
        {
            "metadata": metadata,
            "_physical_spec": _physical_spec(op),
            "__module__": __name__,
            "_calculate_series": staticmethod(fn),
        },
    )
    OperatorRegistry.register(
        cls(),
        canonical=op,
        backend="polars",
        source=_SOURCE,
        status="implemented",
        backend_explicit=True,
    )


_CANONICALS: list[str] = []


def _reg(op: str):
    def _wrap(fn):
        _register_native(op, fn)
        _CANONICALS.append(op)
        return fn

    return _wrap


# ---------------------------------------------------------------------------
# shared window helpers imported from the certified pandas authorities
# ---------------------------------------------------------------------------
from factor_engine.cleaned_operators.rolling_pack import (  # noqa: E402
    check_window,
    map_rolling,
    valid_values,
)
from factor_engine.cleaned_operators.alpha_language_shape import (  # noqa: E402
    _apply_vec_monotonic,
    _ols_fit,
    _ols_slope,
    _trailing_contiguous,
)
from factor_engine.cleaned_operators.alpha_language_volatility import (  # noqa: E402
    _std as _al_std,
)
from factor_engine.cleaned_operators.alpha_language_distribution import (  # noqa: E402
    _mmd_rbf_shift_series,
    _transport_series,
    _wasserstein_shift_series,
)
from factor_engine.cleaned_operators.sequence_complexity import (  # noqa: E402
    _vec_column_permutation_entropy,
)
from factor_engine.cleaned_operators.stateful.episode import (  # noqa: E402
    _dc_states_and_extents,
)
from factor_engine.cleaned_operators.alpha_language_state import (  # noqa: E402
    HysteresisStateKernel,
)
from factor_engine.cleaned_operators.multiscale_trend import (  # noqa: E402
    _consensus_series,
    _normalise_scales,
)
from factor_engine.cleaned_operators.memory_ext import (  # noqa: E402
    _fd_discarded_weight_mass,
    _fractional_difference_series,
)
from factor_engine.cleaned_operators.ts_model.ar_meanrev import (  # noqa: E402
    _ar_apply_vec,
    _mean_reversion_half_life_vec,
    _warn_if_trending_input,
)
from factor_engine.cleaned_operators.spectral import (  # noqa: E402
    _check_spectral_params,
    _spectral_flatness_series,
)
from factor_engine.cleaned_operators.cross_section_ext import (  # noqa: E402
    _isotonic_residual_series,
)
from factor_engine.cleaned_operators.dynamic_knn import (  # noqa: E402
    _check_decision_clock,
    _peer_mean_series,
)
from factor_engine.cleaned_operators.ohlc_spread import (  # noqa: E402
    _abdi_ranaldo_series,
    _edge_series,
)
from factor_engine.cleaned_operators.extrema_divergence import (  # noqa: E402
    _divergence_series,
)
from factor_engine.cleaned_operators.rough_vol import (  # noqa: E402
    _MAX_COVERAGE_IMBALANCE,
    _apply_vec,
    _validate_min_pair_fraction,
    _validate_min_pairs,
    _validate_p,
    _vec_scaling_break,
)
from factor_engine.cleaned_operators.overhaul.base import window_params  # noqa: E402
from factor_engine.cleaned_operators.overhaul.regression import (  # noqa: E402
    _fit_1d,
    _ols_residual_1d,
)
from factor_engine.cleaned_operators.cross_section.robust_cs import (  # noqa: E402
    _quantile_regression_beta,
)
from factor_engine.cleaned_operators.intraday._core import minute_of_day  # noqa: E402
from factor_engine.cleaned_operators.intraday.time_structure_v2 import (  # noqa: E402
    _intraday_returns,
)
from factor_engine.cleaned_operators.volume_clock import (  # noqa: E402
    _volume_clock_efficiency,
    _volume_clock_roughness,
)
from factor_engine.cleaned_operators.fiscal_event_ops import (  # noqa: E402
    _positive as _fiscal_positive,
    _signal_consistency,
)
from factor_engine.cleaned_operators.fiscal_strict import period_ordinal  # noqa: E402
from factor_engine.cleaned_operators.composition import (  # noqa: E402
    _check_zero_policy,
    _close as _comp_close,
)
from factor_engine.cleaned_operators.technical.event_state_derivations_v1 import (  # noqa: E402
    _trailing_map,
)
from factor_engine.cleaned_operators.intraday_activity_duration import (  # noqa: E402
    _day_curvature,
    _official_session_grid,
)


# ===========================================================================
# 1. ashare_fiscal_quarter_from_period_end
# ===========================================================================
_FISCAL_QUARTERS = {(3, 31): 1.0, (6, 30): 2.0, (9, 30): 3.0, (12, 31): 4.0}


def _pd_timestamp_ns(value: float) -> np.datetime64 | None:
    """Reproduce ``pandas.Timestamp(<float>)`` == value nanoseconds after epoch."""
    try:
        if not np.isfinite(value):
            return None
        return np.datetime64(int(np.rint(value)), "ns")
    except (OverflowError, ValueError):
        return None


@_reg("ashare_fiscal_quarter_from_period_end")
def _k_ashare_fiscal_quarter(period_end, **_):
    x = _np(period_end)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return _like(period_end, out)
    # float cells follow the pandas Timestamp(float) branch: value nanoseconds.
    stamps = np.rint(np.where(finite, x, 0.0)).astype(np.int64).astype("datetime64[ns]")
    months = stamps.astype("datetime64[M]")
    day_num = (stamps - months).astype("timedelta64[D]").astype(np.int64) + 1
    month_num = months.astype(np.int64) % 12 + 1
    quarter = np.full(x.shape, np.nan, dtype=np.float64)
    for (m, d), q in _FISCAL_QUARTERS.items():
        quarter[(month_num == m) & (day_num == d)] = q
    out[finite] = quarter[finite]
    return _like(period_end, out)


# ===========================================================================
# 2. ts_cusum_pressure  (verbatim port of the stateful/sequential kernel)
# ===========================================================================
@_reg("ts_cusum_pressure")
def _k_cusum_pressure(x, reference_window: int = 20, drift: float = 0.5, min_periods: Any = None, **_):
    w = int(reference_window)
    if w < 2:
        raise ValueError("reference_window must be >= 2")
    k = float(drift)
    if not np.isfinite(k) or k < 0.0:
        raise ValueError("drift must be a finite number >= 0")
    mp = int(min_periods) if min_periods is not None else min(w, max(5, w // 2))
    xv = _np(x)
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=np.float64)
    for col in range(cols):
        sp = 0.0
        sm = 0.0
        for row in range(rows):
            xt = xv[row, col]
            if not np.isfinite(xt):
                out[row, col] = np.nan
                sp = 0.0
                sm = 0.0
                continue
            lo = max(0, row - w)
            seg = xv[lo:row, col]
            valid = seg[np.isfinite(seg)]
            if valid.size < mp:
                out[row, col] = np.nan
                sp = 0.0
                sm = 0.0
                continue
            magnitude = float(np.max(np.abs(valid)))
            if magnitude == 0.0:
                out[row, col] = np.nan
                sp = 0.0
                sm = 0.0
                continue
            normalized = valid / magnitude
            med = float(np.median(normalized))
            mad = float(np.median(np.abs(normalized - med)))
            if mad == 0.0:
                out[row, col] = np.nan
                sp = 0.0
                sm = 0.0
                continue
            scale = 1.4826 * mad
            z = (float(xt) / magnitude - med) / scale
            sp = max(0.0, sp + z - k)
            sm = min(0.0, sm + z + k)
            out[row, col] = sp + sm
    return _like(x, out)


# ===========================================================================
# 3. composition_normalized_entropy
# ===========================================================================
@_reg("composition_normalized_entropy")
def _k_composition_normalized_entropy(x1, x2, x3, x4=None, x5=None, x6=None, x7=None, x8=None,
                                      zero_policy: str = "reject", composition_id: Any = None, **_):
    _check_zero_policy(zero_policy)
    parts = [f for f in (x1, x2, x3, x4, x5, x6, x7, x8) if f is not None]
    if len(parts) < 3:
        raise ValueError("composition_normalized_entropy requires >= 3 components")
    arrs = [_np(f) for f in parts]
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.stack([np.log(a) for a in arrs])
    valid = np.all(np.isfinite(logs), axis=0)
    p = _comp_close(logs, 0, len(arrs))
    ent = -np.sum(p * np.log(p + _EPS), axis=0)
    out = np.full(logs.shape[1:], np.nan, dtype=np.float64)
    n_parts = len(arrs)
    if n_parts > 1:
        out[valid] = ent[valid] / np.log(n_parts)
    return _like(x1, out)


# ===========================================================================
# 4. cs_quantile_resid
# ===========================================================================
@_reg("cs_quantile_resid")
def _k_cs_quantile_resid(y, x, q: float = 0.5, **_):
    quantile = float(q)
    if not (0.0 < quantile < 1.0):
        raise ValueError("q must be in (0,1)")
    yv, xv = _np(y), _np(x)
    out = np.full(yv.shape, np.nan, dtype=np.float64)
    for row in range(yv.shape[0]):
        yrow, xr = yv[row], xv[row]
        valid = np.isfinite(yrow) & np.isfinite(xr)
        if valid.sum() < 5 or np.std(xr[valid]) <= _EPS:
            continue
        X = np.column_stack([np.ones(int(valid.sum())), xr[valid]])
        yy = yrow[valid]
        beta = _quantile_regression_beta(X, yy, quantile)
        if beta is None:
            continue
        out[row, valid] = yy - X @ beta
    return _like(y, out)


# ===========================================================================
# 5. cs_isotonic_residual
# ===========================================================================
@_reg("cs_isotonic_residual")
def _k_cs_isotonic_residual(y, x, **_):
    return _like(y, _isotonic_residual_series(_np(y), _np(x)))


# ===========================================================================
# 6. ts_edge_effective_spread
# ===========================================================================
@_reg("ts_edge_effective_spread")
def _k_ts_edge_effective_spread(open, high, low, close, window: int = 20,
                                min_valid_pairs: int = 2, min_valid_ratio: float = 0.0, **_):
    if int(window) < 3:
        raise ValueError("ts_edge_effective_spread requires window >= 3")
    out = _edge_series(
        _np(open), _np(high), _np(low), _np(close), int(window),
        int(min_valid_pairs), float(min_valid_ratio),
    )
    return _like(close, out)


# ===========================================================================
# 7. fiscal_direction_consistency  (FiscalEventView ported pandas-free)
# ===========================================================================
def _fiscal_view(values_np: np.ndarray, period_np: np.ndarray):
    rows, cols = values_np.shape
    ordinal_values = np.full((rows, cols), np.nan, dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            ordinal = period_ordinal(period_np[r, c])
            if ordinal is not None:
                ordinal_values[r, c] = ordinal
    snapshots: list[list[dict[int, float]]] = []
    state: list[dict[int, float]] = [dict() for _ in range(cols)]
    first_seen: list[set[int]] = [set() for _ in range(cols)]
    for r in range(rows):
        for c in range(cols):
            ordinal = ordinal_values[r, c]
            value = values_np[r, c]
            if not np.isfinite(ordinal) or not np.isfinite(value):
                continue
            key = int(ordinal)
            if key not in first_seen[c]:
                state[c][key] = float(value)
            first_seen[c].add(key)
        snapshots.append([dict(sorted(column.items())) for column in state])
    return snapshots, ordinal_values


def _fiscal_history(events_row_col, current: float, require_consecutive: bool):
    if not np.isfinite(current):
        return []
    history = list(events_row_col.items())
    if not history:
        return []
    history = [(int(k), float(v)) for k, v in history if k <= int(current) and np.isfinite(v)]
    history.sort()
    if require_consecutive and history:
        contiguous = [history[-1]]
        for item in reversed(history[:-1]):
            if contiguous[0][0] - item[0] != 1:
                break
            contiguous.insert(0, item)
        history = contiguous
    return history


@_reg("fiscal_direction_consistency")
def _k_fiscal_direction_consistency(signal, period_id, periods: int = 8, min_periods: int = 3,
                                    require_consecutive: bool = True,
                                    revision_policy: str = "latest_available", **_):
    periods = _fiscal_positive(periods, "periods")
    min_periods = _fiscal_positive(min_periods, "min_periods")
    if min_periods > periods:
        raise ValueError("min_periods must not exceed periods")
    sv, pv = _np(signal), _np(period_id)
    snapshots, ordinal_values = _fiscal_view(sv, pv)
    out = np.full(sv.shape, np.nan, dtype=np.float64)
    for r in range(sv.shape[0]):
        for c in range(sv.shape[1]):
            history = _fiscal_history(
                snapshots[r][c], ordinal_values[r, c], require_consecutive
            )
            out[r, c] = _signal_consistency(history, periods, min_periods)
    return _like(signal, out)


# ===========================================================================
# 8. ts_spectral_flatness
# ===========================================================================
@_reg("ts_spectral_flatness")
def _k_ts_spectral_flatness(x, window: int = 60, **_):
    return _like(x, _spectral_flatness_series(_np(x), _check_spectral_params(window)))


# ===========================================================================
# 9. ts_wasserstein_shift
# ===========================================================================
@_reg("ts_wasserstein_shift")
def _k_ts_wasserstein_shift(x, recent_window: int = 20, old_window: int = 40, min_periods: int = 5, **_):
    ws = check_window(recent_window, name="recent_window")
    wl = check_window(old_window, name="old_window")
    mp = max(3, int(min_periods))
    return _like(x, _wasserstein_shift_series(_np(x), ws, wl, mp))


# ===========================================================================
# 10. ts_trend_tstat  (overhaul/regression pd_trend_tstat port)
# ===========================================================================
@_reg("ts_trend_tstat")
def _k_ts_trend_tstat(x, window: int = 20, min_periods: Any = None, **_):
    w, mp = window_params(window, min_periods, default_mp=3)
    arr = _np(x)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    design = None
    if 3 <= w <= arr.shape[0]:
        candidate = np.column_stack((np.ones(w), np.arange(w, dtype=float)))
        if np.linalg.matrix_rank(candidate) == 2:
            design = candidate
            slope_geometry = np.linalg.pinv(design.T @ design)[-1, -1]
    for col in range(arr.shape[1]):
        for row in range(arr.shape[0]):
            values = arr[max(0, row - w + 1): row + 1, col]
            mask = np.isfinite(values)
            count = int(mask.sum())
            if count < mp:
                continue
            if count and np.ptp(values[mask]) == 0:
                continue
            if design is not None and values.size == w and count == w:
                yv = values[mask]
                beta, *_ = np.linalg.lstsq(design, yv, rcond=None)
                resid = yv - design @ beta
                slope_var = (float(resid @ resid) / (w - 2)) * slope_geometry
                if np.isfinite(slope_var) and slope_var > 0:
                    out[row, col] = float(beta[-1] / np.sqrt(slope_var))
                continue
            fit = _fit_1d(values, np.arange(values.size, dtype=float), True)
            if fit is not None:
                out[row, col] = fit[4]
    return _like(x, out)


# ===========================================================================
# 11. cs_spline_resid
# ===========================================================================
@_reg("cs_spline_resid")
def _k_cs_spline_resid(y, x, knots: int = 4, **_):
    k = max(2, int(knots))
    yv, xv = _np(y), _np(x)
    out = np.full(yv.shape, np.nan, dtype=np.float64)

    def _hinge(t, knot):
        return np.maximum(t - knot, 0.0)

    for row in range(yv.shape[0]):
        yrow, xr = yv[row], xv[row]
        valid = np.isfinite(yrow) & np.isfinite(xr)
        if valid.sum() < max(6, k * 2):
            continue
        xs, ys = xr[valid], yrow[valid]
        qs = np.quantile(xs, np.linspace(0, 1, k + 1))
        qs = np.unique(qs)
        if len(qs) < 2:
            continue
        X = np.column_stack([np.ones(len(xs)), xs] + [_hinge(xs, knot) for knot in qs[1:]])
        beta, *_ = np.linalg.lstsq(X, ys, rcond=None)
        pred = beta[0] + beta[1] * xr
        for i, knot in enumerate(qs[1:]):
            pred = pred + beta[i + 2] * _hinge(xr, knot)
        out[row, valid] = yrow[valid] - pred[valid]
    return _like(y, out)


# ===========================================================================
# 12. directional_change_state / 13. directional_change_extent
# ===========================================================================
@_reg("directional_change_state")
def _k_directional_change_state(price, threshold: float = 0.03, **_):
    th = float(threshold)
    if not np.isfinite(th) or th <= 0.0:
        raise ValueError("threshold must be > 0")
    pv = _np(price)
    out = np.full(pv.shape, np.nan, dtype=np.float64)
    for col in range(pv.shape[1]):
        st, _ = _dc_states_and_extents(pv[:, col], th)
        out[:, col] = st
    return _like(price, out)


@_reg("directional_change_extent")
def _k_directional_change_extent(price, threshold: float = 0.03, **_):
    th = float(threshold)
    if not np.isfinite(th) or th <= 0.0:
        raise ValueError("threshold must be > 0")
    pv = _np(price)
    out = np.full(pv.shape, np.nan, dtype=np.float64)
    for col in range(pv.shape[1]):
        _, ext = _dc_states_and_extents(pv[:, col], th)
        out[:, col] = ext
    return _like(price, out)


# ===========================================================================
# 14. ts_vol_of_vol
# ===========================================================================
@_reg("ts_vol_of_vol")
def _k_ts_vol_of_vol(ret, inner_window: int = 5, outer_window: int = 40, min_periods: int = 2, **_):
    wi = check_window(inner_window, name="inner_window")
    wo = check_window(outer_window, name="outer_window")
    mp = max(2, int(min_periods))
    if mp > wi or mp > wo:
        raise ValueError("min_periods must be <= inner/outer window")
    rv = _np(ret)
    inner = map_rolling(rv, wi, lambda c: _al_std(c, mp))
    logv = np.log(inner + _EPS)
    outer = map_rolling(logv, wo, lambda c: _al_std(c, mp))
    return _like(ret, outer)


# ===========================================================================
# 15. ts_beta  (rolling_beta Cov(y,x)/Var(x), paired-finite mask)
# ===========================================================================
@_reg("ts_beta")
def _k_ts_beta(y, x, window: int = 20, min_periods: Any = None, **_):
    w = int(window)
    if w < 2:
        raise ValueError("window must be >= 2")
    if min_periods is None:
        mp = min(5, w)
    else:
        mp = int(min_periods)
        if mp > w:
            raise ValueError("min_periods must be <= window")
    yv, xv = _np(y), _np(x)
    valid = np.isfinite(yv) & np.isfinite(xv)
    ym = np.where(valid, yv, 0.0)
    xm = np.where(valid, xv, 0.0)
    # prefix sums (valid-masked) -> trailing-window joint statistics
    def _csum(a):
        out = np.empty_like(a)
        np.cumsum(a, axis=0, out=out)
        return out

    cs_n = _csum(valid.astype(np.float64))
    cs_y = _csum(ym)
    cs_x = _csum(xm)
    cs_yx = _csum(ym * xm)
    cs_xx = _csum(xm * xm)

    def _win(cs):
        out = cs.copy()
        out[w:] = cs[w:] - cs[:-w]
        return out

    # Trailing joint sums; rows before the first full window keep the partial
    # prefix (pandas rolling partial-window semantics: valid count >= mp).
    n = _win(cs_n)
    sy = _win(cs_y)
    sx = _win(cs_x)
    syx = _win(cs_yx)
    sxx = _win(cs_xx)
    ok = n >= mp
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_x = sx / n
        cov = (syx - sy * mean_x) / (n - 1.0)
        var = (sxx - sx * mean_x) / (n - 1.0)
        beta = cov / var
    beta = np.where(ok & (var != 0.0) & np.isfinite(beta), beta, np.nan)
    out = np.where(valid, beta, np.nan)
    return _like(y, out)


# ===========================================================================
# 16/17. intraday_volume_clock_path_efficiency / roughness
# ===========================================================================
def _volume_clock_days(price, activity, open_px, fn, buckets: int):
    pv, av = _np(price), _np(activity)
    ov = _np(open_px) if open_px is not None else None
    day_keys = _day_keys(price)
    groups = _day_index(day_keys) if day_keys is not None else {
        i: np.asarray([i]) for i in range(pv.shape[0])
    }
    day_values: dict[int, np.ndarray] = {}
    for d, idx in groups.items():
        row = np.empty(pv.shape[1], dtype=np.float64)
        for c in range(pv.shape[1]):
            p_vals, a_vals = pv[idx, c], av[idx, c]
            o_vals = ov[idx, c] if ov is not None else np.full(len(idx), np.nan)
            if not np.all(np.isfinite(p_vals)) or not np.all(np.isfinite(a_vals)):
                row[c] = np.nan
                continue
            try:
                row[c] = float(fn(p_vals, a_vals, buckets, o_vals))
            except (ValueError, ZeroDivisionError, OverflowError):
                row[c] = np.nan
        day_values[d] = row
    return _emit_daily(price, day_keys, day_values)


@_reg("intraday_volume_clock_path_efficiency")
def _k_intraday_volume_clock_path_efficiency(price, activity, buckets: int = 16, open=None, **_):
    b = max(4, int(buckets))
    return _volume_clock_days(price, activity, open, _volume_clock_efficiency, b)


@_reg("intraday_volume_clock_roughness")
def _k_intraday_volume_clock_roughness(price, activity, buckets: int = 16, open=None, **_):
    b = max(4, int(buckets))
    return _volume_clock_days(price, activity, open, _volume_clock_roughness, b)


# ===========================================================================
# 18. ts_hysteresis_state
# ===========================================================================
@_reg("ts_hysteresis_state")
def _k_ts_hysteresis_state(z, upper: float = 1.0, lower: float = 0.0,
                           missing_policy: str = "CARRY_STATE_FREEZE_CLOCK", **_):
    hi = float(upper)
    lo = float(lower)
    if not (0.0 <= lo < hi):
        raise ValueError("require 0 <= lower < upper")
    zv = _np(z)
    rows, cols = zv.shape
    out = np.full((rows, cols), np.nan, dtype=np.float64)
    for col in range(cols):
        kernel = HysteresisStateKernel(hi, lo, missing_policy)
        for row in range(rows):
            val = zv[row, col]
            kernel.step(val, row)
            if np.isfinite(val):
                out[row, col] = float(kernel.current_state)
    return _like(z, out)


# ===========================================================================
# 19. intraday_activity_duration_curvature
# ===========================================================================
@_reg("intraday_activity_duration_curvature")
def _k_intraday_activity_duration_curvature(activity, buckets: int = 10, calendar: Any = None, **_):
    b = int(buckets)
    if b < 3:
        raise ValueError("intraday_activity_duration_curvature requires buckets >= 3")
    official = () if calendar is None else _official_session_grid(calendar)
    act = activity
    av = _np(act)
    times = _times_ns(act)
    bcol = _dcols(act)
    day_keys = _day_keys(act)
    groups = _day_index(day_keys) if day_keys is not None else {
        i: np.asarray([i]) for i in range(av.shape[0])
    }
    day_values: dict[int, np.ndarray] = {}
    for d, idx in groups.items():
        row = np.full(av.shape[1], np.nan, dtype=np.float64)
        for c in range(av.shape[1]):
            if not official:
                row[c] = np.nan
                continue
            day_times = times[idx]
            v = np.full(len(official), np.nan, dtype=np.float64)
            minutes = minute_of_day(day_times)
            vals = av[idx, c]
            for j, slot in enumerate(official):
                hit = minutes == slot
                if hit.any():
                    v[j] = vals[hit][0]
            row[c] = _day_curvature(v, b)
        day_values[d] = row
    return _emit_daily(act, day_keys, day_values)


# ===========================================================================
# 20. ts_mean_reversion_half_life
# ===========================================================================
@_reg("ts_mean_reversion_half_life")
def _k_ts_mean_reversion_half_life(x, window: int = 120, min_periods: int = 20, **_):
    xv = _np(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _warn_if_trending_input(xv, "ts_mean_reversion_half_life")
    out = np.full(xv.shape, np.nan, dtype=np.float64)
    for col in range(xv.shape[1]):
        out[:, col] = _mean_reversion_half_life_vec(xv[:, col], int(window), int(min_periods))
    return _like(x, out)


# ===========================================================================
# 21. ts_permutation_entropy
# ===========================================================================
@_reg("ts_permutation_entropy")
def _k_ts_permutation_entropy(x, window: int = 60, order: int = 3, delay: int = 1,
                              normalize: bool = True, min_patterns: Any = None, **_):
    w = check_window(window)
    ord_ = int(order)
    dl = int(delay)
    norm = bool(normalize)
    min_p = 2 if min_patterns is None else int(min_patterns)
    return _like(x, _vec_column_permutation_entropy(_np(x), w, ord_, dl, norm, min_p))


# ===========================================================================
# 22/23. ts_quantile_transport_curvature / slope
# ===========================================================================
@_reg("ts_quantile_transport_curvature")
def _k_ts_quantile_transport_curvature(x, recent_window: int = 20, old_window: int = 40,
                                       min_periods: int = 3, **_):
    ws = check_window(recent_window, name="recent_window")
    wl = check_window(old_window, name="old_window")
    mp = max(3, int(min_periods))
    return _like(x, _transport_series(_np(x), ws, wl, mp, 2))


@_reg("ts_quantile_transport_slope")
def _k_ts_quantile_transport_slope(x, recent_window: int = 20, old_window: int = 40,
                                   min_periods: int = 3, **_):
    ws = check_window(recent_window, name="recent_window")
    wl = check_window(old_window, name="old_window")
    mp = max(3, int(min_periods))
    return _like(x, _transport_series(_np(x), ws, wl, mp, 1))


# ===========================================================================
# 24. ts_monotonicity
# ===========================================================================
@_reg("ts_monotonicity")
def _k_ts_monotonicity(x, window: int = 20, min_periods: int = 3, **_):
    w = check_window(window)
    mp = max(3, int(min_periods))
    return _like(x, _apply_vec_monotonic(_np(x), w, mp))


# ===========================================================================
# 25. intra_close_participation
# ===========================================================================
@_reg("intra_close_participation")
def _k_intra_close_participation(volume, tail_minutes: int = 30, session_tz: Any = None, **_):
    tail = int(tail_minutes)
    if tail < 1:
        raise ValueError("tail_minutes must be >= 1")

    def _fn(vv, times):
        m = np.isfinite(vv)
        if m.sum() < 2:
            return np.nan
        minutes = minute_of_day(times)
        s_open, s_close = int(minutes[m].min()), int(minutes[m].max())
        eff = min(tail, s_close - s_open)
        if eff < 1:
            return np.nan
        mask = (minutes >= s_close - eff) & (minutes <= s_close)
        total = float(np.nansum(vv))
        if total <= _EPS:
            return np.nan
        return float(np.nansum(vv[mask])) / total

    return _daily_map(volume, _fn, min_finite=2)


# ===========================================================================
# 26. ts_mmd_rbf_shift
# ===========================================================================
@_reg("ts_mmd_rbf_shift")
def _k_ts_mmd_rbf_shift(x, recent_window: int = 20, old_window: int = 40, min_periods: int = 5, **_):
    ws = check_window(recent_window, name="recent_window")
    wl = check_window(old_window, name="old_window")
    mp = max(4, int(min_periods))
    return _like(x, _mmd_rbf_shift_series(_np(x), ws, wl, mp))


# ===========================================================================
# 27. state_ewm_if  (verbatim port of the stateful/sequential kernel)
# ===========================================================================
@_reg("state_ewm_if")
def _k_state_ewm_if(x, condition, half_life: float = 10.0, **_):
    hl = float(half_life)
    if not np.isfinite(hl) or hl <= 0.0:
        raise ValueError("half_life must be a finite number > 0")
    alpha = 1.0 - np.exp(-np.log(2.0) / hl)
    xv, cv = _np(x), _np(condition)
    truth = np.isfinite(cv) & (cv == 1.0)
    rows, cols = xv.shape
    out = np.full((rows, cols), np.nan, dtype=np.float64)
    for col in range(cols):
        state = np.nan
        for row in range(rows):
            if not (np.isfinite(xv[row, col]) and np.isfinite(cv[row, col])):
                out[row, col] = np.nan
                state = np.nan
                continue
            if truth[row, col]:
                if not np.isfinite(state):
                    state = float(xv[row, col])
                else:
                    state = alpha * float(xv[row, col]) + (1.0 - alpha) * state
            out[row, col] = state
    return _like(x, out)


# ===========================================================================
# 28. intra_slot_volume_surprise
# ===========================================================================
@_reg("intra_slot_volume_surprise")
def _k_intra_slot_volume_surprise(volume, window: int = 20, session_tz: Any = None, **_):
    times_all = _session_times(volume, session_tz)
    with np.errstate(divide="ignore", invalid="ignore"):
        logvol = np.log1p(_np(volume))
    w = max(2, int(window))
    min_obs = max(2, w // 2)
    day_keys = _day_keys(volume)
    groups = _day_index(day_keys) if day_keys is not None else {
        i: np.asarray([i]) for i in range(logvol.shape[0])
    }
    slots_all = minute_of_day(times_all) if times_all is not None else np.zeros(logvol.shape[0], dtype=int)
    # slot matrix: ordered days x slot -> finite mean (pandas pivot_table mean)
    day_order = list(groups.keys())
    slot_of_day: list[dict[int, float]] = []
    for d in day_order:
        idx = groups[d]
        per_slot: dict[int, list[float]] = {}
        for i in idx:
            s = int(slots_all[i])
            for c in range(logvol.shape[1]):
                v = logvol[i, c]
                if np.isfinite(v):
                    per_slot.setdefault((c, s), []).append(v)
        slot_of_day.append({})
        for c in range(logvol.shape[1]):
            slot_of_day[-1][c] = {
                s: float(np.mean(vs)) for (cc, s), vs in per_slot.items() if cc == c
            }
    all_slots = sorted({s for m in slot_of_day for c in range(logvol.shape[1]) for s in m[c]})
    day_values: dict[int, np.ndarray] = {}
    for di, d in enumerate(day_order):
        row = np.full(logvol.shape[1], np.nan, dtype=np.float64)
        hist_start = max(0, di - w)
        hist_pos = range(hist_start, di)
        for c in range(logvol.shape[1]):
            a = slot_of_day[di][c]
            # pandas rolling-mean semantics: per-slot mean over the past w days
            # (shift(1)), min_periods = max(2, w//2) non-NaN observations.
            b = {}
            for s in a:
                prev = [slot_of_day[hp][c][s] for hp in hist_pos
                        if s in slot_of_day[hp][c] and np.isfinite(slot_of_day[hp][c][s])]
                if len(prev) >= min_obs:
                    b[s] = float(np.mean(prev))
            valid_slots = [s for s in a if s in b and np.isfinite(a[s]) and np.isfinite(b[s])]
            if len(valid_slots) < 2:
                continue
            row[c] = float(np.mean([abs(a[s] - b[s]) for s in valid_slots]))
        day_values[d] = row
    return _emit_daily(volume, day_keys, day_values)


# ===========================================================================
# 29. ts_multiscale_trend_consensus
# ===========================================================================
@_reg("ts_multiscale_trend_consensus")
def _k_ts_multiscale_trend_consensus(x, window: int = 60, scales: Any = (5, 10, 20, 40), **_):
    w = int(window)
    if w < 2:
        raise ValueError("window must be >= 2")
    ss = _normalise_scales(scales)
    for s in ss:
        if s > w:
            raise ValueError("each scale must be <= window")
    return _like(x, _consensus_series(_np(x), ss))


# ===========================================================================
# 30. ts_fractional_difference
# ===========================================================================
@_reg("ts_fractional_difference")
def _k_ts_fractional_difference(x, fd: float = 0.4, cutoff: int = 20,
                                max_discarded_weight_mass: float = 0.5, **_):
    fdv = float(fd)
    if not (np.isfinite(fdv) and -1.0 < fdv < 1.0):
        raise ValueError("fd must satisfy -1 < fd < 1")
    c = int(cutoff)
    if c < 1:
        raise ValueError("cutoff must be >= 1")
    tol = float(max_discarded_weight_mass)
    if not np.isfinite(tol) or not (0.0 <= tol <= 1.0):
        raise ValueError("max_discarded_weight_mass must be in [0, 1]")
    xv = _np(x)
    if xv.shape[0] <= c or xv.shape[1] == 0:
        return _like(x, np.full(xv.shape, np.nan))
    if _fd_discarded_weight_mass(fdv, c) > tol:
        return _like(x, np.full(xv.shape, np.nan))
    return _like(x, _fractional_difference_series(xv, fdv, c))


# ===========================================================================
# 31. ts_ar_coeff_stability
# ===========================================================================
@_reg("ts_ar_coeff_stability")
def _k_ts_ar_coeff_stability(x, window: int = 60, order: int = 1,
                             warmup_policy: str = "expanding", **_):
    xv = _np(x)
    out = np.full(xv.shape, np.nan, dtype=np.float64)
    for col in range(xv.shape[1]):
        out[:, col] = _ar_apply_vec(
            xv[:, col], int(window), int(order), "coeff_stability",
            fit_lag=1, stability_k=5, warmup_policy=str(warmup_policy),
        )
    return _like(x, out)


# ===========================================================================
# 32. state_episode_age
# ===========================================================================
@_reg("state_episode_age")
def _k_state_episode_age(state, window: int = 20, **_):
    w = int(window)
    if w < 2:
        raise ValueError("window must be >= 2")
    arr = _np(state)
    if not bool(np.all(np.isnan(arr) | np.isfinite(arr))):
        raise ValueError("state panel must contain finite state codes or NaN")

    def _age(chunk):
        if np.any(np.isnan(chunk)):
            return np.nan
        start = 0
        for i in range(chunk.size - 1, 0, -1):
            if chunk[i] != chunk[i - 1]:
                start = i
                break
        else:
            return np.nan
        return float(chunk.size - 1 - start)

    return _like(state, _trailing_map(arr, w, _age))


# ===========================================================================
# 33. ts_vol_scaling_break
# ===========================================================================
@_reg("ts_vol_scaling_break")
def _k_ts_vol_scaling_break(x, window: int = 120, p: float = 2.0, min_pairs: int = 5,
                            min_pair_fraction: float = 0.5, **_):
    w = check_window(window)
    pp = _validate_p(p)
    mp = _validate_min_pairs(min_pairs)
    mpf = _validate_min_pair_fraction(min_pair_fraction)
    arr = _np(x)
    out = _apply_vec(arr, _vec_scaling_break, w, pp, mp, mpf, _MAX_COVERAGE_IMBALANCE)
    return _like(x, out)


# ===========================================================================
# 34. ts_abdi_ranaldo_spread
# ===========================================================================
@_reg("ts_abdi_ranaldo_spread")
def _k_ts_abdi_ranaldo_spread(close, high, low, window: int = 20, correction: str = "monthly", **_):
    if int(window) < 3:
        raise ValueError("ts_abdi_ranaldo_spread requires window >= 3")
    if str(correction) != "monthly":
        raise ValueError("ts_abdi_ranaldo_spread correction must be 'monthly'")
    out = _abdi_ranaldo_series(_np(close), _np(high), _np(low), int(window))
    return _like(close, out)


# ===========================================================================
# 35. trade_when
# ===========================================================================
def _truthy_panel(frame: pl.DataFrame) -> np.ndarray:
    v = _np(frame)
    return np.isfinite(v) & (v != 0.0)


@_reg("trade_when")
def _k_trade_when(condition, signal, fallback=0.0, **_):
    panels = [v for v in (condition, signal, fallback) if isinstance(v, pl.DataFrame)]
    if not panels:
        return signal if bool(np.isfinite(condition) and condition != 0) else fallback
    cond = _truthy_panel(condition) if isinstance(condition, pl.DataFrame) else bool(
        np.isfinite(condition) and condition != 0
    )
    signal_value = _np(signal) if isinstance(signal, pl.DataFrame) else signal
    fallback_value = _np(fallback) if isinstance(fallback, pl.DataFrame) else fallback
    values = np.where(cond, signal_value, fallback_value)
    return _like(panels[0], values)


# ===========================================================================
# 36. ts_extrema_divergence_strength
# ===========================================================================
@_reg("ts_extrema_divergence_strength")
def _k_ts_extrema_divergence_strength(x, y, window: int = 60, prominence: float = 0.02,
                                      confirmation: int = 3, match_lag: int = 4,
                                      side: str = "peak", **_):
    if str(side) not in ("peak", "trough"):
        raise ValueError("side must be 'peak' or 'trough'")
    if int(window) < 3:
        raise ValueError("window must be >= 3")
    return _like(
        x,
        _divergence_series(
            _np(x), _np(y), int(window), float(prominence), int(confirmation),
            int(match_lag), str(side),
        ),
    )


# ===========================================================================
# 37. intra_volume_price_alignment
# ===========================================================================
@_reg("intra_volume_price_alignment")
def _k_intra_volume_price_alignment(close, volume, session_tz: Any = None, **_):
    times_all = _session_times(close, session_tz)
    cv_np, vv_np = _np(close), _np(volume)

    def _fn_pair(r_idx, c):
        cv = cv_np[r_idx, c]
        vv = vv_np[r_idx, c]
        times = times_all[r_idx]
        r = _intraday_returns(cv, times)
        m = np.isfinite(r) & np.isfinite(vv)
        if m.sum() < 10:
            return np.nan
        rr, v2 = r[m], vv[m]
        if float(np.std(rr)) < _EPS or float(np.std(v2)) < _EPS:
            return np.nan
        return float(np.corrcoef(rr, v2)[0, 1])

    day_keys = _day_keys(close)
    groups = _day_index(day_keys) if day_keys is not None else {
        i: np.asarray([i]) for i in range(cv_np.shape[0])
    }
    day_values = {
        d: np.asarray([_fn_pair(idx, c) for c in range(cv_np.shape[1])], dtype=np.float64)
        for d, idx in groups.items()
    }
    return _emit_daily(close, day_keys, day_values)


# ===========================================================================
# 38. ts_trend_break_score
# ===========================================================================
@_reg("ts_trend_break_score")
def _k_ts_trend_break_score(x, window: int = 60, split: float = 0.5, min_periods: int = 4, **_):
    w = check_window(window)
    sp = float(split)
    if not 0.0 < sp < 1.0:
        raise ValueError("split must be in (0, 1)")
    mp = max(4, int(min_periods))
    xv = _np(x)

    def _fn(chunk):
        v = _trailing_contiguous(chunk)
        n = v.size
        if n < mp:
            return np.nan
        old_len = max(2, int(np.floor(sp * n)))
        recent_len = n - old_len
        if recent_len < 2:
            return np.nan
        b_old = _ols_slope(v[:old_len])
        b_recent = _ols_slope(v[old_len:])
        if not (np.isfinite(b_old) and np.isfinite(b_recent)):
            return np.nan
        _, _, sigma = _ols_fit(v[old_len:])
        if sigma < _EPS:
            if abs(b_recent - b_old) < _EPS:
                return 0.0
            return np.nan
        return (b_recent - b_old) / sigma

    return _like(x, map_rolling(xv, w, _fn))


# ===========================================================================
# 39. ts_path_efficiency
# ===========================================================================
@_reg("ts_path_efficiency")
def _k_ts_path_efficiency(x, window: int = 20, min_periods: int = 2, **_):
    w = check_window(window)
    mp = max(2, int(min_periods))
    xv = _np(x)

    def _fn(chunk):
        v = _trailing_contiguous(chunk)
        if v.size < mp:
            return np.nan
        net = abs(float(v[-1]) - float(v[0]))
        path = float(np.sum(np.abs(np.diff(v))))
        if path == 0.0:
            return 0.0
        return net / path

    return _like(x, map_rolling(xv, w, _fn))


# ===========================================================================
# 40. intra_tail_volume_share
# ===========================================================================
@_reg("intra_tail_volume_share")
def _k_intra_tail_volume_share(close, volume, tail_quantile: float = 0.75, session_tz: Any = None, **_):
    q = float(tail_quantile)
    if not 0.0 < q < 1.0:
        raise ValueError("tail_quantile must be in (0, 1)")
    times_all = _session_times(close, session_tz)
    cv_np, vv_np = _np(close), _np(volume)

    def _fn_pair(r_idx, c):
        cv = cv_np[r_idx, c]
        vv = vv_np[r_idx, c]
        times = times_all[r_idx]
        r = _intraday_returns(cv, times)
        m = np.isfinite(r) & np.isfinite(vv)
        if m.sum() < 4:
            return np.nan
        rr, v2 = r[m], vv[m]
        thr = float(np.quantile(np.abs(rr), q))
        tail = np.abs(rr) >= thr
        if not tail.any():
            return np.nan
        total = float(np.sum(v2))
        if total <= _EPS:
            return np.nan
        return float(np.sum(v2[tail])) / total

    day_keys = _day_keys(close)
    groups = _day_index(day_keys) if day_keys is not None else {
        i: np.asarray([i]) for i in range(cv_np.shape[0])
    }
    day_values = {
        d: np.asarray([_fn_pair(idx, c) for c in range(cv_np.shape[1])], dtype=np.float64)
        for d, idx in groups.items()
    }
    return _emit_daily(close, day_keys, day_values)


# ===========================================================================
# 41. intraday_impact_decay_rate  (_impact_decay_day ported pandas-free)
# ===========================================================================
def _impact_decay_day_np(rets, amounts, minutes_ns, horizon, shock_quantile):
    n = rets.shape[0]
    if n < horizon + 3:
        return np.nan
    if not np.all(np.isfinite(rets)) or not np.all(np.isfinite(amounts)):
        return np.nan
    if np.any(amounts < 0.0):
        return np.nan
    if np.any(rets <= -1.0):
        return np.nan
    # contiguous session blocks: any deviation from exactly one minute breaks
    minute_steps = np.diff(minutes_ns).astype("timedelta64[ns]").astype(np.int64)
    breaks = np.flatnonzero(minute_steps != 60_000_000_000)
    boundaries = np.concatenate(([0], (breaks + 1).tolist(), [n]))
    blocks = [(int(lo), int(hi)) for lo, hi in zip(boundaries[:-1], boundaries[1:])]
    kappas: list[float] = []
    for start, stop in blocks:
        if stop - start < horizon + 2:
            continue
        r = rets[start:stop]
        a = amounts[start:stop]
        absr = np.abs(r)
        logp = np.log(np.maximum(np.cumprod(1.0 + r), 1e-12))
        next_event = -1
        for e in range(r.size - horizon):
            if e < next_event:
                continue
            prefix = absr[:e]
            if prefix.size < 4:
                continue
            thr = float(np.quantile(prefix, float(shock_quantile)))
            if not (absr[e] > thr and absr[e] > _EPS and a[e] > 0.0):
                continue
            if e < 1:
                continue
            sign_e = 1.0 if r[e] > 0.0 else -1.0
            i0 = sign_e * (logp[e] - logp[e - 1])
            if abs(i0) <= _EPS:
                continue
            impacts = sign_e * (logp[e + 1: e + 1 + horizon] - logp[e - 1])
            norm = impacts / abs(i0)
            absn = np.abs(norm)
            pos = absn[absn > 0.0]
            floor = float(np.min(pos)) if pos.size else 1e-12
            hvals = np.arange(0, horizon + 1, dtype=float)
            if hvals.size < 2:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                logi = np.log(np.maximum(np.concatenate(([1.0], absn)), floor))
            slope = np.polyfit(hvals, logi, 1)[0]
            kappa = float(-slope)
            if np.isfinite(kappa):
                kappas.append(kappa)
                next_event = e + horizon
    if not kappas:
        return np.nan
    return float(np.median(kappas))


@_reg("intraday_impact_decay_rate")
def _k_intraday_impact_decay_rate(ret, amount, horizon: int = 10, shock_quantile: float = 0.9, **_):
    h = max(2, int(horizon))
    sq = float(shock_quantile)
    if not 0.0 < sq < 1.0:
        raise ValueError("intraday_impact_decay_rate requires 0 < shock_quantile < 1")
    rv, av = _np(ret), _np(amount)
    times_all = _times_ns(ret)
    day_keys = _day_keys(ret)
    groups = _day_index(day_keys) if day_keys is not None else {
        i: np.asarray([i]) for i in range(rv.shape[0])
    }
    day_values: dict[int, np.ndarray] = {}
    for d, idx in groups.items():
        minutes = times_all[idx] if times_all is not None else None
        row = np.full(rv.shape[1], np.nan, dtype=np.float64)
        for c in range(rv.shape[1]):
            if minutes is None:
                continue
            row[c] = _impact_decay_day_np(rv[idx, c], av[idx, c], minutes, h, sq)
        day_values[d] = row
    return _emit_daily(ret, day_keys, day_values)


# ===========================================================================
# 42. cs_knn_peer_mean_ex_self
# ===========================================================================
@_reg("cs_knn_peer_mean_ex_self")
def _k_cs_knn_peer_mean_ex_self(target, f1, f2, f3, k: int = 5,
                                feature_available_at: str = "same_day",
                                target_available_at: str = "close_of_t",
                                universe: str = "full_panel", **_):
    kk = int(k)
    if kk < 1:
        raise ValueError("k must be >= 1")
    _check_decision_clock(
        feature_available_at=feature_available_at,
        target_available_at=target_available_at,
    )
    feats = np.stack([_np(f) for f in (f1, f2, f3)], axis=2)
    return _like(target, _peer_mean_series(_np(target), feats, kk))


# ===========================================================================
# 43. ts_vol_term_structure
# ===========================================================================
@_reg("ts_vol_term_structure")
def _k_ts_vol_term_structure(ret, short_window: int = 5, long_window: int = 40,
                             min_periods: int = 2, **_):
    ws = check_window(short_window, name="short_window")
    wl = check_window(long_window, name="long_window")
    if ws >= wl:
        raise ValueError("short_window must be < long_window")
    mp = max(2, int(min_periods))
    rv = _np(ret)
    short = map_rolling(rv, ws, lambda c: _al_std(c, mp))
    longv = map_rolling(rv, wl, lambda c: _al_std(c, mp))
    rows, cols = rv.shape
    out = np.full((rows, cols), np.nan, dtype=np.float64)
    for col in range(cols):
        for row in range(rows):
            if not np.isfinite(short[row, col]) or not np.isfinite(longv[row, col]):
                continue
            if short[row, col] <= _EPS or longv[row, col] <= _EPS:
                continue
            out[row, col] = float(np.log(short[row, col] / longv[row, col]))
    return _like(ret, out)


# ===========================================================================
# 44. ts_realized_quarticity
# ===========================================================================
@_reg("ts_realized_quarticity")
def _k_ts_realized_quarticity(ret, window: int = 20, min_periods: int = 3, **_):
    w = check_window(window)
    mp = max(3, int(min_periods))
    rv = _np(ret)

    def _fn(chunk):
        v = valid_values(chunk)
        n = v.size
        if n < mp:
            return np.nan
        rv2 = float(np.sum(v * v))
        if rv2 < _EPS:
            return np.nan
        rv4 = float(np.sum(v ** 4))
        return float(n * rv4 / (3.0 * rv2 * rv2 + _EPS))

    return _like(ret, map_rolling(rv, w, _fn))


# ===========================================================================
# 45. cs_multi_resid
# ===========================================================================
@_reg("cs_multi_resid")
def _k_cs_multi_resid(y, x1, x2, add_intercept: bool = True, min_obs: Any = None, **_):
    yv, x1v, x2v = _np(y), _np(x1), _np(x2)
    out = np.full(yv.shape, np.nan, dtype=np.float64)
    for row in range(yv.shape[0]):
        out[row] = _ols_residual_1d(
            yv[row], [x1v[row], x2v[row]],
            weights=None,
            add_intercept=bool(add_intercept),
            min_obs=min_obs,
        )
    return _like(y, out)


__all__ = ["_CANONICALS"]
