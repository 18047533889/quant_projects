# -*- coding: utf-8 -*-
"""Round-67: classic trend oscillators missing from the DSL.

Four canonical daily-surface primitives whose formulas were absent from the
FactorEngine DSL surface (verified: none of ``cci`` / ``bias`` / ``psy`` /
``trix`` resolved through ``parse_expr`` before this module):

* ``cci``  — Commodity Channel Index
  ``TP = (high+low+close)/3``; ``MA = rolling_mean(TP, N)``;
  ``MD = mean(|TP - MA|)`` over the same window;
  ``CCI = (TP - MA) / (0.015 * MD)``; ``N`` defaults to 14.
* ``bias`` — BIAS / 乖离率
  ``(close - rolling_mean(close, N)) / rolling_mean(close, N) * 100``;
  ``N`` defaults to 6.
* ``psy``  — Psychological Line / 心理线
  ``100 * count(close > close_prev) / count(valid comparisons)`` over an
  N-bar window (strictly greater; NaN comparisons are not counted);
  ``N`` defaults to 12.
* ``trix`` — TRIX
  three chained ``EMA(close, N)`` then
  ``(EMA3 - prev(EMA3)) / prev(EMA3) * 100``; ``N`` defaults to 12.

Warmup / NaN policy
-------------------
Follows the ``RSI_WILDER`` / ``ATR_WILDER`` family: every rolling window uses
``min_periods == window`` (NaN until a full window of *valid* observations is
available) and a zero denominator publishes NaN.  ``trix`` uses the repo's
``_ema`` convention ``ewm(span=N, adjust=False, min_periods=N)`` (alpha =
2/(N+1)) chained three times, matching ``backend/sql_pushdown/emitter.py``'s
``_ema_span_over_inst`` and its verified NaN-gap recursion.

Backends
--------
``pandas_numpy`` is the logical-contract reference (pandas idioms);
``polars`` is a polars backend with numpy kernels over polars column arrays
(``ExecutionKind.POLARS_NUMPY_KERNEL`` — the same honest pattern used by
``polars_geometry_math``); the ``sql`` backend is provided by the emitter
branches in ``backend/sql_pushdown/emitter.py`` plus the
``SQL_IMPLEMENTED_CANONICALS`` entries in ``backend/sql_tiers.py``.  The daily
surface migration is recorded via
``operator_surface.register_daily_migration``.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from factor_engine.cleaned_operators import base_polars
from factor_engine.cleaned_operators.base import (
    OperatorMetadata,
    ParamRole,
    ParamSpec,
    SeriesOperator,
    register_operator,
)

_SOURCE = "technical.classic_trend_v1"
_META_COLS = frozenset({"date", "stock_code"})
_CCI_SCALE = 0.015


# ---------------------------------------------------------------------------
# scalar validation / frame helpers
# ---------------------------------------------------------------------------


def _window(value: Any, *, minimum: int = 1, name: str = "window") -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        w = int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be an integer") from exc
    if w < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return w


def _value_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in _META_COLS]


def _panel(frame: pd.DataFrame) -> pd.DataFrame:
    cols = _value_columns(frame)
    return frame[cols] if len(cols) != len(frame.columns) else frame


# ---------------------------------------------------------------------------
# pandas_numpy reference kernels
# ---------------------------------------------------------------------------


def _seq_sum(win: np.ndarray) -> np.ndarray:
    """Row-wise sum in *storage order* via ``np.add.accumulate``.

    Deliberately **not** ``win.sum(axis=1)``: numpy's pairwise reduction and the
    DuckDB window aggregates round differently by ~1 ulp, and ``cci``'s
    ``(TP - MA)`` numerator cancels ~6000x (|MA| ~ 31 vs |TP - MA| ~ 5e-3), so a
    1e-15 disagreement in MA/MAD is amplified to ~3e-12 — *outside* the 1e-12
    backend-parity budget.  A plain sequential accumulation matches the DuckDB
    window aggregate / positional self-join bit-for-bit (measured max|diff| =
    0.0), which is what keeps pandas_numpy / polars / sql inside the budget.
    """
    return np.add.accumulate(win, axis=1)[:, -1]


def _rolling_windows(frame: pd.DataFrame, window: int):
    """Yield ``(column_index, windows, all_finite)`` per column.

    ``windows`` has shape ``(n - w + 1, w)`` and ``all_finite`` marks the rows
    whose window is entirely finite (``min_periods == window``).
    """
    w = int(window)
    values = frame.to_numpy(dtype=float)
    n, m = values.shape
    if n < w:
        return
    for j in range(m):
        win = np.lib.stride_tricks.sliding_window_view(values[:, j], w)
        yield j, win, np.isfinite(win).all(axis=1)


def _rolling_seq_mean(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling mean with ``min_periods == window`` using sequential summation.

    Same contract as ``frame.rolling(w, min_periods=w).mean()`` but accumulated
    in storage order so it agrees bit-for-bit with the SQL window aggregate.
    """
    w = int(window)
    out = np.full(frame.shape, np.nan, dtype=float)
    for j, win, ok in _rolling_windows(frame, w):
        out[w - 1 :, j] = np.where(ok, _seq_sum(win) / w, np.nan)
    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


def _rolling_mad(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Windowed mean absolute deviation with ``min_periods=window`` semantics.

    ``MAD_t = mean_i |x_i - mean(window_i)|`` — the same quantity pandas'
    ``rolling(w, min_periods=w).apply(lambda a: np.abs(a - a.mean()).mean(),
    raw=True)`` computes (that callback is only invoked on all-valid windows
    when ``min_periods == w``), but vectorised per column.  A window containing
    any non-finite observation is missing.
    """
    w = int(window)
    out = np.full(frame.shape, np.nan, dtype=float)
    for j, win, ok in _rolling_windows(frame, w):
        with np.errstate(invalid="ignore"):
            ma = np.where(ok, _seq_sum(win) / w, np.nan)
            mad = np.where(ok, _seq_sum(np.abs(win - ma[:, None])) / w, np.nan)
        out[w - 1 :, j] = mad
    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


def _cci_pandas(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int
) -> pd.DataFrame:
    w = _window(window, minimum=2)
    high, low, close = _panel(high), _panel(low), _panel(close)
    tp = (high + low + close) / 3.0
    ma = _rolling_seq_mean(tp, w)
    mad = _rolling_mad(tp, w)
    # MD == 0 exactly when every TP in the window is identical (a limit-up /
    # flat board run).  Testing ``mad == 0`` is not enough: summation residue
    # leaves mad ~1e-14 on such windows, and the numerator (TP - MA) carries a
    # residue of the same order, so the quotient is numerically meaningless.
    # The window range is an exact predicate, so all three backends agree.
    roll = tp.rolling(w, min_periods=w)
    flat = (roll.max() - roll.min()) == 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (tp - ma) / (_CCI_SCALE * mad.replace(0.0, np.nan))
    return out.where(~flat)


def _bias_pandas(close: pd.DataFrame, window: int) -> pd.DataFrame:
    w = _window(window, minimum=1)
    close = _panel(close)
    ma = _rolling_seq_mean(close, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * (close - ma) / ma.replace(0.0, np.nan)


def _psy_pandas(close: pd.DataFrame, window: int) -> pd.DataFrame:
    w = _window(window, minimum=1)
    close = _panel(close)
    prev = close.shift(1)
    valid = close.notna() & prev.notna()
    up = (close > prev) & valid
    num = up.astype(float).rolling(w, min_periods=1).sum()
    den = valid.astype(float).rolling(w, min_periods=1).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        return (100.0 * num / den).where(den >= w)


def _trix_pandas(close: pd.DataFrame, window: int) -> pd.DataFrame:
    w = _window(window, minimum=1)
    close = _panel(close)
    e1 = close.ewm(span=w, adjust=False, min_periods=w).mean()
    e2 = e1.ewm(span=w, adjust=False, min_periods=w).mean()
    e3 = e2.ewm(span=w, adjust=False, min_periods=w).mean()
    prev = e3.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * (e3 - prev) / prev.replace(0.0, np.nan)


# ---------------------------------------------------------------------------
# numpy kernels used by the polars backend
# ---------------------------------------------------------------------------


def _cci_1d(high: np.ndarray, low: np.ndarray, close: np.ndarray, w: int) -> np.ndarray:
    tp = (high + low + close) / 3.0
    n = tp.shape[0]
    out = np.full(n, np.nan, dtype=float)
    if n < w:
        return out
    win = np.lib.stride_tricks.sliding_window_view(tp, w)
    ok = np.isfinite(win).all(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = np.where(ok, _seq_sum(win) / w, np.nan)
        mad = np.where(ok, _seq_sum(np.abs(win - ma[:, None])) / w, np.nan)
        val = (tp[w - 1 :] - ma) / (_CCI_SCALE * mad)
    # flat window (every TP identical) -> MD is 0 -> CCI undefined -> NaN
    flat = win.max(axis=1) == win.min(axis=1)
    gate = ok & (~flat) & (mad != 0.0) & np.isfinite(tp[w - 1 :])
    out[w - 1 :] = np.where(gate, val, np.nan)
    return out


def _bias_1d(close: np.ndarray, w: int) -> np.ndarray:
    n = close.shape[0]
    out = np.full(n, np.nan, dtype=float)
    if n < w:
        return out
    win = np.lib.stride_tricks.sliding_window_view(close, w)
    ok = np.isfinite(win).all(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = np.where(ok, _seq_sum(win) / w, np.nan)
        val = 100.0 * (close[w - 1 :] - ma) / ma
    gate = ok & (ma != 0.0) & np.isfinite(close[w - 1 :])
    out[w - 1 :] = np.where(gate, val, np.nan)
    return out


def _psy_1d(close: np.ndarray, w: int) -> np.ndarray:
    n = close.shape[0]
    out = np.full(n, np.nan, dtype=float)
    if n < 2:
        return out
    cur, prv = close[1:], close[:-1]
    valid = np.isfinite(cur) & np.isfinite(prv)
    cmp = np.where(valid, (cur > prv).astype(float), np.nan)
    if cmp.shape[0] < w:
        return out
    win = np.lib.stride_tricks.sliding_window_view(cmp, w)
    cnt = np.isfinite(win).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        res = np.where(cnt >= w, 100.0 * np.nansum(win, axis=1) / cnt, np.nan)
    out[w:] = res
    return out


def _ewm_adjust_false_minp(
    values: np.ndarray, alpha: float, min_periods: int
) -> np.ndarray:
    """pandas ``ewm(alpha, adjust=False, min_periods, ignore_na=False)`` parity.

    Same verified recursion as ``backend/sql_pushdown/emitter.py``'s
    ``_ewm_adjust_false_sql``: absolute-position decay across NaN gaps, state
    carried forward over missing rows, output gated by the running count of
    valid observations.
    """
    n = values.shape[0]
    out = np.full(n, np.nan, dtype=float)
    state = np.nan
    last_rn: int | None = None
    cnt = 0
    for i in range(n):
        x = values[i]
        if not np.isfinite(x):
            if last_rn is not None and cnt >= min_periods:
                out[i] = state
            continue
        if last_rn is None:
            state = x
        else:
            gap = i - last_rn
            decay = (1.0 - alpha) ** gap
            state = (decay * state + alpha * x) / (decay + alpha)
        last_rn = i
        cnt += 1
        if cnt >= min_periods:
            out[i] = state
    return out


def _trix_1d(close: np.ndarray, w: int) -> np.ndarray:
    alpha = 2.0 / (float(w) + 1.0)
    e1 = _ewm_adjust_false_minp(close, alpha, w)
    e2 = _ewm_adjust_false_minp(e1, alpha, w)
    e3 = _ewm_adjust_false_minp(e2, alpha, w)
    n = close.shape[0]
    if n < 2:
        return np.full(n, np.nan, dtype=float)
    prev = np.full(n, np.nan, dtype=float)
    prev[1:] = e3[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        val = 100.0 * (e3 - prev) / prev
    gate = np.isfinite(e3) & np.isfinite(prev) & (prev != 0.0)
    return np.where(gate, val, np.nan)


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------


def _metadata(
    name: str,
    description: str,
    param_names: list[str],
    *,
    panel_params: tuple[str, ...],
    tags: tuple[str, ...] = (),
    window_minimum: int = 1,
) -> OperatorMetadata:
    return OperatorMetadata(
        name=name,
        category="technical_signal",
        description=description,
        param_names=list(param_names),
        panel_params=panel_params,
        panel_arity=len(panel_params),
        scalar_params=tuple(p for p in param_names if p not in panel_params),
        total_positional_arity=len(param_names),
        param_specs={
            "window": ParamSpec(
                dtype=int, min=window_minimum, param_role=ParamRole.HORIZON
            )
        },
        return_type="series",
        tags=[
            "financial",
            "technical",
            "pit_safe",
            "causal",
            f"signature:{','.join(param_names)}->series",
            *tags,
        ],
    )


# ---------------------------------------------------------------------------
# pandas_numpy reference operators
# ---------------------------------------------------------------------------


@register_operator(
    name="cci",
    category="technical_signal",
    business_category="technical",
    canonical="cci",
    source=_SOURCE,
    backend="pandas_numpy",
    status="production",
)
class Cci(SeriesOperator):
    """商品通道指数：``(TP - MA) / (0.015 * MAD)``，``TP=(high+low+close)/3``。"""

    metadata = _metadata(
        "cci",
        "Commodity Channel Index (TP=(high+low+close)/3, MA=rolling mean, MAD=mean|TP-MA|)。",
        ["high", "low", "close", "window"],
        panel_params=("high", "low", "close"),
        tags=("CCI",),
        window_minimum=2,
    )

    def _calculate_series(
        self,
        high: pd.DataFrame,
        low: pd.DataFrame,
        close: pd.DataFrame,
        window: int = 14,
        **_: Any,
    ) -> pd.DataFrame:
        return _cci_pandas(high, low, close, window)


@register_operator(
    name="bias",
    category="technical_signal",
    business_category="technical",
    canonical="bias",
    source=_SOURCE,
    backend="pandas_numpy",
    status="production",
)
class Bias(SeriesOperator):
    """乖离率：``(close - SMA(close,N)) / SMA(close,N) * 100``。"""

    metadata = _metadata(
        "bias",
        "BIAS / 乖离率：（close - N 均值）/ N 均值 * 100。",
        ["close", "window"],
        panel_params=("close",),
        tags=("BIAS",),
    )

    def _calculate_series(
        self, close: pd.DataFrame, window: int = 6, **_: Any
    ) -> pd.DataFrame:
        return _bias_pandas(close, window)


@register_operator(
    name="psy",
    category="technical_signal",
    business_category="technical",
    canonical="psy",
    source=_SOURCE,
    backend="pandas_numpy",
    status="production",
)
class Psy(SeriesOperator):
    """心理线：窗口内 ``close > close_prev`` 的有效比较占比 * 100。"""

    metadata = _metadata(
        "psy",
        "Psychological Line / 心理线：N 窗口内 close > close_prev 的（有效比较）占比 * 100。",
        ["close", "window"],
        panel_params=("close",),
        tags=("PSY",),
    )

    def _calculate_series(
        self, close: pd.DataFrame, window: int = 12, **_: Any
    ) -> pd.DataFrame:
        return _psy_pandas(close, window)


@register_operator(
    name="trix",
    category="technical_signal",
    business_category="technical",
    canonical="trix",
    source=_SOURCE,
    backend="pandas_numpy",
    status="production",
)
class Trix(SeriesOperator):
    """TRIX：三重 EMA 的变化率 * 100。"""

    metadata = _metadata(
        "trix",
        "TRIX：EMA(EMA(EMA(close,N),N),N) 的单期变化率 * 100（span=N, min_periods=N）。",
        ["close", "window"],
        panel_params=("close",),
        tags=("TRIX",),
    )

    def _calculate_series(
        self, close: pd.DataFrame, window: int = 12, **_: Any
    ) -> pd.DataFrame:
        return _trix_pandas(close, window)


# ---------------------------------------------------------------------------
# polars backend (numpy kernels over polars column arrays)
# ---------------------------------------------------------------------------

_POLARS_NOTE = (
    "Polars backend: per-column numpy kernel over polars column arrays "
    "(SlidingWindowView / explicit EWM recursion), no pandas round-trip. "
    "Eager panel API only; no lazy/streaming claim."
)


def _value_cols_pl(frame: pl.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in _META_COLS]


def _make(base: pl.DataFrame, cols: list[str], values: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame({c: values[:, i] for i, c in enumerate(cols)})


def _np(frame: pl.DataFrame, col: str) -> np.ndarray:
    return frame[col].cast(pl.Float64).to_numpy()


def _cci_polars(
    high: pl.DataFrame, low: pl.DataFrame, close: pl.DataFrame, window: int
) -> pl.DataFrame:
    w = _window(window, minimum=2)
    cols = _value_cols_pl(close)
    out = np.full((close.height, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _cci_1d(_np(high, c), _np(low, c), _np(close, c), w)
    return _make(close, cols, out)


def _bias_polars(close: pl.DataFrame, window: int) -> pl.DataFrame:
    w = _window(window, minimum=1)
    cols = _value_cols_pl(close)
    out = np.full((close.height, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _bias_1d(_np(close, c), w)
    return _make(close, cols, out)


def _psy_polars(close: pl.DataFrame, window: int) -> pl.DataFrame:
    w = _window(window, minimum=1)
    cols = _value_cols_pl(close)
    out = np.full((close.height, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _psy_1d(_np(close, c), w)
    return _make(close, cols, out)


def _trix_polars(close: pl.DataFrame, window: int) -> pl.DataFrame:
    w = _window(window, minimum=1)
    cols = _value_cols_pl(close)
    out = np.full((close.height, len(cols)), np.nan, dtype=float)
    for i, c in enumerate(cols):
        out[:, i] = _trix_1d(_np(close, c), w)
    return _make(close, cols, out)


class _PolarsCci(base_polars.SeriesOperator):
    metadata = _metadata(
        "cci",
        "Commodity Channel Index (polars backend)。",
        ["high", "low", "close", "window"],
        panel_params=("high", "low", "close"),
        tags=("CCI",),
        window_minimum=2,
    )

    def _calculate_series(self, high, low, close, window: int = 14, **_):
        return _cci_polars(high, low, close, window)


class _PolarsBias(base_polars.SeriesOperator):
    metadata = _metadata(
        "bias",
        "BIAS / 乖离率 (polars backend)。",
        ["close", "window"],
        panel_params=("close",),
        tags=("BIAS",),
    )

    def _calculate_series(self, close, window: int = 6, **_):
        return _bias_polars(close, window)


class _PolarsPsy(base_polars.SeriesOperator):
    metadata = _metadata(
        "psy",
        "Psychological Line / 心理线 (polars backend)。",
        ["close", "window"],
        panel_params=("close",),
        tags=("PSY",),
    )

    def _calculate_series(self, close, window: int = 12, **_):
        return _psy_polars(close, window)


class _PolarsTrix(base_polars.SeriesOperator):
    metadata = _metadata(
        "trix",
        "TRIX (polars backend)。",
        ["close", "window"],
        panel_params=("close",),
        tags=("TRIX",),
    )

    def _calculate_series(self, close, window: int = 12, **_):
        return _trix_polars(close, window)


def _build_polars_spec(canonical: str, kernel: str):
    from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec

    return PhysicalImplementationSpec(
        canonical=canonical,
        backend="polars",
        execution_kind=ExecutionKind.POLARS_NUMPY_KERNEL,
        materializes_full_panel=True,
        supports_nulls=True,
        supports_nan=True,
        supports_inf=True,
        implementation_source_hash=f"technical.classic_trend_v1:{kernel}:v1",
        emitter_identity=f"polars_numpy:{canonical}",
        kernel_identity=f"technical.classic_trend_v1:{kernel}",
        parameter_domain_hash=f"{canonical}:declared:v1",
        semantic_contract_hash=f"{canonical}:polars_numpy_kernel:v1",
        notes=_POLARS_NOTE,
    )


_POLARS_IMPLS = (
    ("cci", _PolarsCci, "ClassicTrendCciPolars"),
    ("bias", _PolarsBias, "ClassicTrendBiasPolars"),
    ("psy", _PolarsPsy, "ClassicTrendPsyPolars"),
    ("trix", _PolarsTrix, "ClassicTrendTrixPolars"),
)

for _canonical, _cls, _kernel in _POLARS_IMPLS:
    _cls.__name__ = _kernel
    _cls.__qualname__ = _kernel
    _cls.__module__ = __name__
    _cls._physical_spec = _build_polars_spec(_canonical, _kernel)
    base_polars.register_operator(
        name=_canonical,
        category="technical_signal",
        business_category="technical",
        canonical=_canonical,
        source=_SOURCE,
        backend="polars",
        status="production",
    )(_cls)


__all__ = [
    "Cci",
    "Bias",
    "Psy",
    "Trix",
    "_cci_pandas",
    "_bias_pandas",
    "_psy_pandas",
    "_trix_pandas",
]
