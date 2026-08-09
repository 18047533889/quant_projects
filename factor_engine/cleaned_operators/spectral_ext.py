# -*- coding: utf-8 -*-
"""Spectral-shape primitives (2026-08-08 Gemini round).

Both operators share the certified linear-detrend + Hann + zero-pad-FFT
``_periodogram`` kernel from ``spectral`` (one FFT per window, no recompute):

* ``ts_spectral_entropy``  — normalized Shannon entropy of the power spectrum,
  ``H = -sum p_i log p_i / log(N_f)`` in ``[0, 1]``.  High when energy is spread
  over many frequencies, low when a few dominate.
* ``ts_dominant_cycle_period`` — period of the dominant spectral peak,
  ``1 / f*``.  A quality gate requires the peak's share of total power to reach
  ``min_peak_share``; white noise therefore yields NaN instead of a fabricated
  pseudo-period.

Both are strict-PIT, deterministic and NaN fail-closed.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from cleaned_operators.base_polars import OperatorMetadata as PolarsMetadata
from cleaned_operators.base_polars import SeriesOperator as PolarsSeriesOperator
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators.spectral import _periodogram

_EPS = 1e-12


def _align(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    if not frames:
        return ()
    base = frames[0]
    out = [base]
    for frame in frames[1:]:
        if not frame.index.equals(base.index) or not frame.columns.equals(base.columns):
            frame = frame.reindex(index=base.index, columns=base.columns)
        out.append(frame)
    return tuple(out)


def _frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def _spectral_entropy_series(x2d: np.ndarray, window: int) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            pg = _periodogram(col[i0 : r + 1])
            if pg is None:
                continue
            p, i_max = pg
            total = float(p.sum())
            if total <= _EPS or not np.isfinite(total):
                continue
            pn = p / total
            pn = pn[pn > 0.0]
            if pn.size < 2 or i_max < 2:
                continue
            h = float(-np.sum(pn * np.log(pn)))
            out[r, c] = float(h / np.log(float(i_max)))
    return out


def _dominant_cycle_period_series(
    x2d: np.ndarray,
    window: int,
    min_peak_share: float,
) -> np.ndarray:
    rows, cols = x2d.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    w = int(window)
    gate = float(min_peak_share)
    if gate < 0.0 or gate > 1.0:
        raise ValueError("min_peak_share must be in [0, 1]")
    for c in range(cols):
        col = x2d[:, c]
        for r in range(rows):
            i0 = max(0, r - w + 1)
            chunk = col[i0 : r + 1]
            n = chunk.size
            pg = _periodogram(chunk)
            if pg is None:
                continue
            p, i_max = pg
            total = float(p.sum())
            if total <= _EPS or not np.isfinite(total):
                continue
            peak = float(p.max())
            if peak / total < gate:
                continue
            j = int(np.argmax(p))
            period = float(n) / float(j + 1)
            if period < 2.0 or not np.isfinite(period):
                continue
            out[r, c] = period
    return out


def _ts_spectral_entropy(x: pd.DataFrame, window: int = 60, **_: Any) -> pd.DataFrame:
    w = int(window)
    if w < 4:
        raise ValueError("ts_spectral_entropy requires window >= 4")
    return _frame_like(x, _spectral_entropy_series(x.to_numpy(dtype=float), w))


def _ts_dominant_cycle_period(
    x: pd.DataFrame,
    window: int = 60,
    min_peak_share: float = 0.10,
    **_: Any,
) -> pd.DataFrame:
    w = int(window)
    if w < 4:
        raise ValueError("ts_dominant_cycle_period requires window >= 4")
    return _frame_like(
        x,
        _dominant_cycle_period_series(x.to_numpy(dtype=float), w, float(min_peak_share)),
    )


# ---------------------------------------------------------------------------
# Registration (pandas + polars)
# ---------------------------------------------------------------------------
_DAILY_CANONICALS: tuple[str, ...] = (
    "ts_spectral_entropy",
    "ts_dominant_cycle_period",
)

_KERNELS: dict[str, Callable[..., pd.DataFrame]] = {
    "ts_spectral_entropy": _ts_spectral_entropy,
    "ts_dominant_cycle_period": _ts_dominant_cycle_period,
}

_PARAMS: dict[str, list[str]] = {
    "ts_spectral_entropy": ["x", "window"],
    "ts_dominant_cycle_period": ["x", "window", "min_peak_share"],
}

_CATEGORIES: dict[str, str] = {
    "ts_spectral_entropy": "spectral",
    "ts_dominant_cycle_period": "spectral",
}

# R4-95: the shared ``_periodogram`` kernel (spectral._periodogram) requires the
# trailing window to be FULLY finite (a missing value never zero-pads the FFT),
# so the window semantics are ``trailing_contiguous``.
_WINDOW_SEMANTICS: dict[str, str] = {
    "ts_spectral_entropy": "trailing_contiguous",
    "ts_dominant_cycle_period": "trailing_contiguous",
}

# R4-98: the spectrum describes the return-generating process.  Inputs must be a
# continuous price or a return series (a raw un-adjusted close injects spurious
# low-frequency energy from the level path).
_INPUT_UNITS: dict[str, dict[str, str]] = {
    "ts_spectral_entropy": {"x": "return_or_continuous_price"},
    "ts_dominant_cycle_period": {"x": "return_or_continuous_price"},
}

_SKIP = frozenset({"date", "stock_code"})


def _register() -> None:
    from cleaned_operators.base import Operator as PandasOperator
    from cleaned_operators.base import OperatorMetadata as PandasMetadata
    from cleaned_operators.base import validate_operator_call

    for canonical, fn in _KERNELS.items():
        params = _PARAMS[canonical]
        category = _CATEGORIES[canonical]

        class _PandasOp(PandasOperator):
            metadata = PandasMetadata(
                name=canonical,
                category=category,
                description=canonical,
                examples=[],
                param_names=params,
                return_type="series",
                tags=["daily", "panel", "pit_safe", "causal", "deterministic",
                      f"signature:{','.join(params)}->series",
                      "domain:spectral", "unit:level", "cost:5"],
                window_semantics=_WINDOW_SEMANTICS[canonical],
                input_units=_INPUT_UNITS[canonical],
            )

            _HANDLES_CALL_CONTRACT = True  # R5-02: routes through validate_operator_call

            def calculate(self, *args, _fn=fn, **kwargs):
                # R4-02: route the direct-``calculate`` kernel through the central
                # logical-call validator (integer / panel-axis / param checks).
                processed_args, processed_kwargs = validate_operator_call(self, args, kwargs)
                return _fn(*processed_args, **processed_kwargs)

        OperatorRegistry.register(
            _PandasOp(), canonical=canonical, backend="pandas_numpy",
            source="spectral_ext", backend_explicit=True,
        )

        class _PolarsOp(PolarsSeriesOperator):
            metadata = PolarsMetadata(name=canonical, category="spectral_ext", param_names=[])

            def _calculate_series(self, *frames, _fn=fn, **params):
                import polars as pl  # noqa: F401
                pdfs = [f.select([c for c in f.columns if c not in _SKIP]).to_pandas() for f in frames]
                out = _fn(*pdfs, **params)
                base = frames[0]
                cols = [c for c in base.columns if c not in _SKIP]
                return base.with_columns(
                    [pl.Series(name=c, values=np.asarray(out[c], dtype=np.float64)) for c in cols]
                )

        OperatorRegistry.register(
            _PolarsOp(), canonical=canonical, backend="polars",
            source="spectral_ext_polars", backend_explicit=True,
        )

    import cleaned_operators.operator_surface as _surface

    _surface.extend_extended_only(set(_DAILY_CANONICALS))


_register()
