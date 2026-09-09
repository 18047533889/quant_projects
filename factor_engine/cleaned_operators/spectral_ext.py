# -*- coding: utf-8 -*-
"""Spectral-shape primitives (2026-08-08 Gemini round; R13 split).

All operators share the certified linear-detrend + Hann + zero-pad-FFT
``_periodogram`` kernel from ``spectral`` (one FFT per window, no recompute):

* ``ts_return_spectral_entropy`` — normalized Shannon entropy of the spectrum
  of a (log)return series, ``H = -sum p_i log p_i / log(N_f)`` in ``[0, 1]``.
  High when energy is spread over many frequencies, low when a few dominate.
  A *return* spectrum describes the differenced / return-generating process.
* ``ts_detrended_level_spectral_entropy`` — the SAME entropy statistic, but on
  the spectrum of the *detrended price level* (a ``(log)price`` path), whose
  DC/low-frequency structure differs from the differenced return spectrum.
* ``ts_dominant_cycle_period`` — period of the dominant spectral peak,
  ``1 / f*``.  A quality gate requires the peak's share of total power to reach
  ``min_peak_share``; white noise therefore yields NaN instead of a fabricated
  pseudo-period.

Review #27 (P0): a detrended-price spectrum and a return spectrum are NOT the
same statistical object, so they must not share one canonical.  The two entropy
directions are distinct canonicals with per-direction typed input contracts:
``ts_return_spectral_entropy`` accepts return-typed input only and
``ts_detrended_level_spectral_entropy`` accepts a (log)price-level input only;
each fails closed (``ValueError``) on the opposite type.  The legacy generic
name ``ts_spectral_entropy`` is kept as a LIVE canonical spelling of the
return-direction spectrum (documented mapping: ``ts_spectral_entropy`` ==
``ts_return_spectral_entropy``), so the typed-IR spectral gate keeps rejecting
raw split-sensitive price for the legacy name too.  Both are strict-PIT,
deterministic and NaN fail-closed.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from factor_engine.cleaned_operators.base_polars import OperatorMetadata as PolarsMetadata
from factor_engine.cleaned_operators.base_polars import SeriesOperator as PolarsSeriesOperator
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators.spectral import _periodogram
from factor_engine.cleaned_operators.base import ParamRole

_EPS = 1e-12
# Audit #80: the shared ``_periodogram`` kernel requires >= 16 fully-finite
# rows (spectral._MIN_FINITE); windows 4..15 only ever emit NaN, so they must
# not enter the search surface.
_MIN_WINDOW = 16

# Audit #82: spectral entropy is a normalized Shannon entropy (dimensionless
# ratio); the dominant cycle period is measured in bars, not a raw level.
_OUTPUT_UNITS: dict[str, str] = {
    "ts_return_spectral_entropy": "ratio",
    "ts_spectral_entropy": "ratio",  # legacy spelling of the return direction
    "ts_detrended_level_spectral_entropy": "ratio",
    "ts_dominant_cycle_period": "bars",
}

def _require_input_kind(canonical: str, input_kind: str | None) -> None:
    """Read the typed contract, never infer financial units from sample values.

    The analyzer supplies the immediate child's semantic kind. Direct research
    callers must explicitly declare their assumption with ``input_kind``;
    unknown input is not silently certified from its observed price path.
    """
    from factor_engine.ir.types import OPERATOR_INPUT_TYPE_CONTRACTS
    allowed = OPERATOR_INPUT_TYPE_CONTRACTS[canonical][0].allowed_semantic_kinds
    if input_kind not in allowed:
        raise ValueError(f"{canonical} requires input_kind in {sorted(allowed)}; got {input_kind!r}")


def _check_window(w: int) -> int:
    if w < _MIN_WINDOW:
        raise ValueError(f"window must be >= {_MIN_WINDOW}")
    return w


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
            # Audit #58: NO partial warmup.  The front of a windowed spectrum is
            # a 16-bar spectrum, then 17, ... up to ``w`` — a different factor
            # from the full ``w``-bar spectrum.  Production mining requires the
            # FULL trailing window (NaN until it is available).  ``_periodogram``
            # additionally requires the whole window to be finite.
            if r < w - 1:
                continue
            chunk = col[r - w + 1 : r + 1]
            pg = _periodogram(chunk)
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
            # Audit #58: NO partial warmup (same contract as the entropy kernel).
            if r < w - 1:
                continue
            chunk = col[r - w + 1 : r + 1]
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


def _ts_return_spectral_entropy(x: pd.DataFrame, window: int = 60, *,
                                input_kind: str | None = None, **_: Any) -> pd.DataFrame:
    w = _check_window(int(window))
    _require_input_kind("ts_return_spectral_entropy", input_kind)
    return _frame_like(x, _spectral_entropy_series(x.to_numpy(dtype=float), w))


def _ts_detrended_level_spectral_entropy(
    x: pd.DataFrame,
    window: int = 60,
    *,
    input_kind: str | None = None,
    **_: Any,
) -> pd.DataFrame:
    w = _check_window(int(window))
    _require_input_kind("ts_detrended_level_spectral_entropy", input_kind)
    return _frame_like(x, _spectral_entropy_series(x.to_numpy(dtype=float), w))


def _ts_dominant_cycle_period(
    x: pd.DataFrame,
    window: int = 60,
    min_peak_share: float = 0.10,
    **_: Any,
) -> pd.DataFrame:
    w = _check_window(int(window))
    return _frame_like(
        x,
        _dominant_cycle_period_series(x.to_numpy(dtype=float), w, float(min_peak_share)),
    )


# ---------------------------------------------------------------------------
# Registration (pandas + polars)
# ---------------------------------------------------------------------------
# Review #27 (P0): ``ts_spectral_entropy`` is the LEGACY spelling of the
# return-direction canonical — it IS the return spectrum (same kernel, same
# typed input contract ``return_decimal``, same fail-closed price-level gate).
# The explicit review-required name ``ts_return_spectral_entropy`` is the
# canonical return-direction identity; ``ts_detrended_level_spectral_entropy``
# is the separate detrended price-level direction.  Documented mapping:
# ``ts_spectral_entropy`` == ``ts_return_spectral_entropy`` (return spectrum).
_DAILY_CANONICALS: tuple[str, ...] = (
    "ts_return_spectral_entropy",
    "ts_spectral_entropy",  # legacy spelling of the return-direction canonical
    "ts_detrended_level_spectral_entropy",
    "ts_dominant_cycle_period",
)

_KERNELS: dict[str, Callable[..., pd.DataFrame]] = {
    "ts_return_spectral_entropy": _ts_return_spectral_entropy,
    "ts_spectral_entropy": _ts_return_spectral_entropy,
    "ts_detrended_level_spectral_entropy": _ts_detrended_level_spectral_entropy,
    "ts_dominant_cycle_period": _ts_dominant_cycle_period,
}

_PARAMS: dict[str, list[str]] = {
    "ts_return_spectral_entropy": ["x", "window", "input_kind"],
    "ts_spectral_entropy": ["x", "window", "input_kind"],
    "ts_detrended_level_spectral_entropy": ["x", "window", "input_kind"],
    "ts_dominant_cycle_period": ["x", "window", "min_peak_share"],
}

_CATEGORIES: dict[str, str] = {
    "ts_return_spectral_entropy": "spectral",
    "ts_spectral_entropy": "spectral",
    "ts_detrended_level_spectral_entropy": "spectral",
    "ts_dominant_cycle_period": "spectral",
}

# R4-95: the shared ``_periodogram`` kernel (spectral._periodogram) requires the
# trailing window to be FULLY finite (a missing value never zero-pads the FFT),
# so the window semantics are ``trailing_contiguous``.
_WINDOW_SEMANTICS: dict[str, str] = {
    "ts_return_spectral_entropy": "trailing_contiguous",
    "ts_spectral_entropy": "trailing_contiguous",
    "ts_detrended_level_spectral_entropy": "trailing_contiguous",
    "ts_dominant_cycle_period": "trailing_contiguous",
}

# R4-98 / review #27 (P0): the return spectrum and the detrended-level spectrum
# are DIFFERENT statistical objects and get DIFFERENT typed input contracts.
# ``ts_return_spectral_entropy`` (and its legacy spelling
# ``ts_spectral_entropy``) accepts return-typed input only
# (``return_decimal``); ``ts_detrended_level_spectral_entropy`` accepts a
# (log)price-level input only (``continuous_price``).  Each fails closed on the
# opposite type.  A raw un-adjusted close is excluded from the level canonical
# too — a split gap injects spurious low-frequency energy from the level path.
_INPUT_UNITS: dict[str, dict[str, str]] = {
    "ts_return_spectral_entropy": {"x": "return_decimal"},
    "ts_spectral_entropy": {"x": "return_decimal"},
    "ts_detrended_level_spectral_entropy": {"x": "continuous_price"},
    "ts_dominant_cycle_period": {"x": "return_or_continuous_price"},
}

_SKIP = frozenset({"date", "stock_code"})


def _register() -> None:
    from factor_engine.cleaned_operators.base import Operator as PandasOperator
    from factor_engine.cleaned_operators.base import OperatorMetadata as PandasMetadata
    from factor_engine.cleaned_operators.base import ParamSpec as PandasParamSpec
    from factor_engine.cleaned_operators.base import validate_operator_call

    for canonical, fn in _KERNELS.items():
        params = _PARAMS[canonical]
        category = _CATEGORIES[canonical]
        output_unit = _OUTPUT_UNITS[canonical]

        # Audit #59: spectral PEAK-CONCENTRATION statistics (dominant-cycle
        # period is gated on max(P)/sum(P)) have a mechanical N dependence —
        # the expected max power share of a white-noise spectrum grows with the
        # number of frequency bins, which grows with ``window``.  ``window`` is
        # therefore NOT a free search parameter for this canonical: it is fixed
        # at its default (non-searchable).  The entropy canonicals keep a
        # searchable window (their normalized entropy is N-invariant).
        if canonical == "ts_dominant_cycle_period":
            window_spec = PandasParamSpec(
                dtype=int, min=_MIN_WINDOW, searchable=False,
                param_role=ParamRole.ESTIMATOR_RESOLUTION,
            )
        else:
            window_spec = PandasParamSpec(dtype=int, min=_MIN_WINDOW)

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
                      "domain:spectral", f"unit:{output_unit}", "cost:5"],
                window_semantics=_WINDOW_SEMANTICS[canonical],
                input_units=_INPUT_UNITS[canonical],
                output_unit=output_unit,
                param_specs={"window": window_spec, **({"input_kind": PandasParamSpec(
                    dtype=str, searchable=False,
                    choices=("PriceContinuous",) if canonical == "ts_detrended_level_spectral_entropy"
                    else ("ReturnDecimal",),
                )} if "input_kind" in params else {})},
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
            # Audit #83: canonical backend metadata must be consistent — the
            # polars slot carries the SAME param_names as the pandas/duckdb
            # metadata (not an empty list), so the binder/search grammar sees a
            # uniform parameter contract across backends.
            metadata = PolarsMetadata(name=canonical, category="spectral_ext", param_names=list(params))

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

    import factor_engine.cleaned_operators.operator_surface as _surface

    # ``ts_spectral_entropy`` stays a LIVE canonical (the legacy spelling of the
    # return direction) so the typed-IR spectral gate (``ir.analyzer._SPECTRAL_FAMILY``,
    # which is keyed on the literal resolved canonical name) keeps rejecting raw
    # split-sensitive price for it.  It is therefore NOT retracted from the
    # extended surface, and there is no alias hop away from it.
    _surface.extend_extended_only(set(_DAILY_CANONICALS))


_register()
