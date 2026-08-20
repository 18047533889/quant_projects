# -*- coding: utf-8 -*-
"""Shared causal rolling kernels for the 2026-08 final operator pack.

Every helper iterates per instrument column with a trailing window and never
looks past the current row (prefix-causal).  NaN in the raw window is passed
through to the window function so each family can apply its own missing-value
policy (break / aligned-pair / drop-valid); the helpers never compress time.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    import polars as pl  # noqa: F401  (optional; bridge guards its own import)
except Exception:
    pl = None

from cleaned_operators.base import ParamSpec, ParamRole


def frame_like(template: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    """Re-wrap a numeric array on the template index/columns as float."""
    return pd.DataFrame(values, index=template.index, columns=template.columns, dtype=float)


def map_rolling(values: np.ndarray, window: int, fn: Callable[[np.ndarray], float]) -> np.ndarray:
    """Trailing-window map over a 2D panel, per column, preserving positions.

    ``fn`` receives the raw window slice (may contain NaN) and returns a float
    or ``np.nan``.  The result at row ``r`` depends only on rows ``<= r``.
    """
    rows, cols = values.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            lo = max(0, r - window + 1)
            chunk = values[lo : r + 1, c]
            out[r, c] = fn(chunk)
    return out


def map_pair_rolling(
    a: np.ndarray,
    b: np.ndarray,
    window: int,
    fn: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """Trailing-window map over two aligned panels, per column.

    ``fn(a_chunk, b_chunk)`` receives both raw slices at the same positions; it
    decides how to align (same-position pairs only) and how to treat NaN.
    """
    rows, cols = a.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    for c in range(cols):
        for r in range(rows):
            lo = max(0, r - window + 1)
            out[r, c] = fn(a[lo : r + 1, c], b[lo : r + 1, c])
    return out


def valid_values(chunk: np.ndarray) -> np.ndarray:
    return chunk[np.isfinite(chunk)]


def aligned_pairs(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same-position finite pairs from two equal-length raw window slices."""
    finite = np.isfinite(a) & np.isfinite(b)
    return a[finite].astype(float), b[finite].astype(float)


def check_window(window: int, name: str = "window") -> int:
    # NEW-004: ``int(20.9) -> 20`` is silent coercion that manufactures a false
    # search space (window=20.9 and window=20 compile to the same formula).
    # Route through the shared strict-int gate; a fractional / NaN / Inf /
    # non-numeric value is rejected, never truncated.
    from cleaned_operators.base import strict_int_param

    return strict_int_param(window, name, lower=2)


# ---------------------------------------------------------------------------
# Polars bridge: delegate to the pandas_numpy reference for exact parity.
# ---------------------------------------------------------------------------
_SKIP_PANEL = frozenset(
    {"date", "stock_code", "timestamp", "trade_date", "datetime", "__fe_time__"}
)


def _pl_to_pd(frame: Any) -> pd.DataFrame:
    """polars wide frame -> pandas panel, preserving the true time axis.

    R11 P0-03: the time column (``__fe_time__`` / ``date`` / ``timestamp`` /
    ``trade_date`` / ``datetime``) is not a feature to discard — it IS the row
    axis.  Restore it as a verified unique + monotonic ``DatetimeIndex`` so a
    pandas reference kernel doing index-aware work (session/day grouping,
    ``index.normalize()``, calendar transforms) sees the real dates instead of a
    positional ``RangeIndex``.  Positional kernels are unaffected: ``.to_numpy()``
    rows are unchanged.
    """
    cols = [c for c in frame.columns if c not in _SKIP_PANEL]
    out = frame.select(cols).to_pandas()
    time_col = next(
        (c for c in ("__fe_time__", "date", "timestamp", "trade_date", "datetime")
         if c in frame.columns),
        None,
    )
    if time_col is not None:
        values = frame[time_col].to_list()
        if values:
            idx = pd.DatetimeIndex(values)
            if not idx.is_unique:
                raise ValueError(
                    f"polars panel time axis {time_col!r} has duplicate values "
                    "(R11 P0-03 fail-closed)"
                )
            if not idx.is_monotonic_increasing:
                raise ValueError(
                    f"polars panel time axis {time_col!r} is not monotonically "
                    "increasing (R11 P0-03 fail-closed)"
                )
            out.index = idx
    return out


def _pl_rebuild(base: Any, result: Any) -> Any:
    cols = [c for c in base.columns if c not in _SKIP_PANEL]
    # Some pandas references return a raw numpy array (e.g.
    # ``cs_local_density_score``) rather than a DataFrame; wrap before indexing
    # by column name so the bridge/udf backend stays exact-parity.
    if not isinstance(result, pd.DataFrame):
        pdf = pd.DataFrame(np.asarray(result, dtype=float), columns=cols)
    else:
        pdf = result
    # P0-14: ``base.with_columns`` fundamentally requires a SHAPE-PRESERVING
    # result — one row per base row.  A frequency-transform operator (minute ->
    # daily) returns a strictly shorter panel; rebuilding it onto the minute base
    # would misalign every daily value.  Fail fast instead of silently
    # corrupting.  Such operators must NOT register a generic polars bridge (see
    # ``register_polars_bridge``); they get a purpose-built polars backend or
    # pandas-only admission.
    if len(pdf) != len(base):
        from backend.operator_errors import OperatorShapeError
        raise OperatorShapeError(
            f"polars rebuild of a non-shape-preserving result: {len(pdf)} rows "
            f"vs {len(base)} base rows — a frequency-transform operator cannot be "
            "rebuilt onto the base frame with with_columns (R11 P0-14 fail-closed)"
        )
    return base.with_columns(
        [pl.Series(name=c, values=np.asarray(pdf[c], dtype=np.float64)) for c in cols]
    )


def register_polars_bridge(canonical: str) -> None:
    """Register a polars backend that reproduces the pandas reference exactly.

    The production admission path stays on ``pandas_numpy`` (the certified
    reference backend); the polars slot is an exact-parity accelerated/bridge
    implementation used by the polars runtimes.
    """
    try:
        import polars as pl  # noqa: F401
    except Exception:
        return
    from cleaned_operators.base_polars import OperatorMetadata as PolarsMetadata
    from cleaned_operators.base_polars import SeriesOperator as PolarsSeriesOperator
    from cleaned_operators.registry import OperatorRegistry

    class _PolarsBridge(PolarsSeriesOperator):
        metadata = PolarsMetadata(name=canonical, category="pandas_bridge", param_names=[])

        def _calculate_series(self, *frames, **params):
            from cleaned_operators.registry import OperatorRegistry as _Reg

            pandas_op = _Reg.get(canonical, "pandas_numpy")
            if pandas_op is None:
                raise RuntimeError(f"pandas_numpy reference missing for {canonical}")
            pdfs = [_pl_to_pd(f) for f in frames]
            out = pandas_op.calculate(*pdfs, **params)
            return _pl_rebuild(frames[0], out)

    # Check if there's already a polars backend registered
    existing = OperatorRegistry.get(canonical, "polars")
    if existing is not None:
        # If there's already a polars backend, we need to find its source
        # For now, let's skip registration if there's already a polars backend
        # This avoids the replace issue
        return

    OperatorRegistry.register(
        _PolarsBridge(),
        canonical=canonical,
        backend="polars",
        source="pandas_bridge",
        status="implemented",
        backend_explicit=True,
    )


def register_polars_udf(canonical: str) -> None:
    """Register a polars backend that runs the pandas_numpy reference per panel.

    Follows the ``polars_dynamics._mk`` pattern: a genuine polars
    ``SeriesOperator`` that accepts ``pl.DataFrame`` panels and rebuilds the
    result as polars.  Unlike ``register_polars_bridge`` this is *not* stripped
    by the overhaul cleanup (its source does not declare a compatibility
    bridge), so the canonical keeps a usable polars backend after ``load_all``.
    """
    try:
        import polars as pl  # noqa: F401
    except Exception:
        return
    from cleaned_operators.base_polars import OperatorMetadata as PolarsMetadata
    from cleaned_operators.base_polars import SeriesOperator as PolarsSeriesOperator
    from cleaned_operators.registry import OperatorRegistry

    if canonical == "ts_regression_slope":
        from cleaned_operators.common.polars_ts_rolling import TSRegressionSlopeNative

        native = TSRegressionSlopeNative()
        if OperatorRegistry.get(canonical, backend="polars") is None:
            OperatorRegistry.register(
                native,
                canonical=canonical,
                backend="polars",
                source="factor_dsl_polars_native",
                status="implemented",
                backend_explicit=True,
            )
        return

    class _PolarsUdf(PolarsSeriesOperator):
        metadata = PolarsMetadata(name=canonical, category="polars_udf", param_names=[])

        def _calculate_series(self, *frames, **params):
            from cleaned_operators.registry import OperatorRegistry as _Reg

            pandas_op = _Reg.get(canonical, "pandas_numpy")
            if pandas_op is None:
                raise RuntimeError(f"pandas_numpy reference missing for {canonical}")
            pdfs = [_pl_to_pd(f) for f in frames]
            out = pandas_op.calculate(*pdfs, **params)
            return _pl_rebuild(frames[0], out)

    # R20: Attach param_specs from the pandas_numpy reference so the polars bridge
    # carries the same contract.  The registry backfills param_names but NOT
    # param_specs from a second registration.
    _pandas_ref = OperatorRegistry.get(canonical, "pandas_numpy")
    if _pandas_ref is not None:
        _ref_specs = getattr(getattr(_pandas_ref, "metadata", None), "param_specs", None)
        _ref_names = set(getattr(getattr(_pandas_ref, "metadata", None), "param_names", None) or [])
        _ref_aliases = set(getattr(getattr(_pandas_ref, "metadata", None), "param_aliases", None) or [])
        _udf_names = set(getattr(_PolarsUdf.metadata, "param_names", None) or [])
        _udf_aliases = set(getattr(_PolarsUdf.metadata, "param_aliases", None) or [])
        if _ref_specs:
            from cleaned_operators.base import ParamSpec as _PS
            _filtered = {k: v for k, v in _ref_specs.items()
                         if isinstance(v, _PS)
                         and (k in _ref_names or k in _ref_aliases)
                         and (k in _udf_names or k in _udf_aliases)}
            if _filtered:
                _PolarsUdf.metadata = PolarsMetadata(
                    name=canonical,
                    category="polars_udf",
                    param_names=[],
                    param_specs=_filtered,
                )
        # Copy available_at and same_session_usable from pandas reference
        _ref_available_at = getattr(getattr(_pandas_ref, "metadata", None), "available_at", None)
        _ref_same_session_usable = getattr(getattr(_pandas_ref, "metadata", None), "same_session_usable", None)
        if _ref_available_at is not None or _ref_same_session_usable is not None:
            _PolarsUdf.metadata = PolarsMetadata(
                name=canonical,
                category="polars_udf",
                param_names=[],
                available_at=_ref_available_at,
                same_session_usable=_ref_same_session_usable,
            )

    # Check if there's already a polars backend registered
    existing = OperatorRegistry.get(canonical, "polars")
    if existing is not None:
        # If there's already a polars backend, we need to find its source
        # For now, let's skip registration if there's already a polars backend
        # This avoids the replace issue
        return

    OperatorRegistry.register(
        _PolarsUdf(),
        canonical=canonical,
        backend="polars",
        source="polars_udf",
        status="implemented",
        backend_explicit=True,
    )
