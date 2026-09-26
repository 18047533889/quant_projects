# -*- coding: utf-8 -*-
"""Shared causal rolling kernels for the 2026-08 final operator pack.

Every helper iterates per instrument column with a trailing window and never
looks past the current row (prefix-causal).  NaN in the raw window is passed
through to the window function so each family can apply its own missing-value
policy (break / aligned-pair / drop-valid); the helpers never compress time.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    import polars as pl  # noqa: F401  (optional; bridge guards its own import)
except Exception:
    pl = None

from factor_engine.backend.contracts import (  # noqa: E402  (after optional polars guard)
    ExecutionKind,
    PhysicalImplementationSpec,
)

from factor_engine.cleaned_operators.base import ParamSpec, ParamRole
from factor_engine.cleaned_operators.common._polars_bridge import (
    _is_panel_like, numeric_cols, to_pandas_panel, verify_frames_share_identity, frame_time_index,
)

_DYNAMIC_REGRESSION_DELEGATES = frozenset({
    "ts_expectile_beta_spread",
    "ts_huber_regression_coeff", "ts_huber_regression_coeff_prior",
    "ts_huber_regression_forecast_error", "ts_huber_regression_forecast_error_z",
    "ts_huber_regression_resid_z", "ts_multi_regression_adjusted_r2_prior",
    "ts_multi_regression_coeff", "ts_multi_regression_coeff_prior",
    "ts_multi_regression_coeff_stability", "ts_multi_regression_forecast_error",
    "ts_multi_regression_forecast_error_z", "ts_multi_regression_r2",
    "ts_multi_regression_r2_prior", "ts_multi_regression_resid",
    "ts_multi_regression_resid_z", "ts_quantile_beta_spread",
    "ts_quantile_regression_coeff", "ts_quantile_regression_resid",
    "ts_ridge_regression_coeff", "ts_ridge_regression_coeff_prior",
    "ts_ridge_regression_forecast_error", "ts_ridge_regression_forecast_error_z",
    "ts_ridge_regression_resid_z",
})


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
    from factor_engine.cleaned_operators.base import strict_int_param

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
    # A delegate may mutate its arguments. Never lend it a shared producer's
    # writable NumPy storage, including an already-Pandas panel.
    return (frame if isinstance(frame, pd.DataFrame) else to_pandas_panel(frame)).copy(deep=True)


def _pl_rebuild(base: Any, result: Any) -> Any:
    from factor_engine.backend.operator_errors import OperatorShapeError

    cols = numeric_cols(base)
    # Anonymous arrays have no axis proof. An implementation returning arrays
    # needs an explicit positional output adapter, not a generic relabelling.
    if not isinstance(result, pd.DataFrame):
        raise OperatorShapeError("polars delegate result has no labelled output-axis contract")
    pdf = result
    # P0-14: ``base.with_columns`` fundamentally requires a SHAPE-PRESERVING
    # result — one row per base row.  A frequency-transform operator (minute ->
    # daily) returns a strictly shorter panel; rebuilding it onto the minute base
    # would misalign every daily value.  Fail fast instead of silently
    # corrupting.  Such operators must NOT register a generic polars bridge (see
    # ``register_polars_bridge``); they get a purpose-built polars backend or
    # pandas-only admission.
    if len(pdf) != len(base):
        raise OperatorShapeError(
            f"polars rebuild of a non-shape-preserving result: {len(pdf)} rows "
            f"vs {len(base)} base rows — a frequency-transform operator cannot be "
            "rebuilt onto the base frame with with_columns (R11 P0-14 fail-closed)"
        )
    expected_index = frame_time_index(base)
    if expected_index is None:
        expected_index = pd.RangeIndex(len(base))
    if list(pdf.columns) != cols or not pdf.index.equals(expected_index):
        raise OperatorShapeError("polars delegate output time/instrument axes differ from input")
    return base.with_columns(
        [pl.Series(name=c, values=np.asarray(pdf[c], dtype=np.float64)) for c in cols]
    )


_INTRADAY_DAILY_DELEGATES = frozenset({
    "intraday_rv_signature_slope",
    "intra_impulse_event_detector",
    "intra_multiresolution_resample_reduce",
    "intra_neighbor_event_class",
    "intra_round_price_clustering_share",
    "intra_slice_mask_pair_reduce",
    "intra_slice_mask_reduce",
    "intra_state_dwell_stats",
    "intra_state_interval_moment",
    "intra_state_pair_same_slot_corr",
})


def _pl_rebuild_intraday_result(result: Any) -> Any:
    """Rebuild a labelled frequency-changing result as a fresh Polars panel.

    This is deliberately separate from ``_pl_rebuild``: a minute→daily result
    must carry its own daily index and must never be attached to the minute
    producer with ``with_columns``.
    """
    from factor_engine.backend.operator_errors import OperatorShapeError

    if not isinstance(result, pd.DataFrame):
        raise OperatorShapeError("intraday delegate result has no labelled output-axis contract")
    if not result.index.is_unique or not result.columns.is_unique:
        raise OperatorShapeError("intraday delegate result axes must be unique")
    if not isinstance(result.index, pd.DatetimeIndex):
        raise OperatorShapeError("intraday delegate result must have a DatetimeIndex")
    pdf = result.copy()
    pdf.index.name = "date"
    return pl.from_pandas(pdf.reset_index())


def _call_pandas_delegate(canonical: str, frames: tuple, params: dict):
    """Preserve Python argument binding; never reorder keyword operands."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    pandas_op = OperatorRegistry.get(canonical, "pandas_numpy")
    if pandas_op is None:
        raise RuntimeError(f"pandas_numpy reference missing for {canonical}")
    def iter_panels(value):
        if _is_panel_like(value):
            yield value
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from iter_panels(item)

    panels = [panel for value in (*frames, *params.values()) for panel in iter_panels(value)]
    if not panels:
        raise ValueError(f"{canonical}: pandas delegate received no panel input")
    if canonical in _INTRADAY_DAILY_DELEGATES:
        for panel in panels:
            if frame_time_index(panel) is None:
                raise ValueError(f"{canonical}: intraday delegate panel is missing a time axis")
    verify_frames_share_identity(canonical, *panels)
    # Reuse each conversion once within the call, but keep original positional
    # and keyword slots. The reference's normal binder validates duplicates.
    converted = {}

    def convert(value):
        if isinstance(value, tuple):
            return tuple(convert(item) for item in value)
        if isinstance(value, list):
            return [convert(item) for item in value]
        if not _is_panel_like(value):
            return value
        key = id(value)
        if key not in converted:
            converted[key] = _pl_to_pd(value)
        return converted[key]

    args = tuple(convert(v) for v in frames)
    kwargs = {k: convert(v) for k, v in params.items()}
    result = pandas_op.calculate(*args, **kwargs)
    if canonical in _INTRADAY_DAILY_DELEGATES and len(result) != len(panels[0]):
        return _pl_rebuild_intraday_result(result)
    return _pl_rebuild(panels[0], result)


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
    from factor_engine.cleaned_operators.base_polars import OperatorMetadata as PolarsMetadata
    from factor_engine.cleaned_operators.base_polars import SeriesOperator as PolarsSeriesOperator
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    class _PolarsBridge(PolarsSeriesOperator):
        metadata = PolarsMetadata(name=canonical, category="pandas_bridge", param_names=[])

        def _calculate_series(self, *frames, **params):
            return _call_pandas_delegate(canonical, frames, params)

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
    from factor_engine.cleaned_operators.base_polars import OperatorMetadata as PolarsMetadata
    from factor_engine.cleaned_operators.base_polars import SeriesOperator as PolarsSeriesOperator
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    if canonical in {"ts_dc_overshoot_ratio","ts_dc_event_rate","ts_dc_duration_asymmetry","ts_dc_overshoot_asymmetry"}:
        from factor_engine.cleaned_operators.common.directional_change_polars import make
        OperatorRegistry.register(make(canonical),canonical=canonical,backend="polars",
            source="directional_change_numpy",status="implemented",backend_explicit=True)
        return

    if canonical in {"ts_expectile","ts_expectile_beta"}:
        from factor_engine.cleaned_operators.common.expectile_native import make
        OperatorRegistry.register(make(canonical)(),canonical=canonical,backend="polars",
            source="expectile_numpy",status="implemented",backend_explicit=True)
        return

    if canonical in {"ts_hampel_filter_causal","ts_median3_causal","ts_rolling_median_causal"}:
        from factor_engine.cleaned_operators.common.despike_native import make
        OperatorRegistry.register(make(canonical)(),canonical=canonical,backend="polars",
            source="despike_numpy",status="implemented",backend_explicit=True)
        return

    if canonical in {"ohlc_corwin_schultz_spread","ts_roll_effective_spread"}:
        from factor_engine.cleaned_operators.common.spread_native import make
        OperatorRegistry.register(make(canonical)(),canonical=canonical,backend="polars",
            source="spread_numpy",status="experimental" if canonical=="ts_roll_effective_spread" else "implemented",
            backend_explicit=True)
        return

    if canonical in {"event_mark_autocorr", "event_interval_mark_coupling"}:
        from factor_engine.cleaned_operators.common.marked_event_delegate import register
        register(canonical)
        return

    from factor_engine.cleaned_operators.common import direction_risk_polars as risk_kernels
    if canonical in risk_kernels.PARAMS:
        kernel = getattr(risk_kernels, canonical)

        class _RiskKernel(PolarsSeriesOperator):
            metadata = PolarsMetadata(name=canonical, category="time_series",
                                      **risk_kernels.contract(canonical))
            _physical_spec = risk_kernels.physical_spec(canonical)
            _contract_callable = staticmethod(kernel)

            def _calculate_series(self, *frames, **params):
                return kernel(*frames, **params)

        OperatorRegistry.register(
            _RiskKernel(), canonical=canonical, backend="polars",
            source="direction_risk_polars", status="implemented", backend_explicit=True,
        )
        return

    from factor_engine.cleaned_operators.common import regression_model_polars as model_kernels
    if canonical in model_kernels.PARAMS:
        kernel=getattr(model_kernels,canonical)
        class _ModelKernel(PolarsSeriesOperator):
            metadata=PolarsMetadata(name=canonical,category="time_series",**model_kernels.contract(canonical))
            _physical_spec=model_kernels.physical_spec(canonical)
            _contract_callable=staticmethod(kernel)
            def _calculate_series(self,*frames,**params):
                return kernel(*frames,**params)
        OperatorRegistry.register(
            _ModelKernel(),canonical=canonical,backend="polars",source="regression_model_polars",
            status="implemented",backend_explicit=True)
        return

    if canonical == "ts_regression_slope":
        from factor_engine.cleaned_operators.common.polars_ts_rolling import TSRegressionSlopeNative

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

    if canonical == "session_event_recovery_score":
        from factor_engine.cleaned_operators.common.session_recovery_delegate import register
        register()
        return

    if canonical in {"ts_recovery_fraction", "ts_current_drawdown_area"}:
        from factor_engine.cleaned_operators.common.drawdown_path_delegate import make
        OperatorRegistry.register(
            make(canonical)(), canonical=canonical, backend="polars",
            source="drawdown_path_reference_polars", status="implemented",
            backend_explicit=True,
        )
        return

    if canonical in {
        "beta_residual_z", "beta_divergence_pct", "relative_strength_group_pct",
    }:
        from factor_engine.cleaned_operators.common.group_state_v1_delegate import make
        OperatorRegistry.register(
            make(canonical)(), canonical=canonical, backend="polars",
            source="group_state_v1_reference_polars", status="implemented",
            backend_explicit=True,
        )
        return

    if canonical in {"cross_event", "event_refractory"}:
        from factor_engine.cleaned_operators.common.stateful_events_delegate import make
        OperatorRegistry.register(
            make(canonical)(), canonical=canonical, backend="polars",
            source="stateful_events_reference_polars", status="implemented",
            backend_explicit=True,
        )
        return

    class _PolarsUdf(PolarsSeriesOperator):
        metadata = PolarsMetadata(name=canonical, category="polars_udf", param_names=[])
        # FE-P0-005 / R13 P0-63: explicit physical-spec declaration so production
        # classification (`canonical_polars_kind`, fail-closed) reports this slot
        # as a pandas delegate instead of `UNSUPPORTED` merely because the generic
        # bridge class declares no spec (100k GO §0 backend honesty).
        _physical_spec = PhysicalImplementationSpec(
            canonical=canonical,
            backend="polars",
            execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
            supports_lazy=False,
            supports_streaming=False,
            stateful=canonical == "hump_decay",
            materializes_full_panel=True,
            supports_nulls=True,
            supports_nan=True,
            supports_inf=True,
        )

        def _calculate_series(self, *frames, **params):
            return _call_pandas_delegate(canonical, frames, params)

    regression_contract_twins = {
        "cs_multi_resid", "cs_wls_resid", "ts_max_drawdown", "ts_nth_value",
        "ts_partial_corr", "ts_regression_tstat", "ts_trend_tstat",
        "intra_impulse_event_detector",
    }

    # R20: Attach param_specs from the pandas_numpy reference so the polars bridge
    # carries the same contract.  The registry backfills param_names but NOT
    # param_specs from a second registration.
    _pandas_ref = OperatorRegistry.get(canonical, "pandas_numpy")
    if _pandas_ref is not None:
        _ref_meta = getattr(_pandas_ref, "metadata", None)
        _ref_specs = getattr(_ref_meta, "param_specs", None)
        _ref_names = set(getattr(_ref_meta, "param_names", None) or [])
        _ref_aliases = set(getattr(_ref_meta, "param_aliases", None) or [])
        _udf_names = set(getattr(_PolarsUdf.metadata, "param_names", None) or [])
        _udf_aliases = set(getattr(_PolarsUdf.metadata, "param_aliases", None) or [])
        # R6-196: carry the availability contract from the pandas reference so
        # the polars delegate slot reports the same session-close availability.
        _ref_available_at = getattr(_ref_meta, "available_at", None)
        _ref_same_session_usable = getattr(_ref_meta, "same_session_usable", None)
        if canonical in regression_contract_twins:
            _PolarsUdf.metadata = PolarsMetadata(
                name=canonical,
                category="polars_udf",
                param_names=list(getattr(_ref_meta, "param_names", None) or []),
                param_specs=dict(_ref_specs or {}),
                panel_params=tuple(getattr(_ref_meta, "panel_params", None) or ()),
                scalar_params=tuple(getattr(_ref_meta, "scalar_params", None) or ()),
                mixed_params=tuple(getattr(_ref_meta, "mixed_params", None) or ()),
                nullable_mixed_params=tuple(getattr(_ref_meta, "nullable_mixed_params", None) or ()),
                variadic_mixed_param=getattr(_ref_meta, "variadic_mixed_param", None),
                available_at=_ref_available_at,
                same_session_usable=_ref_same_session_usable,
            )
            _udf_names = set(_PolarsUdf.metadata.param_names)
        _filtered = {}
        if _ref_specs:
            from factor_engine.cleaned_operators.base import ParamSpec as _PS
            _filtered = {k: v for k, v in _ref_specs.items()
                         if isinstance(v, _PS)
                         and (k in _ref_names or k in _ref_aliases)
                         and (k in _udf_names or k in _udf_aliases)}
        if canonical not in regression_contract_twins and (
            _filtered or _ref_available_at is not None or _ref_same_session_usable is not None
        ):
            _PolarsUdf.metadata = PolarsMetadata(
                name=canonical,
                category="polars_udf",
                param_names=[],
                param_specs=_filtered,
                available_at=_ref_available_at,
                same_session_usable=_ref_same_session_usable,
            )
        # The delegate has the same call topology as its pandas authority.
        # In particular, variadic and dynamic-input tags are binder semantics,
        # not documentation: dropping them makes an otherwise honest delegate
        # reject valid multi-panel calls before conversion.
        _PolarsUdf.metadata.param_names = list(getattr(_ref_meta, "param_names", None) or [])
        _PolarsUdf.metadata.panel_params = tuple(getattr(_ref_meta, "panel_params", None) or ())
        _PolarsUdf.metadata.panel_arity = int(getattr(_ref_meta, "panel_arity", 0) or 0)
        _PolarsUdf.metadata.scalar_params = tuple(getattr(_ref_meta, "scalar_params", None) or ())
        _PolarsUdf.metadata.mixed_params = tuple(getattr(_ref_meta, "mixed_params", None) or ())
        _PolarsUdf.metadata.nullable_mixed_params = tuple(getattr(_ref_meta, "nullable_mixed_params", None) or ())
        _PolarsUdf.metadata.variadic_mixed_param = getattr(_ref_meta, "variadic_mixed_param", None)
        _PolarsUdf.metadata.tags = list(getattr(_ref_meta, "tags", None) or [])
    if canonical in _DYNAMIC_REGRESSION_DELEGATES and _pandas_ref is not None:
        _pandas_source = inspect.getsourcefile(type(_pandas_ref))
        if not _pandas_source:
            raise RuntimeError(f"{canonical}: cannot bind delegate to Pandas source")
        _digest = hashlib.sha256(
            Path(__file__).read_bytes() + Path(_pandas_source).read_bytes()
        ).hexdigest()
        _PolarsUdf._physical_spec = PhysicalImplementationSpec(
            canonical=canonical, backend="polars",
            execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
            supports_lazy=False, supports_streaming=False, materializes_full_panel=True,
            supports_nulls=True, supports_nan=True, supports_inf=True,
            implementation_source_hash=_digest,
            kernel_identity=(
                f"{type(_pandas_ref).__module__}.{type(_pandas_ref).__qualname__}"
                "+rolling_pack._call_pandas_delegate"
            ),
        )
    if canonical == "protected_div":
        # This compiler-internal operator is registered before its final GTJA
        # pandas override.  Preserve its stable public call topology instead of
        # freezing the earlier TwoVarOperator panel-only inference.
        _PolarsUdf.metadata.param_names = ["x", "y", "epsilon", "default"]
        _PolarsUdf.metadata.panel_params = ()
        _PolarsUdf.metadata.scalar_params = ("epsilon", "default")
        _PolarsUdf.metadata.mixed_params = ("x", "y")

    # Check if there's already a polars backend registered
    existing = OperatorRegistry.get(canonical, "polars")
    if existing is not None:
        # Test bootstrap may stage legacy single-series pseudo-native classes
        # before the multi-panel regression contracts load.  For this audited
        # family, deterministically replace those classes with the exact
        # Pandas delegate whose content identity is bound above.
        if canonical in _DYNAMIC_REGRESSION_DELEGATES:
            # R68 batch-2/3: a genuine audited POLARS_NATIVE replacement (e.g.
            # r68_native_batch*) is NOT a "legacy single-series pseudo-native"
            # backend — respect "first native registrant wins" and leave the
            # slot alone instead of force-demoting it back to a pandas delegate.
            try:
                from factor_engine.backend.polars_backend_kind import (
                    PolarsImplementationKind,
                    canonical_polars_kind,
                )
                _cur = OperatorRegistry.get(canonical, "polars")
                _cur_meta = (OperatorRegistry._catalog.get(canonical, {}) or {}).get("backend_meta", {}).get("polars") or {}
                # R68 batch-11: canonical_polars_kind is a heuristic that
                # mislabels some audited native replacements (e.g. the
                # ts_quantile_regression_* family) as UNSUPPORTED — the
                # registered op"s own POLARS_NATIVE_EXPR spec is authoritative
                # (same precedent as r68_native_batch9"s atan2 note).
                _cur_native = (
                    canonical_polars_kind(canonical) is PolarsImplementationKind.POLARS_NATIVE
                    or getattr(getattr(_cur, "_physical_spec", None), "execution_kind", None)
                    is ExecutionKind.POLARS_NATIVE_EXPR
                )
                if (
                    _cur is not None
                    and _cur_native
                    and "r68_native_batch" in str(_cur_meta.get("source", ""))
                ):
                    return
            except Exception:
                pass
            OperatorRegistry.register(
                _PolarsUdf(), canonical=canonical, backend="polars",
                source="dynamic_regression_reference_polars",
                status="implemented", backend_explicit=True, replace=True,
                replacement_reason=(
                    "replace legacy single-series regression backend with "
                    "audited multi-panel pandas delegate"
                ),
            )
            return
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
