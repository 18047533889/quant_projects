# -*- coding: utf-8 -*-
"""Pandas/NumPy operator base classes and registration helpers.

Runtime contracts are enforced centrally:
- only parameters declared as integer controls are normalised to ``int``;
- booleans cannot masquerade as windows/lags;
- all panel axes must be unique;
- multi-panel inputs must have exactly matching index/columns unless an
  operator explicitly declares the ``allow_panel_broadcast`` tag;
- ``validate_params`` is executed for every operator call.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from backend.operator_errors import OperatorParameterError

_INTEGER_PARAM_NAMES = frozenset(
    {
        # legacy core names
        "window", "period", "periods", "d", "lag", "n", "m", "k",
        "min_periods", "max_periods", "max_lookback", "periods_per_year",
        "fast_period", "slow_period", "signal_period", "bins", "buckets",
        "order", "degree", "ddof",
        # window-ish names (review P0-06): without central validation these were
        # silently ``int(value)``-truncated inside operator kernels, so
        # fast_window=5.1 / 5.2 / 5.9 all compiled to the same formula — a false
        # search space.  `param_types` in OperatorMetadata is authoritative when
        # set; this name whitelist is the fallback for operators that do not
        # declare it.
        "fast_window", "slow_window", "signal_window", "short_window",
        "long_window", "medium_window", "tenkan_window", "kijun_window",
        "senkou_b_window", "er_window", "atr_window", "ema_window", "adl_window",
        "left_window", "right_window", "outer_window", "inner_window",
        "recent_window", "prior_window", "reference_window", "history_window",
        "old_window", "lookback_window", "baseline_window", "smooth_window",
        "scale_window", "path_window", "window_periods", "average_periods",
        # periods-ish
        "short_periods", "long_periods", "growth_periods", "compare_periods",
        # lag / count / history
        "max_lag", "event_lag", "match_lag", "fit_lag", "lookback_days",
        "history_days", "max_gap", "max_shift", "max_interval", "n_updates",
        "min_updates", "min_transitions", "min_events", "min_peers", "min_pairs",
        "min_patterns", "min_obs", "min_scale", "max_scale", "n_scales", "k_max",
        "min_valid_lags", "min_reference_days", "max_run",
        # structure / search
        "n_components", "embedding_dim", "bucket_count", "n_bins", "n_slots",
        "n_segments", "n_patterns", "steps", "cooldown", "max_spacing",
        "min_spacing", "body_window", "shadow_window", "points", "min_count",
        "top_k", "delay", "grid", "sampling",
        # review #4 integer-parameter sweep: names that kernels were still
        # ``int(value)``-truncating without central validation.  5.9 -> 5 and
        # 3.4 -> 3 compiled to the SAME factor, manufacturing a false search
        # space.  A kernel that legitimately needs a fractional value for one of
        # these must declare ``param_types`` (float) / ``ParamSpec(dtype=float)``.
        "confirmation", "tolerance", "horizon", "min_anchors",
        "min_tail_count", "min_bin_count", "cutoff", "block", "tau",
        "max_iter", "n_iter", "segments", "min_segments", "max_segments",
        "left", "right", "up_count", "down_count", "n_levels", "n_buckets",
    }
)
_NONNEGATIVE_INTEGER_PARAMS = frozenset(
    {"lag", "periods", "d", "ddof", "max_lag", "event_lag", "match_lag",
     "fit_lag", "delay", "max_shift", "max_gap", "lookback_days", "tau",
     "tolerance", "block"}
)


@dataclass(frozen=True)
class ParamSpec:
    """Authoritative contract for a single operator parameter (review #4 R4-01).

    Replaces the name-whitelist heuristics: every operator entering runtime is
    validated against its declared ``ParamSpec``, so a ``5.9`` never silently
    truncates to ``5`` and an inactive parameter never shows up in the search
    grammar.  Falls back to the legacy name whitelist when a parameter has no
    spec (operators that do not declare contracts yet).
    """

    dtype: type | None = None             # int / float / str / bool
    min: float | int | None = None
    max: float | int | None = None
    choices: tuple | None = None          # EnumSpec: canonical allowed values
    searchable: bool = True               # False -> excluded from AlphaProbe/GP grammar
    active_when: tuple | None = None      # (param_name, allowed_values): conditional activation
    history_semantics: str | None = None  # exact_rows/max_rows/finite_observations/
    #                                      # trailing_contiguous/report_events/session_slots


# Typed broadcast tags (review #4 R4-99): a bare ``allow_panel_broadcast`` is a
# blanket waiver of multi-panel axis parity.  Operators that legitimately
# broadcast must declare the SPECIFIC shape they need so a daily scalar cannot
# silently stretch across symbols/sessions.
_TYPED_BROADCAST_TAGS = frozenset(
    {
        "daily_to_minute_broadcast",       # daily scalar/row -> minute panel, same day
        "scalar_to_cross_section_broadcast",  # scalar -> every instrument column
        "same_trading_date_broadcast",     # row aligned on the trading-date level only
        "session_boundary_broadcast",      # session summary row -> each minute slot
    }
)


@dataclass
class OperatorMetadata:
    name: str
    category: str
    description: str = ""
    examples: List[str] = field(default_factory=list)
    param_names: List[str] = field(default_factory=list)
    param_types: Dict[str, type] = field(default_factory=dict)
    return_type: str = "series"
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    # Optional field-semantic metadata.  Existing operators may omit these.
    input_fields: List[str] = field(default_factory=list)
    output_field: str | None = None
    input_units: Dict[str, str] = field(default_factory=dict)
    output_unit: str | None = None
    compatible_units: Dict[str, tuple[str, ...]] = field(default_factory=dict)
    # review #4 R4-01: per-parameter authoritative contracts (dtype/min/max/
    # choices/searchable/active_when/history_semantics).  When a name has a spec
    # here, it OVERRIDES the legacy name whitelist in both directions — a
    # declared float stays float even for a window-ish name, and a spec'd int is
    # validated even for a name outside the whitelist.
    param_specs: Dict[str, ParamSpec] = field(default_factory=dict)
    # review #4 R4-95: meaning of the window-like parameters in this operator.
    window_semantics: str | None = None


def _normalise_integer(
    value: Any,
    name: str,
    declared_type: type | None = None,
    spec: ParamSpec | None = None,
) -> Any:
    """Validate one parameter value against its authoritative contract.

    Resolution order (review #4 R4-01): ``ParamSpec.dtype`` first, then
    ``OperatorMetadata.param_types``, then the legacy name whitelist.  A spec
    may also carry ``min``/``max``/``choices`` that are enforced regardless of
    dtype.
    """
    lower_default = 0 if name in _NONNEGATIVE_INTEGER_PARAMS else 1
    lower = spec.min if spec is not None and spec.min is not None else lower_default
    upper = spec.max if spec is not None and spec.max is not None else None
    choices = spec.choices if spec is not None and spec.choices else None

    def _check_int(result: int) -> int:
        if result < lower:
            raise OperatorParameterError(f"{name} must be >= {lower}")
        if upper is not None and result > upper:
            raise OperatorParameterError(f"{name} must be <= {upper}")
        if choices is not None and result not in choices:
            raise OperatorParameterError(
                f"{name}={result} is not an allowed choice {list(choices)}"
            )
        return result

    is_int_declared = declared_type is int or (
        spec is not None and spec.dtype is int
    )
    if is_int_declared:
        # Authoritative source: a param declared int is validated regardless of
        # its name, so `int(5.9) -> 5` can never slip through (review P0-06 / R4-01).
        if isinstance(value, (bool, np.bool_)):
            raise OperatorParameterError(f"{name} must be an integer, not bool")
        if isinstance(value, (int, float, np.integer, np.floating)):
            if not np.isfinite(float(value)) or float(value) != float(int(value)):
                raise OperatorParameterError(f"{name} must be an integer")
            return _check_int(int(value))
        return value
    if spec is not None and spec.dtype is float:
        # Explicitly declared float: validate numeric and bounds, never truncate.
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            if not np.isfinite(numeric):
                raise OperatorParameterError(f"{name} must be finite")
            if lower is not None and numeric < lower:
                raise OperatorParameterError(f"{name} must be >= {lower}")
            if upper is not None and numeric > upper:
                raise OperatorParameterError(f"{name} must be <= {upper}")
        return value
    if spec is not None and spec.choices is not None and name not in _INTEGER_PARAM_NAMES:
        # EnumSpec: string/bool/numeric choices are canonicalized and checked.
        if isinstance(value, (bool, np.bool_)) and True not in choices and False not in choices:
            raise OperatorParameterError(f"{name} must be one of {list(choices)}")
        if value not in choices:
            raise OperatorParameterError(
                f"{name}={value!r} is not an allowed choice {list(choices)}"
            )
        return value
    if name not in _INTEGER_PARAM_NAMES:
        return value
    if isinstance(value, (bool, np.bool_)):
        raise OperatorParameterError(f"{name} must be an integer, not bool")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return value
    if not np.isfinite(float(value)) or float(value) != float(int(value)):
        raise OperatorParameterError(f"{name} must be an integer")
    return _check_int(int(value))


def _normalise_call(metadata: OperatorMetadata, args: tuple[Any, ...], kwargs: dict[str, Any]):
    # ``getattr`` keeps this compatible with the parallel polars metadata class
    # (base_polars.OperatorMetadata has no param_specs / param_types on every
    # instance); both classes share the param_names contract.
    names = list(getattr(metadata, "param_names", None) or [])
    types = getattr(metadata, "param_types", None) or {}
    specs = getattr(metadata, "param_specs", None) or {}
    processed_args = [
        _normalise_integer(
            value,
            names[index] if index < len(names) else "",
            types.get(names[index]) if index < len(names) else None,
            specs.get(names[index]) if index < len(names) else None,
        )
        for index, value in enumerate(args)
    ]
    processed_kwargs = {
        key: _normalise_integer(value, key, types.get(key), specs.get(key))
        for key, value in kwargs.items()
    }
    return tuple(processed_args), processed_kwargs


def _frame_instruments(frame: pd.DataFrame) -> set | None:
    """Instrument set of a panel, or None when there is no instrument level."""
    index = frame.index
    if isinstance(index, pd.MultiIndex) and "instrument" in index.names:
        return set(index.get_level_values("instrument"))
    return None


def _validate_panel_axes(metadata: OperatorMetadata, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    frames = [value for value in (*args, *kwargs.values()) if isinstance(value, pd.DataFrame)]
    if not frames:
        return
    for position, frame in enumerate(frames):
        if not frame.index.is_unique:
            raise ValueError(f"{metadata.name}: input panel {position} has duplicate index values")
        if not frame.columns.is_unique:
            raise ValueError(f"{metadata.name}: input panel {position} has duplicate columns")
    if len(frames) < 2:
        return
    tags = set(metadata.tags or [])
    if tags & _TYPED_BROADCAST_TAGS or "allow_panel_broadcast" in tags:
        # Review #4 R4-99: a bare ``allow_panel_broadcast`` is the legacy blanket
        # waiver; typed tags (daily_to_minute_broadcast / scalar_to_cross_section
        # _broadcast / same_trading_date_broadcast / session_boundary_broadcast)
        # declare the SPECIFIC shape.  Either way the instrument identity is
        # enforced: a panel cannot be broadcast across a different instrument set.
        base_cols = frames[0].shape[1]
        base_instruments = _frame_instruments(frames[0])
        for position, frame in enumerate(frames[1:], start=1):
            if frame.shape[1] != base_cols:
                raise ValueError(
                    f"{metadata.name}: broadcast input panel {position} has "
                    f"{frame.shape[1]} columns != base {base_cols}"
                )
            if base_instruments is not None:
                other = _frame_instruments(frame)
                if other is not None and not other.issubset(base_instruments):
                    raise ValueError(
                        f"{metadata.name}: broadcast input panel {position} carries "
                        f"instruments outside the base panel (daily->minute or "
                        f"cross-symbol broadcast across different symbols is not allowed)"
                    )
        return
    base = frames[0]
    for position, frame in enumerate(frames[1:], start=1):
        if not frame.index.equals(base.index):
            raise ValueError(f"{metadata.name}: input panel {position} index is misaligned")
        if not frame.columns.equals(base.columns):
            raise ValueError(f"{metadata.name}: input panel {position} columns are misaligned")


def validate_operator_call(
    operator: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Central logical-call validator (review #4 R4-02).

    Every operator — ``SeriesOperator``/``PandasOperator``, the Polars bridge,
    the DuckDB operator, and modules that implement ``calculate`` directly —
    must pass through this gate so integer validation (``ParamSpec`` /
    ``param_types`` / name whitelist), panel-axis alignment (incl. typed
    broadcast), ``validate_params`` and common parameter relations are enforced
    uniformly.  Registry/dispatch layers call this instead of trusting each
    module to remember to call ``_prepare_call``.
    """
    metadata = getattr(operator, "metadata", None)
    if metadata is None:
        raise ValueError(f"{operator!r} has no metadata; cannot validate call")
    processed_args, processed_kwargs = _normalise_call(metadata, args, kwargs)
    _validate_panel_axes(metadata, processed_args, processed_kwargs)
    _validate_common_integer_relations(metadata, processed_args, processed_kwargs)
    valid = operator.validate_params(*processed_args, **processed_kwargs)
    if valid is False:
        raise ValueError(f"{metadata.name}: parameter validation failed")
    return processed_args, processed_kwargs


def _validate_common_integer_relations(metadata: OperatorMetadata, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    names = list(metadata.param_names or [])
    bound = {name: args[index] for index, name in enumerate(names[: len(args)])}
    bound.update(kwargs)
    window = bound.get("window")
    minimum = bound.get("min_periods")
    if isinstance(window, int) and isinstance(minimum, int) and minimum > window:
        raise ValueError(f"{metadata.name}: min_periods must not exceed window")
    k = bound.get("k")
    if isinstance(window, int) and isinstance(k, int) and k > window:
        raise ValueError(f"{metadata.name}: k must not exceed window")


class Operator(ABC):
    metadata: OperatorMetadata

    @abstractmethod
    def calculate(self, *args, **kwargs) -> pd.DataFrame:
        pass

    def validate_params(self, *args, **kwargs) -> bool:
        return True

    def _prepare_call(self, args: tuple[Any, ...], kwargs: dict[str, Any]):
        return validate_operator_call(self, args, kwargs)

    def __repr__(self):
        return f"<Operator: {self.metadata.name}>"

    def __str__(self):
        return f"{self.metadata.name}: {self.metadata.description}"


class SeriesOperator(Operator):
    def calculate(self, *args, **kwargs) -> pd.DataFrame:
        processed_args, processed_kwargs = self._prepare_call(args, kwargs)
        return self._calculate_series(*processed_args, **processed_kwargs)

    @abstractmethod
    def _calculate_series(self, *args, **kwargs) -> pd.DataFrame:
        pass


class ScalarOperator(Operator):
    def calculate(self, *args, **kwargs) -> Any:
        processed_args, processed_kwargs = self._prepare_call(args, kwargs)
        return self._calculate_scalar(*processed_args, **processed_kwargs)

    @abstractmethod
    def _calculate_scalar(self, *args, **kwargs) -> Any:
        pass


class TransformOperator(Operator):
    def _calculate_series(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def calculate(self, x: pd.DataFrame, **kwargs) -> pd.DataFrame:
        processed_args, processed_kwargs = self._prepare_call((x,), kwargs)
        return self._calculate_series(processed_args[0], **processed_kwargs)


class TwoVarOperator(Operator):
    def _calculate_series(self, x: pd.DataFrame, y: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def calculate(self, x: pd.DataFrame, y: pd.DataFrame, **kwargs) -> pd.DataFrame:
        processed_args, processed_kwargs = self._prepare_call((x, y), kwargs)
        return self._calculate_series(processed_args[0], processed_args[1], **processed_kwargs)


def register_operator(
    name: str = None,
    category: str = "general",
    business_category: str = "",
    canonical: str = "",
    source: str = "",
    backend: str | None = None,
    status: str = "implemented",
    replace: bool = False,
    replacement_reason: str = "",
):
    """Instantiate and register an operator class.

    ``replace``/``replacement_reason`` are the explicit escape hatch for the
    registry's silent-overwrite ban (P0-31): a compatibility/override layer
    re-registering an existing canonical+backend must declare it, or load_all
    raises.
    """

    def decorator(cls):
        instance = cls()
        if name:
            instance.metadata.name = name
        if category:
            instance.metadata.category = category
        if business_category:
            instance.metadata.business_category = business_category
        if backend is not None:
            effective_backend = backend
            backend_explicit = True
        elif "Polars" in cls.__name__:
            effective_backend = "polars"
            backend_explicit = False
        else:
            effective_backend = "pandas_numpy"
            backend_explicit = False
        canon = canonical or (name if name else cls.__name__)
        from cleaned_operators.registry import OperatorRegistry

        name_aliases = [name] if name and name != canon else None
        OperatorRegistry.register(
            instance,
            canonical=canon,
            backend=effective_backend,
            source=source or "factor_dsl_np",
            aliases=name_aliases,
            status=status,
            backend_explicit=backend_explicit,
            replace=replace,
            replacement_reason=replacement_reason,
        )
        return cls

    return decorator
