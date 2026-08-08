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
    }
)
_NONNEGATIVE_INTEGER_PARAMS = frozenset(
    {"lag", "periods", "d", "ddof", "max_lag", "event_lag", "match_lag",
     "fit_lag", "delay", "max_shift", "max_gap", "lookback_days"}
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


def _normalise_integer(value: Any, name: str, declared_type: type | None = None) -> Any:
    if declared_type is int:
        # OperatorMetadata.param_types is the authoritative source: a param
        # declared int is validated regardless of its name (review P0-06).
        if isinstance(value, (bool, np.bool_)):
            raise OperatorParameterError(f"{name} must be an integer, not bool")
        if isinstance(value, (int, float, np.integer, np.floating)):
            if not np.isfinite(float(value)) or float(value) != float(int(value)):
                raise OperatorParameterError(f"{name} must be an integer")
            result = int(value)
            lower = 0 if name in _NONNEGATIVE_INTEGER_PARAMS else 1
            if result < lower:
                comparator = ">= 0" if lower == 0 else ">= 1"
                raise OperatorParameterError(f"{name} must be {comparator}")
            return result
        return value
    if name not in _INTEGER_PARAM_NAMES:
        return value
    if isinstance(value, (bool, np.bool_)):
        raise OperatorParameterError(f"{name} must be an integer, not bool")
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return value
    if not np.isfinite(float(value)) or float(value) != float(int(value)):
        raise OperatorParameterError(f"{name} must be an integer")
    result = int(value)
    lower = 0 if name in _NONNEGATIVE_INTEGER_PARAMS else 1
    if result < lower:
        comparator = ">= 0" if lower == 0 else ">= 1"
        raise OperatorParameterError(f"{name} must be {comparator}")
    return result


def _normalise_call(metadata: OperatorMetadata, args: tuple[Any, ...], kwargs: dict[str, Any]):
    names = list(metadata.param_names or [])
    types = metadata.param_types or {}
    processed_args = [
        _normalise_integer(
            value,
            names[index] if index < len(names) else "",
            types.get(names[index]) if index < len(names) else None,
        )
        for index, value in enumerate(args)
    ]
    processed_kwargs = {
        key: _normalise_integer(value, key, types.get(key))
        for key, value in kwargs.items()
    }
    return tuple(processed_args), processed_kwargs


def _validate_panel_axes(metadata: OperatorMetadata, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    frames = [value for value in (*args, *kwargs.values()) if isinstance(value, pd.DataFrame)]
    if not frames:
        return
    for position, frame in enumerate(frames):
        if not frame.index.is_unique:
            raise ValueError(f"{metadata.name}: input panel {position} has duplicate index values")
        if not frame.columns.is_unique:
            raise ValueError(f"{metadata.name}: input panel {position} has duplicate columns")
    if len(frames) < 2 or "allow_panel_broadcast" in set(metadata.tags or []):
        return
    base = frames[0]
    for position, frame in enumerate(frames[1:], start=1):
        if not frame.index.equals(base.index):
            raise ValueError(f"{metadata.name}: input panel {position} index is misaligned")
        if not frame.columns.equals(base.columns):
            raise ValueError(f"{metadata.name}: input panel {position} columns are misaligned")


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
        processed_args, processed_kwargs = _normalise_call(self.metadata, args, kwargs)
        _validate_panel_axes(self.metadata, processed_args, processed_kwargs)
        _validate_common_integer_relations(self.metadata, processed_args, processed_kwargs)
        valid = self.validate_params(*processed_args, **processed_kwargs)
        if valid is False:
            raise ValueError(f"{self.metadata.name}: parameter validation failed")
        return processed_args, processed_kwargs

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
