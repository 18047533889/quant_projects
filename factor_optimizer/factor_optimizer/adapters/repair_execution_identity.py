"""Scoped identity for direct Factor Preprocess repair kernels.

These ValueRepairPlan routes import and invoke kernels directly, bypassing the
FP transform registry. The identity therefore fingerprints the callable the
plan selects and its containing module, without claiming to cover imported
dependencies outside that module.
"""
from __future__ import annotations

import hashlib
import inspect
import platform
import sys


_DIRECT_ROUTES = {
    "rank_shape": ("factor_preprocess.transforms.repair_shapes", "rank_shape"),
    "fp_cs_rank_min": ("factor_preprocess.transforms.repair_shapes", "cross_sectional_rank"),
    "ts_rank_history": ("factor_preprocess.transforms.temporal_representation", "time_series_rank"),
    "ts_zscore_history": ("factor_preprocess.transforms.temporal_representation", "capped_time_series_zscore"),
    "capped_zscore": ("factor_preprocess.transforms.repair_shapes", "capped_zscore"),
    "tail_hinge": ("factor_preprocess.transforms.repair_shapes", "tail_hinge"),
    "robust_scale": ("factor_preprocess.transforms.repair_shapes", "robust_scale"),
}


def _selected_route(plan):
    transform = getattr(plan, "transform", None)
    if transform == "cs_rank":
        parameters = dict(getattr(plan, "parameters", ()))
        if parameters.get("method", "average") == "average":
            raise ValueError("average cs_rank is an FE route, not a direct FP route")
        return "factor_preprocess.transforms.repair_shapes", "cross_sectional_rank"
    try:
        return _DIRECT_ROUTES[transform]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"no direct FP identity route for {transform!r}") from exc


def _source_hash(source: str, *, label: str) -> str:
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"cannot certify empty {label} source")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _resolve_selected_callable(plan):
    """Resolve the same direct FP function imported by ValueRepairPlan.execute."""
    import importlib

    module_name, attribute = _selected_route(plan)
    module = importlib.import_module(module_name)
    function = getattr(module, attribute, None)
    if not callable(function):
        raise ValueError(f"selected FP route {module_name}.{attribute} is not callable")
    return module, function


def _selector_source_fingerprints() -> dict:
    """Fingerprint both the plan's dispatcher and this helper's resolver."""
    from factor_optimizer.adapters.repair_execution import ValueRepairPlan

    sources = {
        "value_repair_execute": ValueRepairPlan.execute,
        "identity_route_selector": _selected_route,
        "identity_callable_resolver": _resolve_selected_callable,
    }
    fingerprints = {}
    for label, function in sources.items():
        try:
            source = inspect.getsource(function)
        except (OSError, TypeError) as exc:
            raise ValueError(f"cannot certify {label} source") from exc
        fingerprints[label] = _source_hash(source, label=label)
    return fingerprints


def build_direct_fp_execution_identity(plan) -> dict:
    """Return a fail-closed, scoped identity for a directly selected FP kernel.

    The selected callable source catches replacement/monkeypatching of the
    imported kernel; the module source catches changes to helpers defined in
    that same module. Imported dependencies in other modules are intentionally
    outside this identity and must not be interpreted as a full runtime closure.
    """
    import numpy as np
    import pandas as pd

    module, function = _resolve_selected_callable(plan)
    try:
        function_source = inspect.getsource(function)
        module_source = inspect.getsource(module)
    except (OSError, TypeError) as exc:
        raise ValueError("cannot certify selected direct FP source") from exc
    return {
        "scope": "selected_callable_and_containing_module_source",
        "route": "direct_fp_kernel",
        "module": module.__name__,
        "qualname": getattr(function, "__qualname__", None),
        "function_source_sha256": _source_hash(function_source, label="selected function"),
        "module_source_sha256": _source_hash(module_source, label="containing module"),
        "selector_source_sha256": _selector_source_fingerprints(),
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "python_implementation": sys.implementation.name,
        },
        "coverage_note": "Imported dependencies outside the containing module are not covered.",
    }


__all__ = ["build_direct_fp_execution_identity"]
