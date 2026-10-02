"""Scoped execution identity for the FactorEngine robust EWMA composite."""
from __future__ import annotations

import hashlib
import json
import math
import numbers
import sys
from typing import Any

from factor_engine.backend.composite_execution_identity import (
    IDENTITY_SCHEMA,
    CompositeExecutionIdentity,
    _callable_digest,
)

ROBUST_EWMA_RECIPE = "FE_COMPOSITE:long_robust_ewm.lagged_robust_ewma:v1"


def build_robust_ewma_identity() -> CompositeExecutionIdentity:
    """Bind the live Pandas wrapper, Polars kernel, helpers, and runtimes.

    This is a scoped identity, not a transitive dependency closure or a
    certification of production admission.
    """
    from factor_engine.backend import (
        long_robust_ewm,
        native_long_robust_ewma,
    )

    wrapper = long_robust_ewm
    kernel = native_long_robust_ewma
    if kernel.pl is not wrapper.pl:
        raise ValueError("robust EWMA wrapper and kernel Polars bindings differ")
    libraries = (("numpy", wrapper.np), ("pandas", wrapper.pd), ("polars", wrapper.pl))
    versions = []
    for label, module in libraries:
        if getattr(module, "__name__", None) != label:
            raise ValueError(f"robust EWMA {label} binding is unsupported")
        version = getattr(module, "__version__", None)
        if type(version) is not str or not version:
            raise ValueError(f"robust EWMA {label} version is unavailable")
        versions.append((label, version))
    versions.append(("python", f"{sys.implementation.name}-{sys.version_info.major}.{sys.version_info.minor}"))

    callables = (
        ("factor_engine.backend.long_robust_ewm.lagged_robust_ewma",
         wrapper.lagged_robust_ewma),
        ("factor_engine.backend.native_long_robust_ewma.lagged_robust_ewma",
         kernel.lagged_robust_ewma),
        ("factor_engine.backend.native_long_robust_ewma._halflife_alpha",
         kernel._halflife_alpha),
        ("factor_engine.backend.native_long_robust_ewma._winsor_width",
         kernel._winsor_width),
        ("factor_engine.backend.native_long_robust_ewma._min_periods",
         kernel._min_periods),
        ("factor_engine.backend.native_long_robust_ewma._temporary_name",
         kernel._temporary_name),
        ("factor_engine.backend.long_robust_ewm.halflife_to_alpha",
         wrapper.halflife_to_alpha),
        ("factor_engine.backend.long_robust_ewm._validate_min_periods",
         wrapper._validate_min_periods),
    )
    callable_digests = tuple((name, _callable_digest(function)) for name, function in callables)
    constants = (
        ("native_long_robust_ewma._WINDOW", kernel._WINDOW),
        ("native_long_robust_ewma._ROLLING_MIN_SAMPLES", kernel._ROLLING_MIN_SAMPLES),
        ("native_long_robust_ewma._STD_FLOOR", kernel._STD_FLOOR),
    )
    for name, value in constants[:2]:
        if type(value) is not int or value <= 0:
            raise ValueError(f"robust EWMA constant {name} must be a positive integer")
    floor = constants[2][1]
    if (isinstance(floor, bool) or not isinstance(floor, numbers.Real)
            or not math.isfinite(float(floor)) or float(floor) <= 0.0):
        raise ValueError("robust EWMA _STD_FLOOR must be finite and positive")
    payload: dict[str, Any] = {
        "schema": IDENTITY_SCHEMA,
        "recipe_identity": ROBUST_EWMA_RECIPE,
        "callable_digests": callable_digests,
        "constants": constants,
        "runtime_versions": tuple(versions),
        "coverage_scope": (
            "live robust EWMA Pandas wrapper, Polars kernel, declared parameter helpers, "
            "window constants, and Python/NumPy/Pandas/Polars versions; "
            "not a full transitive runtime closure"
        ),
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return CompositeExecutionIdentity(
        schema=payload["schema"],
        recipe_identity=ROBUST_EWMA_RECIPE,
        digest=digest,
        callable_digests=callable_digests,
        runtime_versions=tuple(versions),
        coverage_scope=payload["coverage_scope"],
    )


__all__ = ["ROBUST_EWMA_RECIPE", "build_robust_ewma_identity"]
