"""Narrow FE/runtime identity for FP's lagged EWMA and IIR adapters."""
from __future__ import annotations

import hashlib
import json
import sys

from factor_preprocess.errors import GovernanceError

_SCHEMA = "factor-preprocess-execution-identity/v1"
_RECIPES = {
    "ewma": "FE_COMPOSITE:long_smoothing.lagged_ewma:v1",
    "one_sided_iir_lowpass": "FE_COMPOSITE:long_ewm.lagged_iir_lowpass:v1",
}


def get_smoothing_identity(name: str, recipe_identity: str | None) -> dict:
    if name not in _RECIPES or recipe_identity != _RECIPES[name]:
        raise GovernanceError(f"FE smoothing identity is not registered for {name!r}")
    try:
        import numpy
        import pandas
        import polars
        from factor_engine.backend import long_ewm, long_smoothing, native_long_ewm
        from factor_engine.backend.composite_execution_identity import (
            IDENTITY_SCHEMA, _callable_digest,
        )
        from factor_preprocess.adapters import fe_smoothing

        if IDENTITY_SCHEMA != "factor-engine-composite-execution-identity/v1":
            raise ValueError("unsupported FE scoped identity schema")
        if (long_ewm.np is not numpy or long_ewm.pd is not pandas
                or long_smoothing.np is not numpy
                or long_smoothing.pd is not pandas
                or sys.modules.get("polars") is not polars):
            raise ValueError("FE smoothing runtime bindings are unsupported")
        # The native EWMA kernel currently imports Polars inside the function.
        # If a future version binds it in module globals, bind that exact object.
        kernel_polars = native_long_ewm.collect_lagged_ewma.__globals__.get("pl")
        if kernel_polars is not None and kernel_polars is not polars:
            raise ValueError("FE native EWMA Polars binding is unsupported")
        if name == "ewma":
            entrypoint = long_smoothing.lagged_ewma
            callables = (
                ("factor_engine.backend.long_smoothing.lagged_ewma",
                 long_smoothing.lagged_ewma),
                ("factor_engine.backend.long_ewm.lagged_ewma",
                 long_ewm.lagged_ewma),
                ("factor_engine.backend.long_ewm.halflife_to_alpha",
                 long_ewm.halflife_to_alpha),
                ("factor_engine.backend.long_ewm._validate_min_periods",
                 long_ewm._validate_min_periods),
                ("factor_engine.backend.native_long_ewm.collect_lagged_ewma",
                 native_long_ewm.collect_lagged_ewma),
            )
            adapter = fe_smoothing.execute_ewma
        else:
            entrypoint = long_ewm.lagged_iir_lowpass
            callables = (
                ("factor_engine.backend.long_ewm.lagged_iir_lowpass",
                 long_ewm.lagged_iir_lowpass),
                ("factor_engine.backend.native_long_ewm.collect_lagged_ewma",
                 native_long_ewm.collect_lagged_ewma),
            )
            adapter = fe_smoothing.execute_iir_lowpass
        fe_digests = tuple((label, _callable_digest(function))
                           for label, function in callables)
        fp_adapter_digest = _callable_digest(adapter)
        runtime_versions = (
            ("numpy", numpy.__version__), ("pandas", pandas.__version__),
            ("polars", polars.__version__),
            ("python", f"{sys.implementation.name}-{sys.version_info.major}.{sys.version_info.minor}"),
        )
        if any(type(version) is not str or not version
               for _, version in runtime_versions):
            raise ValueError("FE smoothing runtime version is unavailable")
    except (ImportError, ModuleNotFoundError, ValueError, AttributeError,
            TypeError) as exc:
        raise GovernanceError(
            "FE smoothing scoped identity is unavailable; execution identity is unbound"
        ) from exc

    scoped_payload = {
        "schema": IDENTITY_SCHEMA,
        "recipe_identity": recipe_identity,
        "callable_digests": fe_digests,
        "runtime_versions": runtime_versions,
    }
    fe_scoped_digest = hashlib.sha256(json.dumps(
        scoped_payload, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    payload = {
        "schema": _SCHEMA,
        "status": "bound",
        "transform_name": name,
        "execution_origin": "FE_COMPOSITE",
        "identity_kind": "FE_COMPOSITE_SCOPED",
        "recipe_identity": recipe_identity,
        "adapter": (
            "factor_preprocess.adapters.fe_smoothing.execute_ewma"
            if name == "ewma" else
            "factor_preprocess.adapters.fe_smoothing.execute_iir_lowpass"
        ),
        "fp_adapter_digest": fp_adapter_digest,
        "fe_scoped_digest": fe_scoped_digest,
        "fe_callable_digests": [list(item) for item in fe_digests],
        "runtime_versions": [list(item) for item in runtime_versions],
        "coverage_marker": (
            f"{entrypoint.__module__}.{entrypoint.__name__}, its named lagged EWM "
            "helpers/native kernel, FP adapter callable, and listed runtime versions; "
            "not a full transitive runtime closure"
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        **payload,
        "digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


__all__ = ["get_smoothing_identity"]
