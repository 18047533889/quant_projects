"""Scoped identities for FactorEngine-owned long-panel composites.

This module is intentionally FE-owned and imports no operator registry at
module import time. The first supported identity is the long OLS
neutralization recipe. Its scope covers the live entrypoint and explicitly
named helper code, function defaults/closures/annotations, and NumPy/Pandas
versions. It does not claim a full dependency closure for libraries or
transitive runtime behavior.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import math
import marshal
import sys
from types import CodeType, FunctionType
from typing import Any


IDENTITY_SCHEMA = "factor-engine-composite-execution-identity/v1"
OLS_EFFECTIVE_RANK_RECIPE = "FE_COMPOSITE:long_neutralization.ols_effective_rank:v1"


class CompositeExecutionIdentityError(ValueError):
    """Raised when a scoped composite identity cannot be certified safely."""


@dataclass(frozen=True)
class CompositeExecutionIdentity:
    """Immutable, structured identity with an explicit coverage statement."""

    schema: str
    recipe_identity: str
    digest: str
    callable_digests: tuple[tuple[str, str], ...]
    runtime_versions: tuple[tuple[str, str], ...]
    coverage_scope: str

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible representation."""
        return {
            "schema": self.schema,
            "recipe_identity": self.recipe_identity,
            "digest": self.digest,
            "callable_digests": [list(item) for item in self.callable_digests],
            "runtime_versions": [list(item) for item in self.runtime_versions],
            "coverage_scope": self.coverage_scope,
        }


def _stable_value(
    value: Any, *, active_functions: set[int], allow_annotation_class: bool = False
) -> Any:
    """Encode supported immutable state without repr/address fallbacks."""
    value_type = type(value)
    if value is None:
        return {"type": "none"}
    if value_type is bool:
        return {"type": "bool", "value": value}
    if value_type is int:
        return {"type": "int", "value": str(value)}
    if value_type is float:
        if math.isnan(value):
            encoded = "nan"
        elif math.isinf(value):
            encoded = "+inf" if value > 0 else "-inf"
        else:
            encoded = value.hex()
        return {"type": "float", "value": encoded}
    if value_type is str:
        return {"type": "str", "value": value}
    if value_type is bytes:
        return {"type": "bytes", "value": base64.b64encode(value).decode("ascii")}
    # Resolved annotations commonly contain classes such as pandas.DataFrame,
    # as well as builtins. Bind those by qualified name; the library version
    # is separately included in the scoped runtime identity.
    if value_type is type:
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", None)
        if (allow_annotation_class and type(module) is str
                and type(qualname) is str):
            return {"type": "class", "module": module, "qualname": qualname}
        raise CompositeExecutionIdentityError(
            "unsupported dynamic state in callable identity: class value"
        )
    if value_type is tuple:
        return {
            "type": "tuple",
            "items": [_stable_value(item, active_functions=active_functions,
                                     allow_annotation_class=allow_annotation_class)
                      for item in value],
        }
    if value_type is frozenset:
        items = [_stable_value(item, active_functions=active_functions,
                               allow_annotation_class=allow_annotation_class)
                 for item in value]
        items.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
        return {"type": "frozenset", "items": items}
    if isinstance(value, FunctionType):
        return {"type": "function", "identity": _function_payload(
            value, active_functions=active_functions)}
    raise CompositeExecutionIdentityError(
        "unsupported dynamic state in callable identity: "
        f"{value_type.__module__}.{value_type.__qualname__}"
    )


def _canonical_code(code: CodeType) -> CodeType:
    """Strip source location metadata while preserving executable code."""
    constants = tuple(
        _canonical_code(item) if isinstance(item, CodeType) else item
        for item in code.co_consts
    )
    replacements: dict[str, Any] = {
        "co_filename": "",
        "co_firstlineno": 0,
        "co_consts": constants,
    }
    # The line table is debug metadata. Removing it prevents source relocation
    # from changing identity while bytecode and exception tables remain bound.
    if hasattr(code, "co_linetable"):
        replacements["co_linetable"] = b""
    elif hasattr(code, "co_lnotab"):
        replacements["co_lnotab"] = b""
    return code.replace(**replacements)


def _function_payload(function: FunctionType, *, active_functions: set[int]) -> dict[str, Any]:
    if type(function) is not FunctionType or type(function.__code__) is not CodeType:
        raise CompositeExecutionIdentityError("identity requires a Python function")
    marker = id(function)
    if marker in active_functions:
        raise CompositeExecutionIdentityError("recursive callable state is unsupported")
    if not all(type(item) is str for item in (
        function.__module__, function.__qualname__, function.__name__
    )):
        raise CompositeExecutionIdentityError("callable naming metadata is unsupported")

    active_functions.add(marker)
    try:
        code_hash = hashlib.sha256(marshal.dumps(_canonical_code(function.__code__))).hexdigest()
        defaults = _stable_value(function.__defaults__, active_functions=active_functions)
        keyword_defaults = tuple(sorted(
            (name, _stable_value(value, active_functions=active_functions))
            for name, value in (function.__kwdefaults__ or {}).items()
        ))
        annotations = tuple(sorted(
            (name, _stable_value(value, active_functions=active_functions,
                                 allow_annotation_class=True))
            for name, value in (function.__annotations__ or {}).items()
        ))
        closure_cells = []
        for name, cell in zip(function.__code__.co_freevars, function.__closure__ or ()):
            try:
                content = cell.cell_contents
            except ValueError:
                raise CompositeExecutionIdentityError("empty closure cells are unsupported")
            closure_cells.append((
                name, _stable_value(content, active_functions=active_functions)
            ))
        return {
            "module": function.__module__,
            "qualname": function.__qualname__,
            "name": function.__name__,
            "code_sha256": code_hash,
            "defaults": defaults,
            "keyword_defaults": keyword_defaults,
            "annotations": annotations,
            "closure": tuple(closure_cells),
        }
    finally:
        active_functions.remove(marker)


def _callable_digest(function: Any) -> str:
    if type(function) is not FunctionType:
        raise CompositeExecutionIdentityError("identity requires a Python function")
    payload = _function_payload(function, active_functions=set())
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_ols_effective_rank_identity() -> CompositeExecutionIdentity:
    """Identify the live OLS composite entrypoint and its declared local helper.

    The helper references are resolved from the currently loaded FE module on
    each call, so replacing either callable changes the digest. This provider
    deliberately excludes a full Pandas/NumPy/FE transitive implementation
    closure; those inputs must not be inferred from one source-file hash.
    """
    from factor_engine.backend import long_neutralization

    numpy_module = long_neutralization.np
    pandas_module = long_neutralization.pd
    if (getattr(numpy_module, "__name__", None) != "numpy"
            or getattr(pandas_module, "__name__", None) != "pandas"):
        raise CompositeExecutionIdentityError("OLS NumPy/Pandas bindings are unsupported")
    versions = []
    for label, module in (("numpy", numpy_module), ("pandas", pandas_module)):
        version = getattr(module, "__version__", None)
        if type(version) is not str or not version:
            raise CompositeExecutionIdentityError(f"{label} version identity is unavailable")
        versions.append((label, version))
    versions.append(("python", f"{sys.implementation.name}-{sys.version_info.major}.{sys.version_info.minor}"))

    callables = (
        ("factor_engine.backend.long_neutralization.ols_effective_rank",
         long_neutralization.ols_effective_rank),
        ("factor_engine.backend.long_neutralization._private_name",
         long_neutralization._private_name),
    )
    callable_digests = tuple((name, _callable_digest(function))
                             for name, function in callables)
    payload = {
        "schema": IDENTITY_SCHEMA,
        "recipe_identity": OLS_EFFECTIVE_RANK_RECIPE,
        "callable_digests": callable_digests,
        "runtime_versions": tuple(versions),
        "coverage_scope": (
            "live OLS entrypoint and _private_name helper code/defaults/closures/annotations; "
            "NumPy, Pandas, and Python versions; not a full transitive runtime closure"
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return CompositeExecutionIdentity(
        schema=IDENTITY_SCHEMA,
        recipe_identity=OLS_EFFECTIVE_RANK_RECIPE,
        digest=digest,
        callable_digests=callable_digests,
        runtime_versions=tuple(versions),
        coverage_scope=payload["coverage_scope"],
    )


__all__ = (
    "CompositeExecutionIdentity",
    "CompositeExecutionIdentityError",
    "IDENTITY_SCHEMA",
    "OLS_EFFECTIVE_RANK_RECIPE",
    "build_ols_effective_rank_identity",
)
