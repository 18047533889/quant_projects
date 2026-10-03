"""Explicit opt-in TreatmentRecipe plan for the FE native-Polars z-score candidate.

This module does not change registry routing or production identity. Its first
version deliberately accepts only a single stateless ``cs_zscore`` recipe.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import inspect
import json
from pathlib import Path
from numbers import Integral
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from factor_preprocess.errors import InvalidContractError
from factor_preprocess.contracts.treatment_recipe import FitBoundary, TreatmentRecipe
from factor_preprocess.transforms.cross_sectional import cs_zscore
from factor_preprocess.transforms.zscore_numeric import FINITE_ANCHOR_CENTERED_V2

_SCHEMA = "fe-native-zscore-treatment-recipe-candidate-v1"
_ALLOWED_STEP_PARAMETERS = frozenset({"axis", "ddof", "constant_value", "numeric_policy"})
_MAX_IDENTITY_SOURCE_BYTES = 1024 * 1024


def _module_source_identity(path: str | Path, *, max_bytes: int = _MAX_IDENTITY_SOURCE_BYTES):
    """Fingerprint bounded on-disk source, not imported code or its closure."""
    source_path = Path(path).resolve()
    with source_path.open("rb") as stream:
        source = stream.read(max_bytes + 1)
    if len(source) > max_bytes:
        raise ValueError(f"identity source exceeds the {max_bytes}-byte read limit")
    return {
        "path": str(source_path),
        "bytes": len(source),
        "sha256": sha256(source).hexdigest(),
    }


def _identity_digest(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(canonical.encode("utf-8")).hexdigest()


def _jsonable(value: Any):
    """Convert only lossless, deterministic identity values to JSON values."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise InvalidContractError("candidate identity mappings require string keys")
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise InvalidContractError("candidate identity cannot contain non-finite floats")
        return value
    raise InvalidContractError(
        "candidate identity cannot represent parameter type "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


@dataclass(frozen=True)
class FEZScoreRecipeStep:
    """One recipe step with complete, default-bound parameters."""

    step_id: str
    implementation_ref: str
    parameters: Mapping[str, Any]

    def apply(self, values: np.ndarray, *, max_chunk_cells: int,
              max_result_bytes: int | None) -> np.ndarray:
        from factor_preprocess.adapters.fe_zscore import (
            cs_zscore_finite_anchor_fe_native,
        )
        return cs_zscore_finite_anchor_fe_native(
            values,
            **dict(self.parameters),
            max_chunk_cells=max_chunk_cells,
            max_result_bytes=max_result_bytes,
        )


@dataclass(frozen=True)
class FEZScoreRecipePlan:
    """Auditable, opt-in execution plan distinct from the recipe spec identity."""

    spec_identity: Any
    steps: tuple[FEZScoreRecipeStep, ...]
    max_chunk_cells: int
    max_result_bytes: int | None
    _execution_identity_json: str

    @property
    def execution_identity(self) -> dict[str, Any]:
        """Return a fresh JSON-shaped identity record for this candidate route."""
        return json.loads(self._execution_identity_json)

    def apply(self, values: np.ndarray) -> np.ndarray:
        """Run the saved recipe step using only the explicit FE candidate."""
        result = values
        for step in self.steps:
            result = step.apply(
                result,
                max_chunk_cells=self.max_chunk_cells,
                max_result_bytes=self.max_result_bytes,
            )
        return result

    def run(self, values: np.ndarray) -> np.ndarray:
        """Alias emphasizing execution of the compiled candidate plan."""
        return self.apply(values)


def compile_fe_native_zscore_recipe(
    recipe: TreatmentRecipe,
    registry,
    *,
    max_chunk_cells: int = 1_000_000,
    max_result_bytes: int | None = 256 * 1024 * 1024,
) -> FEZScoreRecipePlan:
    """Compile a narrowly scoped, research-only FE z-score recipe candidate.

    The ordinary recipe compiler runs first, preserving all metadata,
    parameter-domain, fit-state and production checks. This binding does not
    admit the adapter as a registry operator and never calls the FP executor.
    """
    if not isinstance(recipe, TreatmentRecipe):
        raise TypeError("recipe must be a TreatmentRecipe")

    # This call is intentionally not replaced: it is the canonical validator.
    recipe.compile(registry)

    from factor_preprocess.adapters.fe_zscore import (
        _validated_memory_bounds,
        _validated_params,
        execution_identity as adapter_execution_identity,
    )

    chunk_limit, result_limit = _validated_memory_bounds(
        max_chunk_cells, max_result_bytes
    )
    if len(recipe.ordered_steps) != 1:
        raise InvalidContractError(
            "FE z-score recipe candidate supports exactly one cs_zscore step"
        )
    step = recipe.ordered_steps[0]
    if step.implementation_ref != "cs_zscore":
        raise InvalidContractError(
            "FE z-score recipe candidate supports only implementation_ref='cs_zscore'"
        )
    if step.requires_fit or step.state_ref is not None:
        raise InvalidContractError("FE z-score recipe candidate does not support fitted state")
    if recipe.fit_boundary != FitBoundary.EXPANDING:
        raise InvalidContractError(
            "FE z-score recipe candidate supports only the EXPANDING fit boundary"
        )
    if recipe.neutralization_spec is not None or recipe.existing_treatment_signature is not None:
        raise InvalidContractError(
            "FE z-score recipe candidate does not support extra recipe context"
        )

    metadata = registry.get(step.implementation_ref)
    if metadata is None:
        raise InvalidContractError("cs_zscore metadata disappeared after recipe validation")
    if (
        metadata.name != "cs_zscore"
        or metadata.func is not cs_zscore
        or metadata.semantic_id != "CROSS_SECTIONAL_ZSCORE:cs"
        or metadata.stage != "representation"
        or metadata.requires_fit
        or metadata.numeric_policy != FINITE_ANCHOR_CENTERED_V2
    ):
        raise InvalidContractError(
            "registered cs_zscore identity is not the supported FP finite-anchor contract"
        )
    if step.semantic_transform_id != metadata.semantic_id or step.stage != metadata.stage:
        raise InvalidContractError("recipe cs_zscore semantic/stage identity is inconsistent")
    unsupported = set(step.parameters) - _ALLOWED_STEP_PARAMETERS
    if unsupported:
        raise InvalidContractError(
            f"unsupported cs_zscore recipe parameters: {sorted(unsupported)!r}"
        )

    try:
        bound = inspect.signature(metadata.func).bind_partial(**dict(step.parameters))
        bound.apply_defaults()
    except TypeError as exc:
        raise InvalidContractError(f"invalid cs_zscore recipe parameters: {exc}") from exc
    parameters = dict(bound.arguments)
    # Match the adapter's exact numeric contract before producing a plan.
    _validated_params(
        parameters["ddof"],
        parameters["constant_value"],
        parameters["numeric_policy"],
    )
    axis = parameters["axis"]
    if axis is not None:
        axes = axis if isinstance(axis, tuple) else (axis,)
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
               for value in axes):
            raise InvalidContractError(
                "FE z-score recipe axis must be None, an integer, or a tuple of integers"
            )
        if isinstance(axis, tuple) and len({int(value) for value in axis}) != len(axis):
            raise InvalidContractError(
                "FE z-score recipe axis contains duplicate values"
            )
    frozen_parameters = MappingProxyType(parameters)
    plan_step = FEZScoreRecipeStep(
        step_id=step.step_id,
        implementation_ref=step.implementation_ref,
        parameters=frozen_parameters,
    )

    adapter_identity = adapter_execution_identity()
    recipe_module_identity = _module_source_identity(Path(__file__))
    identity_payload = {
        "schema": _SCHEMA,
        "status": "opt_in_candidate_not_production_admitted",
        "identity_kind": "recipe_execution_candidate",
        "spec_identity": {
            "spec_id": recipe.spec_identity.spec_id,
            "recipe_ref": recipe.spec_identity.recipe_ref,
            "identity": recipe.spec_identity.identity,
        },
        "steps": [{
            "step_id": plan_step.step_id,
            "implementation_ref": plan_step.implementation_ref,
            "parameters": _jsonable(parameters),
        }],
        "adapter_execution_identity": adapter_identity,
        "configured_bounds": {
            "max_chunk_cells": chunk_limit,
            "max_result_bytes": result_limit,
        },
        "fallback": "disabled",
        "candidate_module_identity": recipe_module_identity,
        "identity_scope": (
            "bounded on-disk source only; not loaded code, transitive imports, or runtime closure"
        ),
    }
    identity_payload["digest"] = _identity_digest(identity_payload)
    return FEZScoreRecipePlan(
        spec_identity=recipe.spec_identity,
        steps=(plan_step,),
        max_chunk_cells=chunk_limit,
        max_result_bytes=result_limit,
        _execution_identity_json=json.dumps(
            identity_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
    )


__all__ = [
    "FEZScoreRecipePlan",
    "FEZScoreRecipeStep",
    "compile_fe_native_zscore_recipe",
]
