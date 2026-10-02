"""FP boundary for FactorEngine's scoped robust EWMA identity."""
from __future__ import annotations

import hashlib
import json

from factor_preprocess.errors import GovernanceError
from factor_preprocess.adapters.fe_smoothing import ROBUST_EWMA_RECIPE

_SCHEMA = "factor-preprocess-execution-identity/v1"


def get_robust_ewma_identity(name: str, recipe_identity: str | None) -> dict:
    if name != "robust_ewma" or recipe_identity != ROBUST_EWMA_RECIPE:
        raise GovernanceError(f"FE robust EWMA identity is not registered for {name!r}")
    try:
        from factor_engine.backend.robust_ewma_execution_identity import (
            build_robust_ewma_identity,
        )
        from factor_engine.backend.composite_execution_identity import IDENTITY_SCHEMA
        scoped = build_robust_ewma_identity()
    except (ImportError, ModuleNotFoundError, ValueError, AttributeError, TypeError) as exc:
        raise GovernanceError(
            "FE robust EWMA scoped identity is unavailable; execution identity is unbound"
        ) from exc
    try:
        scoped_identity_matches = (
            scoped.recipe_identity == recipe_identity
            and scoped.schema == IDENTITY_SCHEMA
        )
    except (AttributeError, TypeError) as exc:
        raise GovernanceError(
            "FE robust EWMA scoped identity is malformed; execution identity is unbound"
        ) from exc
    if not scoped_identity_matches:
        raise GovernanceError(
            "FE robust EWMA scoped identity schema or recipe does not match the FP binding"
        )

    payload = {
        "schema": _SCHEMA,
        "status": "bound",
        "transform_name": name,
        "execution_origin": "FE_COMPOSITE",
        "identity_kind": "FE_COMPOSITE_SCOPED",
        "recipe_identity": recipe_identity,
        "adapter": "factor_preprocess.adapters.fe_smoothing.execute_robust_ewma",
        "fe_scoped_digest": scoped.digest,
        "coverage_marker": (
            f"{scoped.coverage_scope}; does not cover the FP adapter implementation "
            "or the full transitive runtime closure"
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        **payload,
        "digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "scoped_identity": scoped.to_dict(),
    }


__all__ = ["get_robust_ewma_identity"]
