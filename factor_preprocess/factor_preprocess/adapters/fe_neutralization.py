"""Lazy adapter for FE-owned OLS neutralization composites."""
from __future__ import annotations

import hashlib
import json

from factor_preprocess.errors import GovernanceError

OLS_NEUTRALIZATION_RECIPE = "FE_COMPOSITE:long_neutralization.ols_effective_rank:v1"
_EXECUTION_IDENTITY_SCHEMA = "factor-preprocess-execution-identity/v1"
_OLS_NAMES = frozenset({"ols_neutralize", "industry_neutral", "size_neutral", "dual_neutral"})


def get_fe_neutralization_executor(name: str, recipe_identity: str | None):
    """Resolve an explicitly registered neutralization recipe identity."""
    names = {"ols_neutralize", "industry_neutral", "size_neutral", "dual_neutral"}
    if name not in names or recipe_identity != OLS_NEUTRALIZATION_RECIPE:
        raise GovernanceError(f"FE neutralization recipe is not registered for {name!r}")
    try:
        from factor_engine.backend.long_neutralization import ols_effective_rank
    except (ImportError, ModuleNotFoundError):
        return None
    return execute_ols_neutralize

def get_fe_neutralization_identity(name: str, recipe_identity: str | None) -> dict:
    """Return FE's scoped OLS identity with an explicit coverage statement."""
    if name not in _OLS_NAMES or recipe_identity != OLS_NEUTRALIZATION_RECIPE:
        raise GovernanceError(f"FE neutralization identity is not registered for {name!r}")
    try:
        from factor_engine.backend.composite_execution_identity import (
            build_ols_effective_rank_identity,
        )
        fe_identity = build_ols_effective_rank_identity()
    except (ImportError, ModuleNotFoundError) as exc:
        raise GovernanceError(
            "FE OLS composite identity is unavailable; production execution is unbound"
        ) from exc

    coverage = (
        f"{fe_identity.coverage_scope}; does not cover this FP adapter implementation "
        "or the full transitive runtime closure"
    )
    payload = {
        "schema": _EXECUTION_IDENTITY_SCHEMA,
        "status": "bound",
        "transform_name": name,
        "execution_origin": "FE_COMPOSITE",
        "identity_kind": "FE_COMPOSITE_SCOPED",
        "recipe_identity": recipe_identity,
        "adapter": "factor_preprocess.adapters.fe_neutralization.execute_ols_neutralize",
        "fe_scoped_digest": fe_identity.digest,
        "coverage_marker": coverage,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        **payload,
        "digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "scoped_identity": fe_identity.to_dict(),
    }


def fp_research_fallback_identity(name: str, metadata) -> dict:
    """Identify the retained FP kernel without aliasing it to FE execution."""
    payload = {
        "schema": _EXECUTION_IDENTITY_SCHEMA,
        "status": "bound",
        "transform_name": name,
        "execution_origin": "FP_RESEARCH_FALLBACK",
        "identity_kind": "FP_RESEARCH_FALLBACK",
        "implementation_hash": metadata.implementation_hash,
        "numeric_policy_hash": metadata.numeric_policy_hash,
        "signature_hash": metadata.signature_hash,
        "coverage_marker": (
            "FP-native research fallback only; not an FE composite identity"
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        **payload,
        "digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }

def execute_ols_neutralize(
    values, exposures, date_col="date", asset_col="asset_id", value_col="value",
    min_observations=10, add_intercept=True,
):
    """Delegate without duplicating FE's numerical implementation."""
    from factor_engine.backend.long_neutralization import ols_effective_rank
    return ols_effective_rank(
        values, exposures, date_col=date_col, asset_col=asset_col,
        value_col=value_col, min_observations=min_observations,
        add_intercept=add_intercept,
    )
__all__ = [
    "OLS_NEUTRALIZATION_RECIPE", "execute_ols_neutralize",
    "get_fe_neutralization_executor",
    "get_fe_neutralization_identity", "fp_research_fallback_identity",
]
