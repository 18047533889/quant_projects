"""Lazy adapter for FE-owned OLS neutralization composites."""
from __future__ import annotations

from factor_preprocess.errors import GovernanceError

OLS_NEUTRALIZATION_RECIPE = "FE_COMPOSITE:long_neutralization.ols_effective_rank:v1"


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
]
