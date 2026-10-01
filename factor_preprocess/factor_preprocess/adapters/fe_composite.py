"""Central resolver for explicit FactorEngine composite identities."""
from __future__ import annotations


_OLS_NAMES = frozenset({
    "ols_neutralize", "industry_neutral", "size_neutral", "dual_neutral",
})


def get_fe_composite_executor(name: str, recipe_identity: str | None):
    """Dispatch FE composites to a narrow, identity-checked adapter."""
    if name in _OLS_NAMES:
        from factor_preprocess.adapters.fe_neutralization import get_fe_neutralization_executor
        return get_fe_neutralization_executor(name, recipe_identity)
    from factor_preprocess.adapters.fe_smoothing import get_fe_composite_executor as get_smoothing_executor
    return get_smoothing_executor(name, recipe_identity)


__all__ = ["get_fe_composite_executor"]
