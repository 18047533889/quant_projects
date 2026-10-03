"""Central resolver for explicit FactorEngine composite identities."""
from __future__ import annotations

from factor_preprocess.errors import GovernanceError


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


def get_fe_composite_identity(name: str, recipe_identity: str | None) -> dict:
    """Resolve only FE composites with a specifically scoped provider."""
    if name in _OLS_NAMES:
        from factor_preprocess.adapters.fe_neutralization import (
            get_fe_neutralization_identity,
        )
        return get_fe_neutralization_identity(name, recipe_identity)
    if name == "robust_ewma":
        from factor_preprocess.adapters.fe_robust_ewma_identity import (
            get_robust_ewma_identity,
        )
        return get_robust_ewma_identity(name, recipe_identity)
    if name == "event_decay":
        from factor_preprocess.adapters.fe_smoothing_identity import (
            get_smoothing_identity,
        )
        return get_smoothing_identity(name, recipe_identity)
    if name in {"ewma", "one_sided_iir_lowpass"}:
        from factor_preprocess.adapters.fe_smoothing_identity import get_smoothing_identity
        return get_smoothing_identity(name, recipe_identity)
    raise GovernanceError(
        f"No scoped FE composite identity provider is bound for {name!r}; "
        "execution identity is not bound"
    )


__all__ = ["get_fe_composite_executor", "get_fe_composite_identity"]
