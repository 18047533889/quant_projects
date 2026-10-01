"""Shared, deterministic static candidate catalog for batch research."""
from __future__ import annotations


def optimizer_candidate_specs(config):
    """Return the optimizer's ordered, prespecified (family, parameters) grid.

    Per-factor smoothing and diagnosis-derived proposals are intentionally
    added by ``optimize_factor_batch`` after this static catalog is built.
    """
    from factor_optimizer.policy.repair_registry import RepairFamilyRegistry

    registry = RepairFamilyRegistry.default()
    families = config.families or tuple(registry.family_names)
    specs = []
    for family in sorted(families):
        prior = registry.get(family).parameter_prior.to_dict()
        if family == "NO_OP_RAW":
            continue
        if family in {"U_SHAPE_REPAIR", "INVERTED_U_REPAIR"}:
            choices = [dict(prior, center=c, power=p, asymmetry=False)
                       for c in (.35, .5, .65) for p in (1., 2.)]
        elif family == "CAUSAL_SMOOTHING":
            # Compiled per factor with its TRAIN evidence reference below.
            choices = []
        elif family == "DECAY_REFINEMENT":
            choices = [dict(prior, decay=d, half_life_relative=r)
                       for d in (.5, .8) for r in (False, True)]
        elif family == "MISSINGNESS_FRESHNESS":
            choices = [dict(prior, mode="flag")]
            choices += [dict(prior, mode="fill", freshness_window=w) for w in (1, 3, 5)]
        elif family == "TAIL_HINGE":
            choices = [dict(prior, hinge=side) for side in ("top", "bottom")]
        elif family == "TAIL_SATURATION":
            choices = [dict(prior, saturate=side) for side in ("top", "bottom", "both")]
        elif family == "REPRESENTATION_RANK":
            choices = [prior] + [dict(prior, rank_axis="ts", tie_method=tie, window=window)
                                for tie in ("average", "min") for window in (5, 10, 20)]
        elif family == "REPRESENTATION_ZSCORE":
            choices = [prior] + [dict(prior, zscore_axis="ts", window=window) for window in (5, 10, 20)]
        else:
            choices = [prior]
        specs.extend((family, p) for p in choices)
    compose = config.compose_smoothing_sign and (
        not config.families or 'SIGN_ORIENTATION' in config.families)
    declared_count = sum(2 if compose and family in {'CAUSAL_SMOOTHING', 'DECAY_REFINEMENT'}
                         else 1 for family, _ in specs)
    if declared_count > config.maximum_candidates:
        raise ValueError("candidate budget is too small for declared families")
    return specs
