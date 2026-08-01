from __future__ import annotations

from factor_cold_start.autofactor.provider import (
    load_ashare_daily_pack,
    load_ashare_extended_pack,
    load_us_daily_pack,
    load_us_extended_pack,
)


def test_factor_pack_providers_are_immutable_and_unique() -> None:
    daily_packs = [load_ashare_daily_pack(), load_us_daily_pack()]
    research_packs = [load_ashare_extended_pack(), load_us_extended_pack()]
    for pack in daily_packs + research_packs:
        assert pack.factors
        assert len(pack.names()) == len(set(pack.names()))
        assert len({factor.formula_hash for factor in pack.factors}) == len(pack.factors)
        assert pack.metadata["factor_count"] == len(pack.factors)
        assert pack.metadata["output_frequency"] == "daily"
        assert pack.metadata["dsl_surface"] == "compat"

    for pack in daily_packs:
        assert pack.metadata["readiness"] == "production"
        assert all(factor.metadata["production_admitted"] for factor in pack.factors)
        assert all(factor.metadata["certified_backends"] for factor in pack.factors)
        assert all(factor.metadata["physical_plan_backend"] for factor in pack.factors)
        assert all(factor.metadata["physical_plan_candidates"] for factor in pack.factors)
        assert all(
            factor.metadata["physical_plan_backend"]
            in factor.metadata["physical_plan_candidates"]
            for factor in pack.factors
        )

    for pack in research_packs:
        assert pack.metadata["readiness"] == "research"
        assert all(not factor.metadata["production_admitted"] for factor in pack.factors)
        assert all(not factor.metadata["physical_plan_backend"] for factor in pack.factors)