from __future__ import annotations

from factor_cold_start.autofactor.provider import (
    load_ashare_daily_pack,
    load_ashare_extended_pack,
    load_us_daily_pack,
    load_us_extended_pack,
)


def test_factor_pack_providers_are_immutable_and_unique() -> None:
    packs = [
        load_ashare_daily_pack(),
        load_us_daily_pack(),
        load_ashare_extended_pack(),
        load_us_extended_pack(),
    ]
    for pack in packs:
        assert pack.factors
        assert len(pack.names()) == len(set(pack.names()))
        assert len({factor.formula_hash for factor in pack.factors}) == len(pack.factors)
        assert pack.metadata["factor_count"] == len(pack.factors)
