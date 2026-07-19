from __future__ import annotations

from factor_cold_start.catalog import load_catalog
from factor_cold_start.generator import MARKET_FIELDS
from factor_cold_start.sampler import sample_factors


def test_sampler_is_deterministic_and_diverse() -> None:
    first = sample_factors(market="ashare", surface="daily", size=64, seed="alpha")
    second = sample_factors(market="ashare", surface="daily", size=64, seed="alpha")
    assert [row.factor_id for row in first] == [row.factor_id for row in second]
    assert len({row.family for row in first}) >= 12
    assert len({row.complexity for row in first}) >= 2
    assert len({row.factor_id for row in first}) == 64


def test_sampler_respects_available_fields_and_tiers() -> None:
    fields = MARKET_FIELDS["us"]["core"] | MARKET_FIELDS["us"]["derived"]
    rows = sample_factors(
        market="us",
        surface="daily",
        size=80,
        seed=7,
        available_fields=fields,
        availability_tiers=("core", "derived"),
    )
    assert all(set(row.required_fields) <= fields for row in rows)
    assert all(row.availability_tier in {"core", "derived"} for row in rows)
    assert any(row.availability_tier == "derived" for row in rows)


def test_sampler_returns_catalog_when_request_exceeds_pool() -> None:
    catalog = load_catalog("ashare", "daily")
    rows = sample_factors(market="ashare", surface="daily", size=len(catalog) + 10, seed=0)
    assert rows == catalog
