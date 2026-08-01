"""Load and filter committed cold-start factor catalogs."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from .model import ColdStartFactor, ensure_unique

PACKAGE_ROOT = Path(__file__).resolve().parent
CATALOG_ROOT = PACKAGE_ROOT / "catalogs"

_RECIPE_OWNED_TA_CANONICALS = frozenset({
    "AROON",
    "AROON_up",
    "AROON_down",
    "CCI",
    "StochasticK",
    "StochasticD",
    "WilliamsR",
})


@lru_cache(maxsize=1)
def _generated_catalogs():
    from .generator import build_catalogs

    return build_catalogs(PACKAGE_ROOT.parent)


def _load_rows(path: Path) -> tuple[ColdStartFactor, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = ensure_unique(ColdStartFactor.from_dict(row) for row in data["factors"])
    if int(data.get("factor_count", -1)) != len(rows):
        raise ValueError(f"catalog count mismatch: {path}")
    return rows


def _extend_unique(
    rows: list[ColdStartFactor], additions: Iterable[ColdStartFactor]
) -> None:
    hashes = {row.formula_hash for row in rows}
    identifiers = {row.factor_id for row in rows}
    for row in additions:
        if row.formula_hash in hashes:
            continue
        if row.factor_id in identifiers:
            raise ValueError(
                f"duplicate cold-start factor id with distinct formula: {row.factor_id}"
            )
        rows.append(row)
        hashes.add(row.formula_hash)
        identifiers.add(row.factor_id)


def _extended_seed_families(market: str) -> tuple[Iterable[ColdStartFactor], ...]:
    from .candle_pattern_seeds import candle_pattern_seeds
    from .expectation_seeds import expectation_seeds
    from .fundamental_flow_seeds import fundamental_flow_seeds
    from .intraday_daily_seeds import intraday_daily_seeds
    from .structure_extra_seeds import structure_extra_seeds
    from .technical_extension_seeds import technical_extension_seeds
    from .v2_operator_seeds import v2_operator_seeds

    filtered_technical = (
        seed
        for seed in technical_extension_seeds(market)
        if not (_RECIPE_OWNED_TA_CANONICALS & set(seed.operators))
    )
    return (
        filtered_technical,
        candle_pattern_seeds(market),
        v2_operator_seeds(market),
        structure_extra_seeds(market),
        fundamental_flow_seeds(market),
        expectation_seeds(market),
        intraday_daily_seeds(market),
    )


@lru_cache(maxsize=8)
def load_catalog(
    market: str, surface: str = "daily"
) -> tuple[ColdStartFactor, ...]:
    if market not in {"ashare", "us"}:
        raise ValueError("market must be ashare or us")
    if surface not in {"daily", "extended"}:
        raise ValueError("surface must be daily or extended")

    path = CATALOG_ROOT / f"{market}_{surface}.json"
    base = (
        _load_rows(path)
        if path.is_file()
        else _generated_catalogs()[(market, surface)]
    )
    rows = list(base)

    supplemental = CATALOG_ROOT / f"{market}_{surface}_production.json"
    if supplemental.is_file():
        _extend_unique(rows, _load_rows(supplemental))

    if surface == "extended":
        for additions in _extended_seed_families(market):
            _extend_unique(rows, additions)

    return ensure_unique(rows)


def load_all_catalogs() -> tuple[ColdStartFactor, ...]:
    rows: list[ColdStartFactor] = []
    for market in ("ashare", "us"):
        for surface in ("daily", "extended"):
            rows.extend(load_catalog(market, surface))
    identifiers = [row.factor_id for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate factor ids across cold-start catalogs")
    return tuple(rows)


def filter_catalog(
    market: str,
    surface: str = "daily",
    *,
    available_fields: Iterable[str] | None = None,
    families: Iterable[str] | None = None,
    availability_tiers: Iterable[str] | None = None,
) -> tuple[ColdStartFactor, ...]:
    rows = load_catalog(market, surface)
    fields = None if available_fields is None else set(available_fields)
    family_set = None if families is None else set(families)
    tier_set = None if availability_tiers is None else set(availability_tiers)
    return tuple(
        row
        for row in rows
        if (fields is None or set(row.required_fields) <= fields)
        and (family_set is None or row.family in family_set)
        and (tier_set is None or row.availability_tier in tier_set)
    )
