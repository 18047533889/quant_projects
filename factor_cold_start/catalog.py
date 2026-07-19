"""Load and filter committed cold-start factor catalogs."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from .model import ColdStartFactor, ensure_unique

PACKAGE_ROOT = Path(__file__).resolve().parent
CATALOG_ROOT = PACKAGE_ROOT / "catalogs"


@lru_cache(maxsize=1)
def _generated_catalogs():
    from .generator import build_catalogs

    return build_catalogs(PACKAGE_ROOT.parent)


@lru_cache(maxsize=8)
def load_catalog(market: str, surface: str = "daily") -> tuple[ColdStartFactor, ...]:
    if market not in {"ashare", "us"}:
        raise ValueError("market must be ashare or us")
    if surface not in {"daily", "extended"}:
        raise ValueError("surface must be daily or extended")
    path = CATALOG_ROOT / f"{market}_{surface}.json"
    if not path.is_file():
        return _generated_catalogs()[(market, surface)]
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = ensure_unique(ColdStartFactor.from_dict(row) for row in data["factors"])
    if int(data.get("factor_count", -1)) != len(rows):
        raise ValueError(f"catalog count mismatch: {path}")
    return rows


def load_all_catalogs() -> tuple[ColdStartFactor, ...]:
    rows: list[ColdStartFactor] = []
    for market in ("ashare", "us"):
        for surface in ("daily", "extended"):
            rows.extend(load_catalog(market, surface))
    ids = [row.factor_id for row in rows]
    if len(ids) != len(set(ids)):
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
