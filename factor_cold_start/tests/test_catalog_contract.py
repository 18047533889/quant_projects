from __future__ import annotations

import json
from pathlib import Path

from factor_cold_start.catalog import load_all_catalogs, load_catalog
from factor_cold_start.generator import MARKET_FIELDS, existing_formula_hashes
from factor_cold_start.model import formula_dependencies, formula_tree_key

ROOT = Path(__file__).resolve().parents[2]


def _allowed(market: str, tier: str) -> set[str]:
    out = set(MARKET_FIELDS[market]["core"])
    if tier != "core":
        out.update(MARKET_FIELDS[market].get(tier, set()))
    return out


def test_catalog_counts_and_global_ids() -> None:
    expected_minimums = {
        ("ashare", "daily"): 800,
        ("us", "daily"): 950,
        ("ashare", "extended"): 1300,
        ("us", "extended"): 1400,
    }
    for key, minimum in expected_minimums.items():
        assert len(load_catalog(*key)) >= minimum
    rows = load_all_catalogs()
    assert len(rows) >= 4500
    assert len({row.factor_id for row in rows}) == len(rows)


def test_formula_metadata_fields_and_existing_pack_dedup() -> None:
    manifest = set(json.loads((ROOT / "factor_engine" / "docs" / "field_manifest.json").read_text())["fields"])
    existing = existing_formula_hashes(ROOT)
    for row in load_all_catalogs():
        ops, fields = formula_dependencies(row.formula)
        assert ops == row.operators
        assert fields == row.required_fields
        assert set(fields) <= manifest
        assert set(fields) <= _allowed(row.market, row.availability_tier)
        assert row.formula_hash not in existing


def test_market_specific_field_isolation() -> None:
    for row in load_catalog("ashare", "daily") + load_catalog("ashare", "extended"):
        assert "adj_factor" not in row.required_fields
        assert not any(field.startswith("ret__") for field in row.required_fields)
    for row in load_catalog("us", "daily") + load_catalog("us", "extended"):
        assert "factor" not in row.required_fields
        assert "circulating_cap" not in row.required_fields
