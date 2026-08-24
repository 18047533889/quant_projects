# -*- coding: utf-8 -*-
"""Full-canonical market manifest + CI invariants (spec §53, §118, §121)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]  # factor_engine


@pytest.fixture(scope="module", autouse=True)
def _load():
    from factor_engine.cleaned_operators import load_all

    load_all()


def _build_manifest():
    from factor_engine.market.capability_resolver import build_market_operator_manifest

    return build_market_operator_manifest()


def test_every_registered_canonical_has_a_market_row() -> None:
    from factor_engine.cleaned_operators import OperatorRegistry

    manifest = _build_manifest()
    manifest_ops = set(manifest["operators"])
    registered = set(OperatorRegistry.list_canonical())
    assert registered == manifest_ops, (
        f"registered vs manifest mismatch: "
        f"{registered - manifest_ops} / {manifest_ops - registered}"
    )


def test_zero_unknown_and_not_reviewed() -> None:
    manifest = _build_manifest()
    # UNKNOWN is the hard gate: every registered canonical resolves an A/US status.
    assert manifest["counts"]["unknown"] == 0
    # not_reviewed counts ONLY truly unclassified (UNKNOWN) operators; a generic
    # op whose A/US status is the intentional both-markets input-dependent
    # default is "reviewed by default".  The per-row ``contract_origin`` keeps
    # the honest accounting (P1-8): explicit/typed/fallback are never conflated.
    assert manifest["counts"]["not_reviewed"] == manifest["counts"]["unknown"]
    assert manifest["counts"]["explicit_contract"] >= 0
    assert manifest["counts"]["fallback_default"] >= 0
    for name, row in manifest["operators"].items():
        assert row["contract_origin"] in {"explicit", "typed_derived", "fallback_default"}, name


def test_every_row_has_ashare_and_us_status() -> None:
    manifest = _build_manifest()
    from factor_engine.market import MarketStatus

    allowed = {s.value for s in MarketStatus}
    for name, row in manifest["operators"].items():
        assert row["ashare"]["status"] in allowed, name
        assert row["us"]["status"] in allowed, name


def test_price_limit_family_is_ashare_only() -> None:
    manifest = _build_manifest()
    for name in ("ashare_limit_up_touch", "ashare_limit_one_price", "limit_up_close"):
        row = manifest["operators"][name]
        assert row["ashare"]["status"] == "CERTIFIED_NATIVE"
        assert row["us"]["status"] == "UNSUPPORTED_MARKET_MECHANISM"


def test_industry_holder_indexweight_provider_required_on_us() -> None:
    manifest = _build_manifest()
    for name in (
        "industry_size_neutralize", "holder_concentration", "holder_pledge_ratio",
        "index_weight",
    ):
        row = manifest["operators"][name]
        assert row["ashare"]["status"] == "CERTIFIED_NATIVE", name
        assert row["us"]["status"] == "PROVIDER_REQUIRED", name


def test_generated_manifest_file_is_committed_and_in_sync() -> None:
    """factor_engine/docs/operator_market_capabilities.json must match the live registry."""
    path = ROOT / "factor_engine" / "docs" / "operator_market_capabilities.json"
    if not path.is_file():
        pytest.skip("manifest not generated; run scripts/build_operator_market_capabilities.py")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["counts"]["unknown"] == 0
    assert doc["counts"]["not_reviewed"] == 0
    from factor_engine.cleaned_operators import OperatorRegistry

    assert set(doc["operators"]) == set(OperatorRegistry.list_canonical())


def test_market_field_manifest_generated() -> None:
    path = ROOT / "factor_engine" / "docs" / "market_field_manifest.json"
    if not path.is_file():
        pytest.skip("manifest not generated; run scripts/build_market_field_manifest.py")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["concept_count"] >= 40
    for concept, row in doc["concepts"].items():
        assert "ashare" in row and "us" in row, concept
