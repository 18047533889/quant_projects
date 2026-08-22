# -*- coding: utf-8 -*-
"""R23 P1 certification tests: intraday_microstructure + market_microstructure + intraday_session.

Asserts that the 14 intraday operators certified by R23 P1 (intraday family)
have production_admitted=True in the DirectUseOperator verdict.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators import load_all
from mining.direct_use import build_direct_use_operator

FE_ROOT = Path(__file__).resolve().parents[2]

# The 14 intraday canonicals certified by R23 P1 (minute-shape parity evidence
# in evidence/intraday_minute_parity.json with status == "certified").
R23_CERTIFIED_CANONICALS = frozenset({
    "intra_amihud",
    "intra_bipower_variation",
    "intra_concentration",
    "intra_entropy",
    "intra_extreme_bar_return",
    "intra_jump_ratio",
    "intra_kyle_lambda_proxy",
    "intra_path_efficiency",
    "intra_realized_semivariance",
    "intra_realized_variance",
    "intra_segment_return",
    "intra_segment_volume_share",
    "intra_signed_imbalance_proxy",
    "intra_vwap_above_ratio",
})


@pytest.fixture(scope="module", autouse=True)
def load_ops():
    load_all()


def test_all_certified_have_production_admitted():
    """Every R23-certified intraday canonical must have production_admitted=True."""
    failures = []
    for canonical in sorted(R23_CERTIFIED_CANONICALS):
        catalog = OperatorRegistry._catalog.get(canonical)
        assert catalog is not None, f"{canonical} not in registry catalog"
        row = build_direct_use_operator(canonical, catalog)
        if not row.production_admitted:
            failures.append(
                f"{canonical}: production_admitted={row.production_admitted} "
                f"(prod_certified={row.production_certified}, "
                f"mining_visible={row.mining_visible})"
            )
    assert not failures, (
        f"R23-certified intraday operators with production_admitted=False:\n"
        + "\n".join(failures)
    )


def test_all_certified_have_production_certified():
    """Every R23-certified intraday canonical must have production_certified=True."""
    for canonical in sorted(R23_CERTIFIED_CANONICALS):
        catalog = OperatorRegistry._catalog.get(canonical)
        assert catalog is not None
        assert catalog.get("production_certified") is True, (
            f"{canonical}: catalog production_certified is not True"
        )


def test_all_certified_have_production_status():
    """Every R23-certified intraday canonical must have status=production."""
    for canonical in sorted(R23_CERTIFIED_CANONICALS):
        catalog = OperatorRegistry._catalog.get(canonical)
        assert catalog is not None
        assert catalog.get("status") == "production", (
            f"{canonical}: status is {catalog.get('status')!r}, expected 'production'"
        )


def test_experimental_not_certified():
    """In-scope intraday operators that are NOT certified must NOT be admitted."""
    audit_path = FE_ROOT / "docs/R23_PER_CANONICAL_AUDIT.json"
    d = json.loads(audit_path.read_text(encoding="utf-8"))
    cats = ("intraday_microstructure", "market_microstructure", "intraday_session")
    scope = {x["canonical"] for x in d if x.get("category") in cats}
    non_certified = scope - R23_CERTIFIED_CANONICALS

    failures = []
    for canonical in sorted(non_certified):
        catalog = OperatorRegistry._catalog.get(canonical)
        if catalog is None:
            continue
        row = build_direct_use_operator(canonical, catalog)
        if row.production_admitted:
            failures.append(
                f"{canonical}: production_admitted=True but not in certified set"
            )
    assert not failures, (
        f"Non-certified intraday scope operators with production_admitted=True:\n"
        + "\n".join(failures)
    )


def test_certified_count():
    """Exactly 14 intraday operators are certified."""
    catalog = OperatorRegistry._catalog
    actual = sum(
        1 for c in catalog if c in R23_CERTIFIED_CANONICALS
        and catalog[c].get("production_certified") is True
    )
    assert actual == 14, f"Expected 14 certified, got {actual}"
