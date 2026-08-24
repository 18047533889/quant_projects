# -*- coding: utf-8 -*-
"""R23 P1 certification tests: time_series + price_structure + price_volume_extension.

Asserts that the 18 operators certified by R23 P1 have production_admitted=True
in the DirectUseOperator verdict.
"""
from __future__ import annotations

import pytest
from factor_engine.cleaned_operators.registry import OperatorRegistry
from factor_engine.cleaned_operators import load_all
from factor_engine.mining.direct_use import build_direct_use_operator

# The 18 canonicals certified by R23 P1 certification
R23_CERTIFIED_CANONICALS = frozenset({
    "ts_autocorr", "ts_beta", "ts_corr", "ts_cov",
    "ts_delay", "ts_delta", "ts_log_return",
    "ts_max", "ts_mean", "ts_median", "ts_min",
    "ts_pct", "ts_rank", "ts_sharpe",
    "ts_std", "ts_sum", "ts_var", "ts_zscore",
})


@pytest.fixture(scope="module", autouse=True)
def load_ops():
    load_all()


def test_all_certified_have_production_admitted():
    """Every R23 P1 certified canonical must have production_admitted=True."""
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
        f"R23 P1 certified operators with production_admitted=False:\n"
        + "\n".join(failures)
    )


def test_all_certified_have_production_certified():
    """Every R23 P1 certified canonical must have production_certified=True."""
    for canonical in sorted(R23_CERTIFIED_CANONICALS):
        catalog = OperatorRegistry._catalog.get(canonical)
        assert catalog is not None
        assert catalog.get("production_certified") is True, (
            f"{canonical}: catalog production_certified is not True"
        )


def test_all_certified_have_production_status():
    """Every R23 P1 certified canonical must have status=production."""
    for canonical in sorted(R23_CERTIFIED_CANONICALS):
        catalog = OperatorRegistry._catalog.get(canonical)
        assert catalog is not None
        assert catalog.get("status") == "production", (
            f"{canonical}: status is {catalog.get('status')!r}, expected 'production'"
        )


def test_experimental_not_certified():
    """Operators in scope that are NOT certified must NOT have production_admitted."""
    from factor_engine.cleaned_operators.operator_surface import DAILY_CANONICALS
    import json

    d = json.load(open("/home/shw/quant_projects/docs/R23_PER_CANONICAL_AUDIT.json"))
    cats = ("time_series", "price_structure", "price_volume_extension")
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
        f"Non-certified scope operators with production_admitted=True:\n"
        + "\n".join(failures)
    )


def test_certified_count():
    """Exactly 18 operators are certified."""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    catalog = OperatorRegistry._catalog
    actual = sum(
        1 for c in catalog if c in R23_CERTIFIED_CANONICALS
        and catalog[c].get("production_certified") is True
    )
    assert actual == 18, f"Expected 18 certified, got {actual}"