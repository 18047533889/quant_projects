# -*- coding: utf-8 -*-
"""A-share bp/% vendor units vs US decimal — binding-level parity (P1-012).

The audit (P1-012) requires that every entry point into FactorEngine produces the
SAME canonical units: A Return bp must become decimal (/10000) exactly once, A
TurnoverRatio / ROE / ROA / Margin / ShareRatio / Index Weight are percent (/100),
while US Ret / ROE / dividend_yield are already decimal and MUST NOT be scaled
again.  These tests pin the provider binding transforms so no single entry point
can return bp where another returns decimal.
"""
from __future__ import annotations

import pytest

from factor_engine.fields.providers import PROVIDER_REGISTRY
from factor_engine.market.capabilities import ProviderQuality


def _binding(concept: str, market: str):
    return PROVIDER_REGISTRY.binding(concept, market)


def test_a_share_return_bp_to_decimal() -> None:
    b = _binding("return_decimal", "ashare")
    assert b is not None
    assert b.transform_description.startswith("x * 0.0001")  # bp -> decimal
    assert float(b.transform(125.0)) == pytest.approx(0.0125)


def test_us_return_decimal_identity() -> None:
    b = _binding("return_decimal", "us")
    assert b is not None
    assert b.transform_description.startswith("identity")
    assert float(b.transform(0.0125)) == pytest.approx(0.0125)  # NOT scaled again


def test_a_share_percent_to_decimal() -> None:
    # TurnoverRatio / ROE / dividend yield are percent in the A-share COS dict:
    # a vendor value of 12.5% must enter FactorEngine as 0.125 decimal.
    for concept in ("turnover_ratio_decimal", "roe_decimal", "dividend_yield_decimal"):
        b = _binding(concept, "ashare")
        assert b is not None, concept
        assert float(b.transform(12.5)) == pytest.approx(0.125), concept


def test_us_roe_dividend_yield_identity() -> None:
    # US valuation fields are already decimal; they must not be /100 again.
    for concept in ("roe_decimal", "dividend_yield_decimal"):
        b = _binding(concept, "us")
        assert b is not None, concept
        assert b.transform_description.startswith("identity"), (concept, b.transform_description)
        assert float(b.transform(0.125)) == pytest.approx(0.125), concept


def test_a_share_index_weight_percent_to_decimal() -> None:
    b = _binding("index_weight", "ashare")
    assert b is not None
    assert float(b.transform(2.0)) == pytest.approx(0.02)  # Weight is percent


def test_us_turnover_unavailable_not_silently_volume() -> None:
    # US has no isomorphic D1 turnover; the binding must be unavailable so US
    # expressions never substitute volume for turnover.
    b = _binding("turnover_ratio_decimal", "us")
    assert b is not None
    assert b.quality == ProviderQuality.UNAVAILABLE
