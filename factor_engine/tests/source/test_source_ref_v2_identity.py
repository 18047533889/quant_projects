# -*- coding: utf-8 -*-
"""R24-136..138 + R24-241: SourceRef v2 carries market-scoped semantic identity;
A StockIncome.revenue vs US StockIncome.revenue encode to DIFFERENT identities."""
from __future__ import annotations

import pytest

from api.source_ref import (
    decode_source_ref,
    encode_source_ref,
    make_source_ref,
)


def test_cross_market_identity_differs() -> None:
    # R24-241 golden: A StockIncome.revenue vs US StockIncome.revenue must NOT
    # share an encoded identity.
    a = make_source_ref("StockIncome", "revenue", market="ashare", dataset="ashare_stock_income")
    us = make_source_ref("StockIncome", "revenue", market="us", dataset="us_stock_income")
    assert encode_source_ref(a) != encode_source_ref(us)


def test_v2_decodes_market_scoped_fields() -> None:
    spec = make_source_ref(
        "StockIncome", "revenue", market="us",
        concept_id="concept_revenue", field_id="us.stock_income.revenue",
        provider_id="p1", dataset="us_stock_income", timeframe="ttm",
        temporal_policy_digest="d123", catalog_hash="c456", source_version="v2",
    )
    assert spec.is_v2
    d = decode_source_ref(encode_source_ref(spec))
    assert d.market == "us"
    assert d.concept_id == "concept_revenue"
    assert d.field_id == "us.stock_income.revenue"
    assert d.provider_id == "p1"
    assert d.dataset == "us_stock_income"
    assert d.timeframe == "ttm"
    assert d.catalog_hash == "c456"


def test_v1_back_compat() -> None:
    # A legacy v1 ref (no market) still decodes.
    v1 = make_source_ref("StockIncome", "revenue")
    d = decode_source_ref(encode_source_ref(v1))
    assert d.market is None
    assert d.is_v2 is False
