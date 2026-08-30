from __future__ import annotations

import pytest


def test_field_keeps_catalog_identity_without_explicit_table():
    from factor_engine.api.columns import field
    from factor_engine.expr.field import FieldRef
    from factor_engine.ir.analyzer import Analyzer

    # ADJ_FIELD_MIGRATION: bare field('close') resolves to the adj authority
    # table StockDailyBarAdj (AdjClose).
    ref = field("close")
    assert isinstance(ref, FieldRef)
    assert ref.field_id == "StockDailyBarAdj.close"
    assert ref.source_name == "AdjClose"
    assert ref.table == "StockDailyBarAdj"

    analysis = Analyzer().lower(ref)
    assert analysis.ir.attrs["field_id"] == "StockDailyBarAdj.close"
    assert analysis.ir.attrs["field_registry_hash"]


def test_secondary_field_uses_source_transport_and_keeps_identity():
    from factor_engine.api.columns import field
    from factor_engine.api.source_ref import decode_source_ref
    from factor_engine.expr.field import FieldRef

    ref = field("pe_ratio")
    assert isinstance(ref, FieldRef)
    assert ref.table == "StockValuationDaily"
    encoded = decode_source_ref(ref.name)
    assert encoded is not None
    assert encoded.table == "StockValuationDaily"
    assert encoded.field == "PeRatio"


def test_unknown_field_remains_strict_by_default():
    from factor_engine.api.columns import field

    with pytest.raises(KeyError, match="unknown field"):
        field("definitely_not_a_field")


def test_string_dsl_field_preserves_catalog_identity():
    from factor_engine.api.dsl_parser import parse_expr
    from factor_engine.expr.field import FieldRef
    from factor_engine.ir.analyzer import Analyzer

    expr = parse_expr("field('pe_ratio')")
    assert isinstance(expr, FieldRef)
    analysis = Analyzer().lower(expr)
    assert analysis.ir.attrs["field_id"] == "StockValuationDaily.pe_ratio"
    assert analysis.ir.attrs["source_table"] == "StockValuationDaily"


def test_production_rejects_bare_secondary_catalog_field():
    from factor_engine.api.mining_integration import validate_production_dsl

    ok, msg = validate_production_dsl("rank(col('pe_ratio'))", market="ashare")
    assert ok is False
    assert "requires field(...)" in msg


def test_stale_field_ref_is_rejected():
    from factor_engine.expr.field import FieldRef
    from factor_engine.ir.analyzer import Analyzer, FieldCatalogMismatchError

    stale = FieldRef(
        name="close",
        field_id="StockDailyBar.close",
        canonical_name="close",
        table="StockDailyBar",
        source_name="Close",
        catalog_hash="stale",
    )
    with pytest.raises(FieldCatalogMismatchError, match="does not match active catalog"):
        Analyzer().lower(stale)
