from __future__ import annotations

import pytest


def test_field_keeps_catalog_identity_without_explicit_table():
    from api.columns import field
    from expr.field import FieldRef
    from ir.analyzer import Analyzer

    ref = field("close")
    assert isinstance(ref, FieldRef)
    assert ref.field_id == "StockDailyBar.close"
    assert ref.source_name == "Close"

    analysis = Analyzer().lower(ref)
    assert analysis.ir.attrs["field_id"] == "StockDailyBar.close"
    assert analysis.ir.attrs["field_registry_hash"]


def test_secondary_field_uses_source_transport_and_keeps_identity():
    from api.columns import field
    from api.source_ref import decode_source_ref
    from expr.field import FieldRef

    ref = field("pe_ratio")
    assert isinstance(ref, FieldRef)
    assert ref.table == "StockValuationDaily"
    encoded = decode_source_ref(ref.name)
    assert encoded is not None
    assert encoded.table == "StockValuationDaily"
    assert encoded.field == "PeRatio"


def test_unknown_field_remains_strict_by_default():
    from api.columns import field

    with pytest.raises(KeyError, match="unknown field"):
        field("definitely_not_a_field")
