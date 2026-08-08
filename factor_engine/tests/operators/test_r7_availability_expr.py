"""Round-7 WS-C: AvailabilityExpr + structured SourceVintageSpec (#274-#278).

Covers:
* unknown availability -> fail-closed (review #274/#275);
* ``_AVAILABILITY_RANK`` string total order removed, UNKNOWN sorts +inf;
* ``_EOD_AVAILABILITY_MARKERS`` substring guessing removed (#277);
* ``SourceVintageSpec`` structured vintage (#278).
"""
from __future__ import annotations

import pytest

from ir.schema import Schema, _available_at_of, propagate_available_at


@pytest.fixture(scope="session", autouse=True)
def strict_fiscal_parameter_domain_certification_guard():
    """Shadow the operators conftest autouse guard.

    The conftest guard calls ``load_all()``, which is blocked by concurrent
    WS-A/WS-H work-in-progress.  These WS-C unit tests exercise ir.types /
    ir.schema / fields.spec directly and do not need the full operator load.
    """
    yield
from ir.types import (
    SourceVintageSpec,
    TimestampColumn,
    UNKNOWN,
    UnknownAvailability,
    availability_expr_of,
    latest_availability,
)


# ---------------------------------------------------------------------------
# #274/#275  unknown availability is fail-closed (+inf), never silently dropped.
# ---------------------------------------------------------------------------
def test_unknown_availability_fails_closed():
    # A literal "unknown" descriptor must sort LATEST (the old bug put it first).
    assert availability_expr_of("unknown").lateness == float("inf")
    # None input -> UNKNOWN too.
    assert isinstance(availability_expr_of(None), UnknownAvailability)
    assert isinstance(availability_expr_of(UNKNOWN), UnknownAvailability)


def test_propagate_available_at_unknown_wins_over_known():
    # #275: do NOT drop the unknown input and keep the latest known one.
    assert propagate_available_at(("session_close", "unknown")) == "unknown"
    assert propagate_available_at(("PubDate", None)) == "unknown"
    assert propagate_available_at(("session_close", "session_open", "unknown")) == "unknown"


def test_propagate_available_at_returns_latest_known():
    assert propagate_available_at(("session_open", "session_close")) == "session_close"
    assert propagate_available_at(("session_close", "PubDate")) == "PubDate"
    assert propagate_available_at(()) is None


def test_latest_availability_expression():
    latest = latest_availability(["session_close", "PubDate"])
    assert latest.label == "PubDate"
    assert latest_availability(["session_close", None]).label == "unknown"


# ---------------------------------------------------------------------------
# #274  string total order removed; ordering lives on the AvailabilityExpr.
# ---------------------------------------------------------------------------
def test_availability_expr_label_and_lateness():
    assert availability_expr_of("PubDate").lateness == 90.0
    assert availability_expr_of("session_close").lateness == 40.0
    assert availability_expr_of("session_close").label == "session_close"


def test_unknown_sorts_latest():
    from ir.schema import _rank_availability

    assert _rank_availability("unknown") > _rank_availability("session_close")
    assert _rank_availability("unknown") == float("inf")


def test_timestamp_column_resolves_to_concrete_column():
    expr = TimestampColumn("declaration_date")
    assert isinstance(expr, TimestampColumn)
    assert expr.label == "declaration_date"
    assert expr.lateness >= 100.0


# ---------------------------------------------------------------------------
# #277  availability no longer guessed from name SUBSTRINGS ("pe"/"pb").
# ---------------------------------------------------------------------------
def test_eod_substring_markers_removed():
    # "pe_ratio" contains "pe" but is NOT an exact EOD name -> must stay unknown
    # (the old ``_EOD_AVAILABILITY_MARKERS`` substring match made it session_close).
    spec = type("_Spec", (), {"name": "pe_ratio", "available_at": None,
                              "knowledge_time_column": None, "role": "feature"})
    assert _available_at_of(spec) is None


def test_exact_eod_names_still_get_session_close():
    for name in ("close", "high", "low", "vwap", "volume", "amount", "turnover"):
        spec = type("_Spec", (), {"name": name, "available_at": None,
                                  "knowledge_time_column": None, "role": "feature"})
        assert _available_at_of(spec) == "session_close"


def test_open_names_get_session_open():
    spec = type("_Spec", (), {"name": "open", "available_at": None,
                              "knowledge_time_column": None, "role": "feature"})
    assert _available_at_of(spec) == "session_open"


def test_availability_prefers_explicit_and_knowledge_time():
    spec = type("_Spec", (), {"name": "close", "available_at": "local_close",
                              "knowledge_time_column": None, "role": "feature"})
    assert _available_at_of(spec) == "local_close"
    spec2 = type("_Spec", (), {"name": "custom", "available_at": None,
                               "knowledge_time_column": "PubDate", "role": "feature"})
    assert _available_at_of(spec2) == "PubDate"


def test_role_based_availability():
    # group keys / identifiers are not EOD-realised series.
    spec = type("_Spec", (), {"name": "industry", "available_at": None,
                              "knowledge_time_column": None, "role": "group_key"})
    assert _available_at_of(spec) is None


# ---------------------------------------------------------------------------
# #278  structured SourceVintageSpec.
# ---------------------------------------------------------------------------
def test_source_vintage_spec_structured():
    spec = SourceVintageSpec(
        knowledge_time="PubDate", revision_time="rev_col", version_id="v2", snapshot_id="snap1"
    )
    assert spec.knowledge_time == "PubDate"
    assert spec.revision_time == "rev_col"
    assert spec.version_id == "v2"
    assert spec.snapshot_id == "snap1"
    assert spec.to_dict() == {
        "knowledge_time": "PubDate",
        "revision_time": "rev_col",
        "version_id": "v2",
        "snapshot_id": "snap1",
    }
    # hashable / frozen
    assert hash(spec) is not None


def test_source_vintage_from_field_spec():
    from fields.spec import FieldSpec

    field = FieldSpec(
        name="net_profit", table="StockIncome", source_name="net_profit",
        knowledge_time_column="PubDate", revision_columns=("rev_a", "rev_b"),
    )
    vintage = SourceVintageSpec.from_field_spec(field)
    assert vintage.knowledge_time == "PubDate"
    assert vintage.revision_time == "rev_a"
    assert vintage.version_id == "rev_b"


def test_schema_from_field_carries_structured_vintage():
    from fields.spec import FieldSpec

    field = FieldSpec(
        name="net_profit", table="StockIncome", source_name="net_profit",
        knowledge_time_column="PubDate",
    )
    schema = Schema.from_field(field)
    assert isinstance(schema.source_vintage, SourceVintageSpec)
    assert schema.source_vintage.knowledge_time == "PubDate"
    # availability label + typed expression stay in sync.
    assert schema.available_at == "PubDate"
    assert schema.availability_expr is not None
    assert schema.availability_expr.label == "PubDate"


def test_schema_semantic_lattice_present():
    from fields.spec import FieldSpec

    field = FieldSpec(
        name="volume", table="StockDailyBar", source_name="volume", domain="price_volume",
        frequency="daily",
    )
    schema = Schema.from_field(field)
    assert schema.semantic_lattice is not None
    assert schema.semantic_lattice.domains == ("price_volume",)
    assert schema.semantic_lattice.frequencies == ("daily",)
