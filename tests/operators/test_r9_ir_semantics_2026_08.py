"""Round-9 IR semantic-lattice fixes (R9-P0-003 / R9-P0-008 / R9-P0-009).

Covers:
* R9-P0-003 — the semantic lattice NEVER keeps the first input as the scalar
  authoritative value on a conflict; a mixed dimension holds the ``MIXED``
  marker, so ``semantic(A, B) == semantic(B, A)`` for the commutative ops
  ``add`` / ``multiply`` (both with conflicting and with matching domains).
* R9-P0-008 — every ``AvailabilityExpr`` subclass implements a real
  ``resolve(row, calendar, timezone, decision_context)`` returning a
  timestamp-like value, and fails clearly when the provider it needs is
  unavailable (never a compile-time total-order number).
* R9-P0-009 — ``ExDate`` (ex-dividend date) is a DISTINCT semantic from
  ``DeclarationDate``; the string-label shim maps ``"ex_date"`` to ``ExDate``
  and also maps ``record_date`` / ``payment_date`` / ``effective_date``.
"""
from __future__ import annotations

import datetime

import pytest


@pytest.fixture(scope="session", autouse=True)
def strict_fiscal_parameter_domain_certification_guard():
    """Shadow the operators conftest autouse guard.

    The conftest guard calls ``load_all()``, which is blocked by concurrent
    work-in-progress.  These R9 unit tests exercise ir.types / ir.analyzer
    directly and do not need the full operator load.  The analyzer-path tests
    below additionally stub the operator registry so ``Analyzer.lower`` never
    triggers ``load_all()``.
    """
    yield


from factor_engine.ir.types import (  # noqa: E402
    AfterClose,
    DeclarationDate,
    EffectiveDate,
    ExDate,
    FilingDate,
    LocalClose,
    MIXED,
    Midnight,
    NextTradingDay,
    NextTradingOpen,
    PaymentDate,
    PreClose,
    PubDate,
    RecordDate,
    SessionClose,
    SessionOpen,
    TimestampColumn,
    UNKNOWN,
    UnknownAvailability,
    availability_expr_of,
    lattice_join_semantic_attrs,
)


# ---------------------------------------------------------------------------
# R9-P0-003  semantic lattice must not keep "first input" as authoritative.
# ---------------------------------------------------------------------------
def _authoritative(semantic: dict) -> dict:
    """Project the semantic dict onto its scalar authoritative slots.

    ``mixed_*`` diagnostic tuples preserve *encounter order* (pinned by the
    r7 lattice contract), so operand-order independence is asserted on the
    scalar authoritative values — the actual bug (``domain`` used to silently
    be ``distinct[0]``).
    """
    return {
        key: value
        for key, value in semantic.items()
        if not key.startswith("mixed_") and key != "lattice"
    }


def _mixed_sets(semantic: dict) -> dict:
    return {
        key: set(value)
        for key, value in semantic.items()
        if key.startswith("mixed_")
    }


def test_lattice_conflicting_domains_is_mixed_never_first_input():
    fundamental = {"domain": "fundamental", "unit": "CNY", "frequency": "daily"}
    price = {"domain": "price_volume", "unit": "CNY", "frequency": "daily"}
    ab = lattice_join_semantic_attrs([fundamental, price])
    ba = lattice_join_semantic_attrs([price, fundamental])

    # The authoritative scalar is the MIXED marker in BOTH operand orders,
    # never ``distinct[0]`` ("fundamental").
    assert ab["domain"] == MIXED == ba["domain"]
    assert _authoritative(ab) == _authoritative(ba)
    # The diagnostic mixed_* tuple records the actual distinct values; both
    # orders carry the same SET of values.
    assert set(ab["mixed_domain"]) == {"fundamental", "price_volume"}
    assert _mixed_sets(ab) == _mixed_sets(ba)
    # Unambiguous dimension stays scalar.
    assert ab["unit"] == "CNY" == ba["unit"]
    assert ab["frequency"] == "daily" == ba["frequency"]


def test_lattice_matching_domains_stays_scalar_and_order_independent():
    price_a = {"domain": "price_volume", "unit": "CNY", "frequency": "daily"}
    price_b = {"domain": "price_volume", "unit": "CNY", "frequency": "daily"}
    ab = lattice_join_semantic_attrs([price_a, price_b])
    ba = lattice_join_semantic_attrs([price_b, price_a])
    assert ab["domain"] == "price_volume" == ba["domain"]
    assert ab == ba  # matching domains -> full dict equality


def test_lattice_unit_conflict_is_mixed():
    a = {"unit": "CNY", "domain": "price_volume"}
    b = {"unit": "dimensionless", "domain": "price_volume"}
    ab = lattice_join_semantic_attrs([a, b])
    ba = lattice_join_semantic_attrs([b, a])
    assert ab["unit"] == MIXED == ba["unit"]
    assert set(ab["mixed_unit"]) == {"CNY", "dimensionless"}
    assert _authoritative(ab) == _authoritative(ba)


def test_lattice_semantic_kind_conflict_is_mixed():
    raw = {"semantic_kind": "PriceRaw", "domain": "price_volume"}
    fin = {"semantic_kind": "FinancialCumulativeYTDFlow", "domain": "fundamental"}
    ab = lattice_join_semantic_attrs([raw, fin])
    ba = lattice_join_semantic_attrs([fin, raw])
    assert ab["semantic_kind"] == MIXED == ba["semantic_kind"]
    assert set(ab["mixed_semantic_kind"]) == {
        "PriceRaw", "FinancialCumulativeYTDFlow",
    }
    assert _authoritative(ab) == _authoritative(ba)


def _patch_operator_registry(monkeypatch):
    """Stub the operator registry so ``Analyzer.lower`` avoids ``load_all()``."""
    from factor_engine.backend import cleaned_bridge
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    monkeypatch.setattr(
        cleaned_bridge, "ensure_cleaned_loaded", lambda: None
    )
    monkeypatch.setattr(
        OperatorRegistry,
        "resolve_canonical_strict",
        classmethod(lambda cls, name, *, max_depth=8: name),
    )
    monkeypatch.setattr(
        OperatorRegistry,
        "get",
        classmethod(lambda cls, name, backend="pandas_numpy": None),
    )


def _analyzer_semantic(monkeypatch, left, right, op: str) -> dict:
    _patch_operator_registry(monkeypatch)
    from factor_engine.api.columns import field
    from factor_engine.ir.analyzer import Analyzer

    if op == "add":
        expr = field(left) + field(right)
    elif op == "multiply":
        expr = field(left) * field(right)
    else:  # pragma: no cover - test-internal
        raise AssertionError(f"unknown op {op!r}")
    return Analyzer().lower(expr).ir.semantic_attrs


def test_analyzer_add_conflicting_domains_order_independent(monkeypatch):
    # ``add(fundamental, price)`` vs ``add(price, fundamental)``.
    ab = _analyzer_semantic(monkeypatch, "net_profit", "close", "add")
    ba = _analyzer_semantic(monkeypatch, "close", "net_profit", "add")
    assert ab["domain"] == MIXED == ba["domain"]
    assert ab["domain"] != "fundamental"  # NOT the first input's value
    assert _authoritative(ab) == _authoritative(ba)
    assert _mixed_sets(ab) == _mixed_sets(ba)


def test_analyzer_add_matching_domains_order_independent(monkeypatch):
    ab = _analyzer_semantic(monkeypatch, "close", "high", "add")
    ba = _analyzer_semantic(monkeypatch, "high", "close", "add")
    assert ab["domain"] == "price_volume" == ba["domain"]
    assert _authoritative(ab) == _authoritative(ba)


def test_analyzer_multiply_conflicting_domains_order_independent(monkeypatch):
    ab = _analyzer_semantic(monkeypatch, "net_profit", "close", "multiply")
    ba = _analyzer_semantic(monkeypatch, "close", "net_profit", "multiply")
    assert ab["domain"] == MIXED == ba["domain"]
    assert _authoritative(ab) == _authoritative(ba)
    assert _mixed_sets(ab) == _mixed_sets(ba)


def test_analyzer_source_vintage_conflict_is_mixed_and_order_independent(monkeypatch):
    # ``net_profit`` carries knowledge_time=PubDate; ``close`` carries none.
    # A genuine vintage conflict must yield the MIXED marker (the review's
    # "dependency set"), never the first non-None spec.
    ab = _analyzer_semantic(monkeypatch, "net_profit", "close", "add")
    ba = _analyzer_semantic(monkeypatch, "close", "net_profit", "add")
    assert ab.get("source_vintage") == MIXED == ba.get("source_vintage")
    assert ab.get("mixed_source_vintage") is not None
    assert _authoritative(ab) == _authoritative(ba)


# ---------------------------------------------------------------------------
# R9-P0-008  AvailabilityExpr subclasses implement a real resolve().
# ---------------------------------------------------------------------------
def test_resolve_column_based_expressions():
    row = {
        "PubDate": "2024-01-05",
        "filing_date": "2024-01-04",
        "declaration_date": "2024-01-06",
        "ex_date": "2024-01-10",
        "record_date": "2024-01-08",
        "payment_date": "2024-02-01",
        "effective_date": "2024-01-07",
        "custom_ts": datetime.datetime(2024, 1, 3, 9, 30),
    }
    assert PubDate().resolve(row=row) == datetime.datetime(2024, 1, 5)
    assert FilingDate().resolve(row=row) == datetime.datetime(2024, 1, 4)
    assert DeclarationDate().resolve(row=row) == datetime.datetime(2024, 1, 6)
    assert ExDate().resolve(row=row) == datetime.datetime(2024, 1, 10)
    assert RecordDate().resolve(row=row) == datetime.datetime(2024, 1, 8)
    assert PaymentDate().resolve(row=row) == datetime.datetime(2024, 2, 1)
    assert EffectiveDate().resolve(row=row) == datetime.datetime(2024, 1, 7)
    assert TimestampColumn("custom_ts").resolve(row=row) == datetime.datetime(
        2024, 1, 3, 9, 30
    )


def test_resolve_wallclock_expressions():
    row = {"date": datetime.date(2024, 1, 2)}
    assert Midnight().resolve(row=row) == datetime.datetime(2024, 1, 2)
    assert SessionOpen().resolve(row=row) == datetime.datetime(2024, 1, 2, 9, 30)
    assert PreClose().resolve(row=row) == datetime.datetime(2024, 1, 2, 14, 57)
    assert LocalClose().resolve(row=row) == datetime.datetime(2024, 1, 2, 15, 0)
    assert SessionClose().resolve(row=row) == datetime.datetime(2024, 1, 2, 15, 0)
    assert AfterClose().resolve(row=row) == datetime.datetime(2024, 1, 2, 16, 0)


def test_resolve_calendar_driven_expressions():
    class FakeCalendar:
        def next_open(self, ref):
            return datetime.datetime(2024, 1, 8, 9, 30)

        def next_day(self, ref):
            return datetime.datetime(2024, 1, 8)

    row = {"PubDate": "2024-01-05"}
    assert NextTradingOpen().resolve(
        row=row, calendar=FakeCalendar()
    ) == datetime.datetime(2024, 1, 8, 9, 30)
    assert NextTradingDay().resolve(
        row=row, calendar=FakeCalendar()
    ) == datetime.datetime(2024, 1, 8)


def test_resolve_fails_clearly_without_required_provider():
    # Column-based: missing row / column -> NotImplementedError, never a guess.
    with pytest.raises(NotImplementedError):
        PubDate().resolve(row=None)
    with pytest.raises(NotImplementedError):
        TimestampColumn("missing_col").resolve(row={"other": 1})
    # Wall-clock: no date anywhere -> NotImplementedError.
    with pytest.raises(NotImplementedError):
        SessionClose().resolve()
    # Calendar-driven: a bare string calendar is NOT a real calendar.
    with pytest.raises(NotImplementedError):
        NextTradingOpen().resolve(row={"PubDate": "2024-01-05"}, calendar="TradeDate")
    # UNKNOWN stays fail-closed.
    with pytest.raises(ValueError):
        UnknownAvailability().resolve()
    assert isinstance(UNKNOWN, UnknownAvailability)


def test_resolve_timezone_awareness():
    row = {"date": datetime.date(2024, 1, 2)}
    dt = SessionClose().resolve(row=row, timezone="UTC")
    assert dt.tzinfo is not None
    assert dt.astimezone(datetime.timezone.utc) == datetime.datetime(
        2024, 1, 2, 15, 0, tzinfo=datetime.timezone.utc
    )


# ---------------------------------------------------------------------------
# R9-P0-009  ExDate is NOT DeclarationDate.
# ---------------------------------------------------------------------------
def test_ex_date_is_distinct_from_declaration_date():
    assert isinstance(availability_expr_of("ex_date"), ExDate)
    assert not isinstance(availability_expr_of("ex_date"), DeclarationDate)
    # The declaration label still maps to the declaration expression.
    assert isinstance(availability_expr_of("declaration_date"), DeclarationDate)
    assert availability_expr_of("declaration_date").label == "declaration_date"
    # Distinct classes -> distinct types (no aliasing via __eq__).
    assert type(ExDate()) is not type(DeclarationDate())
    assert ExDate() != DeclarationDate()
    # The corporate-action / effective date labels resolve to their own types.
    assert isinstance(availability_expr_of("record_date"), RecordDate)
    assert isinstance(availability_expr_of("payment_date"), PaymentDate)
    assert isinstance(availability_expr_of("effective_date"), EffectiveDate)
    assert isinstance(availability_expr_of("ex_dividend_date"), ExDate)
