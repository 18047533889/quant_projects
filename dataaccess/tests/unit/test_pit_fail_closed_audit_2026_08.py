"""Synthetic regressions for audited COS event PIT fail-open paths."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from data_access.core.exceptions import (
    AvailabilityLatencyError,
    CalendarUnavailableError,
    SemanticCatalogUnavailableError,
)
from data_access.cos_event_runtime import (
    _apply_latency_minutes,
    _resolve_event_availability,
    _select,
)
from data_access.cos_contract import COSDatasetContract


def _contract() -> COSDatasetContract:
    return COSDatasetContract(
        name="synthetic_event",
        market="ashare",
        temporal_model="E1",
        join_policy="event",
        instrument_column="Symbol",
        panel_policy="event_only",
        pit_policy="strict",
        availability_column="knowledge",
        period_column=None,
        event_column=None,
    )


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions = pd.DataFrame(
        {
            "__pit_position": [0],
            "instrument": ["A"],
            "decision_timestamp": [pd.Timestamp("2024-01-02 10:00", tz="UTC")],
        }
    )
    events = pd.DataFrame(
        {
            "Symbol": ["A"],
            "knowledge": [pd.Timestamp("2024-01-02 09:00", tz="UTC")],
            "value": [7],
        }
    )
    return decisions, events


@pytest.mark.parametrize("availability", ["next_trading_day", "next_session_open"])
@pytest.mark.parametrize("calendar", [None, type("EmptyCalendar", (), {"has_data": False})()])
def test_calendar_required_availability_never_uses_raw_time_fallback(
    availability: str, calendar: object | None
) -> None:
    decisions, events = _frames()
    with pytest.raises(CalendarUnavailableError, match="日历不可用"):
        _select(
            decisions,
            events,
            _contract(),
            "decision_timestamp",
            "instrument",
            "knowledge",
            "latest_available",
            availability=availability,
            calendar=calendar,
        )


class _BrokenAvailableFrom:
    def __add__(self, other: object) -> object:
        raise TypeError("synthetic unsupported temporal value")


def test_nonzero_latency_failure_is_typed_and_preserves_cause() -> None:
    with pytest.raises(AvailabilityLatencyError) as caught:
        _apply_latency_minutes(_BrokenAvailableFrom(), 5)
    assert isinstance(caught.value.__cause__, TypeError)


def test_calendar_path_latency_failure_is_typed_and_preserves_cause(monkeypatch) -> None:
    import data_access.read.session_calendar as session_calendar

    decisions, events = _frames()
    calendar = type("Calendar", (), {"has_data": True, "timezone": "UTC"})()
    monkeypatch.setattr(
        session_calendar,
        "compile_available_from",
        lambda *args, **kwargs: _BrokenAvailableFrom(),
    )
    with pytest.raises(AvailabilityLatencyError) as caught:
        _select(
            decisions,
            events,
            _contract(),
            "decision_timestamp",
            "instrument",
            "knowledge",
            "latest_available",
            availability="next_trading_day",
            calendar=calendar,
            latency=5,
        )
    assert isinstance(caught.value.__cause__, TypeError)


def test_semantic_catalog_bootstrap_failure_never_defaults_same_day(monkeypatch) -> None:
    import data_access.read.semantic_catalog as semantic_catalog

    def fail_bootstrap() -> object:
        raise OSError("synthetic catalog storage unavailable")

    monkeypatch.setattr(semantic_catalog, "get_semantic_catalog", fail_bootstrap)
    with pytest.raises(SemanticCatalogUnavailableError) as caught:
        _resolve_event_availability(object(), "synthetic_event")
    assert isinstance(caught.value.__cause__, OSError)


def test_authoritative_catalog_no_declaration_explicitly_resolves_same_day(monkeypatch) -> None:
    import data_access.read.semantic_catalog as semantic_catalog

    catalog = type("Catalog", (), {"_fields": {}})()
    monkeypatch.setattr(semantic_catalog, "get_semantic_catalog", lambda: catalog)
    assert _resolve_event_availability(object(), "synthetic_event") == ("same_day", None)
