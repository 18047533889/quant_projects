from __future__ import annotations

import pytest

from data_access.core.exceptions import ValidationError
from data_access.read.aggregation import (
    AggregationSemantic,
    AggregationSpec,
    SessionWindow,
    _require_production_aggregation_contract,
)
from data_access.read.minute_filter import FilterSignature


def test_minute_at_requires_hhmm_direct_constructor() -> None:
    with pytest.raises(ValidationError, match="hhmm"):
        AggregationSpec(aggregation="minute_at")


def test_full_session_is_explicit_typed_window() -> None:
    spec = AggregationSpec(
        aggregation="minute_range",
        session_window=SessionWindow.FULL_REGULAR_SESSION,
    )
    sig = FilterSignature.canonicalize(spec)
    assert sig.kind == "full_regular_session"
    assert sig.to_sql(minute_expr="m", elapsed_expr="e") is None


def test_custom_range_requires_both_bounds() -> None:
    with pytest.raises(ValidationError, match="start 和 end"):
        AggregationSpec(
            aggregation="minute_range",
            session_window=SessionWindow.CUSTOM_RANGE,
            start="09:31",
        )


def test_typed_semantic_derives_metric_without_field_name() -> None:
    spec = AggregationSpec(semantic=AggregationSemantic.ADDITIVE_FLOW)
    assert spec.effective_metric("turnover_value", production=True) == "sum"


def test_production_rejects_name_inference() -> None:
    spec = AggregationSpec()
    with pytest.raises(ValidationError, match="禁止从字段名推断"):
        spec.effective_metric("buy_volume_x", production=True)


def test_research_keeps_legacy_name_inference() -> None:
    assert AggregationSpec().effective_metric("Volume", production=False) == "sum"


def test_production_contract_requires_market_and_calendar(monkeypatch) -> None:
    monkeypatch.setattr(
        "data_access.read.query_budget.is_strict_semantics", lambda: True
    )
    typed = AggregationSpec(semantic=AggregationSemantic.STOCK_LAST)
    with pytest.raises(ValidationError, match="market"):
        _require_production_aggregation_contract(typed, field="Close")

    with_market = AggregationSpec(
        semantic=AggregationSemantic.STOCK_LAST,
        market="us",
    )
    with pytest.raises(ValidationError, match="calendar"):
        _require_production_aggregation_contract(with_market, field="Close")

    complete = AggregationSpec(
        semantic=AggregationSemantic.STOCK_LAST,
        market="us",
        calendar="us_calendar",
    )
    _require_production_aggregation_contract(complete, field="not_name_inferred")

    mismatched = AggregationSpec(
        semantic=AggregationSemantic.STOCK_LAST,
        market="us",
        calendar="ashare_calendar",
    )
    with pytest.raises(ValidationError, match="不一致"):
        _require_production_aggregation_contract(mismatched, field="Close")


def test_minute_of_day_invariants() -> None:
    with pytest.raises(ValidationError, match="period"):
        AggregationSpec(aggregation="minute_of_day", period=None)
    with pytest.raises(ValidationError, match="index"):
        AggregationSpec(aggregation="minute_of_day", period=5, index=-1)
