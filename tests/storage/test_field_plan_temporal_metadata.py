# -*- coding: utf-8 -*-
"""R24-062..067: NormalizedFieldPlan carries the FULL temporal metadata (never
dropped), and MissingSemantic distinguishes OUT_OF_COVERAGE (source has no
history) from NOT_APPLICABLE (economically not applicable)."""
from __future__ import annotations

import pytest

from factor_engine.storage.sources.field_plan import (
    MissingSemantic,
    NormalizedFieldPlan,
    missing_semantic_for_plan,
    plan_from_field_spec,
)


class _FakeSpec:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        for attr in (
            "source_name", "canonical_unit", "scale_to_canonical", "frequency",
            "grain", "knowledge_time_column", "effective_time_column",
            "revision_columns", "domain", "universe_id", "temporal_model",
            "mining_allowed", "price_basis", "flow_semantics", "required_filters",
            "applicability", "allowed_operator_families", "null_policy",
            "semantic_kind", "strict_pit_allowed", "role", "table", "dataset",
        ):
            if not hasattr(self, attr):
                setattr(self, attr, None)


def test_plan_from_field_spec_carries_temporal_metadata() -> None:
    spec = _FakeSpec(
        source_name="revenue",
        dataset="us_stock_income",
        table="StockIncome",
        name="revenue",
        concept_id="concept_revenue",
        field_id="us.stock_income.revenue",
        knowledge_time_column="pub_date",
        effective_time_column="period_end",
        period_id_column="report_period_end_date",
        revision_columns=("update_time", "version_id"),
        temporal_model="financial_pit",
        timeframe="ttm",
        availability_precision="date",
        availability_expr="NextTradingOpen(pub_date)",
        source_vintage="v2026",
        snapshot_policy="current_snapshot",
        missing_semantic="unknown",
        market="us",
        strict_pit_allowed=True,
        mining_allowed=True,
    )
    plan = plan_from_field_spec("revenue", spec, market="us")
    assert plan.concept_id == "concept_revenue"
    assert plan.field_id == "us.stock_income.revenue"
    assert plan.knowledge_time == "pub_date"
    assert plan.effective_time == "period_end"
    assert plan.period_id_column == "report_period_end_date"
    assert plan.revision_order == ("update_time", "version_id")
    assert plan.temporal_model == "financial_pit"
    assert plan.timeframe == "ttm"
    assert plan.availability_precision == "date"
    assert plan.availability_expr == "NextTradingOpen(pub_date)"
    assert plan.source_vintage == "v2026"
    assert plan.snapshot_policy == "current_snapshot"
    assert plan.market == "us"


def test_current_snapshot_historical_absence_is_out_of_coverage() -> None:
    # R24-065/066: current-snapshot table with no historical row = OUT_OF_COVERAGE,
    # NOT NOT_APPLICABLE.
    plan = NormalizedFieldPlan(logical_concept="x", coverage="current_snapshot", role="feature")
    assert missing_semantic_for_plan(plan) == MissingSemantic.OUT_OF_COVERAGE


def test_not_applicable_stays_distinct() -> None:
    # NOT_APPLICABLE = economically not applicable (role columns).
    plan = NormalizedFieldPlan(logical_concept="x", role="knowledge_time")
    assert missing_semantic_for_plan(plan) == MissingSemantic.NOT_APPLICABLE


def test_financial_nan_is_unknown() -> None:
    plan = NormalizedFieldPlan(
        logical_concept="revenue", temporal_model="financial_pit", role="feature"
    )
    assert missing_semantic_for_plan(plan) == MissingSemantic.UNKNOWN
