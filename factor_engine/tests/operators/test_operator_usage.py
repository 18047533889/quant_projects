# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import pytest

from cleaned_operators import load_all
from operator_usage_expanded import build_usage_report, expand_formula_recipes, extract_call_names


@pytest.fixture(scope="module", autouse=True)
def _load() -> None:
    load_all()


def test_call_extraction_and_recipe_expansion_are_safe() -> None:
    calls = extract_call_names("rank(ts_mean(close, 20) / ts_std(close, 20))")
    assert calls.count("rank") == 1
    assert calls.count("ts_mean") == 1
    assert calls.count("ts_std") == 1
    assert extract_call_names("not valid python (") == []
    expanded = expand_formula_recipes("momentum(ts_mean(close, 5), 20)")
    assert expanded is not None
    assert extract_call_names(expanded).count("ts_mean") == 1
    assert extract_call_names(expanded).count("ts_delta") == 1
    assert "momentum" not in extract_call_names(expanded)


def test_usage_resolves_aliases_and_counts_recipe_primitives() -> None:
    records = [
        (Path("recent.json"), {
            "candidate_id": "recent",
            "formula": "ADX(high, low, close, 14) + WMA(close, 10)",
            "expression_type": "dsl",
            "born_timestamp": "2026-07-10T00:00:00Z",
        }),
        (Path("old.json"), {
            "candidate_id": "old",
            "formula": "MACD_line(close, 12, 26)",
            "expression_type": "dsl",
            "born_timestamp": "2025-01-01T00:00:00Z",
        }),
        (Path("recipe.json"), {
            "candidate_id": "recipe",
            "formula": "momentum(close, 20)",
            "expression_type": "dsl",
            "born_timestamp": "2026-07-11T00:00:00Z",
        }),
    ]
    report = build_usage_report(records, as_of=datetime(2026, 7, 17, tzinfo=timezone.utc), lookback_days=90)
    assert report.schema_version == "operator_usage.v2"
    assert report.operator_usage["ADX"].call_count == 1
    assert report.operator_usage["ts_decay_linear"].direct_call_count == 1
    assert report.recipe_usage["momentum"].call_count == 1
    assert report.operator_usage["ts_delta"].call_count == 1
    assert report.operator_usage["ts_delta"].direct_call_count == 0
    assert report.operator_usage["ts_delta"].recipe_expanded_call_count == 1
    assert report.fused_composite_review["ADX"] == "retain_active"
    assert report.fused_composite_review["MACD_line"] == "review_dormant"
    assert report.fused_composite_review["MACD_hist"] == "review_unused"


def test_reference_timestamp_is_deterministic_max_manifest_time() -> None:
    records = [
        (Path("a.json"), {"formula": "ts_mean(close, 5)", "born_timestamp": "2026-01-01T00:00:00Z"}),
        (Path("b.json"), {"formula": "ts_mean(close, 5)", "born_timestamp": "2026-03-01T00:00:00Z"}),
    ]
    first = build_usage_report(records)
    second = build_usage_report(list(reversed(records)))
    assert first.reference_timestamp == "2026-03-01T00:00:00+00:00"
    assert first.to_dict() == second.to_dict()


def test_invalid_lookback_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        build_usage_report([], lookback_days=0)
