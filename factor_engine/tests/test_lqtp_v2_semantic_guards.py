from __future__ import annotations

import pytest


def _analysis(formula: str):
    from api.dsl_parser import parse_expr
    from ir.analyzer import Analyzer
    return Analyzer().lower(parse_expr(formula, surface="lqtp"))


def test_known_derived_fields_become_source_refs_not_daily_columns() -> None:
    from api.source_ref import decode_source_ref
    analysis = _analysis("enterprise_value / ebitda_approx")
    specs = [decode_source_ref(name) for name in analysis.referenced_columns]
    specs = [spec for spec in specs if spec is not None]
    assert {(spec.table, spec.field) for spec in specs} == {
        ("DerivedField", "enterprise_value"),
        ("DerivedField", "ebitda_approx"),
    }


def test_missing_derived_field_definition_fails_closed(monkeypatch) -> None:
    from runtime.derived_field_registry import load_derived_field_definition
    monkeypatch.delenv("FACTOR_ENGINE_DERIVED_FIELD_REGISTRY", raising=False)
    with pytest.raises(RuntimeError, match="no configured definition"):
        load_derived_field_definition("enterprise_value", path="/definitely/missing/derived_fields.yaml")


def test_industry_neutralize_injects_industry_source() -> None:
    from api.source_ref import decode_source_ref
    analysis = _analysis("industry_neutralize(close)")
    assert "close" in analysis.referenced_columns
    specs = [decode_source_ref(name) for name in analysis.referenced_columns]
    assert any(
        spec is not None
        and spec.table == "IndustryDaily"
        and spec.field == "IndustryCode"
        for spec in specs
    )


def test_one_arg_neutralize_and_size_aliases_fail_closed() -> None:
    from api.dsl_parser import DSLParseError, parse_expr
    formulas = [
        "neutralize(close)",
        "size_neutralize(close)",
        "market_cap_neutralize(close)",
        "size_industry_neutralize(close)",
    ]
    for formula in formulas:
        with pytest.raises(DSLParseError):
            parse_expr(formula, surface="lqtp")


def test_default_market_return_uses_registered_return_field() -> None:
    from api.source_ref import decode_source_ref
    analysis = _analysis("market_ret")
    specs = [decode_source_ref(name) for name in analysis.referenced_columns]
    specs = [spec for spec in specs if spec is not None]
    assert len(specs) == 1
    assert specs[0].table == "BenchmarkIndexDailyBar"
    assert specs[0].field == "Return"
    assert specs[0].params_dict() == {"index": "000985.SH"}
