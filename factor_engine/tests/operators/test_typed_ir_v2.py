from __future__ import annotations

import pytest

from factor_engine.api.dsl_parser import DSLParseError, parse_expr
from factor_engine.backend.operator_types import check_operator_types
from factor_engine.expr.field import FieldRef
from factor_engine.planner.logical_plan import PlanNode


def test_canonical_expression_is_stable_and_field_aware() -> None:
    from factor_engine.expr.canonical import canonical_expression

    first = parse_expr("ts_mean(field('close'), 3)")
    second = parse_expr("ts_mean(close, 3)")
    assert canonical_expression(first) == canonical_expression(second)
    assert "StockDailyBar.close" in canonical_expression(first)


def test_dsl_known_field_is_catalog_bound() -> None:
    expr = parse_expr("close")
    assert isinstance(expr, FieldRef)
    assert expr.field_id == "StockDailyBar.close"


def test_dsl_unknown_column_remains_legacy_column() -> None:
    expr = parse_expr("custom_alpha_input")
    assert not isinstance(expr, FieldRef)
    assert expr.name == "custom_alpha_input"


def test_ir_propagates_field_semantics() -> None:
    from factor_engine.ir.analyzer import Analyzer

    analysis = Analyzer().lower(parse_expr("ts_mean(close, 3)"))
    semantic = analysis.ir.semantic_attrs
    assert semantic["domain"] == "price_volume"
    assert semantic["frequency"] == "daily"
    assert semantic["cardinality"] == "many_to_one"
    assert semantic["pit_safe"] is True


def test_max_domains_is_enforced() -> None:
    from factor_engine.ir.analyzer import Analyzer, validate_max_domains

    analysis = Analyzer().lower(parse_expr("add(field('close'), field('pe_ratio'))"))
    assert validate_max_domains(analysis, max_domains=2) == []
    assert "exceeding max_domains=1" in validate_max_domains(analysis, max_domains=1)[0]


def test_type_checker_rejects_excess_arity() -> None:
    node = PlanNode(
        op="rank",
        inputs=[
            PlanNode(op="column", attrs={"name": "close", "dtype": "float64"}),
            PlanNode(op="literal", attrs={"value": 2}),
        ],
    )
    assert any("最多接受" in error for error in check_operator_types(node))


def test_boolean_signatures_are_strict() -> None:
    invalid = PlanNode(
        op="not_",
        inputs=[PlanNode(op="column", attrs={"name": "close", "dtype": "float64"})],
    )
    assert any("bool Series" in error for error in check_operator_types(invalid))

    valid = PlanNode(
        op="not_",
        inputs=[PlanNode(op="column", attrs={"name": "flag", "dtype": "bool"})],
    )
    assert check_operator_types(valid) == []


def test_semantic_signature_rejects_frequency_cardinality_and_pit() -> None:
    from factor_engine.backend.operator_types import ArgSpec, OperatorSignature, TypeKind

    signature = OperatorSignature(
        "typed_test",
        (ArgSpec("x", TypeKind.SERIES_FLOAT),),
        input_frequencies=(("daily",),),
        input_cardinalities=(("many_to_one",),),
        input_domains=(("price", "return"),),
    )
    node = PlanNode(
        op="typed_test",
        inputs=[PlanNode(op="column", attrs={
            "name": "holder_rows", "dtype": "float64", "frequency": "minute",
            "cardinality": "one_to_many", "domain": "shareholder", "pit_safe": False,
        })],
    )
    errors = signature.validate_node(node)
    assert any("frequency" in error for error in errors)
    assert any("cardinality" in error for error in errors)
    assert any("domain" in error for error in errors)
    assert any("PIT" in error for error in errors)


def test_numeric_signature_rejects_category_field() -> None:
    node = PlanNode(
        op="ts_mean",
        inputs=[PlanNode(op="column", attrs={"name": "industry", "dtype": "string"})],
        attrs={"window": 3},
    )
    assert any("numeric Series" in error for error in check_operator_types(node))
