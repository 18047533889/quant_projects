from __future__ import annotations


def test_typed_field_constructor_is_not_treated_as_runtime_operator() -> None:
    from factor_engine.api.mining_integration import validate_production_dsl

    ok, message = validate_production_dsl(
        'ts_mean(field("close"), 2)', market="ashare"
    )

    assert ok, message


def test_unknown_raw_column_remains_rejected() -> None:
    from factor_engine.api.mining_integration import validate_production_dsl

    ok, message = validate_production_dsl(
        'ts_mean(col("not_a_catalog_field"), 2)', market="ashare"
    )

    assert not ok
    assert "unknown raw column" in message


def test_unknown_typed_field_remains_rejected() -> None:
    from factor_engine.api.mining_integration import validate_production_dsl

    import pytest

    with pytest.raises(KeyError, match="not_a_catalog_field"):
        validate_production_dsl(
            'ts_mean(field("not_a_catalog_field"), 2)', market="ashare"
        )


def test_malformed_typed_field_remains_rejected() -> None:
    from factor_engine.api.mining_integration import validate_production_dsl

    ok, message = validate_production_dsl('ts_mean(field(), 2)', market="ashare")

    assert not ok
    assert "field" in message.lower()


def test_forbidden_operator_remains_rejected_with_typed_field() -> None:
    from factor_engine.api.mining_integration import validate_production_dsl

    ok, message = validate_production_dsl(
        'bfill(field("close"))', market="ashare"
    )

    assert not ok
    assert "bfill" in message


def test_fastpath_ast_scanner_only_classifies_runtime_operators() -> None:
    from factor_engine.backend.production_fastpath_gate import (
        check_production_fastpath_formula_ops,
    )

    result = check_production_fastpath_formula_ops(
        'field("close")', use_real_plan=False
    )

    assert result.ok
    assert result.ops_checked == ()
