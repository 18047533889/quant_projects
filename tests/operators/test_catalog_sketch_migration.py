from __future__ import annotations

import pytest

from factor_engine.tools.catalog_sketch_migration import (
    convert_complete_parameter_sketch,
    migrate_catalog_sketch_formula,
)


def test_conversion_is_explicitly_opt_in() -> None:
    source = "ts_mean [x=close; window=20]"
    result = migrate_catalog_sketch_formula(source)
    assert result.formula == source
    assert result.status == "disabled"


def test_complete_registered_sketch_converts_and_binds() -> None:
    result = migrate_catalog_sketch_formula(
        "ts_mean [x=close; window=add(10, 10)]", enabled=True
    )
    assert result.formula == "ts_mean(x=close, window=add(10, 10))"
    assert result.converted
    assert result.status == "converted"


@pytest.mark.parametrize(
    ("source", "status"),
    [
        ("not_a_registered_operator [x=close]", "dsl_validation_failed"),
        ("ts_mean [x=close; x=volume]", "invalid_or_duplicate_key"),
        ("ts_mean [x=close; window=]", "invalid_value"),
        ("ts_mean [price field=close]", "invalid_or_duplicate_key"),
        ("ts_mean [x=natural language value]", "invalid_value"),
        ("ts_mean [x=close", "not_complete_sketch"),
        ("prefix + ts_mean [x=close]", "not_complete_sketch"),
        ("ts_mean[x=close][window=20]", "invalid_assignment_list"),
    ],
)
def test_unresolved_sketches_are_retained(source: str, status: str) -> None:
    result = migrate_catalog_sketch_formula(source, enabled=True)
    assert result.formula == source
    assert not result.converted
    assert result.status == status


def test_pure_converter_does_not_enable_python_subscripts() -> None:
    assert not convert_complete_parameter_sketch("series[window]").converted
    assert not convert_complete_parameter_sketch("obj.method [x=close]").converted


def test_supplied_frozen_parser_is_reused_as_the_only_validator() -> None:
    class RejectingParser:
        def __init__(self) -> None:
            self.seen: list[str] = []

        def parse(self, text: str) -> None:
            self.seen.append(text)
            raise ValueError("rejected by frozen snapshot")

    parser = RejectingParser()
    source = "ts_mean [x=close; window=20]"
    result = migrate_catalog_sketch_formula(source, enabled=True, parser=parser)
    assert result.formula == source
    assert result.status == "dsl_validation_failed"
    assert parser.seen == ["ts_mean(x=close, window=20)"]


def test_non_string_and_budget_fail_closed() -> None:
    with pytest.raises(TypeError):
        migrate_catalog_sketch_formula(None, enabled=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="budget"):
        migrate_catalog_sketch_formula("x" * 65_537, enabled=True)
