import pytest

from factor_engine.api.dsl_parser import ComplexityBudget, DSLParseError, parse_expr


@pytest.mark.parametrize("text",['col("x", window=5, window=10)', 'ts_mean(close, window=5, d=10)', 'ts_mean(close, 5, window=10)'])
def test_duplicate_keyword_spellings_are_rejected(text):
    with pytest.raises(DSLParseError):
        parse_expr(text)


@pytest.mark.parametrize("text",['col("a" * 9)', 'col("a" + "12345678")', 'col(1000000000 * 1000000000)', 'col(1 / 0)'])
def test_constant_intermediates_are_bounded_before_use(text):
    budget=ComplexityBudget(max_string_literal_bytes=8,max_literal_magnitude=1e9)
    with pytest.raises(DSLParseError):
        parse_expr(text,budget=budget)


def test_normal_constant_and_column_arithmetic_still_parse():
    parse_expr('close / (volume + 1)')
    parse_expr('ts_mean(close, window=5)')
    parse_expr('col("ab" + "cd")',budget=ComplexityBudget(max_string_literal_bytes=8))


def test_registry_binding_failure_is_not_silently_ignored(monkeypatch):
    from factor_engine.api.dsl_parser import _ExprBuilder
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    builder=_ExprBuilder()
    monkeypatch.setattr(OperatorRegistry,"resolve_canonical",classmethod(lambda cls,name:(_ for _ in ()).throw(RuntimeError("registry unavailable"))))
    with pytest.raises(DSLParseError,match="registry unavailable"):
        builder.build('ts_mean(close, window=5)')
