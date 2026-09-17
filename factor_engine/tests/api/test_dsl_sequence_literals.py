import numpy as np
import pandas as pd
import pytest

from factor_engine.api.dsl_parser import DSLParseError, DSLParser
from factor_engine.cleaned_operators.registry import OperatorRegistry


def _literal_value(value):
    return getattr(value, "value", value)


def test_declared_tuple_sequence_parses_and_matches_operator_calculation():
    parser = DSLParser(surface="compat_research")
    expr = parser.parse(
        "ts_multiscale_trend_consensus(close, 60, (5, 10, 20, 40))"
    )
    assert expr.op == "ts_multiscale_trend_consensus"
    window = _literal_value(expr.args[1])
    scales = _literal_value(expr.args[2])
    assert window == 60
    assert scales == (5, 10, 20, 40)

    index = pd.date_range("2024-01-01", periods=100)
    close = pd.DataFrame({"A": 100 + np.arange(100) + np.sin(np.arange(100))}, index=index)
    operator = OperatorRegistry.get(expr.op, mode="any")
    from_dsl_literals = operator.calculate(close, window, scales)
    explicit = operator.calculate(close, window=60, scales=(5, 10, 20, 40))
    np.testing.assert_allclose(from_dsl_literals, explicit, equal_nan=True)
    assert np.isfinite(from_dsl_literals.to_numpy(dtype=float)).any()


def test_declared_list_sequence_is_supported_without_changing_values():
    expr = DSLParser(surface="compat_research").parse(
        "ts_multiscale_trend_consensus(close, scales=[5, 10, 20, 40])"
    )
    assert _literal_value(dict(expr.kwargs)["scales"]) == [5, 10, 20, 40]


@pytest.mark.parametrize(
    "formula, message",
    [
        ("ts_mean(close, (5, 10))", "does not declare"),
        ("ts_multiscale_trend_consensus(close, scales=(5, close))", "flat scalar"),
        ("ts_multiscale_trend_consensus(close, scales=((5, 10),))", "flat scalar"),
        ("ts_multiscale_trend_consensus(close, scales=[x for x in close])", "Unsupported syntax"),
    ],
)
def test_unsafe_or_undeclared_sequence_forms_are_rejected(formula, message):
    with pytest.raises(DSLParseError, match=message):
        DSLParser(surface="compat_research").parse(formula)


def test_sequence_length_keeps_existing_variadic_bound():
    values = ",".join(str(value + 2) for value in range(33))
    with pytest.raises(DSLParseError, match="max_variadic_inputs"):
        DSLParser(surface="compat_research").parse(
            f"ts_multiscale_trend_consensus(close, scales=({values}))"
        )
