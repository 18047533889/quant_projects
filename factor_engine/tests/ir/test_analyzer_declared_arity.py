from types import SimpleNamespace

import pytest

from factor_engine.cleaned_operators.base import (
    OperatorMetadata,
    OperatorParameterError,
    _normalise_call,
    validate_operator_call_arity,
)
from factor_engine.expr import CleanedCall, ColumnRef, Literal
from factor_engine.ir.analyzer import Analyzer


def test_analyzer_rejects_positional_call_beyond_named_contract():
    # Regression: Analyzer used to lower five inputs even though the selected
    # implementation declares only close/high/low/window.
    expression = CleanedCall(
        "ts_abdi_ranaldo_spread",
        (
            ColumnRef("open"),
            ColumnRef("high"),
            ColumnRef("low"),
            ColumnRef("close"),
            Literal(20),
        ),
    )

    with pytest.raises(OperatorParameterError, match="at most 4"):
        Analyzer().lower(expression)


def test_analyzer_rejects_amihud_call_missing_declared_window():
    # The three positional values bind ret/close/volume; the four-parameter
    # contract still has no window, even when the third literal looks like one.
    expression = CleanedCall(
        "amihud_illiquidity",
        (ColumnRef("ret"), ColumnRef("amount"), Literal(20)),
    )

    with pytest.raises(OperatorParameterError, match="missing required.*window"):
        Analyzer().lower(expression)


def test_analyzer_rejects_markov_undeclared_min_periods_keyword():
    expression = CleanedCall(
        "ts_markov_transition_surprisal",
        (ColumnRef("ret"),),
        (("window", 60), ("bins", 3), ("lag", 1), ("min_periods", 3)),
    )

    with pytest.raises(OperatorParameterError, match="undeclared keyword.*min_periods"):
        Analyzer().lower(expression)


def test_analyzer_accepts_markov_declared_alias_and_authored_defaults():
    expression = CleanedCall(
        "ts_markov_transition_surprisal",
        (ColumnRef("ret"),),
        (("min_transition_count", 3),),
    )

    Analyzer().lower(expression)


def _operator(metadata, calculate=lambda: None):
    return SimpleNamespace(metadata=metadata, calculate=calculate)


def test_unnamed_legacy_contract_without_explicit_arity_remains_kernel_implied():
    metadata = OperatorMetadata(name="legacy", category="elementwise")
    validate_operator_call_arity(_operator(metadata), (object(), object()), {})


def test_explicit_input_arity_remains_exact_for_unnamed_contract():
    metadata = OperatorMetadata(
        name="binary", category="elementwise", input_arity=2
    )
    operator = _operator(metadata)
    validate_operator_call_arity(operator, (object(), object()), {})
    with pytest.raises(OperatorParameterError, match="input_arity=2"):
        validate_operator_call_arity(operator, (object(), object(), object()), {})


def test_named_topology_honors_total_arity_mixed_input_and_optional_default():
    metadata = OperatorMetadata(
        name="power_like",
        category="elementwise",
        param_names=["base", "exponent", "clip"],
        panel_params=("base",),
        mixed_params=("exponent",),
        scalar_params=("clip",),
        total_positional_arity=3,
    )

    def calculate(base, exponent, clip=None):
        return base

    operator = _operator(metadata, calculate)
    validate_operator_call_arity(operator, (object(), 2), {})
    validate_operator_call_arity(operator, (object(),), {"exponent": 2})
    with pytest.raises(OperatorParameterError, match="at most 3"):
        validate_operator_call_arity(operator, (object(), 2, None, "extra"), {})


@pytest.mark.parametrize("tag", ["variadic", "dynamic_inputs"])
def test_explicit_variadic_contract_remains_unbounded(tag):
    metadata = OperatorMetadata(
        name="row_sum", category="cross_sectional", tags=[tag]
    )
    validate_operator_call_arity(
        _operator(metadata), tuple(object() for _ in range(30)), {"custom_input": object()}
    )


def test_runtime_normaliser_retains_variadic_scope_after_arity_validation():
    metadata = OperatorMetadata(
        name="mean_like",
        category="time_series",
        param_names=["x", "window"],
    )
    args, kwargs = _normalise_call(metadata, (object(), 20), {})
    assert len(args) == 2
    assert kwargs == {}
