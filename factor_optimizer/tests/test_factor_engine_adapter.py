"""Focused contracts for the optional FactorEngine adapter."""
import pytest

pytest.importorskip("factor_engine.cleaned_operators.registry")
from factor_optimizer.adapters.factor_engine import create_fe_adapter


@pytest.fixture(scope="module")
def adapter():
    return create_fe_adapter()


def test_operator_alias_is_canonicalized(adapter):
    result = adapter.validate_mutation({"target_operator": "CS_RANK"}, spec=None)
    assert result["is_legal"]
    assert result["metadata"]["canonical_operators"]["target_operator"] == "rank"


def test_operator_alias_and_valid_parameter_alias_are_canonicalized(adapter):
    result = adapter.validate_mutation(
        {"target_operator": "ts_sma", "target_operator_parameters": {"d": 5}},
        spec=None,
    )
    assert result["is_legal"], result
    assert result["metadata"]["canonical_operators"]["target_operator"] == "ts_mean"
    assert result["metadata"]["normalized_operator_parameters"]["target_operator"] == {"window": 5}


def test_invalid_operator_parameter_is_rejected_by_fe_contract(adapter):
    result = adapter.validate_mutation(
        {"target_operator": "ts_mean", "target_operator_parameters": {"window": 0}},
        spec=None,
    )
    assert not result["is_legal"]
    assert "window must be >= 1" in result["reason"]


def test_malformed_nonstring_operator_reference_fails_closed(adapter):
    import numpy as np

    result = adapter.validate_mutation({"target_operator": np.array(["rank"])}, spec=None)
    assert not result["is_legal"]
    assert result["reason"] == "target_operator must be a string"


def test_replacement_operator_parameters_use_fe_domain_validation(adapter):
    result = adapter.validate_mutation(
        {
            "target_operator": "rank",
            "replacement_operator": "ts_mean",
            "replacement_operator_parameters": {"window": 0},
        },
        spec=None,
    )
    assert not result["is_legal"]
    assert "window must be >= 1" in result["reason"]


def test_unknown_operator_is_rejected(adapter):
    result = adapter.validate_mutation({"target_operator": "not_a_real_operator"}, spec=None)
    assert not result["is_legal"]
    assert "not found in FE catalog" in result["reason"]


def test_mutation_without_operator_remains_not_applicable_to_fe(adapter):
    result = adapter.validate_mutation({"new_window": 12}, spec=None)
    assert result["is_legal"]
    assert result["metadata"]["operator_check"] == "not_applicable"


def test_analyzer_evidence_is_additive_and_legacy_score_is_unchanged(adapter):
    from factor_engine.expr.cleaned_call import CleanedCall
    from factor_engine.expr.column import ColumnRef

    expr = CleanedCall("ts_mean", (ColumnRef("close"),), (("window", 5),))
    result = adapter.estimate_complexity(expr)
    assert result["operator_count"] == 1
    assert result["lookback_periods"] == 0
    assert result["estimated_cost"] == 1.0
    assert result["fe_analysis"]["available"]
    assert result["fe_analysis"]["lookback"] == 4
    assert result["fe_analysis"]["has_ts_op"]
    assert result["fe_analysis"]["referenced_columns"] == ["close"]


def test_malformed_mutation_parameters_fail_closed(adapter):
    class Mutation:
        parameters = ["rank"]

    result = adapter.validate_mutation(Mutation(), spec=None)
    assert not result["is_legal"]
    assert "must be a mapping" in result["reason"]


def test_empty_wrong_type_operator_parameters_fail_closed(adapter):
    result = adapter.validate_mutation(
        {"target_operator": "ts_mean", "target_operator_parameters": []}, spec=None,
    )
    assert not result["is_legal"]
    assert "must be a mapping" in result["reason"]


def test_malformed_parameter_name_fails_closed(adapter):
    result = adapter.validate_mutation(
        {"operator_name": "ts_mean", "parameter_name": ["window"], "new_value": 5},
        spec=None,
    )
    assert not result["is_legal"]
    assert "parameter_name must be a nonempty string" in result["reason"]
