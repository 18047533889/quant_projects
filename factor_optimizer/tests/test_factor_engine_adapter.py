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


def test_analyzer_lookback_is_authoritative_for_complexity(adapter):
    from factor_engine.expr.cleaned_call import CleanedCall
    from factor_engine.expr.column import ColumnRef

    expr = CleanedCall("ts_mean", (ColumnRef("close"),), (("window", 5),))
    result = adapter.estimate_complexity(expr)
    assert result["operator_count"] == 1
    assert result["lookback_periods"] == 4
    assert result["estimated_cost"] == 1.2
    assert result["fe_analysis"]["available"]
    assert result["fe_analysis"]["lookback"] == 4
    assert result["fe_analysis"]["has_ts_op"]
    assert result["fe_analysis"]["referenced_columns"] == ["close"]


def test_analyzer_lookback_uses_fe_canonical_lag_parameter_n(adapter):
    from factor_engine.expr.cleaned_call import CleanedCall
    from factor_engine.expr.column import ColumnRef

    expr = CleanedCall("ts_delay", (ColumnRef("close"),), (("n", 5),))
    result = adapter.estimate_complexity(expr)
    assert result["lookback_periods"] == 5
    assert result["estimated_cost"] == 1.25
    assert result["fe_analysis"]["lookback"] == 5


def test_analyzer_preserves_two_row_floor_for_stateless_call(adapter):
    from factor_engine.expr.cleaned_call import CleanedCall
    from factor_engine.expr.column import ColumnRef

    result = adapter.estimate_complexity(CleanedCall("rank", (ColumnRef("close"),)))
    assert result["lookback_periods"] == 2
    assert result["estimated_cost"] == 1.1
    assert result["fe_analysis"]["lookback"] == 2


def test_analyzer_failure_uses_cleaned_call_kwargs_fallback(adapter, monkeypatch):
    from factor_engine.expr.cleaned_call import CleanedCall
    from factor_engine.expr.column import ColumnRef
    from factor_engine.ir.analyzer import Analyzer

    def fail_lower(self, expr):
        raise ValueError("analyzer unavailable")

    monkeypatch.setattr(Analyzer, "lower", fail_lower)
    expr = CleanedCall("ts_mean", (ColumnRef("close"),), (("window", 5),))
    result = adapter.estimate_complexity(expr)
    assert result["lookback_periods"] == 5
    assert result["estimated_cost"] == 1.25
    assert result["fe_analysis"] == {
        "available": False, "reason": "analyzer unavailable",
    }


def test_analyzer_window_one_preserves_two_row_floor(adapter):
    from factor_engine.expr.cleaned_call import CleanedCall
    from factor_engine.expr.column import ColumnRef

    expr = CleanedCall("ts_mean", (ColumnRef("close"),), (("window", 1),))
    result = adapter.estimate_complexity(expr)
    assert result["lookback_periods"] == 2
    assert result["estimated_cost"] == 1.1
    assert result["fe_analysis"]["lookback"] == 2


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


def test_target_operator_scalar_parameter_is_validated(adapter):
    result = adapter.validate_mutation(
        {"target_operator": "ts_mean", "parameter_name": "window", "new_value": 0},
        spec=None,
    )
    assert not result["is_legal"]
    assert "window must be >= 1" in result["reason"]


def test_target_operator_scalar_parameter_is_canonicalized(adapter):
    result = adapter.validate_mutation(
        {"target_operator": "ts_sma", "parameter_name": "d", "new_value": 5},
        spec=None,
    )
    assert result["is_legal"], result
    assert result["metadata"]["canonical_operators"]["target_operator"] == "ts_mean"
    assert result["metadata"]["normalized_operator_parameters"]["target_operator"] == {"window": 5}


def test_scalar_parameter_is_rejected_when_target_and_replacement_are_both_named(adapter):
    result = adapter.validate_mutation(
        {
            "target_operator": "ts_mean",
            "replacement_operator": "ts_delay",
            "parameter_name": "window",
            "new_value": 5,
        },
        spec=None,
    )
    assert not result["is_legal"]
    assert "ambiguous" in result["reason"]


def test_malformed_parameter_name_fails_closed(adapter):
    result = adapter.validate_mutation(
        {"operator_name": "ts_mean", "parameter_name": ["window"], "new_value": 5},
        spec=None,
    )
    assert not result["is_legal"]
    assert "parameter_name must be a nonempty string" in result["reason"]


def test_operator_metadata_uses_fe_canonical_names(adapter):
    metadata = adapter.get_operator_metadata(["CS_RANK", "rank", "not_a_real_operator"])
    assert tuple(metadata) == ("rank",)
    assert metadata["rank"]["status"] in {"implemented", "production"}


def test_adapter_adds_only_sibling_package_parent_once(monkeypatch):
    import sys
    from pathlib import Path

    repo_root = str(Path(__file__).resolve().parents[2])
    package_dir = str(Path(repo_root) / "factor_engine")
    monkeypatch.setattr(sys, "path", [entry for entry in sys.path
                                     if entry not in {repo_root, package_dir}])
    create_fe_adapter()
    create_fe_adapter()
    assert sys.path.count(repo_root) == 1
    assert package_dir not in sys.path


def test_semantic_hash_normalizes_fe_parameter_aliases_for_seen_dedup(adapter):
    from factor_engine.expr.cleaned_call import CleanedCall
    from factor_engine.expr.column import ColumnRef
    from factor_optimizer.seen.identity import SeenCache

    close = ColumnRef("close")
    alias = CleanedCall("ts_sma", (close,), (("d", 5),))
    canonical = CleanedCall("ts_mean", (close,), (("window", 5),))
    other_value = CleanedCall("ts_mean", (close,), (("window", 6),))
    other_operator = CleanedCall("ts_delay", (close,), (("n", 5),))

    assert adapter.compute_canonical_hash(alias) == adapter.compute_canonical_hash(canonical)
    assert adapter.compute_canonical_hash(alias) != adapter.compute_canonical_hash(other_value)
    assert adapter.compute_canonical_hash(alias) != adapter.compute_canonical_hash(other_operator)

    seen = SeenCache(adapter)
    assert seen.check_and_mark(alias, "trial-1")[0] is False
    assert seen.check_and_mark(canonical, "trial-2")[0] is True
    assert seen.size() == 1


def test_semantic_hash_rejects_invalid_fe_parameters(adapter):
    from factor_engine.expr.cleaned_call import CleanedCall
    from factor_engine.expr.column import ColumnRef

    invalid_window = CleanedCall("ts_sma", (ColumnRef("close"),), (("d", 0),))
    with pytest.raises(ValueError, match="window must be >= 1"):
        adapter.compute_canonical_hash(invalid_window)

    unknown_operator = CleanedCall("not_a_real_operator", (ColumnRef("close"),))
    with pytest.raises((KeyError, ValueError)):
        adapter.compute_canonical_hash(unknown_operator)
