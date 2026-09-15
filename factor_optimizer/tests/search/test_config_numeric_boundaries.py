import pytest

from factor_optimizer.contracts.search_budget import SearchBudget
from factor_optimizer.search.runner import SearchConfig


@pytest.mark.parametrize("field", ["plateau_window", "max_concurrency"])
@pytest.mark.parametrize("value", [True, 1.0, float("nan"), "1"])
def test_counts_require_integers(field, value):
    with pytest.raises(TypeError, match=field):
        SearchConfig(SearchBudget(2, 2, 2.0), **{field: value})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_plateau_threshold_requires_finite_number(value):
    with pytest.raises(ValueError, match="plateau_threshold"):
        SearchConfig(SearchBudget(2, 2, 2.0), plateau_threshold=value)


@pytest.mark.parametrize("value", [True, False, "0.1", None])
def test_plateau_threshold_rejects_wrong_types(value):
    with pytest.raises(TypeError, match="plateau_threshold"):
        SearchConfig(SearchBudget(2, 2, 2.0), plateau_threshold=value)


def test_zero_threshold_and_one_window_remain_supported():
    config = SearchConfig(SearchBudget(2, 2, 2.0), plateau_threshold=0, plateau_window=1)
    assert config.plateau_threshold == 0


@pytest.mark.parametrize("field", ["enable_multifidelity", "require_evaluation_protocol"])
@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, [], {}])
def test_search_switches_require_booleans(field, value):
    with pytest.raises(TypeError, match=field):
        SearchConfig(SearchBudget(2, 2, 2.0), **{field: value})


@pytest.mark.parametrize("value", [False, True])
def test_explicit_boolean_switches_roundtrip(value):
    config = SearchConfig(SearchBudget(2, 2, 2.0), enable_multifidelity=value,
                          require_evaluation_protocol=value)
    restored = SearchConfig.from_dict(config.to_dict())
    assert restored.enable_multifidelity is value
    assert restored.require_evaluation_protocol is value


@pytest.mark.parametrize("value", [None, False, 1, [], {}])
def test_execution_mode_rejects_non_enum_non_string(value):
    with pytest.raises(TypeError, match="execution_mode"):
        SearchConfig(SearchBudget(2, 2, 2.0), execution_mode=value)


def test_production_lookalike_cannot_bypass_mode_validation():
    from types import SimpleNamespace
    with pytest.raises(TypeError, match="execution_mode"):
        SearchConfig(SearchBudget(2, 2, 2.0),
                     execution_mode=SimpleNamespace(value="production"))


def test_research_mode_string_still_normalizes():
    from factor_optimizer.capabilities import ExecutionMode
    config = SearchConfig(SearchBudget(2, 2, 2.0), execution_mode="research_only")
    assert config.execution_mode is ExecutionMode.RESEARCH_ONLY


@pytest.mark.parametrize("value", [False, True])
def test_boolean_evaluation_cost_is_not_a_budget(value):
    with pytest.raises(TypeError, match="evaluation_cost_units"):
        SearchConfig(SearchBudget(2, 2, 2.0), evaluation_cost_units=value)


@pytest.mark.parametrize("value", [None, 0, 0.0, 1.5])
def test_valid_evaluation_costs_roundtrip(value):
    config = SearchConfig(SearchBudget(2, 2, 2.0), evaluation_cost_units=value)
    assert SearchConfig.from_dict(config.to_dict()).evaluation_cost_units == value
