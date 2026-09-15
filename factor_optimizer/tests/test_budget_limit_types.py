import pytest

from factor_optimizer.contracts.search_budget import SearchBudget


@pytest.mark.parametrize("field", ["max_trials", "max_evaluations", "max_llm_calls"])
@pytest.mark.parametrize("value", [True, False, 1.5, float("nan"), float("inf"), "2"])
def test_count_limits_require_actual_integers(field, value):
    with pytest.raises(TypeError, match=field):
        SearchBudget(**{field: value})


@pytest.mark.parametrize("value", [True, False])
def test_boolean_cost_limit_is_rejected(value):
    with pytest.raises(TypeError, match="max_cost_units"):
        SearchBudget(max_cost_units=value)


@pytest.mark.parametrize("llm", [None, 0, 2])
def test_valid_integer_limits_roundtrip(llm):
    budget = SearchBudget(1, 1, 0.5, max_llm_calls=llm)
    assert SearchBudget.from_dict(budget.to_dict()) == budget
