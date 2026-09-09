import numpy as np
import pandas as pd

from factor_engine.backend.cleaned_bridge import (
    compile_operator_recipe, execute_operator_recipe,
)


def test_multi_step_recipe_uses_one_input_load_and_existing_plan_nodes():
    panel = pd.DataFrame(
        [[1.0, 3.0], [np.nan, 5.0], [7.0, 9.0]],
        index=pd.date_range("2024-01-01", periods=3), columns=["A", "B"],
    )
    steps = (("ffill_limit", {"max_periods": 1}), ("rank", {}))
    plan = compile_operator_recipe(steps)
    assert plan.op == "rank"
    assert plan.inputs[0].op == "ffill_limit"
    assert plan.inputs[0].inputs[0].op == "column"
    stats = {}
    result = execute_operator_recipe(
        panel, steps, runtime_stats=stats, allow_research=True,
    )
    assert result.shape == panel.shape
    # Forward-filled A remains below B on every date; canonical rank is 0..1.
    np.testing.assert_array_equal(result.to_numpy(), [[0., 1.]] * 3)
    assert result.index.equals(panel.index)
    assert result.columns.equals(panel.columns)
    assert stats["recipe_execution_authority"] == "research_local"
    assert stats["recipe_input_load_count"] == 1
    assert stats["recipe_operator_count"] == 2


def test_unknown_recipe_operator_fails_before_execution():
    try:
        compile_operator_recipe((("definitely_unknown", {}),))
    except (KeyError, ValueError):
        pass
    else:
        raise AssertionError("unknown operator must fail closed")
