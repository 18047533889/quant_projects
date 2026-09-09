from types import SimpleNamespace

import pandas as pd
import pytest

from factor_engine.backend.cleaned_bridge import execute_operator_recipe
from factor_engine.backend.context import ExecutionContext


class _Source:
    def load_column(self, name):
        raise AssertionError("spy backend should not read")


class _Backend:
    def __init__(self, result):
        self.result = result
        self.context = None

    def execute(self, plan, context):
        self.context = context
        return self.result


def _panel():
    return pd.DataFrame([[1.0]], index=pd.Index([1], name="timestamp"), columns=["A"])


def test_production_recipe_propagates_caller_authority_and_backend():
    panel = _panel()
    budget = SimpleNamespace(name="approved-budget")
    shared = {}
    context = ExecutionContext(
        data_source=_Source(), run_mode="production", execution_id="approved-run",
        query_budget=budget, shared_result_cache=shared, selected_backend="approved-auto",
    )
    backend = _Backend(panel)
    stats = {}
    out = execute_operator_recipe(
        panel, (), runtime_stats=stats, execution_context=context, backend=backend,
    )
    pd.testing.assert_frame_equal(out, panel)
    assert backend.context.run_mode == "production"
    assert backend.context.execution_id == "approved-run"
    assert backend.context.query_budget is budget
    assert backend.context.shared_result_cache is shared
    assert backend.context.selected_backend == "approved-auto"
    assert stats["recipe_execution_authority"] == "supplied_production"
    assert stats["recipe_backend"] == "_Backend"


def test_production_or_implicit_research_without_authority_fails_closed():
    panel = _panel()
    production = ExecutionContext(data_source=_Source(), run_mode="production")
    with pytest.raises(ValueError, match="caller-selected backend"):
        execute_operator_recipe(panel, (), execution_context=production)
    with pytest.raises(ValueError, match="approved ExecutionContext"):
        execute_operator_recipe(panel, ())


def test_explicit_research_path_is_labeled_with_actual_backend():
    panel = _panel()
    backend = _Backend(panel)
    stats = {}
    execute_operator_recipe(panel, (), runtime_stats=stats, backend=backend, allow_research=True)
    assert backend.context.run_mode == "research"
    assert stats == {
        "recipe_execution_authority": "research_local",
        "recipe_backend": "_Backend",
        "recipe_input_load_count": 0,
        "recipe_operator_count": 0,
    }
