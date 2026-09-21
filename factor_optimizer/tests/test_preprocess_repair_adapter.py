"""Real proposal -> FP execution, with independent lag and prefix oracles."""
import numpy as np
import pandas as pd
import pytest
pytest.importorskip("factor_preprocess.registry.transforms")
from factor_optimizer.adapters import preprocessing
from factor_optimizer.search.conditional_search import HierarchicalConditionalSearch, build_conditional_tree


def frame():
    return pd.DataFrame({"asset_id": ["A"] * 30, "date": pd.date_range("2020-01-01", periods=30),
                         "value": np.arange(1., 60., 2.)})


def compile_plan(family, params):
    return preprocessing.compile_smoothing_repair(family, params, natural_time_scale=20,
        training_context_ref="train-profile:fixed")


def test_sma_executes_actual_trailing_kernel_and_keeps_raw_input():
    values = frame(); original = values.copy(deep=True)
    plan = compile_plan("CAUSAL_SMOOTHING", {"method": "SMA", "natural_time_scale_relative": 0.15})
    result = plan.execute(values, allow_research=True)
    np.testing.assert_allclose(result.iloc[:6], [np.nan, np.nan, np.nan, 3., 5., 7.], equal_nan=True)
    pd.testing.assert_frame_equal(values, original)


@pytest.mark.parametrize("method", ["EWMA", "KAMA", "IIR", "Kalman", "SMA"])
def test_smoothing_plan_excludes_current_bar_and_future(method):
    plan = compile_plan("CAUSAL_SMOOTHING", {"method": method, "natural_time_scale_relative": 0.5})
    values = frame(); changed = values.copy()
    changed.loc[20:, "value"] = 9999.
    full = plan.execute(values, allow_research=True)
    altered = plan.execute(changed, allow_research=True)
    np.testing.assert_allclose(full.iloc[:21], altered.iloc[:21], equal_nan=True)
    np.testing.assert_allclose(full.iloc[:21], plan.execute(values.iloc[:21], allow_research=True), equal_nan=True)
    assert full.notna().any()


@pytest.mark.parametrize("relative", [True, False])
def test_zero_decay_is_rejected_before_evaluation_if_registry_forbids_unit_gain(relative):
    with pytest.raises(preprocessing.IneligibleSmoothingRepair, match="alpha"):
        compile_plan("DECAY_REFINEMENT", {"decay": 0., "half_life_relative": relative})


def test_absolute_decay_is_previous_state_retention_not_new_sample_weight():
    plan = compile_plan("DECAY_REFINEMENT", {"decay": 0.75, "half_life_relative": False})
    np.testing.assert_allclose(plan.execute(frame(), allow_research=True).iloc[:4],
                               [np.nan, 1., 1.5, 2.375], equal_nan=True)


def test_relative_decay_maps_to_declared_train_scale():
    plan = compile_plan("DECAY_REFINEMENT", {"decay": 0.5, "half_life_relative": True})
    assert dict(plan.parameters) == {"halflife": 10., "min_periods": 1}


def test_real_conditional_proposals_are_executable():
    search = HierarchicalConditionalSearch(build_conditional_tree(["CAUSAL_SMOOTHING", "DECAY_REFINEMENT"]), seed=71)
    executed = 0
    for _ in range(20):
        candidate = search.propose_recipe()
        try:
            plan = compile_plan(candidate["family"], candidate["params"])
        except preprocessing.IneligibleSmoothingRepair:
            continue  # Count as ineligible, never as a scored success or raw fallback.
        assert len(plan.execute(frame(), allow_research=True)) == len(frame())
        executed += 1
    assert executed >= 10


def test_research_adapter_cannot_claim_production_or_invent_time_scale():
    plan = compile_plan("CAUSAL_SMOOTHING", {"method": "SMA", "natural_time_scale_relative": 0.5})
    with pytest.raises(ValueError):
        plan.execute(frame())
    with pytest.raises(ValueError):
        preprocessing.compile_smoothing_repair("DECAY_REFINEMENT", {"decay": .5, "half_life_relative": True},
            natural_time_scale=float("nan"), training_context_ref="train:1")
    with pytest.raises(ValueError):
        compile_plan("DECAY_REFINEMENT", {"decay": .5, "half_life_relative": 1})
