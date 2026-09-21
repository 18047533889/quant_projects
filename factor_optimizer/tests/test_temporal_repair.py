import numpy as np
import pandas as pd
import pytest

from factor_optimizer.adapters.repair_execution import compile_value_repair


@pytest.mark.parametrize("family,params", [
    ("REPRESENTATION_RANK", {"rank_axis": "ts", "tie_method": "average", "window": 3}),
    ("REPRESENTATION_RANK", {"rank_axis": "ts", "tie_method": "min", "window": 3}),
    ("REPRESENTATION_ZSCORE", {"zscore_axis": "ts", "cap": 3., "window": 3}),
])
def test_declared_temporal_repairs_execute_and_freeze_window(family, params):
    plan = compile_value_repair(family, params, natural_time_scale=10., training_context_ref="TRAIN")
    data = pd.DataFrame({"date": range(6), "asset_id": "a", "value": [1., 2., 3., 4., 5., 6.]})
    out = plan.execute(data, allow_research=True)
    assert np.isnan(out[:3]).all()
    assert np.isfinite(out[3:]).all()
    np.testing.assert_array_equal(out[:5], plan.execute(data.iloc[:5], allow_research=True))
    other = compile_value_repair(family, dict(params, window=4), natural_time_scale=10.,
                                training_context_ref="TRAIN")
    assert plan.identity != other.identity


def test_automatic_grid_contains_both_temporal_rank_ties_and_zscore_windows():
    from factor_optimizer.research_batch import _specs, BatchOptimizationConfig
    specs = _specs(BatchOptimizationConfig())
    rank = [p for f,p in specs if f=="REPRESENTATION_RANK" and p["rank_axis"]=="ts"]
    zscore = [p for f,p in specs if f=="REPRESENTATION_ZSCORE" and p["zscore_axis"]=="ts"]
    assert {p["tie_method"] for p in rank} == {"average", "min"}
    assert {p["window"] for p in rank} == {5, 10, 20}
    assert {p["window"] for p in zscore} == {5, 10, 20}

def test_temporal_candidates_and_selection_ignore_test_values_and_labels():
    from dataclasses import replace
    from .test_research_batch_boundaries import _fixture
    from factor_optimizer.research_batch import BatchOptimizationConfig, optimize_factor_batch
    batch, labels = _fixture()
    config = BatchOptimizationConfig(selection_objective="rank_ic", bootstrap_draws=99,
        families=("REPRESENTATION_RANK", "REPRESENTATION_ZSCORE"))
    original = optimize_factor_batch(batch, labels, config=config, allow_research=True)
    values = batch.values.copy()
    targets = labels.values.copy()
    idx = list(original.split.test_indices)
    values[idx] = 1e200
    targets[idx] = np.nan
    changed = optimize_factor_batch(replace(batch, values=values), replace(labels, values=targets),
                                    config=config, allow_research=True)
    for name in batch.factor_ids:
        left, right = original.factors[name], changed.factors[name]
        assert left.candidates == right.candidates
        assert left.plan_identity == right.plan_identity
        assert left.selected_family == right.selected_family
        assert left.validation_lower_bound == right.validation_lower_bound
