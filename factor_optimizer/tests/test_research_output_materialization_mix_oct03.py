"""Mixed RAW and failed-materialization output keeps factor order and masks."""

import numpy as np
import pytest

from factor_preprocess.contracts.treatment_lineage import TransformLineage
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle


def test_middle_materialization_failure_preserves_prefix_and_neighbor_outputs(monkeypatch):
    from factor_optimizer.research_batch import (
        BatchOptimizationConfig,
        optimize_factor_batch,
    )
    from factor_optimizer.research_baseline import BaselineRepairPlan

    rng = np.random.default_rng(82)
    times, assets = 240, 40
    raw = rng.normal(size=(times, assets))
    time_axis = AxisRef("time", "int", times, np.arange(times))
    asset_axis = AxisRef(
        "asset", "str", assets, np.array([f"a{i}" for i in range(assets)])
    )
    values = np.stack((raw + 3.0, raw, raw - 2.0), axis=-1).astype(np.float32)
    factor_ids = ("left_raw", "middle_baseline", "right_raw")
    batch = FactorBatch(factor_ids, time_axis, asset_axis, values)
    labels = LabelBundle(
        "synthetic", raw, 1,
        decision_time=tuple(range(times)),
        label_start_time=tuple(range(1, times + 1)),
        label_end_time=tuple(range(2, times + 2)),
        asset_axis=asset_axis,
    )
    config = BatchOptimizationConfig(
        selection_objective="rank_ic", families=("SIGN_ORIENTATION",)
    )
    kwargs = dict(
        allow_research=True,
        config=config,
        lineages={"middle_baseline": TransformLineage()},
    )

    clean = optimize_factor_batch(batch, labels, **kwargs)
    assert clean.factors["middle_baseline"].selected_family == "BASELINE"
    assert clean.factors["middle_baseline"].status == "baseline_accepted"
    cut = clean.split.test_start

    original_execute = BaselineRepairPlan.execute
    failures = []

    def fail_middle_test_execution(self, frame, *args, **call_kwargs):
        dates = np.asarray(frame["date"])
        if dates.size and np.max(dates) >= cut:
            failures.append((self.identity, int(np.max(dates))))
            raise RuntimeError("synthetic frozen-plan TEST materialization failure")
        return original_execute(self, frame, *args, **call_kwargs)

    monkeypatch.setattr(BaselineRepairPlan, "execute", fail_middle_test_execution)
    mixed = optimize_factor_batch(batch, labels, **kwargs)

    assert len(failures) == 1
    assert mixed.optimized.factor_ids == factor_ids
    assert tuple(mixed.factors) == factor_ids
    assert mixed.optimized.values.shape == batch.values.shape
    assert mixed.optimized.values.dtype == np.dtype(np.float64)
    assert mixed.optimized.validity.dtype == np.dtype(bool)

    left, middle, right = (mixed.factors[factor_id] for factor_id in factor_ids)
    assert left.selected_family == right.selected_family == "NO_OP_RAW"
    assert left.status == right.status == "raw_retained"
    assert middle.selected_family == clean.factors["middle_baseline"].selected_family
    assert middle.plan_identity == clean.factors["middle_baseline"].plan_identity
    assert middle.status == "materialization_failed"
    assert middle.materialization_error

    output = mixed.optimized.values
    mask = mixed.optimized.validity
    np.testing.assert_array_equal(output[:, :, 0], batch.values[:, :, 0])
    np.testing.assert_array_equal(output[:, :, 2], batch.values[:, :, 2])
    np.testing.assert_array_equal(output[:cut, :, 1], clean.optimized.values[:cut, :, 1])
    assert np.isnan(output[cut:, :, 1]).all()
    np.testing.assert_array_equal(mask, np.isfinite(output))
    assert not mask[cut:, :, 1].any()

    assert not np.shares_memory(output, batch.values)
    assert not output.flags.writeable
    with pytest.raises(ValueError):
        output.flags.writeable = True
