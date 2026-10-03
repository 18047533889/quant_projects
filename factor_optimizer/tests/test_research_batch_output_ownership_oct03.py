"""Characterize batch output ordering, masking, dtype, and ownership."""

import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle


def test_overbudget_four_factor_output_preserves_raw_values_and_ownership(monkeypatch):
    # Reuse the established four-factor analytical fixture: RAW, reverse, U,
    # and all-invalid columns. The synthetic diagnosis adds two distinct U
    # proposals so the 12-entry declared grid exceeds this per-factor budget.
    from factor_optimizer.research_batch import BatchOptimizationConfig, optimize_factor_batch
    import factor_optimizer.research_batch_diagnostics as batch_diagnostics

    rng = np.random.default_rng(82)
    times, assets_count = 240, 40
    z = np.stack([
        rng.permutation(np.linspace(-1, 1, assets_count))
        for _ in range(times)
    ])
    target_values = z ** 2
    original_values = np.stack(
        (target_values, -target_values, z, np.full_like(z, np.nan)), axis=-1
    )
    time_axis = AxisRef("time", "int", times, np.arange(times))
    asset_axis = AxisRef(
        "asset", "str", assets_count,
        np.array([f"a{i}" for i in range(assets_count)]),
    )
    original_batch = FactorBatch(
        ("good", "reverse", "u", "invalid"), time_axis, asset_axis,
        original_values,
    )
    labels = LabelBundle(
        "synthetic", target_values, 1,
        decision_time=tuple(range(times)),
        label_start_time=tuple(range(1, times + 1)),
        label_end_time=tuple(range(2, times + 2)), asset_axis=asset_axis,
    )
    source_values = original_batch.values.copy()
    source_values[0, 0, 0] = 123.5  # finite value hidden by explicit validity
    source_values[1, 1, 1] = np.nan  # explicit non-finite input in a mixed column
    validity = np.ones(source_values.shape, dtype=bool)
    validity[0, 0, 0] = False
    source_before_mutation = source_values.copy()
    batch = FactorBatch(
        original_batch.factor_ids, original_batch.time_axis,
        original_batch.asset_axis, source_values, validity=validity,
    )
    expected = source_before_mutation.astype(np.float64, copy=True)
    expected[~validity | ~np.isfinite(expected)] = np.nan

    # FactorBatch owns its construction buffer; subsequent caller mutation
    # cannot alter the input contract or the eventual optimizer result.
    source_values[:] = -777.0
    np.testing.assert_array_equal(batch.values, source_before_mutation)
    np.testing.assert_array_equal(batch.validity, validity)

    diagnosis = {
        "issues": [],
        "proposed_shape_family": "U_SHAPE_REPAIR",
        "proposed_center": 0.41,
        "layer_decay": {"proposed_half_lives": []},
    }
    monkeypatch.setattr(
        batch_diagnostics, "diagnose_raw_batch_in_chunks",
        lambda chunk, target, **kwargs: {
            factor_id: dict(diagnosis) for factor_id in chunk.factor_ids
        },
    )
    config = BatchOptimizationConfig(
        families=("U_SHAPE_REPAIR",), maximum_candidates=12,
        selection_objective="rank_ic",
    )

    result = optimize_factor_batch(batch, labels, config=config, allow_research=True)

    output = result.optimized
    assert output.factor_ids == batch.factor_ids
    assert tuple(result.factors) == batch.factor_ids
    assert output.values.shape == batch.values.shape
    assert output.values.dtype == np.dtype(np.float64)
    assert output.validity.dtype == np.dtype(bool)
    assert np.array_equal(output.values, expected, equal_nan=True)
    assert np.array_equal(output.validity, np.isfinite(expected))
    assert all(item.status == "budget_exceeded_raw_retained"
               and item.selected_family == "NO_OP_RAW"
               for item in result.factors.values())
    assert all(item.training_diagnostics["candidate_budget"]["status"] == "exceeded"
               for item in result.factors.values())

    assert not np.shares_memory(output.values, batch.values)
    assert not output.values.flags.writeable
    with pytest.raises(ValueError):
        output.values.flags.writeable = True
    np.testing.assert_array_equal(output.values, expected)
