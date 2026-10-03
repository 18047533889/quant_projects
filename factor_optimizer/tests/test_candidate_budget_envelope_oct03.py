"""Bound the default candidate budget against the real adaptive envelope."""
import numpy as np
import pytest

from factor_optimizer.candidate_catalog import optimizer_candidate_specs
from factor_optimizer.research_batch import BatchOptimizationConfig, optimize_factor_batch
from factor_optimizer.adapters.preprocessing import compile_admissible_smoothing_grid
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle


def _small_batch():
    rng = np.random.default_rng(20261003)
    times, assets = 240, 24
    values = rng.normal(size=(times, assets, 1))
    returns = rng.normal(size=(times, assets))
    time_axis = AxisRef("time", "int", times, np.arange(times))
    asset_axis = AxisRef("asset", "str", assets,
                         np.asarray([f"a{i}" for i in range(assets)]))
    batch = FactorBatch(("budget-envelope",), time_axis, asset_axis, values)
    labels = LabelBundle(
        "synthetic-forward", returns, 1,
        decision_time=tuple(range(times)),
        label_start_time=tuple(range(1, times + 1)),
        label_end_time=tuple(range(2, times + 2)),
        asset_axis=asset_axis,
    )
    return batch, labels


@pytest.mark.parametrize("maximum_candidates", [128, 127])
def test_default_candidate_budget_covers_or_atomically_rejects_catalogue(monkeypatch, maximum_candidates):
    import factor_optimizer.research_batch_diagnostics as batch_diagnostics
    import factor_optimizer.research_batch as research_batch

    config = BatchOptimizationConfig(maximum_candidates=maximum_candidates)
    static = optimizer_candidate_specs(config)
    smoothing_grid = compile_admissible_smoothing_grid(
        natural_time_scale=config.natural_time_scale,
        training_context_ref="candidate-budget-envelope",
    )
    assert len(static) == 64
    assert len(smoothing_grid) == 27

    # Two non-grid TRAIN half-lives exercise the bounded adaptive smoother
    # route; stable tail layers add the two documented layered-decay scales.
    diagnosis = {
        "issues": [],
        "proposed_shape_family": "U_SHAPE_REPAIR",
        "proposed_center": 0.41,
        "layer_decay": {
            "status": "available",
            "lags": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
            "proposed_half_lives": [7.0, 14.0],
            "layers": [
                {"stable_initial_direction": True, "half_life_bars": 17}
                for _ in range(20)
            ],
        },
    }
    monkeypatch.setattr(
        batch_diagnostics,
        "diagnose_raw_batch_in_chunks",
        lambda batch, labels, **kwargs: {batch.factor_ids[0]: dict(diagnosis)},
    )
    pair_calls = []

    def insufficient_raw(raw, candidate, batch, labels, indices, config, **kwargs):
        pair_calls.append(tuple(indices))
        return (np.zeros(len(indices)), np.zeros(len(indices), dtype=bool), 0.0)

    monkeypatch.setattr(research_batch, "_pair_ic", insufficient_raw)
    batch, labels = _small_batch()
    result = optimize_factor_batch(batch, labels, allow_research=True, config=config)

    outcome = result.factors["budget-envelope"]
    budget = outcome.training_diagnostics["candidate_budget"]
    # 64 static + 4 static decay signs + 54 smoothing-grid signs +
    # 2 fitted shapes + 4 adaptive EWMA signs + 4 layered-decay signs.
    expected_raw = len(static) + 4 + 2 * len(smoothing_grid) + 2 + 4 + 4
    assert expected_raw == 132
    assert budget["raw_required"] == expected_raw
    assert budget["required"] == budget["unique_required"] == 128
    assert budget["required"] + budget["deduplicated"] == expected_raw
    assert budget["deduplicated"] == 4
    assert budget["maximum"] == config.maximum_candidates == maximum_candidates
    assert budget["evaluated"] == 0
    assert outcome.candidates == ()
    np.testing.assert_array_equal(result.optimized.values, batch.values)

    if maximum_candidates == 127:
        assert budget["status"] == "exceeded"
        assert outcome.status == "budget_exceeded_raw_retained"
        assert outcome.selected_family == "NO_OP_RAW"
        assert pair_calls == []  # Overflow is rejected before even RAW eligibility.
    else:
        assert budget["status"] == "admitted"
        assert outcome.status == "invalid_raw"
        assert len(pair_calls) == 1  # Only RAW eligibility; no candidate was partially scored.
