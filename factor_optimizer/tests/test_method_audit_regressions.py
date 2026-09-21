"""Numerical and selection failures found by the per-method data audit."""
from dataclasses import replace
import numpy as np
import pytest

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from factor_optimizer.research_batch import BatchOptimizationConfig, _pair_ic, optimize_factor_batch


def panel(seed=193, times=300, assets=32):
    rng = np.random.default_rng(seed)
    latent = np.empty((times, assets))
    latent[0] = rng.normal(size=assets)
    for t in range(1, times):
        latent[t] = .98 * latent[t-1] + rng.normal(scale=.15, size=assets)
    observed = latent + rng.normal(scale=1.4, size=latent.shape)
    ta = AxisRef("time", "int", times, np.arange(times))
    aa = AxisRef("asset", "str", assets, np.array([f"a{i}" for i in range(assets)]))
    batch = FactorBatch(("negative_noisy",), ta, aa, -observed[:, :, None])
    labels = LabelBundle("synthetic", latent, 1, decision_time=tuple(range(times)),
                         label_start_time=tuple(range(1, times+1)),
                         label_end_time=tuple(range(2, times+2)), asset_axis=aa)
    return batch, labels


def test_constant_candidate_days_cannot_disappear_from_coverage():
    batch, labels = panel()
    raw = batch.values[:, :, 0]
    candidate = raw.copy()
    candidate[::2] = 0.
    _, good, retention = _pair_ic(raw, candidate, batch, labels, tuple(range(100)), BatchOptimizationConfig(selection_objective='rank_ic', ))
    assert good.sum() == 50
    assert retention == pytest.approx(.5)


def test_missing_whole_candidate_days_count_against_valid_ic_coverage():
    batch, labels = panel()
    raw = batch.values[:, :, 0]
    candidate = raw.copy()
    candidate[::2] = np.nan
    _, good, retention = _pair_ic(raw, candidate, batch, labels, tuple(range(100)), BatchOptimizationConfig(selection_objective='rank_ic', ))
    assert good.sum() == 50
    assert retention == pytest.approx(.5)


@pytest.mark.parametrize("seed", [193, 817])
def test_negative_noisy_factor_can_combine_orientation_and_smoothing(seed):
    batch, labels = panel(seed)
    config = BatchOptimizationConfig(selection_objective='rank_ic', families=("SIGN_ORIENTATION", "CAUSAL_SMOOTHING"),
                                     bootstrap_draws=99)
    result = optimize_factor_batch(batch, labels, config=config, allow_research=True)
    outcome = result.factors["negative_noisy"]
    assert outcome.status == "improved"
    assert outcome.selected_family == "CAUSAL_SMOOTHING"
    assert outcome.plan.multiplier == -1
    assert any(r.get("orientation") == -1 for r in outcome.candidates)
    assert outcome.validation_candidate_identity == outcome.plan_identity
    # The frozen compound plan must actually change values, not just its label.
    from scipy.stats import spearmanr
    idx = result.split.validation_indices
    repaired = np.mean([spearmanr(result.optimized.values[i,:,0], labels.values[i]).statistic for i in idx])
    sign_only = np.mean([spearmanr(-batch.values[i,:,0], labels.values[i]).statistic for i in idx])
    assert repaired > sign_only + .10


def test_auto_grid_covers_both_tail_directions():
    from factor_optimizer.research_batch import _specs
    specs = _specs(BatchOptimizationConfig(selection_objective='rank_ic', families=("TAIL_HINGE", "TAIL_SATURATION")))
    assert {p["hinge"] for f,p in specs if f == "TAIL_HINGE"} == {"top", "bottom"}
    assert {p["saturate"] for f,p in specs if f == "TAIL_SATURATION"} == {"top", "bottom", "both"}


def test_scale_is_rejected_at_configuration_boundary():
    with pytest.raises(ValueError, match="scale"):
        BatchOptimizationConfig(selection_objective='rank_ic', natural_time_scale=10001)


def test_compound_selection_does_not_consume_test_labels():
    batch, labels = panel()
    config = BatchOptimizationConfig(selection_objective='rank_ic', families=("SIGN_ORIENTATION", "DECAY_REFINEMENT"), bootstrap_draws=99)
    a = optimize_factor_batch(batch, labels, config=config, allow_research=True)
    values = labels.values.copy()
    values[240:] = np.nan
    b = optimize_factor_batch(batch, replace(labels, values=values), config=config, allow_research=True)
    left, right = a.factors["negative_noisy"], b.factors["negative_noisy"]
    assert left.selected_family == "DECAY_REFINEMENT"
    assert left.plan.multiplier == -1
    assert (left.plan_identity, left.train_gain, left.validation_lower_bound) == (
        right.plan_identity, right.train_gain, right.validation_lower_bound)


def test_disabling_composition_retains_standalone_sign_and_candidate_budget_is_enforced():
    batch, labels = panel()
    conf = BatchOptimizationConfig(selection_objective='rank_ic', families=("SIGN_ORIENTATION", "DECAY_REFINEMENT"),
                                    compose_smoothing_sign=False, bootstrap_draws=99)
    outcome = optimize_factor_batch(batch, labels, config=conf, allow_research=True).factors["negative_noisy"]
    assert outcome.selected_family == "SIGN_ORIENTATION"
    assert not any(r["orientation"] == -1 for r in outcome.candidates)
    with pytest.raises(ValueError, match="budget"):
        optimize_factor_batch(batch, labels, config=replace(conf, compose_smoothing_sign=True,
                              maximum_candidates=6), allow_research=True)


@pytest.mark.parametrize("value", [1, "yes", None])
def test_orientation_configuration_requires_real_boolean(value):
    with pytest.raises(ValueError, match="compose_smoothing_sign"):
        BatchOptimizationConfig(selection_objective='rank_ic', compose_smoothing_sign=value)


def test_signed_candidates_reuse_one_base_materialization(monkeypatch):
    from collections import Counter
    from factor_optimizer.adapters.repair_execution import ValueRepairPlan
    calls = Counter()
    original = ValueRepairPlan.execute

    def measured(self, values, **kwargs):
        calls[(self.identity, len(values))] += 1
        return original(self, values, **kwargs)

    monkeypatch.setattr(ValueRepairPlan, "execute", measured)
    batch, labels = panel()
    config = BatchOptimizationConfig(selection_objective='rank_ic', families=("SIGN_ORIENTATION", "DECAY_REFINEMENT"),
                                     bootstrap_draws=99)
    result = optimize_factor_batch(batch, labels, config=config, allow_research=True)
    outcome = result.factors["negative_noisy"]
    assert outcome.status == "improved"
    assert outcome.selected_family == "DECAY_REFINEMENT"
    assert outcome.plan.multiplier == -1
    prefix_size = result.split.test_start * batch.values.shape[1]
    prefix_calls = [count for (_, size), count in calls.items() if size == prefix_size]
    assert prefix_calls and max(prefix_calls) == 1
