"""Data-backed automatic TRAIN/VALIDATION selection; TEST stays unused."""
from dataclasses import replace
import numpy as np
import pytest
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle

def fixture():
    rng = np.random.default_rng(82)
    t, n = 240, 40
    z = np.stack([rng.permutation(np.linspace(-1, 1, n)) for _ in range(t)])
    y = z ** 2
    values = np.stack((y, -y, z, np.full_like(z, np.nan)), axis=-1)
    ta = AxisRef("time", "int", t, np.arange(t))
    aa = AxisRef("asset", "str", n, np.array([f"a{i}" for i in range(n)]))
    batch = FactorBatch(("good", "reverse", "u", "invalid"), ta, aa, values)
    labels = LabelBundle("synthetic", y, 1, decision_time=tuple(range(t)),
                         label_start_time=tuple(range(1,t+1)), label_end_time=tuple(range(2,t+2)), asset_axis=aa)
    return batch, labels

def api():
    from factor_optimizer.research_batch import optimize_factor_batch, BatchOptimizationConfig
    return optimize_factor_batch, BatchOptimizationConfig

def test_automatic_batch_repairs_sign_and_u_but_preserves_good_and_invalid_raw():
    optimize, Config = api()
    batch, labels = fixture()
    result = optimize(batch, labels, config=Config(families=("SIGN_ORIENTATION", "U_SHAPE_REPAIR")), allow_research=True)
    assert tuple(result.factors) == batch.factor_ids
    assert result.factors["good"].selected_family == "NO_OP_RAW"
    assert result.factors["reverse"].selected_family == "SIGN_ORIENTATION"
    assert result.factors["u"].selected_family == "U_SHAPE_REPAIR"
    assert result.factors["invalid"].status == "invalid_raw"
    assert np.isnan(result.optimized.values[:,:,3]).all()
    assert result.factors["u"].validation_lower_bound > .1
    assert result.split.test_indices[0] == 192
    assert max(labels.label_end_time[i] for i in result.split.train_indices) < labels.decision_time[result.split.validation_indices[0]]
    assert max(labels.label_end_time[i] for i in result.split.validation_indices) < labels.decision_time[result.split.test_indices[0]]

def test_test_labels_never_change_selected_method_or_scores():
    optimize, Config = api()
    batch, labels = fixture()
    conf = Config(families=("SIGN_ORIENTATION", "U_SHAPE_REPAIR"))
    baseline = optimize(batch, labels, config=conf, allow_research=True)
    poisoned = labels.values.copy()
    poisoned[192:] = np.nan
    altered = optimize(batch, replace(labels, values=poisoned), config=conf, allow_research=True)
    for factor in batch.factor_ids:
        a, b = baseline.factors[factor], altered.factors[factor]
        assert a.selected_family == b.selected_family
        assert a.plan_identity == b.plan_identity
        assert a.train_gain == b.train_gain
        assert a.validation_lower_bound == b.validation_lower_bound

def test_missing_axes_and_missing_opt_in_are_rejected():
    optimize, Config = api()
    batch, labels = fixture()
    with pytest.raises(ValueError, match="research"):
        optimize(batch, labels)
    wrong = replace(batch, asset_axis=AxisRef("asset", "str", 40))
    with pytest.raises(ValueError, match="axes"):
        optimize(wrong, labels, allow_research=True)

def test_short_data_does_not_invent_a_valid_split():
    optimize, Config = api()
    batch, labels = fixture()
    from factor_optimizer.research_batch import automatic_time_split
    with pytest.raises(ValueError, match="observations"):
        automatic_time_split(replace(labels, values=labels.values[:30], decision_time=labels.decision_time[:30],
            label_start_time=labels.label_start_time[:30], label_end_time=labels.label_end_time[:30]), Config())

def test_candidate_order_does_not_change_selection():
    optimize, Config = api()
    batch, labels = fixture()
    a = optimize(batch, labels, config=Config(families=("SIGN_ORIENTATION", "U_SHAPE_REPAIR")), allow_research=True)
    b = optimize(batch, labels, config=Config(families=("U_SHAPE_REPAIR", "SIGN_ORIENTATION")), allow_research=True)
    assert {k:v.plan_identity for k,v in a.factors.items()} == {k:v.plan_identity for k,v in b.factors.items()}

def test_validation_label_failure_falls_back_instead_of_trying_other_candidates():
    optimize, Config = api()
    batch, labels = fixture()
    poisoned = labels.values.copy()
    poisoned[144:192] = 0.
    result = optimize(batch, replace(labels, values=poisoned),
        config=Config(families=("SIGN_ORIENTATION", "U_SHAPE_REPAIR")), allow_research=True)
    assert all(item.selected_family == "NO_OP_RAW" for item in result.factors.values())
    assert result.factors["reverse"].train_gain > 1.
    assert result.factors["reverse"].validation_lower_bound is None

def test_factor_order_and_test_features_do_not_change_training_decisions():
    optimize, Config = api()
    batch, labels = fixture()
    conf = Config(families=("SIGN_ORIENTATION",))
    baseline = optimize(batch, labels, config=conf, allow_research=True)
    values = batch.values.copy()
    values[192:] = 1e10
    order = [3, 2, 1, 0]
    permuted = replace(batch, factor_ids=tuple(batch.factor_ids[i] for i in order),
                       values=values[:,:,order])
    changed = optimize(permuted, labels, config=conf, allow_research=True)
    assert {k:v.plan_identity for k,v in baseline.factors.items()} == {k:v.plan_identity for k,v in changed.factors.items()}

@pytest.mark.parametrize("field,bad", [
    ("minimum_improvement", True), ("natural_time_scale", True),
    ("block_length", 0), ("seed", True), ("minimum_assets", 1),
    ("train_fraction", float("nan")), ("bootstrap_draws", 3),
])
def test_invalid_configuration_cannot_drive_automatic_selection(field, bad):
    _, Config = api()
    with pytest.raises((ValueError, TypeError)):
        Config(**{field: bad})


def test_full_coverage_requirement_is_allowed():
    _, Config = api()
    assert Config(minimum_coverage=1.).minimum_coverage == 1.


def test_default_missingness_search_executes_fill_not_only_a_diagnostic_flag():
    optimize, Config = api()
    batch, labels = fixture()
    values = batch.values[:,:,:1].copy()
    values[::3, :3] = np.nan
    batch = replace(batch, factor_ids=("good",), values=values)
    result = optimize(batch, labels, config=Config(families=("MISSINGNESS_FRESHNESS",)),
                      allow_research=True)
    records = result.factors["good"].candidates
    assert any(r["parameters"].get("mode") == "fill" and r["status"] == "train_evaluated"
               for r in records)
    # Filling gaps alone cannot claim higher RankIC on shared observations.
    assert result.factors["good"].selected_family == "NO_OP_RAW"
