import numpy as np
import pandas as pd
import pytest
from factor_preprocess.contracts.treatment_lineage import TransformLineage, TransformStep


def fixture():
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle
    rng = np.random.default_rng(82)
    values = rng.normal(size=(240, 40))
    ta = AxisRef('time', 'int', 240, np.arange(240))
    aa = AxisRef('asset', 'str', 40, np.array([f'a{i}' for i in range(40)]))
    batch = FactorBatch(('good',), ta, aa, values[:, :, None])
    labels = LabelBundle('synthetic', values, 1, decision_time=tuple(range(240)),
        label_start_time=tuple(range(1, 241)), label_end_time=tuple(range(2, 242)), asset_axis=aa)
    return batch, labels


def compile_plan(lineage, **kwargs):
    from factor_optimizer.research_baseline import compile_baseline
    return compile_baseline(lineage, training_context_ref='train-only', **kwargs)


def frame():
    return pd.DataFrame({'date': [1]*5, 'asset_id': list('abcde'),
                         'value': [1., 4., 2., 10., 1000.]}, index=[8, 8, 2, 1, 6])


def test_known_untreated_factor_gets_winsor_then_rank():
    plan = compile_plan(TransformLineage())
    assert plan.operations == ('winsor', 'cs_rank')
    result = plan.execute(frame(), allow_research=True)
    np.testing.assert_allclose(result, [0., .5, .25, .75, 1.])
    assert result.index.equals(frame().index)


def test_existing_rank_anywhere_in_lineage_is_not_repeated():
    lineage = TransformLineage((TransformStep('CS_RANK:pct', 'representation', 'rank'),))
    plan = compile_plan(lineage)
    assert 'cs_rank' not in plan.operations
    # Non-rank magnitudes prove we did not silently rank during execution.
    result = plan.execute(frame(), allow_research=True)
    assert result.max() > 100


def test_unknown_lineage_is_not_assumed_untreated():
    plan = compile_plan(None)
    assert plan.operations == ()
    assert 'lineage_unknown' in plan.omissions
    np.testing.assert_array_equal(plan.execute(frame(), allow_research=True), frame().value)


def test_unknown_step_does_not_erase_confirmed_rank_deduplication():
    lineage = TransformLineage((
        TransformStep('CS_RANK:pct', 'representation', 'rank'),
        TransformStep('UNKNOWN:custom', 'representation', 'custom'),
    ))
    plan = compile_plan(lineage)
    assert plan.operations == ()
    assert 'lineage_unknown' in plan.omissions
    assert 'cs_rank_already_present' in plan.omissions


def test_partial_rank_evidence_suppresses_rank_search_without_inventing_baseline():
    from factor_preprocess.contracts.treatment_lineage import ExistingTreatmentSignature
    from factor_optimizer.research_batch import optimize_factor_batch, BatchOptimizationConfig
    batch, labels = fixture()
    signature = ExistingTreatmentSignature(cs_rank=True, status='incomplete')
    result = optimize_factor_batch(batch, labels, allow_research=True,
        config=BatchOptimizationConfig(selection_objective='rank_ic',
            families=('REPRESENTATION_RANK',), bootstrap_draws=99),
        lineages={'good': signature})
    item = result.factors['good']
    assert item.candidates == ()
    assert item.selected_family == 'NO_OP_RAW'
    assert item.baseline_diagnostics['operations'] == ()
    assert 'lineage_unknown' in item.baseline_diagnostics['omissions']
    np.testing.assert_array_equal(result.optimized.values, batch.values)


def test_exposure_availability_is_checked_before_neutralization():
    plan = compile_plan(TransformLineage(), exposure_columns=('size',))
    exposures = frame()[['date', 'asset_id']].assign(size=np.arange(5), available_time=2)
    with pytest.raises(ValueError, match='available'):
        plan.execute(frame(), exposures=exposures, allow_research=True)


def test_neutralization_requires_declared_exposures_and_preserves_alignment():
    plan = compile_plan(TransformLineage(), exposure_columns=('size',), minimum_assets=3)
    assert plan.operations == ('winsor', 'neutralize', 'cs_rank')
    exposures = frame()[['date', 'asset_id']].assign(size=np.arange(5), available_time=1)
    result = plan.execute(frame(), exposures=exposures.iloc[::-1], allow_research=True)
    assert result.index.equals(frame().index)
    assert result.notna().all()
    with pytest.raises(ValueError, match='exposure'):
        plan.execute(frame(), allow_research=True)


def test_baseline_training_guard_rejects_large_loss_and_ignores_future_labels():
    from factor_optimizer.research_baseline import assess_baseline_training
    from factor_optimizer.research_batch import BatchOptimizationConfig, automatic_time_split
    from dataclasses import replace
    batch, labels = fixture()
    config = BatchOptimizationConfig(selection_objective='rank_ic', )
    split = automatic_time_split(labels, config)
    raw = batch.values[:, :, 0]
    candidate = -raw
    verdict = assess_baseline_training(raw, candidate, batch, labels, split, config)
    assert not verdict['accepted']
    assert verdict['reason'] == 'substantial_training_degradation'
    poison = labels.values.copy()
    poison[split.validation_start:] = np.nan
    changed = assess_baseline_training(raw, candidate, batch,
                                      replace(labels, values=poison), split, config)
    assert changed == verdict
    assert assess_baseline_training(raw, raw, batch, labels, split, config)['accepted']


def test_joint_baseline_rejects_tail_portfolio_damage_hidden_by_small_ic_change():
    from dataclasses import replace
    from factor_optimizer.research_baseline import assess_baseline_training
    from factor_optimizer.research_batch import BatchOptimizationConfig, automatic_time_split
    from quant_evaluator.contracts.factor_batch import AxisRef
    batch, labels = fixture()
    raw = np.tile(np.arange(400, dtype=float), (240, 1))
    candidate = raw.copy()
    candidate[:, [319, 399]] = candidate[:, [399, 319]]
    rng = np.random.default_rng(993)
    y = rng.normal(0, .01, raw.shape)
    y[:, 399] += .3
    assets = AxisRef('asset', 'str', 400, np.array([f'a{i}' for i in range(400)]))
    batch = replace(batch, asset_axis=assets, values=raw[:, :, None])
    labels = replace(labels, asset_axis=assets, values=y)
    config = BatchOptimizationConfig(bootstrap_draws=99)
    split = automatic_time_split(labels, config)
    legacy = assess_baseline_training(raw, candidate, batch, labels, split,
                                     replace(config, selection_objective='rank_ic'))
    assert legacy['accepted']
    verdict = assess_baseline_training(raw, candidate, batch, labels, split, config)
    assert not verdict['accepted']
    assert verdict['reason'] == 'substantial_joint_training_degradation'
    assert verdict['joint_candidate']['sharpe'] < verdict['joint_raw']['sharpe'] - .25
    assert assess_baseline_training(raw, raw, batch, labels, split, config)['accepted']
    y[split.validation_start:] = np.nan
    assert assess_baseline_training(raw, candidate, batch, replace(labels, values=y),
                                    split, config) == verdict


@pytest.mark.parametrize("failure", ["held_return", "overlap"])
def test_joint_baseline_does_not_accept_unavailable_portfolio_evidence(failure):
    from dataclasses import replace
    from factor_optimizer.research_baseline import assess_baseline_training
    from factor_optimizer.research_batch import BatchOptimizationConfig, automatic_time_split
    batch, labels = fixture()
    raw = batch.values[:, :, 0]
    y = .01*raw.copy()
    if failure == "held_return":
        y[70, np.argmax(raw[70])] = np.nan
    labels = replace(labels, values=y)
    if failure == "overlap":
        labels = replace(labels, label_end_time=tuple(range(3, 243)))
    config = BatchOptimizationConfig(bootstrap_draws=99)
    verdict = assess_baseline_training(raw, raw, batch, labels,
                                      automatic_time_split(labels, config), config)
    assert not verdict['accepted']
    assert verdict['reason'] == 'joint_training_metrics_unavailable'
    assert verdict['joint_unavailable_reason']


def test_batch_applies_baseline_before_search_and_freezes_composed_plan():
    from factor_optimizer.research_batch import optimize_factor_batch, BatchOptimizationConfig
    from dataclasses import replace
    batch, labels = fixture()
    batch = replace(batch, factor_ids=('good',), values=batch.values[:, :, :1])
    result = optimize_factor_batch(batch, labels, allow_research=True,
        config=BatchOptimizationConfig(selection_objective='rank_ic', families=('SIGN_ORIENTATION',)),
        lineages={'good': TransformLineage()})
    factor = result.factors['good']
    assert factor.selected_family == 'BASELINE'
    assert factor.baseline_diagnostics['accepted']
    assert result.optimized.values.min() >= 0
    assert result.optimized.values.max() <= 1
    long = pd.DataFrame({'date': np.repeat(batch.time_axis.values, 40),
                         'asset_id': np.tile(batch.asset_axis.values, 240),
                         'value': batch.values[:, :, 0].ravel()})
    np.testing.assert_allclose(factor.plan.execute(long, allow_research=True),
                               result.optimized.values[:, :, 0].ravel())

@pytest.mark.parametrize("failure", ["availability", "duplicate"])
def test_test_period_exposure_failure_preserves_frozen_selection_and_prefix(failure):
    from factor_optimizer.research_batch import optimize_factor_batch, BatchOptimizationConfig
    batch, labels = fixture()
    exposures = pd.DataFrame({'date': np.repeat(batch.time_axis.values, 40),
        'asset_id': np.tile(batch.asset_axis.values, 240),
        'available_time': np.repeat(batch.time_axis.values, 40), 'size': 1.})
    kwargs = dict(allow_research=True, lineages={'good': TransformLineage()},
        exposure_columns=('size',), config=BatchOptimizationConfig(
            selection_objective='rank_ic', families=('SIGN_ORIENTATION',)))
    clean = optimize_factor_batch(batch, labels, exposures=exposures, **kwargs)
    assert clean.factors['good'].selected_family == 'BASELINE'
    cut = clean.split.test_start
    poisoned = exposures.copy()
    if failure == "availability":
        poisoned.loc[poisoned.date >= cut, 'available_time'] += 1000
    else:
        poisoned = pd.concat([poisoned, poisoned.loc[poisoned.date == cut].iloc[:1]])
    bad = optimize_factor_batch(batch, labels, exposures=poisoned, **kwargs)
    expected, actual = clean.factors['good'], bad.factors['good']
    assert actual.plan_identity == expected.plan_identity
    assert actual.selected_family == expected.selected_family
    assert actual.train_gain == expected.train_gain
    assert actual.validation_lower_bound == expected.validation_lower_bound
    assert actual.status == 'materialization_failed'
    assert actual.materialization_error
    np.testing.assert_array_equal(bad.optimized.values[:cut], clean.optimized.values[:cut])
    assert np.isnan(bad.optimized.values[cut:]).all()
    assert not bad.optimized.validity[cut:].any()
    assert not bad.test_evaluated
    from factor_optimizer.research_final_report import freeze_selection
    with pytest.raises(ValueError, match='materialization'):
        freeze_selection(batch, bad, dataset_identity='synthetic')


def test_validation_rejects_frozen_baseline_without_retry():
    from factor_optimizer.research_batch import optimize_factor_batch, BatchOptimizationConfig
    from dataclasses import replace
    batch, labels = fixture()
    batch = replace(batch, factor_ids=('good',), values=batch.values[:, :, :1])
    y = labels.values.copy()
    y[144:192] = 0
    result = optimize_factor_batch(batch, replace(labels, values=y), allow_research=True,
        config=BatchOptimizationConfig(selection_objective='rank_ic', families=('SIGN_ORIENTATION',)),
        lineages={'good': TransformLineage()})
    assert result.factors['good'].selected_family == 'NO_OP_RAW'
    np.testing.assert_array_equal(result.optimized.values, batch.values)


@pytest.mark.parametrize("failure", ["availability", "duplicate"])
def test_validation_exposure_error_cannot_change_training_baseline_decision(failure):
    from factor_optimizer.research_batch import optimize_factor_batch, BatchOptimizationConfig
    batch, labels = fixture()
    exposures = pd.DataFrame({'date': np.repeat(batch.time_axis.values, 40),
        'asset_id': np.tile(batch.asset_axis.values, 240),
        'available_time': np.repeat(batch.time_axis.values, 40), 'size': 1.})
    kwargs = dict(allow_research=True, lineages={'good': TransformLineage()},
        exposure_columns=('size',), config=BatchOptimizationConfig(selection_objective='rank_ic', families=('SIGN_ORIENTATION',)))
    clean = optimize_factor_batch(batch, labels, exposures=exposures, **kwargs)
    poisoned = exposures.copy()
    if failure == "availability":
        poisoned.loc[poisoned.date >= 144, 'available_time'] += 1000
    else:
        poisoned = pd.concat([poisoned, poisoned.loc[poisoned.date == 144].iloc[:1]])
    bad = optimize_factor_batch(batch, labels, exposures=poisoned, **kwargs)
    assert clean.factors['good'].baseline_diagnostics == bad.factors['good'].baseline_diagnostics
    assert clean.factors['good'].validation_candidate_identity == bad.factors['good'].validation_candidate_identity
    assert bad.factors['good'].selected_family == 'NO_OP_RAW'


def test_baseline_plus_sign_is_replayed_as_one_frozen_pipeline():
    from factor_optimizer.research_batch import optimize_factor_batch, BatchOptimizationConfig
    from dataclasses import replace
    batch, labels = fixture()
    batch = replace(batch, values=-batch.values)
    result = optimize_factor_batch(batch, labels, allow_research=True,
        lineages={'good': TransformLineage()},
        config=BatchOptimizationConfig(selection_objective='rank_ic', families=('SIGN_ORIENTATION',)))
    selected = result.factors['good']
    assert selected.selected_family == 'SIGN_ORIENTATION'
    assert hasattr(selected.plan, 'baseline')
    assert result.optimized.values.min() >= -1
    assert result.optimized.values.max() <= 0


def test_unavailable_training_neutralization_keeps_other_baseline_steps():
    from factor_optimizer.research_batch import optimize_factor_batch, BatchOptimizationConfig
    batch, labels = fixture()
    exposures = pd.DataFrame({'date': np.repeat(batch.time_axis.values, 40),
        'asset_id': np.tile(batch.asset_axis.values, 240),
        'available_time': np.repeat(batch.time_axis.values, 40), 'size': np.nan})
    result = optimize_factor_batch(batch, labels, exposures=exposures, allow_research=True,
        lineages={'good': TransformLineage()}, exposure_columns=('size',),
        config=BatchOptimizationConfig(selection_objective='rank_ic', families=('SIGN_ORIENTATION',)))
    selected = result.factors['good']
    assert selected.selected_family == 'BASELINE'
    assert selected.plan.baseline.operations == ('winsor', 'cs_rank')
    assert 'neutralization_train_rejected' in selected.plan.baseline.omissions
    assert not selected.baseline_diagnostics['neutralization_attempt']['accepted']
    assert selected.baseline_diagnostics['accepted']
    assert result.optimized.values.min() >= 0
    assert result.optimized.values.max() <= 1
