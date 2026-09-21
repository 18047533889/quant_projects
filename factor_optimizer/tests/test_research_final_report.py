import numpy as np
import pytest


def sample():
    from factor_optimizer.research_batch import optimize_factor_batch, BatchOptimizationConfig
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle
    rng = np.random.default_rng(28)
    x = rng.normal(size=(240, 30))
    y = .004*x + rng.normal(0, .01, x.shape)
    aa = AxisRef('asset', 'str', 30, np.array([f'a{i}' for i in range(30)]))
    ta = AxisRef('time', 'int', 240, np.arange(240))
    raw = FactorBatch(('f',), ta, aa, -x[:, :, None])
    labels = LabelBundle('returns', y, 1, decision_time=tuple(range(240)),
        label_start_time=tuple(range(1,241)), label_end_time=tuple(range(2,242)), asset_axis=aa)
    result = optimize_factor_batch(raw, labels, allow_research=True,
        config=BatchOptimizationConfig(families=('SIGN_ORIENTATION',), bootstrap_draws=99))
    return raw, labels, result


class Store:
    def __init__(self, labels):
        self.labels = labels
        self.reads = 0

    def read(self):
        self.reads += 1
        return self.labels


def authority(tmp_path, frozen, store):
    from factor_optimizer.contracts.campaign_store import SQLiteCampaignStore
    from factor_optimizer.data_capabilities import TestAuthorityBroker, TestStoreRef
    broker = TestAuthorityBroker(dataset_identity=frozen.dataset_identity,
        provider_identity='trusted-test-provider', search_session_id='finished-search',
        split_id=frozen.split_identity, campaign_store=SQLiteCampaignStore(tmp_path/'campaign.sqlite'),
        campaign_id='final', candidate_set_hash=frozen.selection_hash,
        profile_hash=frozen.profile_hash, purpose='research_final_report')
    broker.attach_store_ref(TestStoreRef(store, dataset_identity=frozen.dataset_identity))
    return broker


def test_report_separates_test_and_description_and_reuses_immutable_result(tmp_path):
    from factor_optimizer.research_final_report import freeze_selection, evaluate_frozen
    raw, labels, result = sample()
    frozen = freeze_selection(raw, result, dataset_identity='dataset-v1')
    store = Store(labels)
    report = evaluate_frozen(frozen, authority(tmp_path, frozen, store))
    assert report['test']['role'] == 'held_out_final'
    assert report['full_sample']['role'] == 'descriptive_only'
    assert report['test']['days'] == 48
    assert report['full_sample']['days'] == 240
    assert report['test']['factors']['f']['selected']['rank_ic'] > 0
    assert not result.test_evaluated  # Original search artifact stays unmodified.
    second_store = Store(None)  # A restarted authority must not re-read labels.
    assert evaluate_frozen(frozen, authority(tmp_path, frozen, second_store)) == report
    assert store.reads == 1 and second_store.reads == 0


def test_modified_frozen_outputs_are_rejected_before_test_read(tmp_path):
    from factor_optimizer.research_final_report import freeze_selection, evaluate_frozen
    raw, labels, result = sample()
    frozen = freeze_selection(raw, result, dataset_identity='dataset-v1')
    store = Store(labels)
    broker = authority(tmp_path, frozen, store)
    # Contract arrays can be replaced by constructing a different batch result;
    # try mutating the retained buffer directly if writable.
    object.__setattr__(result.optimized, 'values', result.optimized.values + 1)
    with pytest.raises(ValueError, match='frozen|changed'):
        evaluate_frozen(frozen, broker)
    assert store.reads == 0


def test_failed_post_exposure_report_cannot_reopen_test(tmp_path):
    from factor_optimizer.research_final_report import freeze_selection, evaluate_frozen
    from factor_optimizer.contracts.campaign_store import CampaignStateError
    raw, labels, result = sample()
    frozen = freeze_selection(raw, result, dataset_identity='dataset-v1')
    store = Store('wrong payload')
    with pytest.raises((TypeError, ValueError)):
        evaluate_frozen(frozen, authority(tmp_path, frozen, store))
    with pytest.raises(CampaignStateError):
        evaluate_frozen(frozen, authority(tmp_path, frozen, Store(labels)))
    assert store.reads == 1


@pytest.mark.parametrize('profile_change', [
    {'research_cost_rate': 0.},
    {'research_empty_leg_policy': 'unavailable'},
])
def test_cost_profile_cannot_be_changed_to_reopen_same_holdout(tmp_path, profile_change):
    from dataclasses import replace
    from factor_optimizer.research_final_report import freeze_selection, evaluate_frozen
    from factor_optimizer.contracts.campaign_store import CampaignStateError
    raw, labels, result = sample()
    frozen = freeze_selection(raw, result, dataset_identity='dataset-v1')
    store = Store(labels)
    evaluate_frozen(frozen, authority(tmp_path, frozen, store))
    changed = replace(result, config=replace(result.config, **profile_change))
    alternate = freeze_selection(raw, changed, dataset_identity='dataset-v1')
    with pytest.raises(CampaignStateError):
        authority(tmp_path, alternate, Store(labels))
    assert store.reads == 1


def test_report_includes_worst_block_sharpe_used_by_selection(tmp_path):
    from factor_optimizer.research_final_report import freeze_selection, evaluate_frozen
    raw, labels, result = sample()
    frozen = freeze_selection(raw, result, dataset_identity='dataset-v1')
    report = evaluate_frozen(frozen, authority(tmp_path, frozen, Store(labels)))
    assert report['test']['factors']['f']['selected']['worst_block_sharpe'] > 0


def test_non_durable_authority_is_rejected_before_read():
    from factor_optimizer.research_final_report import freeze_selection, evaluate_frozen
    from factor_optimizer.data_capabilities import TestAuthorityBroker, TestStoreRef
    raw, labels, result = sample()
    frozen = freeze_selection(raw, result, dataset_identity='dataset-v1')
    store = Store(labels)
    broker = TestAuthorityBroker(dataset_identity='dataset-v1', provider_identity='provider',
                                  search_session_id='finished', split_id=frozen.split_identity)
    broker.attach_store_ref(TestStoreRef(store, dataset_identity='dataset-v1'))
    with pytest.raises(ValueError, match='durable'):
        evaluate_frozen(frozen, broker)
    assert store.reads == 0
