import numpy as np
import pytest
from quant_evaluator.metrics.shape_evidence import (compute_left_right_asymmetry,
    compute_shape_stability,compute_shape_regime_stability,compute_shape_bootstrap_confidence)


@pytest.mark.parametrize('q',[5,6,20])
def test_mirror_contrast_does_not_confuse_symmetric_curvature(q):
    x = np.linspace(-1,1,q)
    assert compute_left_right_asymmetry((x*x)[:,None])[0] == pytest.approx(0,abs=1e-15)
    assert compute_left_right_asymmetry((-x*x)[:,None])[0] == pytest.approx(0,abs=1e-15)
    assert compute_left_right_asymmetry((x*x+x)[:,None])[0] > 0


def test_stability_uses_excluded_reference_and_correlation_scale():
    p = np.arange(5.)[:,None]
    assert compute_shape_stability(np.stack([p,p,p]))[0] == pytest.approx(1)
    assert compute_shape_stability(np.stack([p,-p]))[0] == pytest.approx(-1)
    assert compute_shape_regime_stability(np.stack([p,-p,p]))[0] == pytest.approx(-1)
    assert np.isnan(compute_shape_stability(p)[0])
    assert np.isnan(compute_shape_stability(np.ones((3,5,1)))[0])


def test_block_resampling_constant_ties_not_fabricated_confidence():
    assert np.isnan(compute_shape_bootstrap_confidence(np.ones((4,5,1)))[0])
    with pytest.raises(ValueError,match='block_length'):
        compute_shape_bootstrap_confidence(np.ones((4,5,1)),block_length=10)


def test_bootstrap_factor_permutation_and_singleton_invariance():
    windows = np.random.default_rng(1103).normal(size=(8,7,3))
    batch = compute_shape_bootstrap_confidence(windows,random_seed=97)
    for f in range(3):
        assert batch[f] == compute_shape_bootstrap_confidence(windows[:,:,f:f+1],random_seed=97)[0]
    np.testing.assert_equal(batch[::-1],compute_shape_bootstrap_confidence(windows[:,:,::-1],random_seed=97))


def test_public_shape_stability_builds_real_disjoint_windows():
    from quant_evaluator.contracts.factor_batch import FactorBatch,AxisRef
    from quant_evaluator.contracts.label_bundle import LabelBundle
    from quant_evaluator.runtime.evaluator import evaluate
    values = np.broadcast_to(np.arange(100.)[None,:,None],(60,100,1))
    batch = FactorBatch(('f',),AxisRef('time','int',60),AxisRef('asset','int',100),values)
    labels = LabelBundle('h1',np.broadcast_to(np.arange(100.)[None,:],(60,100)),1,
        decision_time=tuple(range(60)),label_start_time=tuple(range(1,61)),label_end_time=tuple(range(2,62)))
    bundle = evaluate(batch,labels,metrics=['shape_stability','shape_regime_stability'],
        quantile_builder_parameters={'n_quantiles':10,'window_size':20})
    for mid in bundle.artifacts:
        assert bundle.artifacts[mid].values[0] == pytest.approx(1)
        assert bundle.artifacts[mid].provenance['quantile_builder']['n_quantiles'] == 10


@pytest.mark.parametrize('value',[True,1.5,-1])
def test_adaptive_counts_never_truncate_or_drop_invalid_dates(value):
    from quant_evaluator.contracts.adaptive_bins_policy import resolve_bin_count
    with pytest.raises(ValueError,match='integer'):
        resolve_bin_count([2000,value])


def test_empty_fallback_is_explicit_insufficient_and_actual_buckets_required():
    from quant_evaluator.contracts.adaptive_bins_policy import AdaptiveBinsPolicy,resolve_bin_count,resolve_bin_count_from_counts
    policy = AdaptiveBinsPolicy(fallback_bins=())
    assert resolve_bin_count([0],policy).is_insufficient
    assert resolve_bin_count_from_counts({20:[100,0,100]},policy).is_insufficient
    with pytest.raises(ValueError,match='descending'):
        AdaptiveBinsPolicy(preferred_bins=5,fallback_bins=(10,))
    with pytest.raises(ValueError):
        AdaptiveBinsPolicy(fallback_bins=(10.2,5))


def test_public_binary_factor_cannot_claim_twenty_usable_buckets():
    from quant_evaluator.contracts.factor_batch import FactorBatch,AxisRef
    from quant_evaluator.contracts.label_bundle import LabelBundle
    from quant_evaluator.runtime.evaluator import evaluate
    values = np.broadcast_to(np.repeat([0.,1.],1000)[None,:,None],(30,2000,1))
    batch = FactorBatch(('binary',),AxisRef('time','int',30),AxisRef('asset','int',2000),values)
    labels = LabelBundle('h1',np.ones((30,2000)),1,decision_time=tuple(range(30)),
        label_start_time=tuple(range(1,31)),label_end_time=tuple(range(2,32)))
    bundle = evaluate(batch,labels,metrics=['adaptive_quantile_count'],quantile_builder_parameters={'n_quantiles':20})
    assert not bundle.get_metric('adaptive_quantile_count','binary').valid
    coverage = bundle.artifacts['adaptive_quantile_count'].provenance['adaptive_bins_coverage'][0]
    assert coverage['selected_q'] is None
    assert coverage['fallback_reason'] == 'insufficient'
    assert coverage['tie_policy'] == 'max'
    assert all(row['distinct_levels'] == 2 for row in coverage['dates'])
    assert all(row['candidate_min_bucket_counts']['20'] == 0 for row in coverage['dates'])


def _adaptive_public_bundle(values, factor_ids):
    from quant_evaluator.contracts.factor_batch import FactorBatch,AxisRef
    from quant_evaluator.contracts.label_bundle import LabelBundle
    from quant_evaluator.runtime.evaluator import evaluate
    times, assets, _ = values.shape
    batch = FactorBatch(tuple(factor_ids),AxisRef('time','int',times),
        AxisRef('asset','int',assets),values)
    labels = LabelBundle('h1',np.ones((times,assets)),1,
        decision_time=tuple(range(times)),label_start_time=tuple(range(1,times+1)),
        label_end_time=tuple(range(2,times+2)))
    return evaluate(batch,labels,metrics=['adaptive_quantile_count'])


def test_public_continuous_factor_selects_twenty_and_exposes_coverage():
    values = np.broadcast_to(np.arange(2000.)[None,:,None],(20,2000,1))
    bundle = _adaptive_public_bundle(values, ('continuous',))
    assert bundle.get_metric('adaptive_quantile_count','continuous').value == 20.0
    artifact = bundle.artifacts['adaptive_quantile_count']
    coverage = artifact.provenance['adaptive_bins_coverage'][0]
    assert coverage['selected_q'] == 20
    assert coverage['comparison_policy'] == 'largest_fixed_q_feasible_on_every_date'
    assert coverage['applicable_dates'] == coverage['total_dates'] == 20
    assert all(row['candidate_min_bucket_counts']['20'] == 100 for row in coverage['dates'])


def test_adaptive_factor_resolution_is_singleton_batch_invariant():
    continuous = np.broadcast_to(np.arange(2000.)[None,:,None],(20,2000,1))
    binary = np.broadcast_to(np.repeat([0.,1.],1000)[None,:,None],(20,2000,1))
    together = _adaptive_public_bundle(np.concatenate([continuous,binary],axis=2),
        ('continuous','binary'))
    alone_continuous = _adaptive_public_bundle(continuous,('continuous',))
    alone_binary = _adaptive_public_bundle(binary,('binary',))
    assert together.get_metric('adaptive_quantile_count','continuous').value == \
        alone_continuous.get_metric('adaptive_quantile_count','continuous').value == 20.0
    assert not together.get_metric('adaptive_quantile_count','binary').valid
    assert not alone_binary.get_metric('adaptive_quantile_count','binary').valid


def test_missing_date_remains_explicit_and_prevents_silent_fixed_q_claim():
    values = np.broadcast_to(np.arange(2000.)[None,:,None],(20,2000,1)).copy()
    values[7,:,0] = np.nan
    bundle = _adaptive_public_bundle(values,('sparse',))
    assert not bundle.get_metric('adaptive_quantile_count','sparse').valid
    coverage = bundle.artifacts['adaptive_quantile_count'].provenance['adaptive_bins_coverage'][0]
    assert coverage['applicable_dates'] == 19
    assert coverage['total_dates'] == 20
    assert coverage['dates'][7]['applicable'] is False
    assert coverage['dates'][7]['reason'] == 'missing_factor_values'
    assert coverage['dates'][7]['candidate_min_bucket_counts']['20'] is None


def test_public_adaptive_fallback_builds_real_ten_bucket_daily_and_profile():
    values = np.broadcast_to(np.repeat(np.arange(10.),200)[None,:,None],
        (20,2000,1))
    from quant_evaluator.contracts.factor_batch import FactorBatch,AxisRef
    from quant_evaluator.contracts.label_bundle import LabelBundle
    from quant_evaluator.runtime.evaluator import evaluate
    batch = FactorBatch(('ten_level',),AxisRef('time','int',20),
        AxisRef('asset','int',2000),values)
    labels = LabelBundle('h1',np.broadcast_to(np.arange(2000.)[None,:],(20,2000)),1,
        decision_time=tuple(range(20)),label_start_time=tuple(range(1,21)),
        label_end_time=tuple(range(2,22)))
    bundle = evaluate(batch,labels,metrics=[
        'adaptive_quantile_count','quantile_returns_daily','quantile_returns_full'])
    assert bundle.get_metric('adaptive_quantile_count','ten_level').value == 10.0
    daily = bundle.artifacts['quantile_returns_daily']
    profile = bundle.artifacts['quantile_returns_full']
    assert daily.values.shape == (20,10,1)
    assert daily.counts.shape == (20,10,1)
    assert tuple(daily.quantile_axis) == tuple(range(10))
    assert np.all(daily.counts == 200)
    assert profile.values.shape == (10,1)
    assert profile.quantile_axis.quantile_labels == tuple(f'Q{i}' for i in range(1,11))
    for artifact in (daily,profile):
        binding = artifact.provenance['adaptive_quantile_binding']
        assert binding['selected_q_by_factor'] == (10,)
        assert binding['count_artifact_policy']['policy_id'] == 'QE_ADAPTIVE_BINS'


def test_public_adaptive_profiles_transport_mixed_q_axes_without_padding():
    continuous = np.broadcast_to(np.arange(2000.)[None,:,None],(20,2000,1))
    ten_level = np.broadcast_to(np.repeat(np.arange(10.),200)[None,:,None],
        (20,2000,1))
    from quant_evaluator.contracts.factor_batch import FactorBatch,AxisRef
    from quant_evaluator.contracts.label_bundle import LabelBundle
    from quant_evaluator.runtime.evaluator import evaluate
    batch = FactorBatch(('q20','q10'),AxisRef('time','int',20),AxisRef('asset','int',2000),
        np.concatenate([continuous,ten_level],axis=2))
    labels = LabelBundle('h1',np.ones((20,2000)),1,decision_time=tuple(range(20)),
        label_start_time=tuple(range(1,21)),label_end_time=tuple(range(2,22)))
    bundle = evaluate(batch,labels,metrics=[
        'adaptive_quantile_count','quantile_returns_daily','quantile_returns_full'])
    assert bundle.artifacts.get('quantile_returns_full') is None
    assert bundle.artifacts.get('quantile_returns_daily') is None
    assert bundle.factor_artifacts['q20']['quantile_returns_full'].values.shape == (20,1)
    assert bundle.factor_artifacts['q10']['quantile_returns_full'].values.shape == (10,1)
    assert bundle.factor_artifacts['q20']['quantile_returns_daily'].values.shape == (20,20,1)
    assert bundle.factor_artifacts['q10']['quantile_returns_daily'].values.shape == (20,10,1)
    from quant_evaluator.api.requests import EvaluationBundle
    import json
    restored = EvaluationBundle.from_dict(json.loads(json.dumps(bundle.to_dict(),allow_nan=False)))
    assert restored.factor_artifacts['q20']['quantile_returns_full'].values.shape == (20,1)
    assert restored.factor_artifacts['q10']['quantile_returns_full'].values.shape == (10,1)
    reversed_batch = FactorBatch(('q10','q20'),AxisRef('time','int',20),
        AxisRef('asset','int',2000),np.concatenate([ten_level,continuous],axis=2))
    reversed_bundle = evaluate(reversed_batch,labels,metrics=[
        'adaptive_quantile_count','quantile_returns_daily','quantile_returns_full'])
    for fid in ('q20','q10'):
        for mid in ('quantile_returns_daily','quantile_returns_full'):
            np.testing.assert_equal(bundle.factor_artifacts[fid][mid].values,
                reversed_bundle.factor_artifacts[fid][mid].values)


def test_serializable_request_custom_adaptive_policy_is_canonically_validated():
    from quant_evaluator.api.requests import EvaluationRequest
    from quant_evaluator.contracts.factor_batch import FactorBatch,AxisRef
    from quant_evaluator.contracts.label_bundle import LabelBundle
    from quant_evaluator.runtime.evaluator import evaluate
    import json
    values = np.broadcast_to(np.repeat(np.arange(4.),500)[None,:,None],(20,2000,1))
    batch = FactorBatch(('four_level',),AxisRef('time','int',20),
        AxisRef('asset','int',2000),values)
    labels = LabelBundle('h1',np.ones((20,2000)),1,decision_time=tuple(range(20)),
        label_start_time=tuple(range(1,21)),label_end_time=tuple(range(2,22)))
    policy = {'preferred_bins':4,'fallback_bins':[2],
        'min_effective_names_per_bin':500,'policy_id':'CUSTOM_Q','policy_version':'1.0'}
    request = EvaluationRequest(batch,labels,metric_ids=('adaptive_quantile_count',),
        tier='extended',metric_parameters={'adaptive_quantile_count':{'policy':policy}})
    transported = EvaluationRequest.from_dict(json.loads(json.dumps(request.to_dict())))
    assert dict(transported.metric_parameters['adaptive_quantile_count']['policy'])['preferred_bins'] == 4
    rebound = EvaluationRequest(batch,labels,metric_ids=transported.metric_ids,
        tier=transported.tier,metric_parameters=transported.metric_parameters)
    bundle = evaluate(rebound)
    assert bundle.get_metric('adaptive_quantile_count','four_level').value == 4.0
    recorded = bundle.artifacts['adaptive_quantile_count'].provenance['adaptive_bins_policy']
    assert recorded['policy_id'] == 'CUSTOM_Q'
    assert recorded['fallback_bins'] == (2,)
    with pytest.raises(ValueError,match='Unknown AdaptiveBinsPolicy fields'):
        evaluate(EvaluationRequest(batch,labels,metric_ids=('adaptive_quantile_count',),
            tier='extended',metric_parameters={'adaptive_quantile_count':{
                'policy':dict(policy,untrusted='ignored')}}))


def test_l20_cuda_adaptive_mixed_q_ties_custom_and_missing_parity():
    pytest.importorskip('cupy')
    from quant_evaluator.contracts.factor_batch import FactorBatch,AxisRef
    from quant_evaluator.contracts.label_bundle import LabelBundle
    from quant_evaluator.runtime.evaluator import evaluate
    continuous = np.broadcast_to(np.arange(2000.)[None,:,None],(20,2000,1))
    ten_level = np.broadcast_to(np.repeat(np.arange(10.),200)[None,:,None],(20,2000,1))
    batch = FactorBatch(('q20','q10'),AxisRef('time','int',20),AxisRef('asset','int',2000),
        np.concatenate([continuous,ten_level],axis=2))
    labels = LabelBundle('h1',np.broadcast_to(np.arange(2000.)[None,:],(20,2000)),1,
        decision_time=tuple(range(20)),label_start_time=tuple(range(1,21)),
        label_end_time=tuple(range(2,22)))
    metrics = ['adaptive_quantile_count','quantile_returns_daily','quantile_returns_full']
    cpu = evaluate(batch,labels,metrics=metrics)
    gpu = evaluate(batch,labels,metrics=metrics,backend='cuda_strict')
    assert gpu.metadata['adaptive_planning_backend'] == 'cpu'
    assert gpu.metadata['adaptive_profile_backend'] == 'cuda_strict'
    assert gpu.metadata['adaptive_cuda_no_fallback'] is True
    assert gpu.metadata['adaptive_q_groups'] == (10,20)
    for fid in ('q20','q10'):
        assert gpu.get_metric('adaptive_quantile_count',fid).value == \
            cpu.get_metric('adaptive_quantile_count',fid).value
        for mid in ('quantile_returns_daily','quantile_returns_full'):
            ga = gpu.factor_artifacts[fid][mid]
            ca = cpu.factor_artifacts[fid][mid]
            assert ga.provenance['execution_backend'] == 'cuda_strict'
            assert ga.provenance['no_fallback'] is True
            np.testing.assert_allclose(ga.values,ca.values,rtol=0,atol=1e-12)

    binary = np.broadcast_to(np.repeat([0.,1.],1000)[None,:,None],(20,2000,1))
    missing = continuous.copy(); missing[3,:,0] = np.nan
    for fid, values in [('binary',binary),('missing',missing)]:
        sparse = FactorBatch((fid,),AxisRef('time','int',20),AxisRef('asset','int',2000),values)
        cpu_count = evaluate(sparse,labels,metrics=['adaptive_quantile_count'])
        gpu_count = evaluate(sparse,labels,metrics=['adaptive_quantile_count'],backend='cuda_strict')
        assert not cpu_count.get_metric('adaptive_quantile_count',fid).valid
        assert not gpu_count.get_metric('adaptive_quantile_count',fid).valid

    four_level = np.broadcast_to(np.repeat(np.arange(4.),500)[None,:,None],(20,2000,1))
    custom_batch = FactorBatch(('q4',),AxisRef('time','int',20),AxisRef('asset','int',2000),four_level)
    params = {'adaptive_quantile_count':{'policy':{'preferred_bins':4,'fallback_bins':[2],
        'min_effective_names_per_bin':500,'policy_id':'CUSTOM_Q','policy_version':'1.0'}}}
    custom_gpu = evaluate(custom_batch,labels,metrics=metrics,metric_parameters=params,
        backend='cuda_strict')
    assert custom_gpu.get_metric('adaptive_quantile_count','q4').value == 4
    assert custom_gpu.factor_artifacts['q4']['quantile_returns_full'].values.shape == (4,1)
