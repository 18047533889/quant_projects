import numpy as np
import pytest
from quant_evaluator.api.requests import EvaluationRequest
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.contracts.artifact_types import ProbePortfolioArtifact
from quant_evaluator.contracts.metric_artifacts import SeriesMetricArtifact, VectorMetricArtifact
from quant_evaluator.contracts.errors import InvalidContractError
from quant_evaluator.runtime.evaluator import evaluate, Evaluator
from quant_evaluator.registry.metrics import get_metric


def inputs():
    rng = np.random.default_rng(452)
    t, n = 30, 60
    x = rng.normal(size=(t, n))
    batch = FactorBatch(('up', 'down'), AxisRef('time', 'int', t), AxisRef('asset', 'int', n),
                        np.stack([x, -x], axis=-1))
    label = LabelBundle('forward', x, 1, decision_time=tuple(range(t)),
                        label_start_time=tuple(range(1, t+1)), label_end_time=tuple(range(2, t+2)))
    return batch, label


def test_series_and_vector_are_not_implicit_scalar_objectives():
    batch, label = inputs()
    out = evaluate(batch, label, metrics=['rank_ic_series', 'quantile_returns_full'])
    assert isinstance(out.artifacts['rank_ic_series'], SeriesMetricArtifact)
    assert isinstance(out.artifacts['quantile_returns_full'], VectorMetricArtifact)
    assert out.artifacts['rank_ic_series'].values.shape == (30, 2)
    assert out.artifacts['quantile_returns_full'].values.shape == (5, 2)
    assert not out.metric_values
    assert all(not v for v in out.grouped_metrics.values())
    for artifact in out.artifacts.values():
        restored = type(artifact).from_dict(artifact.to_dict())
        np.testing.assert_equal(restored.values, artifact.values)


def test_parameters_change_public_values_identity_and_cached_execution():
    batch, label = inputs()
    runtime = Evaluator()
    a = evaluate(batch, label, metrics=['rank_ic'], evaluator=runtime,
                 metric_parameters={'rank_ic': {'min_assets': 10}})
    b = evaluate(batch, label, metrics=['rank_ic'], evaluator=runtime,
                 metric_parameters={'rank_ic': {'min_assets': 100}})
    assert a.get_metric('rank_ic', 'up').valid
    assert not b.get_metric('rank_ic', 'up').valid
    assert a.config_hash != b.config_hash
    assert a.metric_versions['rank_ic'] == get_metric('rank_ic').metric_version
    assert a.get_metric('rank_ic', 'up').metric_version == a.metric_versions['rank_ic']


def test_parameter_roundtrip_and_unknown_parameter_rejection():
    request = EvaluationRequest(None, None, metric_parameters={'rank_ic': {'min_assets': 25}})
    assert EvaluationRequest.from_dict(request.to_dict()).metric_parameters == request.metric_parameters
    batch, label = inputs()
    with pytest.raises(InvalidContractError, match='Invalid parameters'):
        evaluate(batch, label, metrics=['rank_ic'], metric_parameters={'rank_ic': {'min_asssets': 10}})


def test_portfolio_risk_uses_each_portfolio_not_the_label_panel():
    batch, label = inputs()
    returns = np.column_stack([np.tile([.1, -.02], 15), np.tile([.01, -.1], 15)])
    portfolio = ProbePortfolioArtifact(returns, time_index=label.decision_time, factor_ids=batch.factor_ids)
    out = evaluate(batch, label, metrics=['sharpe_ratio'], portfolio_returns=portfolio)
    assert out.get_metric('sharpe_ratio', 'up').value > 0
    assert out.get_metric('sharpe_ratio', 'down').value < 0
    with pytest.raises(Exception, match='ProbePortfolioArtifact'):
        evaluate(batch, label, metrics=['sharpe_ratio'])


def test_real_cuda_uses_same_typed_contract_and_true_counts():
    pytest.importorskip('cupy')
    batch, label = inputs()
    metrics = ['rank_ic', 'rank_ic_series', 'coverage', 'quantile_returns_full', 'ic_ir']
    cpu = evaluate(batch, label, metrics=metrics)
    gpu = evaluate(batch, label, metrics=metrics, backend='cuda')
    assert type(cpu) is type(gpu)
    assert cpu.metric_versions == gpu.metric_versions
    assert cpu.config_hash == gpu.config_hash
    for mid in metrics:
        assert type(cpu.artifacts[mid]) is type(gpu.artifacts[mid])
        np.testing.assert_allclose(cpu.artifacts[mid].values, gpu.artifacts[mid].values, atol=1e-10)
    assert gpu.get_metric('rank_ic', 'up').observation_count == 30
    assert gpu.get_metric('coverage', 'up').observation_count == 1800
    assert not gpu.get_metric('ic_ir', 'up').valid  # constant +1 IC is not infinite skill


def test_daily_quantile_public_cpu_gpu_keeps_all_axes_counts_and_masks():
    pytest.importorskip('cupy')
    batch, label = inputs()
    kwargs = dict(metrics=['quantile_returns_daily'],
                  metric_parameters={'quantile_returns_daily': {'n_quantiles': 3, 'min_assets': 5}})
    cpu = evaluate(batch,label,**kwargs)
    gpu = evaluate(batch,label,backend='cuda',**kwargs)
    left, right = cpu.artifacts['quantile_returns_daily'], gpu.artifacts['quantile_returns_daily']
    assert left.values.shape == (30,3,2)
    np.testing.assert_allclose(left.values,right.values,atol=1e-12)
    np.testing.assert_equal(left.counts,right.counts)
    np.testing.assert_equal(left.valid_mask,right.valid_mask)
    assert not cpu.metric_values
    assert not left.provenance_refs_present
    assert left.provenance['config_hash'] == cpu.config_hash == gpu.config_hash


@pytest.mark.parametrize('backend', [None, 'cuda'])
def test_request_budget_and_tier_are_enforced_before_dispatch(backend, monkeypatch):
    from quant_evaluator.runtime.device_session import DeviceEvaluationSession
    batch, label = inputs()
    monkeypatch.setattr(DeviceEvaluationSession, '_open', lambda self: pytest.fail('allocated device before rejecting plan'))
    with pytest.raises(InvalidContractError, match='cost_budget'):
        evaluate(EvaluationRequest(batch,label,metric_ids=('rank_ic','coverage'),cost_budget=1),backend=backend)
    with pytest.raises(InvalidContractError, match='tier'):
        evaluate(EvaluationRequest(batch,label,metric_ids=('quantile_returns_daily',),tier='core'),backend=backend)


def test_budget_charges_one_real_shared_artifact_builder_node():
    batch, label = inputs()
    metrics = ('quantile_monotonicity', 'quantile_curvature')
    with pytest.raises(InvalidContractError, match=r'artifact_builders=2\.0'):
        evaluate(EvaluationRequest(batch, label, metric_ids=metrics, tier='extended', cost_budget=3))
    bundle = evaluate(EvaluationRequest(batch, label, metric_ids=metrics, tier='extended', cost_budget=4))
    plan = bundle.metadata['execution_plan']
    assert plan['planned_cost'] == 4.0
    assert plan['artifact_builder_cost'] == 2.0
    assert plan['artifact_builder_nodes'] == ({
        'artifact': 'QuantileReturnArtifact',
        'required_by': metrics,
        'cost': 2.0,
        'tier': 'extended',
    },)


def test_quantile_consumers_share_raw_builder_across_profile_policies(monkeypatch):
    import quant_evaluator.metrics.quantile as kernels
    original = kernels.compute_quantile_returns_fast
    calls = []

    def counted(*args, **kwargs):
        calls.append((kwargs['n_quantiles'], kwargs['min_assets']))
        return original(*args, **kwargs)

    monkeypatch.setattr(kernels, 'compute_quantile_returns_fast', counted)
    batch, label = inputs()
    evaluate(batch, label, metrics=['quantile_monotonicity', 'shape_stability'])
    assert calls == [(5, 10)]


def test_unavailable_required_input_fails_during_plan_compilation():
    from types import SimpleNamespace
    from quant_evaluator.runtime.evaluator import _compile_public_artifact_plan

    specs = {'needs_exposure': SimpleNamespace(requires=['exposure_panel'])}
    with pytest.raises(InvalidContractError, match='missing exposure_panel input'):
        _compile_public_artifact_plan(specs)


def test_runtime_provenance_never_trusts_external_factor_value_reference():
    batch, label = inputs()
    validity = np.ones_like(batch.values, dtype=bool)
    validity[0, 0, 0] = False
    claimed = FactorBatch(
        batch.factor_ids, batch.time_axis, batch.asset_axis, batch.values,
        validity=validity, value_hash='externally-claimed-ref',
    )
    changed_values = np.array(batch.values, copy=True)
    changed_values[0, 0, 0] += 1
    changed = FactorBatch(
        batch.factor_ids, batch.time_axis, batch.asset_axis,
        changed_values, validity=validity,
        value_hash='externally-claimed-ref',
    )

    first = evaluate(claimed, label, metrics=['coverage'])
    second = evaluate(changed, label, metrics=['coverage'])
    p1, p2 = first.metadata['provenance'], second.metadata['provenance']
    assert p1['factor_value_hash'] == p2['factor_value_hash'] == 'externally-claimed-ref'
    assert p1['factor_value_bytes_hash'] != p2['factor_value_bytes_hash']
    assert p1['factor_validity_hash'] == p2['factor_validity_hash']
    assert first.artifacts['coverage'].provenance['factor_value_bytes_hash'] == p1['factor_value_bytes_hash']


def test_authoritative_array_hash_uses_object_values_not_pointer_bytes():
    import hashlib
    from quant_evaluator.runtime.evaluator import authoritative_array_hash

    left = np.array([('asset', 1), {'session': 'A'}], dtype=object)
    right = np.array([('asset', 1), {'session': 'A'}], dtype=object)
    changed = np.array([('asset', 2), {'session': 'A'}], dtype=object)
    assert authoritative_array_hash(left) == authoritative_array_hash(right)
    assert authoritative_array_hash(left) != authoritative_array_hash(changed)

    scalar = np.asarray(7, dtype=np.int64)
    legacy = hashlib.sha256()
    legacy.update(str(scalar.dtype).encode())
    legacy.update(str(scalar.shape).encode())
    legacy.update(np.ascontiguousarray(scalar).tobytes())
    assert authoritative_array_hash(scalar) == legacy.hexdigest()


def test_public_exposure_builder_is_axis_bound_shared_and_missing_aware(monkeypatch):
    from quant_evaluator.metrics.exposure_evidence import ExposurePanel
    import quant_evaluator.metrics.exposure_evidence as exposure_builder

    rng = np.random.default_rng(91)
    t, n = 30, 60
    times = np.arange(t)
    assets = np.array([f'A{i:03d}' for i in range(n)], dtype=object)
    risk = rng.normal(size=(t, n, 2))
    factors = np.stack([
        2.0 * risk[:, :, 0] - risk[:, :, 1],
        -2.0 * risk[:, :, 0] + risk[:, :, 1],
    ], axis=-1)
    validity = np.ones_like(risk, dtype=bool)
    validity[0, 0, :] = False
    panel = ExposurePanel(
        risk, style_names=('industry', 'size'), source_ref='da://risk/v1',
        provider='data_access', date_index=tuple(times),
        security_ids=tuple(assets), factor_ids=('up', 'down'),
        universe_snapshot_ref='snapshot://u1', validity=validity,
    )
    batch = FactorBatch(
        ('up', 'down'), AxisRef('time', 'int', t, times),
        AxisRef('asset', 'object', n, assets), factors,
    )
    labels = LabelBundle(
        'forward', factors[:, :, 0] + rng.normal(scale=.1, size=(t, n)), 1,
        decision_time=tuple(times), label_start_time=tuple(times + 1),
        label_end_time=tuple(times + 2),
    )
    metrics = ('industry_exposure', 'size_exposure', 'neutralized_rank_ic')
    with pytest.raises(InvalidContractError, match='missing exposure_panel'):
        evaluate(batch, labels, metrics=metrics)
    with pytest.raises(InvalidContractError, match='artifact_builders=2.0'):
        evaluate(EvaluationRequest(
            batch, labels, metric_ids=metrics, tier='extended', cost_budget=4,
            exposure_panel=panel,
        ))
    original = exposure_builder.compute_factor_loadings
    builder_calls = []
    def counted_builder(*args, **kwargs):
        builder_calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(exposure_builder, 'compute_factor_loadings', counted_builder)
    bundle = evaluate(EvaluationRequest(
        batch, labels, metric_ids=metrics, tier='extended', cost_budget=5,
        exposure_panel=panel,
    ))
    # Public values are standardized loadings, not raw regression betas.
    # Derive the gold independently as beta_k * sd(risk_k) / sd(factor)
    # on each date's shared valid support.
    expected = {fid: [] for fid in batch.factor_ids}
    for day in range(t):
        keep = validity[day].all(axis=1)
        x = np.column_stack((np.ones(keep.sum()), risk[day, keep]))
        for index, fid in enumerate(batch.factor_ids):
            y = factors[day, keep, index]
            beta = np.linalg.solve(x.T @ x, x.T @ y)[1:]
            expected[fid].append(beta * np.std(risk[day, keep], axis=0) / np.std(y))
    expected = {fid: np.asarray(rows).mean(axis=0) for fid, rows in expected.items()}
    assert bundle.get_metric('industry_exposure', 'up').value == pytest.approx(expected['up'][0], abs=1e-12)
    assert bundle.get_metric('industry_exposure', 'down').value == pytest.approx(expected['down'][0], abs=1e-12)
    assert bundle.get_metric('size_exposure', 'up').value == pytest.approx(expected['up'][1], abs=1e-12)
    assert bundle.metadata['execution_plan']['artifact_builder_nodes'][0]['required_by'] == metrics
    # One loading-series fit and one residual-IC fit per factor; the two
    # exposure scalar metrics share the loading-series artifact.
    assert len(builder_calls) == 2 * len(batch.factor_ids)
    # Both factors are exact linear combinations of risk styles, leaving a
    # constant residual. Residual rank IC is therefore explicitly undefined.
    assert all(not bundle.get_metric('neutralized_rank_ic', fid).valid for fid in batch.factor_ids)

    swapped = ExposurePanel(
        risk, style_names=('industry', 'size'), source_ref='da://risk/v1',
        provider='data_access', date_index=tuple(times),
        security_ids=tuple(assets[::-1]), factor_ids=('up', 'down'),
        universe_snapshot_ref='snapshot://u1', validity=validity,
    )
    with pytest.raises(InvalidContractError, match='security axis'):
        evaluate(batch, labels, metrics=['industry_exposure'], exposure_panel=swapped)
    from quant_evaluator.runtime.device_session import DeviceEvaluationSession
    monkeypatch.setattr(
        DeviceEvaluationSession, '_open',
        lambda self: pytest.fail('allocated CUDA session for unsupported exposure plan'),
    )
    with pytest.raises(Exception, match='CUDA.*implementation'):
        evaluate(batch, labels, metrics=['neutralized_rank_ic'],
                 exposure_panel=panel, backend='cuda')


def test_public_multiple_testing_derives_axis_bound_hac_pvalues_once(monkeypatch):
    import quant_evaluator.metrics.ic as ic_kernels
    from quant_evaluator.metrics.registry_adapters import compute_hac_pvalue_value

    rng = np.random.default_rng(412)
    t, n = 45, 60
    labels_array = rng.normal(size=(t, n))
    factors = np.stack([
        .20 * labels_array + rng.normal(size=(t, n)),
        -.15 * labels_array + rng.normal(size=(t, n)),
        rng.normal(size=(t, n)),
    ], axis=-1)
    batch = FactorBatch(
        ('a', 'b', 'c'), AxisRef('time', 'int', t), AxisRef('asset', 'int', n),
        factors,
    )
    labels = LabelBundle(
        'forward', labels_array, 1, decision_time=tuple(range(t)),
        label_start_time=tuple(range(1, t + 1)),
        label_end_time=tuple(range(2, t + 2)),
    )
    metrics = (
        'bonferroni_correction', 'benjamini_hochberg_correction',
        'holm_bonferroni_correction', 'sidak_correction',
    )
    with pytest.raises(InvalidContractError, match=r'artifact_builders=2\.0'):
        evaluate(EvaluationRequest(
            batch, labels, metric_ids=metrics, tier='research', cost_budget=5,
        ))

    original = ic_kernels.compute_daily_ic
    calls = []
    def counted(*args, **kwargs):
        calls.append((kwargs['method'], kwargs['min_assets']))
        return original(*args, **kwargs)
    monkeypatch.setattr(ic_kernels, 'compute_daily_ic', counted)
    bundle = evaluate(EvaluationRequest(
        batch, labels, metric_ids=metrics, tier='research', cost_budget=6,
    ))
    assert calls == [('spearman', 20)]
    daily_ic, _ = original(batch, labels, method='spearman', min_assets=20)
    raw_p = compute_hac_pvalue_value(
        daily_ic, min_periods=30, max_lag=5, kernel='bartlett'
    )
    expected_bonf = np.minimum(raw_p * np.isfinite(raw_p).sum(), 1.0)
    np.testing.assert_allclose(
        bundle.artifacts['bonferroni_correction'].values, expected_bonf,
        equal_nan=True,
    )
    assert all(mid in bundle.artifacts for mid in metrics)
    plan = bundle.metadata['execution_plan']
    assert tuple(node['artifact'] for node in plan['artifact_builder_nodes']) == (
        'ICSeriesArtifact', 'HACPValueVector',
    )
    assert bundle.artifacts['bonferroni_correction'].provenance['p_value_builder']['sample_unit'] == 'daily_ic'
    assert bundle.get_metric('bonferroni_correction', 'a').observation_count == 45

    short_batch = FactorBatch(
        batch.factor_ids, AxisRef('time', 'int', 20), batch.asset_axis,
        batch.values[:20],
    )
    short_labels = LabelBundle(
        'forward', labels_array[:20], 1, decision_time=tuple(range(20)),
        label_start_time=tuple(range(1, 21)), label_end_time=tuple(range(2, 22)),
    )
    missing = evaluate(short_batch, short_labels, metrics=['bonferroni_correction'])
    assert all(not missing.get_metric('bonferroni_correction', fid).valid
               for fid in short_batch.factor_ids)


def test_bundle_json_roundtrip_preserves_scalar_series_profile_daily_evidence():
    import json
    from quant_evaluator.api.requests import EvaluationBundle
    batch,label = inputs()
    bundle = evaluate(batch,label,metrics=['rank_ic','rank_ic_series','quantile_returns_full','quantile_returns_daily'])
    restored = EvaluationBundle.from_dict(json.loads(json.dumps(bundle.to_dict(),allow_nan=False)))
    assert restored.config_hash == bundle.config_hash
    assert restored.metric_versions == bundle.metric_versions
    assert restored.grouped_metrics == bundle.grouped_metrics
    assert restored.diagnostics == bundle.diagnostics
    for mid in bundle.artifacts:
        assert type(restored.artifacts[mid]) is type(bundle.artifacts[mid])
        np.testing.assert_equal(restored.artifacts[mid].values,bundle.artifacts[mid].values)
    daily = restored.artifacts['quantile_returns_daily']
    np.testing.assert_equal(daily.counts,bundle.artifacts['quantile_returns_daily'].counts)
    assert not daily.provenance_refs_present


def test_derived_ic_and_counts_reuse_one_request_local_series_per_method(monkeypatch):
    import quant_evaluator.metrics.ic as kernels
    original = kernels.compute_daily_ic
    calls = []
    def counted(*args, **kwargs):
        calls.append((kwargs.get('method'),kwargs.get('min_assets')))
        return original(*args, **kwargs)
    monkeypatch.setattr(kernels,'compute_daily_ic',counted)
    batch,label = inputs()
    bundle = evaluate(batch,label,metrics=['ic_ir','ic_std','pearson_ic_ir','pearson_ic_std'])
    assert sorted(calls) == [('pearson',20),('spearman',20)]
    assert bundle.get_metric('ic_std','up').observation_count == 30
