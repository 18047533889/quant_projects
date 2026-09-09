"""Independent public-function regression witnesses from V8 (CPU domain)."""
import numpy as np
import pytest

from quant_evaluator.metrics.robustness import compute_hac_variance
from quant_evaluator.metrics.temporal import compute_autocorrelation
from quant_evaluator.metrics.multiple_testing import sidak_correction
from quant_evaluator.metrics.portfolio_stats import compute_maximum_drawdown
from quant_evaluator.metrics.risk.var_cvar import compute_cvar, compute_var_cvar
from quant_evaluator.metrics.risk.tail_risk import compute_upside_potential_ratio


def test_t31_hac_bartlett_normalization():
    x = np.tile([-1., 1.], 10)
    assert compute_hac_variance(x, 1)[0] == pytest.approx(.0025)
    assert compute_hac_variance(x + .04, 1)[0] == pytest.approx(.0025)


@pytest.mark.parametrize('n,lag', [(20,0),(20,1),(20,10),(60,5)])
def test_t32_hac_matches_positive_semidefinite_quadratic_form(n,lag):
    distances=np.abs(np.arange(n)[:,None]-np.arange(n)[None,:])
    weight=np.maximum(0.,1.-distances/(lag+1))
    eigenvalues,eigenvectors=np.linalg.eigh(weight)
    assert eigenvalues.min() >= -1e-12
    x=np.column_stack([np.random.default_rng(32).normal(size=(n,4)),eigenvectors[:,0]])
    centered=x-x.mean(axis=0)
    expected=np.einsum('if,ij,jf->f',centered,weight,centered)/(n*n)
    actual=compute_hac_variance(x,lag)
    np.testing.assert_allclose(actual,expected,atol=1e-15,rtol=1e-12)
    assert np.all(actual>=0)


def test_t38_nonmonotone_lag_pair_counts():
    x = np.full(90, np.nan)
    x[::2] = np.arange(45.)
    acf = compute_autocorrelation(x, max_lag=100)
    assert np.isnan(acf[1])
    assert acf[2] == pytest.approx(1.)


def test_t77_fixed_tail_mass():
    x = np.r_[-.1, np.zeros(99)]
    assert compute_cvar(x) == pytest.approx(.02)


def test_t78_fractional_tail_mass():
    x = np.array([-.07, -.02, 0., .01, .02, .03, .04])
    expected = (.07 + .4 * .02) / 1.4
    assert compute_cvar(x, .8, min_periods=7) == pytest.approx(expected)
    assert compute_cvar(np.tile(x, 3), .8, min_periods=7) == pytest.approx(expected)


@pytest.mark.parametrize('wins,expected', [(90, 2.846049894151541), (10, .105409255338946)])
def test_t79_unconditional_partial_moments(wins, expected):
    x = np.r_[np.full(wins, .01), np.full(100-wins, -.01)]
    assert compute_upside_potential_ratio(x) == pytest.approx(expected)


def test_t81_no_implicit_mixed_var_es_model():
    with pytest.raises(ValueError, match='Cornish|cornish'):
        compute_var_cvar(np.linspace(-.1, .2, 100), method='cornish_fisher')


def test_t87_sidak_tiny_values():
    x = np.full(100000, 1e-20)
    adjusted, reject = sidak_correction(x, alpha=1e-16)
    assert adjusted[0] == pytest.approx(1e-15, rel=1e-12, abs=0)
    assert not reject.any()


def test_t107_peak_not_underwater_trough():
    dd, _, peak = compute_maximum_drawdown(np.array([0., -1e-6]))
    assert dd > 0
    assert peak == 0


def test_a12_no_feasible_additional_break():
    from quant_evaluator.metrics.stats.structural_breaks import bai_perron_test
    rng = np.random.default_rng(701)
    x = np.arange(20.)
    y = rng.normal(size=20)
    count, points, bic = bai_perron_test(y, x, min_obs=10)
    edges = [0] + points + [20]
    rss = 0.
    for a, b in zip(edges, edges[1:]):
        design = np.column_stack([np.ones(b-a), x[a:b]])
        rss += np.sum((y[a:b] - design @ np.linalg.lstsq(design, y[a:b], rcond=None)[0]) ** 2)
    assert count == len(points)
    assert bic == pytest.approx(20*np.log(rss/20) + 2*(count+1)*np.log(20))


def test_a13_rank_deficient_chow_rejected():
    from quant_evaluator.metrics.stats.structural_breaks import chow_test
    x = np.arange(60.)
    with pytest.raises(ValueError, match='rank'):
        chow_test(np.sin(x), np.column_stack([x, x]), 30)


def test_t39_centered_persistence_translation():
    from quant_evaluator.metrics.temporal import compute_half_life
    rng = np.random.default_rng(431)
    x = np.zeros((500, 1))
    for t in range(1, 500):
        x[t] = .7*x[t-1] + rng.normal(scale=.01)
    np.testing.assert_allclose(compute_half_life(x), compute_half_life(x+1), rtol=1e-10)


@pytest.mark.parametrize('method', ['min', 'max'])
def test_t71_t72_quantile_axes(method):
    from quant_evaluator.metrics.quantile import assign_quantiles, assign_quantiles_fast, assign_quantiles_batch
    from quant_evaluator.metrics.quantile_numba import assign_quantiles_numba
    x = np.arange(10.)
    assert len(np.unique(assign_quantiles(x, 5, method))) == 5
    for fn in (assign_quantiles_fast, assign_quantiles_batch, assign_quantiles_numba):
        one = fn(x[None, :, None], 5, method)
        two = fn(np.tile(x[None, :, None], (1, 1, 2)), 5, method)
        assert one.shape == (1, 10, 1)
        np.testing.assert_array_equal(one[..., 0], two[..., 0])


@pytest.mark.parametrize('k', [1, 2, 3, 6, 12])
@pytest.mark.parametrize('det', [-1, 0, 1])
@pytest.mark.parametrize('lags', [1, 2, 4])
@pytest.mark.parametrize('alpha', [.01, .05, .10])
def test_t55_johansen_reference(k, det, lags, alpha):
    from quant_evaluator.metrics.stats.cointegration import johansen_test
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    x = np.random.default_rng(901).normal(size=(200, k)).cumsum(axis=0)
    if k == 1 and det >= 0:
        # Installed statsmodels squeezes K=1 deterministic residuals to 1D;
        # do not claim certification for an unsupported reference domain.
        with pytest.raises(ValueError, match='unsupported'):
            johansen_test(x, det_order=det, lags=lags, alpha=alpha)
        with pytest.raises(np.linalg.LinAlgError):
            coint_johansen(x, det, 1)
        return
    actual = johansen_test(x, det_order=det, lags=lags, alpha=alpha)
    expected = coint_johansen(x, det, lags-1)
    np.testing.assert_allclose(actual[0], expected.lr1)
    np.testing.assert_allclose(actual[1], expected.lr2)
    rank = 0
    for statistic, critical in zip(expected.lr1, expected.cvt[:, {.10:0,.05:1,.01:2}[alpha]]):
        if statistic <= critical:
            break
        rank += 1
    assert actual[2] == rank
    np.testing.assert_allclose(actual[3], expected.eig)


def test_t34_t35_joint_plan_identity_and_permutation():
    from quant_evaluator.contracts.resampling import ResamplingPlan
    from quant_evaluator.metrics.robustness import compute_joint_block_bootstrap
    plan = ResamplingPlan(tuple(range(80)), 'trading_bar:synthetic', 5, 30, 512)
    x = np.random.default_rng(781).normal(size=80)
    artifact = compute_joint_block_bootstrap(np.column_stack([x, x]), plan, ('a', 'b'))
    np.testing.assert_array_equal(artifact.samples[:, 0], artifact.samples[:, 1])
    single = compute_joint_block_bootstrap(x[:, None], plan, ('b',))
    np.testing.assert_array_equal(artifact.samples[:, 1], single.samples[:, 0])
    assert artifact.provenance['replicate_ids'] == single.provenance['replicate_ids']
    gap = x.copy(); gap[30] = np.nan
    assert np.isnan(compute_joint_block_bootstrap(gap[:, None], plan, ('g',)).samples).all()


def test_t42_active_geometry_and_no_duplicate_rf():
    from quant_evaluator.metrics.probe_portfolio.sharpe import compute_active_metrics
    rp = np.array([.2, -.1]); rb = np.array([.1, 0.])
    result = compute_active_metrics(rp, rb, min_periods=2)
    assert result['relative_nav'][-1] == pytest.approx(1.08/1.1)
    same = compute_active_metrics(rp, rp, risk_free_rate=.05, min_periods=2)
    assert np.isnan(same['information_ratio'])
    np.testing.assert_array_equal(same['relative_nav'], np.ones(2))


def test_t44_t46_short_risk_schema():
    from quant_evaluator.metrics.probe_portfolio.sharpe import compute_drawdown_persistence, compute_portfolio_metrics
    assert np.isnan(compute_drawdown_persistence(np.full(9, -.1)))
    empty = compute_portfolio_metrics(np.array([]))
    full = compute_portfolio_metrics(np.linspace(-.03, .04, 100))
    assert set(empty) == set(full)


def test_t40_t41_membership_units_and_unknown_exit():
    from quant_evaluator.metrics.temporal import compute_factor_turnover_rate
    x = np.arange(100., dtype=float)
    y = np.roll(x, 10)
    values = np.array([x, y])[:, :, None]
    assert compute_factor_turnover_rate(values)[0, 0] == pytest.approx(.2)
    assert compute_factor_turnover_rate(values, measure='top_exit_fraction')[0, 0] == 1
    values[1, 90:, 0] = np.nan
    assert np.isnan(compute_factor_turnover_rate(values)[0, 0])


def test_t160_t161_cache_metadata_and_preallocation_guard():
    from quant_evaluator.runtime.cache_v2 import MemoryCacheLayer, ZlibCompressor
    cache = MemoryCacheLayer(1_000_000, ZlibCompressor(), workspace_budget_bytes=20000)
    for _ in range(1100):
        assert cache.put('same', np.arange(8.))
        cache.clear()
    assert len(cache._compression_stats) == 1000
    assert not cache.put('huge_compressible', np.zeros(200000))
    assert cache.get_stats()['current_size_bytes'] == 0


def test_t73_t74_t75_real_cuda_parameter_domain():
    cp = pytest.importorskip('cupy')
    from quant_evaluator.kernels.gpu.rank import batched_quantile_assignment, batched_rank
    x = cp.array([[0., 1., 1., 2., 3.]])
    np.testing.assert_array_equal(cp.asnumpy(batched_quantile_assignment(x, 2, 'min')), [[0, 0, 0, 1, 1]])
    np.testing.assert_array_equal(cp.asnumpy(batched_quantile_assignment(x, 2, 'max')), [[0, 1, 1, 1, 1]])
    ranked = batched_rank(cp.array([[-cp.inf, 1., 2., cp.inf, cp.nan]]))
    np.testing.assert_allclose(cp.asnumpy(ranked), [[np.nan, 1., 2., np.nan, np.nan]], equal_nan=True)
    tile = cp.tile(x, (100, 1))
    np.testing.assert_array_equal(cp.asnumpy(batched_quantile_assignment(tile, 2, workspace_bytes=4096)), np.tile([[0, 1, 1, 1, 1]], (100, 1)))
    with pytest.raises(MemoryError):
        batched_quantile_assignment(x, 2, workspace_bytes=1)


def test_t89_t90_complete_family_never_shrinks_failed_members():
    from quant_evaluator.contracts.hypothesis_family import HypothesisFamilyArtifact
    from quant_evaluator.metrics.multiple_testing import compute_family_correction
    def member(i, status, p=None):
        return dict(hypothesis_id=str(i), effective_spec_hash=f'spec{i}', evaluation_intent_hash=f'intent{i}',
                    horizon=1, label_ref='h1', direction='higher', status=status, pvalue=p,
                    pvalue_ref=f'p{i}' if p is not None else None)
    members = (member(0, 'COMPUTED', .04), member(1, 'FAILED'))
    family = HypothesisFamilyArtifact('family', 'campaign', 'policy', 'context', members, 'seal', 4, 'bonferroni')
    result = compute_family_correction(family)
    assert result['m'] == 2 and result['members']['0']['adjusted_p'] == pytest.approx(.08)
    assert result['members']['1']['adjusted_p'] is None
    reversed_family = HypothesisFamilyArtifact('family', 'campaign', 'policy', 'context', members[::-1], 'seal', 4, 'bonferroni')
    assert reversed_family.content_hash == family.content_hash
    pending = HypothesisFamilyArtifact('family', 'campaign', 'policy', 'context', (members[0], member(1, 'PENDING')), 'seal', 4)
    assert compute_family_correction(pending)['status'] == 'WAIT'
    assert not compute_family_correction(pending)['members']['0']['reject']


def test_t64_t67_qualification_domain_is_not_portable():
    from quant_evaluator.contracts.qualification import NumericalQualificationReceipt
    fields = dict(source_tree_hash='source', implementation_hash='impl', route='root.evaluate', backend='cpu_fp64',
                  parameter_domain_hash='domain', metric_instance_hash='instance')
    receipt = NumericalQualificationReceipt(**fields, test_run_ref='run', assertions={'golden:assert1':'PASS'})
    assert receipt.require_scope(**fields) is receipt
    for field in fields:
        with pytest.raises(ValueError, match='mismatch'):
            receipt.require_scope(**{**fields, field:fields[field]+':unverified-change'})
    skipped = NumericalQualificationReceipt(**fields, test_run_ref='run', assertions={'golden:assert1':'SKIP'})
    with pytest.raises(ValueError, match='PASS'):
        skipped.require_scope(**fields)


def test_t47_t50_factor_validity_storage_identity():
    from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
    source = np.array([[[np.inf], [3.]]], dtype=np.float32)
    mask = np.ones(source.shape, dtype=bool)
    batch = FactorBatch(('f',), AxisRef('time','int64',1,np.array([0])), AxisRef('asset','int64',2,np.array([0,1])), source, mask)
    assert not batch.is_valid(0, 0, 0)
    assert batch.dtype == 'float32'
    source[0, 1, 0] = 99
    assert batch.values[0, 1, 0] == 3
    with pytest.raises(ValueError):
        AxisRef('time','int64',2,np.array([1,0]))


def test_t116_method_version_changes_identity_and_rejects_stale_serialized_instance(monkeypatch):
    from dataclasses import replace
    from quant_evaluator.contracts.metric_instance import MetricInstance
    from quant_evaluator.contracts.qualification import NumericalQualificationReceipt
    from quant_evaluator.registry import metrics
    old = MetricInstance('rank_ic')
    unaffected = MetricInstance('pearson_ic')
    original_get = metrics.get_metric
    def upgraded(name):
        spec = original_get(name)
        return replace(spec, metric_version=spec.metric_version + '.next') if name == old.metric_id else spec
    monkeypatch.setattr(metrics, 'get_metric', upgraded)
    new = MetricInstance('rank_ic')
    assert old.metric_id == new.metric_id
    assert old.instance_id != new.instance_id
    assert unaffected.instance_id == MetricInstance('pearson_ic').instance_id
    with pytest.raises(ValueError, match='version is unavailable'):
        MetricInstance.from_dict(old.to_dict())
    fields = dict(source_tree_hash='tree', implementation_hash='impl', route='rank_ic',
                  backend='cpu', parameter_domain_hash='domain', metric_instance_hash=old.instance_id)
    receipt = NumericalQualificationReceipt(**fields, test_run_ref='executed-fixture', assertions={'golden':'PASS'})
    with pytest.raises(ValueError, match='mismatch'):
        receipt.require_scope(**{**fields, 'metric_instance_hash':new.instance_id})


def test_t163_t165_actual_device_storage_and_transfer_capability():
    cp = pytest.importorskip('cupy')
    from quant_evaluator.contracts.backend_policy import GPUExecutionPolicy, PrecisionPolicy
    from quant_evaluator.runtime.device_session import DeviceEvaluationSession, UnsupportedBackendCapability
    host = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    for policy, expected in ((PrecisionPolicy.GPU_FP64, 'float64'), (PrecisionPolicy.GPU_MIXED, 'float32')):
        with DeviceEvaluationSession(GPUExecutionPolicy(precision_policy=policy)) as session:
            device = session.stage_factors(host, ('a', 'b'))
            assert str(device.dtype) == expected
            before = cp.asnumpy(device)
            host[:] += 1
            np.testing.assert_array_equal(cp.asnumpy(device), before)
            meta = session.metadata()
            assert meta['storage_dtypes']['factors'] == expected
            assert meta['effective_transfer'] == 'synchronous'
            assert not meta['effective_pinned'] and meta['effective_buffers'] == 1
    with pytest.raises(UnsupportedBackendCapability):
        with DeviceEvaluationSession(GPUExecutionPolicy(device_ids=(0, 1))):
            pass
    with pytest.raises(UnsupportedBackendCapability):
        with DeviceEvaluationSession(GPUExecutionPolicy(required_capabilities=('async_transfer',))):
            pass


def test_a14_ols_cusum_reference_and_unsupported_recursive_scope():
    from quant_evaluator.metrics.stats.structural_breaks import cusum_test
    from statsmodels.stats.diagnostic import breaks_cusumolsresid
    x = np.random.default_rng(782).normal(size=80); x -= x.mean()
    path, boundary, actual, stable = cusum_test(x, residual_kind='ols')
    expected, _, critical = breaks_cusumolsresid(x)
    assert actual == pytest.approx(expected)
    assert boundary == dict(critical)[5]
    with pytest.raises(ValueError, match='not qualified'):
        cusum_test(x)
    with pytest.raises(ValueError, match='alpha'):
        cusum_test(x, alpha=.001, residual_kind='ols')


def test_t38_lag_counts_are_explicit():
    from quant_evaluator.metrics.temporal import compute_autocorrelation_evidence
    x = np.full(90, np.nan); x[::2] = np.arange(45.)
    result = compute_autocorrelation_evidence(x, max_lag=2, clock_ref='trading_grid')
    assert result['pair_counts'][1] == 0 and result['pair_counts'][2] == 44
    assert result['status'][1] == 'INSUFFICIENT_DATA'


def test_t37_invalid_values_cannot_change_rank_stability():
    from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
    from quant_evaluator.metrics.registry_adapters import compute_rank_stability_value
    x = np.random.default_rng(792).normal(size=(30, 30, 1))
    mask = np.ones(x.shape, dtype=bool); mask[:, :5] = False
    def batch(values):
        return FactorBatch(('f',), AxisRef('time','int64',30,np.arange(30)), AxisRef('asset','int64',30,np.arange(30)), values, mask)
    expected = compute_rank_stability_value(batch(x))
    x[:, :5] = 1e30
    np.testing.assert_allclose(compute_rank_stability_value(batch(x)), expected, equal_nan=True)
@pytest.mark.parametrize('q', [10, 20, 40])
def test_quantile_returns_bounded_workspace_matches_full(q):
    cp = pytest.importorskip('cupy')
    from quant_evaluator.kernels.gpu.quantile import batched_quantile_returns
    rng = np.random.default_rng(816)
    x = rng.normal(size=(3, 2, 200))
    r = rng.normal(size=(3, 200))
    x[1, 0, 3] = np.inf
    r[2, 5] = np.nan
    full, full_counts = batched_quantile_returns(x, r, q, min_assets=1, return_counts=True)
    tile, tile_counts = batched_quantile_returns(x, r, q, min_assets=1, return_counts=True, workspace_bytes=65536)
    np.testing.assert_allclose(cp.asnumpy(tile), cp.asnumpy(full), equal_nan=True)
    np.testing.assert_array_equal(cp.asnumpy(tile_counts), cp.asnumpy(full_counts))
    for t in range(3):
        for f in range(2):
            finite = np.isfinite(x[t, f])
            bounds = np.quantile(x[t, f, finite], np.arange(1, q) / q)
            buckets = np.searchsorted(bounds, x[t, f], side='right')
            for bucket in range(q):
                members = finite & np.isfinite(r[t]) & (buckets == bucket)
                expected = r[t, members].mean() if members.any() else np.nan
                np.testing.assert_allclose(float(tile[t, bucket, f]), expected, equal_nan=True)
    with pytest.raises(MemoryError):
        batched_quantile_returns(x, r, q, workspace_bytes=1)


def test_t162_cache_replacement_does_not_evict_other_key():
    from quant_evaluator.runtime.cache_v2 import MemoryCacheLayer, NoCompressor
    cache = MemoryCacheLayer(3000, NoCompressor(), workspace_budget_bytes=100000)
    assert cache.put('other', b'a' * 1300)
    assert cache.put('replace', b'b' * 1300)
    assert cache.put('replace', b'c' * 1300)
    assert cache.get('other')[0] == b'a' * 1300
    assert cache.get('replace')[0] == b'c' * 1300
    assert cache.get_stats()['current_size_bytes'] <= 3000


def test_t161_cache_read_checks_budget_before_decompression(monkeypatch):
    from quant_evaluator.runtime.cache_v2 import MemoryCacheLayer, ZlibCompressor
    compressor = ZlibCompressor()
    cache = MemoryCacheLayer(65536, compressor, workspace_budget_bytes=100000)
    assert cache.put('large', np.zeros(1000))
    cache.workspace_budget_bytes = 100
    monkeypatch.setattr(compressor, 'decompress', lambda _: pytest.fail('decompressed before budget admission'))
    assert cache.get('large') is None
    assert cache.get_stats()['num_entries'] == 1


def test_cache_dict_budget_and_custom_array_reducer_are_rejected_before_pickle(monkeypatch):
    import quant_evaluator.runtime.cache_v2 as module
    cache = module.MemoryCacheLayer(100000, module.ZlibCompressor(), workspace_budget_bytes=100000)
    values = {str(i): np.arange(2000, dtype=float) for i in range(100)}
    monkeypatch.setattr(module.pickle, 'dumps', lambda *a, **k: pytest.fail('serialized inadmissible payload'))
    assert not cache.put('large-dict', values)
    class CustomArray(np.ndarray):
        def __reduce__(self):
            pytest.fail('executed arbitrary reducer')
    assert not cache.put('custom-array', np.arange(3).view(CustomArray))
    nested = []
    for _ in range(1000): nested = [nested]
    assert not cache.put('too-deep', nested)
