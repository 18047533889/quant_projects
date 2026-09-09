import numpy as np
import pytest
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import evaluate


def contracts(values, labels):
    t, n, f = values.shape
    return FactorBatch(tuple(f'f{i}' for i in range(f)), AxisRef('time','int',t),
                       AxisRef('asset','int',n), values), LabelBundle('ret',labels,1,
                       decision_time=tuple(range(t)),label_start_time=tuple(range(1,t+1)),
                       label_end_time=tuple(range(2,t+2)))


def test_gpu_turnover_is_not_membership_turnover():
    pytest.importorskip('cupy')
    values = np.tile(np.arange(20.)[None, :, None], (35, 1, 1))
    values[1::2, :10, 0] = np.arange(10.)[::-1]
    batch, labels = contracts(values, np.ones((35,20)))
    metrics = ['turnover','factor_turnover_rate']
    cpu = evaluate(batch, labels, metrics=metrics)
    gpu = evaluate(batch, labels, metrics=metrics, backend='cuda')
    for mid in metrics:
        assert gpu.get_metric(mid,'f0').value == pytest.approx(cpu.get_metric(mid,'f0').value)
    assert gpu.get_metric('turnover','f0').value > 0
    assert gpu.get_metric('factor_turnover_rate','f0').value == 0


def test_gpu_shape_uses_mean_profile_not_daily_direction_votes():
    pytest.importorskip('cupy')
    values = np.tile(np.arange(100.)[None,:,None], (30,1,1))
    labels = np.tile(np.arange(100.)[None,:], (30,1))
    labels[1::2] *= -.1
    batch, label = contracts(values, labels)
    cpu = evaluate(batch,label,metrics=['quantile_monotonicity'])
    gpu = evaluate(batch,label,metrics=['quantile_monotonicity'],backend='cuda')
    assert cpu.get_metric('quantile_monotonicity','f0').value == 1
    assert gpu.get_metric('quantile_monotonicity','f0').value == 1


def test_nondefault_gpu_min_assets_matches_cpu():
    pytest.importorskip('cupy')
    rng = np.random.default_rng(59)
    batch,label = contracts(rng.normal(size=(5,19,2)),rng.normal(size=(5,19)))
    for minimum in [10,20]:
        options = dict(metrics=['rank_ic'],metric_parameters={'rank_ic': {'min_assets':minimum}})
        cpu = evaluate(batch,label,**options)
        gpu = evaluate(batch,label,backend='cuda',**options)
        np.testing.assert_allclose(cpu.artifacts['rank_ic'].values,gpu.artifacts['rank_ic'].values,atol=1e-12)
