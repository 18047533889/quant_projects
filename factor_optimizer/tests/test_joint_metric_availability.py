import numpy as np
import pytest
from factor_optimizer import research_fitness as fitness


@pytest.mark.parametrize('block', [0, 1, 2])
def test_unavailable_sharpe_in_any_chronological_block_is_not_hidden(block):
    rng = np.random.default_rng(214)
    a = np.column_stack((rng.normal(0, .1, 60), rng.normal(0, .01, 60), np.ones(60)))
    a[block*20:(block+1)*20, 1] = 0.
    with pytest.raises(fitness.JointMetricsUnavailable) as exc:
        fitness.summarize(a)
    assert exc.value.metrics == ('worst_block_sharpe',)


@pytest.mark.parametrize('case,code', [('short', 'insufficient_observations'),
    ('missing', 'missing_observations'), ('constant', 'undefined_ratios')])
def test_unavailable_metrics_have_typed_reason(case, code):
    error = getattr(fitness, 'JointMetricsUnavailable', None)
    assert error is not None
    rng = np.random.default_rng(201)
    a = np.column_stack((rng.normal(0, .1, 40), rng.normal(0, .01, 40), np.ones(40)))
    if case == 'short':
        a = a[:10]
    elif case == 'missing':
        a[3, 0] = np.nan
    else:
        a[:, 1] = 0.
    with pytest.raises(error) as exc:
        fitness.summarize(a)
    assert exc.value.code == code
    if case == 'constant':
        assert set(exc.value.metrics) == {'sharpe', 'worst_block_sharpe'}


@pytest.mark.parametrize('unexpected', [False, True])
@pytest.mark.parametrize('stage', ['summary', 'bootstrap'])
def test_validation_evidence_failure_is_not_operator_failure_or_retry(monkeypatch, unexpected, stage):
    from factor_optimizer.research_batch import optimize_factor_batch, BatchOptimizationConfig
    from factor_optimizer.adapters.repair_execution import ValueRepairPlan
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle
    error = getattr(fitness, 'JointMetricsUnavailable', None)
    assert error is not None
    rng = np.random.default_rng(211)
    x = rng.normal(size=(240, 40))
    y = -.005*x + rng.normal(0, .001, x.shape)
    ta = AxisRef('time', 'int', 240, np.arange(240))
    aa = AxisRef('asset', 'str', 40, np.array([f'a{i}' for i in range(40)]))
    b = FactorBatch(('negative',), ta, aa, x[:, :, None])
    labels = LabelBundle('synthetic', y, 1, decision_time=tuple(range(240)),
        label_start_time=tuple(range(1, 241)), label_end_time=tuple(range(2, 242)), asset_axis=aa)
    original = fitness.summarize
    calls = []
    def summarize(a, **kw):
        if len(a) < 80 and stage == 'summary':
            calls.append(len(a))
            if unexpected:
                raise ValueError('implementation failure')
            raise error('undefined_ratios', 'zero variance', metrics=('sharpe',))
        return original(a, **kw)
    monkeypatch.setattr(fitness, 'summarize', summarize)
    if stage == 'bootstrap':
        def compare(*args, **kwargs):
            calls.append('bootstrap')
            if unexpected:
                raise ValueError('implementation failure')
            raise error('undefined_ratios', 'zero variance', metrics=('sharpe',))
        monkeypatch.setattr(fitness, 'compare_joint', compare)
    executions = []
    execute = ValueRepairPlan.execute
    def measured(self, values, **kwargs):
        executions.append(len(values))
        return execute(self, values, **kwargs)
    monkeypatch.setattr(ValueRepairPlan, 'execute', measured)
    result = optimize_factor_batch(b, labels, allow_research=True,
        config=BatchOptimizationConfig(families=('SIGN_ORIENTATION',), bootstrap_draws=99))
    row = result.factors['negative']
    assert row.validation_candidate_identity is not None
    assert row.train_gain > 0
    assert row.validation_lower_bound is None
    assert row.status == ('error_raw_retained' if unexpected else 'raw_retained')
    assert row.selected_family == 'NO_OP_RAW'
    assert len(calls) == 1
    assert executions.count(result.split.test_start*40) == 1
    np.testing.assert_array_equal(result.optimized.values, b.values)
    assert not result.test_evaluated
    if not unexpected:
        assert row.joint_diagnostics['comparison_status'] == 'METRICS_UNAVAILABLE'
        assert row.joint_diagnostics['unavailable_code'] == 'undefined_ratios'
        assert row.joint_diagnostics['unavailable_metrics'] == ('sharpe',)
        assert row.joint_diagnostics['unavailable_role'] == ('raw' if stage == 'summary' else 'bootstrap')
