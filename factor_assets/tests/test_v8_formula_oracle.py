"""Independent section-15 formula oracle: no production score as expected."""
import hashlib
import json
import numpy as np
import pytest
from factor_assets.selection.decision import JointUtilityEvidence


def evidence(values, fw=(.10, .25, .65), sw=(.2, .3, .5)):
    values = np.asarray(values)
    semantic = dict(context='synthetic-formula-only', plan='same-clock-plan',
        replicates=tuple(f'b{i}' for i in range(len(values))),
        blocks=('P','E','R','S','I','O'), block_weights=(.2,.3,.1,.15,.15,.1),
        window_ids=('early','middle','late'), window_weights=fw,
        scenario_ids=('base','adverse','severe'), scenario_weights=sw,
        samples=values.tolist(), qualification='math-oracle-only',
        coefficients=(.55,.75,.60,.20,.10))
    digest = hashlib.sha256(json.dumps(semantic,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return JointUtilityEvidence(evidence_id='oracle-input',content_hash=digest,
        comparison_context_hash=semantic['context'],resampling_plan_ref=semantic['plan'],
        replicate_ids=semantic['replicates'],block_ids=semantic['blocks'],
        block_weights=semantic['block_weights'],window_weights=fw,scenario_weights=sw,
        samples=semantic['samples'],qualification_scope=semantic['qualification'],
        window_ids=semantic['window_ids'],scenario_ids=semantic['scenario_ids'])


def independent_score_samples(x, fw=(.10,.25,.65), sw=(.2,.3,.5)):
    w = np.array([.2,.3,.1,.15,.15,.1])
    geom = np.exp((np.log(np.where(x == 0, 1, x)) * w).sum(axis=-1))
    geom = np.where((x == 0).any(axis=-1), 0, geom)
    u = .55 * (x * w).sum(axis=-1) + .45 * geom
    q = .75 * (u * np.asarray(sw)).sum(axis=-1) + .25 * u.min(axis=-1)
    order = np.argsort(q, axis=-1, kind='stable')
    sorted_q = np.take_along_axis(q, order, axis=-1)
    weights = np.take_along_axis(np.broadcast_to(fw, q.shape), order, axis=-1)
    before = np.cumsum(weights, axis=-1) - weights
    admitted = np.minimum(weights, np.maximum(.20 - before, 0))
    tail = (admitted * sorted_q).sum(axis=-1) / .20
    return 100 * (.60 * (q * fw).sum(axis=-1) + .40 * tail)


def test_v8_full_4d_formula_unequal_weights_fractional_tail_linear_q10():
    x = np.random.default_rng(891).uniform(.01, .99, size=(17, 3, 3, 6))
    x[3,1,2,4] = 0
    actual = evidence(x)
    reference = independent_score_samples(x)
    np.testing.assert_allclose(100*np.asarray(actual.replicate_scores()), reference, rtol=1e-13, atol=1e-13)
    mean, lower = actual.fitness()
    assert mean == pytest.approx(reference.mean(), abs=1e-12)
    assert lower == pytest.approx(np.quantile(reference,.1,method='linear'), abs=1e-12)
    assert abs(lower - np.sort(reference)[1]) > 1e-5


def test_v8_joint_formula_monotonicity_and_measured_zero():
    x = np.random.default_rng(892).uniform(.01,.7,size=(17,3,3,6))
    better = x.copy(); better[...,2] += .2
    assert evidence(better).fitness()[1] >= evidence(x).fitness()[1]
    zero = evidence(np.zeros_like(x))
    assert zero.fitness() == (0.,0.)
    one = evidence(np.ones_like(x))
    np.testing.assert_allclose(one.fitness(), (100.,100.))


def test_v8_joint_formula_pairing_is_per_replicate_not_difference_of_quantiles():
    x = np.random.default_rng(893).uniform(.01,.99,size=(17,3,3,6))
    y = x[::-1].copy()
    ax, ay = evidence(x), evidence(y)
    actual = 100 * (np.asarray(ax.replicate_scores()) - np.asarray(ay.replicate_scores()))
    expected = independent_score_samples(x) - independent_score_samples(y)
    np.testing.assert_allclose(actual,expected,atol=1e-12)
    assert np.quantile(actual,.1) < -1
    assert ax.fitness()[1]-ay.fitness()[1] == pytest.approx(0)


def test_t06_display_only_zero_weight_cannot_zero_geometric_utility():
    from types import SimpleNamespace
    from factor_assets.selection.decision import DecisionProvider, SelectionPolicySpec, MetricRule, UtilityDirection
    active=MetricRule('active',UtilityDirection.HIGHER_IS_BETTER,1.,0.,1.)
    display=MetricRule('display',UtilityDirection.HIGHER_IS_BETTER,0.,0.,1.)
    base=SelectionPolicySpec('p','1',(active,),{})
    extended=SelectionPolicySpec('p','1',(active,display),{})
    def source(names, values):
        return SimpleNamespace(metric_ids=names,evidence_id='test',comparison_context_hash='context',
            resampling_plan_ref='plan',replicate_ids=('a','b'),window_ids=('window',),
            scenario_ids=('scenario',),qualification_scope='arithmetic-test-only',
            samples=tuple(((tuple(values),),) for _ in range(2)))
    expected=DecisionProvider(base)._map_raw_evidence(source(('active',),(.5,))).fitness()
    actual=DecisionProvider(extended)._map_raw_evidence(source(('active','display'),(.5,0.))).fitness()
    assert actual==pytest.approx(expected)
