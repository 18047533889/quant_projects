import numpy as np
import pytest


def series():
    rng = np.random.default_rng(92)
    return np.column_stack((rng.normal(.05, .08, 120),
                            rng.normal(.001, .008, 120), np.full(120, .4)))


def test_joint_utility_rejects_ic_gain_with_material_drawdown_loss():
    from factor_optimizer.research_fitness import summarize, joint_utility, passes_floors
    raw = series()
    candidate = raw.copy()
    candidate[:, 0] += .02
    candidate[50:55, 1] = -.15
    a, b = summarize(raw), summarize(candidate)
    assert b['rank_ic'] > a['rank_ic']
    assert not passes_floors(a, b)
    assert joint_utility(b) < joint_utility(a)


def test_missing_portfolio_return_does_not_fabricate_zero_drawdown():
    from factor_optimizer.research_fitness import summarize
    sample = series()
    sample[40, 1] = np.nan
    with pytest.raises(ValueError, match='missing|finite'):
        summarize(sample)


def test_paired_bootstrap_identical_evidence_has_exact_zero_interval():
    from factor_optimizer.research_fitness import compare_joint
    from factor_optimizer.research_batch import BatchOptimizationConfig
    result = compare_joint(series(), series(), BatchOptimizationConfig(bootstrap_draws=99))
    assert result.difference_interval == (0., 0.)
    assert result.mean_difference == 0.


def test_joint_portfolio_membership_never_drops_missing_selected_return():
    from factor_optimizer.research_fitness import portfolio_series
    values = np.tile(np.arange(20, dtype=float), (60, 1))
    returns = np.full_like(values, .01)
    returns[10, -1] = np.nan
    pnl, turnover = portfolio_series(values, returns, cost_rate=0.)
    assert np.isnan(pnl[10])
    np.testing.assert_allclose(pnl[np.arange(60) != 10], 0., atol=1e-12)
    assert turnover[0] == 1.
    assert (turnover[1:] == 0).all()


def test_cost_charged_on_entry_and_full_notional_rebalance():
    from factor_optimizer.research_fitness import portfolio_series
    values = np.tile(np.arange(20, dtype=float), (60, 1))
    values[1:] *= -1
    pnl, turnover = portfolio_series(values, np.zeros_like(values), cost_rate=.001)
    np.testing.assert_allclose(turnover[:3], [1., 2., 0.])
    np.testing.assert_allclose(pnl[:3], [-.001, -.002, 0.])


def test_constantly_tied_signal_is_unavailable_not_cash_success():
    from factor_optimizer.research_fitness import portfolio_series
    pnl, turnover = portfolio_series(np.ones((60, 20)), np.ones((60, 20))*.01)
    assert np.isnan(pnl).all()


def test_nonconstant_ties_that_empty_a_leg_follow_qe_cash_and_exit_cost():
    from factor_optimizer.research_fitness import portfolio_series
    values = np.vstack((np.arange(20, dtype=float), np.r_[np.zeros(19), 1.],
                        np.r_[np.zeros(19), 1.], np.arange(20, dtype=float)))
    # The sparse signal is known and nonconstant, but all percentile cutoffs
    # are zero, so tie='max' leaves the bottom bucket empty.
    pnl, turnover = portfolio_series(values, np.zeros_like(values), cost_rate=.001)
    np.testing.assert_array_equal(turnover, [1., 1., 0., 1.])
    np.testing.assert_allclose(pnl, [-.001, -.001, 0., -.001])
    legacy, _ = portfolio_series(values, np.zeros_like(values), empty_leg_policy='unavailable')
    assert np.isnan(legacy[1:3]).all()
    np.testing.assert_allclose(legacy[[0, 3]], [-.001, -.001])


def test_signal_cash_rule_is_label_independent_but_missing_signal_is_unknown():
    from factor_optimizer.research_fitness import portfolio_series
    values = np.tile(np.r_[np.zeros(19), 1.], (3, 1))
    returns = np.full_like(values, np.nan)
    pnl, turnover = portfolio_series(values, returns)
    np.testing.assert_array_equal(pnl, np.zeros(3))
    np.testing.assert_array_equal(turnover, np.zeros(3))
    values[1, :] = np.nan
    values[2, :] = 1.
    missing, _ = portfolio_series(values, returns)
    assert missing[0] == 0.
    assert np.isnan(missing[1:]).all()


def test_default_batch_uses_joint_metrics_and_test_labels_do_not_select():
    from dataclasses import replace
    from factor_optimizer.research_batch import optimize_factor_batch, BatchOptimizationConfig
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle
    rng = np.random.default_rng(192)
    x = rng.normal(size=(240, 40))
    y = .003*x + rng.normal(0, .01, x.shape)
    ta = AxisRef('time', 'int', 240, np.arange(240))
    aa = AxisRef('asset', 'str', 40, np.array([f'a{i}' for i in range(40)]))
    batch = FactorBatch(('reverse',), ta, aa, -x[:, :, None])
    labels = LabelBundle('daily_return', y, 1, decision_time=tuple(range(240)),
        label_start_time=tuple(range(1,241)), label_end_time=tuple(range(2,242)), asset_axis=aa)
    conf = BatchOptimizationConfig(families=('SIGN_ORIENTATION',), bootstrap_draws=99)
    a = optimize_factor_batch(batch, labels, config=conf, allow_research=True)
    selected = a.factors['reverse']
    assert selected.selected_family == 'SIGN_ORIENTATION'
    assert selected.joint_diagnostics['objective'] == 'joint'
    assert selected.joint_diagnostics['validation_candidate']['sharpe'] > 0
    assert selected.validation_lower_bound > 0
    assert selected.training_diagnostics['input_stage'] == 'raw'
    assert all('negative_rank_ic' in c['diagnosed_issues']
               for c in selected.candidates if c['family'] == 'SIGN_ORIENTATION')
    poisoned = y.copy()
    poisoned[192:] = np.nan
    b = optimize_factor_batch(batch, replace(labels, values=poisoned), config=conf, allow_research=True)
    assert b.factors['reverse'].joint_diagnostics == selected.joint_diagnostics
    assert b.factors['reverse'].plan_identity == selected.plan_identity


def test_raw_series_cache_is_exact_across_masks_costs_and_mutated_returns():
    from factor_optimizer import research_fitness as fitness
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle
    from dataclasses import replace
    assert hasattr(fitness, "RawSeriesCache"), "bounded RAW metric reuse is missing"
    rng = np.random.default_rng(881)
    raw = rng.normal(size=(60, 40))
    candidate = -raw.copy()
    ta = AxisRef("time", "int", 60, np.arange(60))
    aa = AxisRef("asset", "str", 40, np.array([str(i) for i in range(40)]))
    batch = FactorBatch(("x",), ta, aa, raw[:, :, None])
    labels = LabelBundle("y", rng.normal(0, .01, raw.shape), 1,
        decision_time=tuple(range(60)), label_start_time=tuple(range(60)),
        label_end_time=tuple(range(1,61)), asset_axis=aa)
    cache = fitness.RawSeriesCache()
    cases = [(candidate, labels, .001), (candidate*2, labels, .001)]
    missing = candidate.copy()
    missing[10:15, :5] = np.nan
    cases += [(missing, labels, .001), (candidate, labels, .002)]
    y = labels.values.copy()
    y[20] *= -2
    cases += [(candidate, replace(labels, values=y), .002)]
    for c, target, cost in cases:
        expected = fitness.paired_series(raw, c, batch, target, tuple(range(60)), cost_rate=cost)
        actual = fitness.paired_series(raw, c, batch, target, tuple(range(60)),
                                       cost_rate=cost, raw_cache=cache)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        actual[0][:] = 999  # Caller mutation must not poison a reused baseline.
        again = fitness.paired_series(raw, c, batch, target, tuple(range(60)),
                                      cost_rate=cost, raw_cache=cache)
        np.testing.assert_array_equal(again[0], expected[0])
    raw[10] = 0.
    raw[10, -1] = 1.
    for policy in ('unavailable', 'signal_cash', 'unavailable'):
        expected = fitness.paired_series(raw, raw, batch, labels, tuple(range(60)),
                                         empty_leg_policy=policy)
        actual = fitness.paired_series(raw, raw, batch, labels, tuple(range(60)),
                                       raw_cache=cache, empty_leg_policy=policy)
        np.testing.assert_array_equal(actual[0], expected[0])
        assert bool(np.isnan(actual[0][10, 1])) == (policy == 'unavailable')
