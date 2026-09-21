from dataclasses import replace
import numpy as np

from factor_optimizer.research_batch import automatic_time_split
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle

def test_turnover_diagnosis_counts_signal_driven_cash_exit_and_reentry():
    from factor_optimizer.research_decay import diagnose_layer_decay
    from factor_optimizer.research_batch import BatchOptimizationConfig
    batch, labels = panel(40)
    x = np.tile(np.arange(40, dtype=float), (300, 1))
    x[70] = np.r_[np.zeros(39), 1.]
    batch = replace(batch, values=x[:, :, None])
    cfg = BatchOptimizationConfig()
    split = automatic_time_split(labels, cfg)
    row = diagnose_layer_decay(batch, labels, split, cfg, 0)
    assert np.isclose(row['full_notional_turnover'], 3/len(split.train_indices))
    assert row['turnover_unavailable_reason'] is None
    old = diagnose_layer_decay(batch, labels, split,
                              replace(cfg, research_empty_leg_policy='unavailable'), 0)
    assert old['full_notional_turnover'] is None
    assert old['turnover_unavailable_reason']


def test_selection_diagnosis_matches_costed_joint_scoring_and_reports_issues():
    from factor_optimizer.research_batch import BatchOptimizationConfig
    from factor_optimizer.research_diagnostics import diagnose_training_batch
    from factor_optimizer.research_fitness import paired_series, summarize
    batch, labels = panel(40)
    rng = np.random.default_rng(231)
    x = rng.normal(size=(300, 40))
    y = -.002*x + rng.normal(0, .01, x.shape)
    batch, labels = replace(batch, values=x[:, :, None]), replace(labels, values=y)
    cfg = BatchOptimizationConfig()
    row = diagnose_training_batch(batch, labels, config=cfg)['u_shape']
    raw, _ = paired_series(x, x, batch, labels, automatic_time_split(labels, cfg).train_indices,
                          cost_rate=cfg.research_cost_rate,
                          empty_leg_policy=cfg.research_empty_leg_policy)
    assert row['selection_portfolio']['metrics'] == summarize(raw)
    assert row['selection_portfolio']['status'] == 'available'
    codes = {issue['code'] for issue in row['issues']}
    assert 'high_turnover' in codes
    assert 'negative_rank_ic' in codes
    poisoned = y.copy()
    poisoned[automatic_time_split(labels, cfg).validation_start:] = np.nan
    changed = diagnose_training_batch(batch, replace(labels, values=poisoned), config=cfg)['u_shape']
    assert changed == row



def panel(n=400):
    t = 300
    rng = np.random.default_rng(291)
    x = np.tile(np.linspace(0, 1, n), (t, 1))
    y = .04*(x-.5)**2 + rng.normal(0, .0001, (t, n))
    ta = AxisRef("time", "int", t, np.arange(t))
    aa = AxisRef("asset", "str", n, np.array([f"a{i}" for i in range(n)]))
    batch = FactorBatch(("u_shape",), ta, aa, x[:, :, None])
    labels = LabelBundle("synthetic", y, 1, decision_time=tuple(range(t)),
        label_start_time=tuple(range(1, t+1)), label_end_time=tuple(range(2, t+2)),
        asset_axis=aa)
    return batch, labels


def test_twenty_bin_profile_detects_u_shape_with_actual_bin_counts():
    from factor_optimizer.research_diagnostics import diagnose_training_batch
    batch, labels = panel()
    row = diagnose_training_batch(batch, labels)["u_shape"]
    assert row["partition"] == "TRAIN"
    assert len(row["quantile_mean_returns"]) == 20
    assert row["minimum_bin_count"] == 20
    assert row["u_contrast"] > .006
    assert row["complete_quantile_days"] == len(automatic_time_split(labels).train_indices)
    assert row["proposed_shape_family"] == "U_SHAPE_REPAIR"
    assert .4 <= row["proposed_center"] <= .6


def test_insufficient_assets_do_not_fake_twenty_layers():
    from factor_optimizer.research_diagnostics import diagnose_training_batch
    batch, labels = panel(100)
    row = diagnose_training_batch(batch, labels)["u_shape"]
    assert row["quantile_mean_returns"] is None
    assert row["proposed_shape_family"] is None
    assert row["long_short_sharpe"] is None
    assert row["complete_quantile_days"] == 0


def test_future_factor_and_label_poison_cannot_change_training_diagnosis():
    from factor_optimizer.research_diagnostics import diagnose_training_batch
    batch, labels = panel()
    expected = diagnose_training_batch(batch, labels)
    cut = automatic_time_split(labels).validation_start
    values, y = batch.values.copy(), labels.values.copy()
    values[cut:] = np.nan
    y[cut:] = 999.
    other = replace(batch, values=values)
    assert diagnose_training_batch(other, replace(labels, values=y)) == expected


def test_overlapping_forward_labels_are_not_compounded_as_daily_pnl():
    from factor_optimizer.research_diagnostics import diagnose_training_batch
    batch, labels = panel()
    row = diagnose_training_batch(batch, replace(labels, horizon=2))["u_shape"]
    assert row["quantile_mean_returns"] is not None
    assert row["long_short_sharpe"] is None
    assert row["long_short_max_drawdown"] is None
    assert row["portfolio_unavailable_reason"] == "multi-bar labels are not daily PnL"


def test_missing_day_does_not_become_zero_pnl_or_known_full_drawdown():
    from factor_optimizer.research_diagnostics import diagnose_training_batch
    batch, labels = panel()
    values = batch.values.copy()
    values[70] = np.nan
    row = diagnose_training_batch(replace(batch, values=values), labels)["u_shape"]
    assert row["complete_quantile_days"] == len(automatic_time_split(labels).train_indices)-1
    assert row["long_short_max_drawdown"] is None


def test_single_bar_tag_cannot_hide_overlapping_label_timestamps():
    from factor_optimizer.research_diagnostics import diagnose_training_batch
    batch, labels = panel()
    overlapping = replace(labels, label_end_time=tuple(range(3, 303)))
    row = diagnose_training_batch(batch, overlapping)["u_shape"]
    assert row["quantile_mean_returns"] is not None
    assert row["long_short_sharpe"] is None
    assert row["long_short_max_drawdown"] is None
    assert "overlapping" in row["portfolio_unavailable_reason"]


def test_automatic_search_uses_train_fitted_twenty_bin_center():
    from factor_optimizer.research_batch import optimize_factor_batch, BatchOptimizationConfig
    batch, labels = panel()
    x = batch.values[:, :, 0]
    y = .04*(x-.2)**2
    config = BatchOptimizationConfig(selection_objective='rank_ic', families=("U_SHAPE_REPAIR",), bootstrap_draws=99)
    result = optimize_factor_batch(batch, replace(labels, values=y),
                                   config=config, allow_research=True)
    outcome = result.factors["u_shape"]
    assert outcome.training_diagnostics["proposed_shape_family"] == "U_SHAPE_REPAIR"
    center = outcome.training_diagnostics["proposed_center"]
    assert .15 < center < .25
    assert any(r["parameters"].get("center") == center for r in outcome.candidates)
    assert outcome.status == "improved"
    assert dict(outcome.plan.parameters)["center"] == center


def test_off_center_u_shape_is_not_required_to_have_valley_at_median():
    from factor_optimizer.research_diagnostics import diagnose_training_batch
    batch, labels = panel()
    y = .04*(batch.values[:, :, 0]-.2)**2
    row = diagnose_training_batch(batch, replace(labels, values=y))["u_shape"]
    assert row["proposed_shape_family"] == "U_SHAPE_REPAIR"
    assert .15 < row["proposed_center"] < .25


def test_nonreturn_scale_labels_do_not_crash_other_factor_diagnoses():
    from factor_optimizer.research_diagnostics import diagnose_training_batch
    batch, labels = panel()
    x = batch.values[:, :, 0]
    factors = replace(batch, factor_ids=("up", "down"), values=np.stack((x, -x), axis=-1))
    result = diagnose_training_batch(factors, replace(labels, values=5*x))
    assert result["up"]["rank_ic"] > .99
    assert result["down"]["rank_ic"] < -.99
    assert result["down"]["long_short_max_drawdown"] is None
    assert result["down"]["portfolio_unavailable_reason"]


def test_layer_decay_preserves_persistent_twenty_bin_signal_and_right_censoring():
    from factor_optimizer.research_diagnostics import diagnose_training_batch
    batch, labels = panel()
    row = diagnose_training_batch(batch, labels)["u_shape"]["layer_decay"]
    assert row["partition"] == "TRAIN"
    assert row["status"] == "available"
    assert len(row["layers"]) == 20
    for layer in row["layers"]:
        np.testing.assert_allclose(layer["mean_excess_returns"],
                                   layer["mean_excess_returns"][0])
        assert layer["half_life_bars"] is None
        assert layer["right_censored"]
    assert row["full_notional_turnover"] < .02


def test_layer_decay_detects_fast_loss_without_recommending_slow_smoothing():
    from factor_optimizer.research_diagnostics import diagnose_training_batch
    batch, labels = panel()
    rng = np.random.default_rng(216)
    x = rng.normal(size=batch.values.shape[:2])
    batch = replace(batch, values=x[:, :, None])
    labels = replace(labels, values=.01*x + rng.normal(0, .001, x.shape))
    row = diagnose_training_batch(batch, labels)["u_shape"]["layer_decay"]
    assert row["status"] == "available"
    assert row["layers"][0]["half_life_bars"] == 1
    assert row["layers"][-1]["half_life_bars"] == 1
    assert row["full_notional_turnover"] > 1
    assert row["proposed_half_lives"] == []


def test_layer_decay_insufficient_bins_is_not_an_invented_half_life():
    from factor_optimizer.research_diagnostics import diagnose_training_batch
    batch, labels = panel(100)
    row = diagnose_training_batch(batch, labels)["u_shape"]["layer_decay"]
    assert row["status"] == "unavailable"
    assert row["layers"] == []
    assert row["proposed_half_lives"] == []


def test_automatic_smoothing_adds_train_measured_decay_scale():
    from factor_optimizer.research_batch import optimize_factor_batch, BatchOptimizationConfig
    batch, labels = panel(240)
    rng = np.random.default_rng(217)
    x = rng.normal(size=batch.values.shape[:2])
    for t in range(1, len(x)):
        x[t] = .9*x[t-1] + np.sqrt(1-.9**2)*x[t]
    batch = replace(batch, values=x[:, :, None])
    labels = replace(labels, values=.01*x + rng.normal(0, .005, x.shape))
    config = BatchOptimizationConfig(families=("CAUSAL_SMOOTHING",), bootstrap_draws=99)
    result = optimize_factor_batch(batch, labels, config=config, allow_research=True)
    factor = result.factors["u_shape"]
    targets = factor.training_diagnostics["layer_decay"]["proposed_half_lives"]
    assert targets
    targeted = [r for r in factor.candidates if r.get("proposal_source") == "TRAIN_layer_decay"]
    assert targeted
    assert all(r["parameters"]["halflife"] in targets for r in targeted)
    assert all(r["status"] in ("train_evaluated", "ineligible") for r in targeted)
