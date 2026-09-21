from dataclasses import replace
import numpy as np

from factor_optimizer.research_batch import automatic_time_split
from quant_evaluator.contracts.factor_batch import FactorBatch, AxisRef
from quant_evaluator.contracts.label_bundle import LabelBundle


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
