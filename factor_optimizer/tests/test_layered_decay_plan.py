import importlib.util
import numpy as np
import pandas as pd


def test_layered_plan_replays_prefix_and_preserves_duplicate_caller_index():
    assert importlib.util.find_spec("factor_optimizer.adapters.layered_decay") is not None
    from factor_optimizer.adapters.layered_decay import LayeredDecayPlan
    rng = np.random.default_rng(523)
    x = rng.normal(size=(60, 40))
    frame = pd.DataFrame({"date": np.repeat(np.arange(60), 40),
                          "asset_id": np.tile(np.arange(40), 60), "value": x.ravel()})
    frame.index = np.arange(len(frame)) % 11
    plan = LayeredDecayPlan((2., 5.)*10, "train-only")
    result = plan.execute(frame, allow_research=True)
    assert result.index.equals(frame.index)
    assert np.isnan(result.iloc[:40]).all()
    prefix = plan.execute(frame.iloc[:1200], allow_research=True)
    np.testing.assert_array_equal(result.iloc[:1200], prefix)
    order = np.arange(len(frame)).reshape(60,40)[:,::-1].ravel()
    permuted = plan.execute(frame.iloc[order], allow_research=True)
    np.testing.assert_array_equal(result.to_numpy(), permuted.to_numpy()[np.argsort(order)])


def test_training_layer_curves_feed_real_layered_candidates():
    import runpy
    from pathlib import Path
    from factor_optimizer.research_batch import optimize_factor_batch, BatchOptimizationConfig
    fixture = runpy.run_path(str(Path(__file__).with_name("test_research_diagnostics.py")))
    batch, labels = fixture["panel"]()
    result = optimize_factor_batch(batch, labels, allow_research=True,
        config=BatchOptimizationConfig(families=("DECAY_REFINEMENT",), bootstrap_draws=99))
    candidates = [c for c in result.factors["u_shape"].candidates
                  if c.get("transform") == "layered_decay"]
    assert candidates
    assert all(len(c["parameters"]["half_lives"]) == 20 for c in candidates)
    assert all(c["proposal_source"] == "TRAIN_layer_states" for c in candidates)
    assert all(c.get("valid_train_days", 0) >= 60 for c in candidates)
    assert all("train_candidate_metrics" in c for c in candidates)


def test_validation_labels_cannot_change_layered_train_scores_but_can_reject_winner():
    from dataclasses import replace
    import runpy
    from pathlib import Path
    from factor_optimizer.research_batch import optimize_factor_batch, BatchOptimizationConfig

    fixture = runpy.run_path(str(Path(__file__).with_name("test_research_diagnostics.py")))
    batch, labels = fixture["panel"](400)
    rng = np.random.default_rng(721)
    innovations = rng.normal(size=labels.values.shape)
    latent = np.empty_like(innovations)
    latent[0] = .2 * innovations[0]
    for t in range(1, len(latent)):
        latent[t] = .98 * latent[t - 1] + .2 * innovations[t]
    factor_values = latent + .8 * rng.normal(size=latent.shape)
    labels = replace(labels, values=latent + .05 * rng.normal(size=latent.shape))
    batch = replace(batch, values=factor_values[:, :, None])
    config = BatchOptimizationConfig(families=("DECAY_REFINEMENT",), bootstrap_draws=99,
                                     selection_objective="rank_ic", minimum_improvement=.001)
    clean = optimize_factor_batch(batch, labels, allow_research=True, config=config)
    split = clean.split
    assert np.isfinite(labels.values[list(split.validation_indices)]).all()
    assert clean.factors["u_shape"].selected_family == "DECAY_REFINEMENT"
    assert clean.factors["u_shape"].status == "improved"
    assert clean.factors["u_shape"].validation_coverage >= config.minimum_coverage
    poisoned_values = labels.values.copy()
    poisoned_values[list(split.validation_indices)] *= -1.
    poisoned = optimize_factor_batch(batch, replace(labels, values=poisoned_values),
                                      allow_research=True, config=config)

    def train_candidate_records(result):
        return [dict(record) for record in result.factors["u_shape"].candidates
                if record.get("transform") == "layered_decay"]

    clean_records = train_candidate_records(clean)
    poisoned_records = train_candidate_records(poisoned)
    assert clean_records and len(clean_records) == len(poisoned_records)
    assert any(record["train_gain"] > config.minimum_improvement for record in clean_records)
    for before, after in zip(clean_records, poisoned_records):
        assert before["parameters"] == after["parameters"]
        assert before["train_gain"] == after["train_gain"]
        assert before["valid_train_days"] == after["valid_train_days"]
    assert poisoned.factors["u_shape"].status == "raw_retained"
    assert poisoned.factors["u_shape"].selected_family == "NO_OP_RAW"
    assert poisoned.factors["u_shape"].reason.startswith("TRAIN winner not confirmed on VALIDATION")
    assert poisoned.factors["u_shape"].validation_coverage >= config.minimum_coverage

    # The joint-scoring path also records portfolio metric dictionaries; compare
    # these directly so validation poisoning cannot alter any nested TRAIN score.
    joint_config = replace(config, selection_objective="joint")
    joint_clean = optimize_factor_batch(batch, labels, allow_research=True, config=joint_config)
    joint_poisoned = optimize_factor_batch(
        batch, replace(labels, values=poisoned_values), allow_research=True, config=joint_config)
    joint_clean_records = train_candidate_records(joint_clean)
    joint_poisoned_records = train_candidate_records(joint_poisoned)
    assert joint_clean_records and len(joint_clean_records) == len(joint_poisoned_records)

    def assert_nested_equal(left, right):
        if isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                assert_nested_equal(a, b)
        elif isinstance(left, (float, np.floating)) and isinstance(right, (float, np.floating)):
            assert (left == right) or (np.isnan(left) and np.isnan(right))
        else:
            assert left == right

    for before, after in zip(joint_clean_records, joint_poisoned_records):
        assert_nested_equal(before["train_raw_metrics"], after["train_raw_metrics"])
        assert_nested_equal(before["train_candidate_metrics"], after["train_candidate_metrics"])


def test_layered_plan_equal_date_rows_have_strict_lag_isolation():
    from factor_optimizer.adapters.layered_decay import LayeredDecayPlan

    rng = np.random.default_rng(914)
    values = rng.normal(size=(8, 40))
    frame = pd.DataFrame({"date": np.repeat(np.arange(8), 40),
                          "asset_id": np.tile(np.arange(40), 8),
                          "value": values.ravel()})
    plan = LayeredDecayPlan((3.,) * 20, "train-only")
    original = frame.copy(deep=True)
    baseline = plan.execute(frame, allow_research=True)

    changed = frame.copy()
    changed.loc[changed["date"] == 4, "value"] *= -100.
    counterfactual = plan.execute(changed, allow_research=True)
    dates = frame["date"].to_numpy()
    same_date = dates == 4
    earlier = dates < 4
    future = dates > 4
    np.testing.assert_array_equal(counterfactual.to_numpy()[same_date], baseline.to_numpy()[same_date])
    np.testing.assert_array_equal(counterfactual.to_numpy()[earlier], baseline.to_numpy()[earlier])
    pd.testing.assert_frame_equal(frame, original)
    future_values, baseline_future = counterfactual.to_numpy()[future], baseline.to_numpy()[future]
    comparable = np.isfinite(future_values) & np.isfinite(baseline_future)
    assert comparable.any()
    assert np.any(future_values[comparable] != baseline_future[comparable])


def test_layered_quantile_blocks_match_full_batch_at_small_middle_and_full_sizes(monkeypatch):
    from quant_evaluator.metrics import quantile
    from factor_preprocess.transforms.layered_decay import layered_decay
    from factor_optimizer.adapters.layered_decay import _assign_daily_quantiles

    rng = np.random.default_rng(2201)
    values = rng.normal(size=(7, 45))
    values[2, ::5] = 3.0
    values[4, ::7] = np.nan
    values[6, ::9] = -2.0
    panel = values.copy()
    full_panel_bins = quantile.assign_quantiles_batch(panel, n_quantiles=20)
    expected = layered_decay(panel, full_panel_bins, (4.,) * 20, allow_research=True)

    calls = []
    original = quantile.assign_quantiles_batch

    def bounded_assignments(chunk, *, n_quantiles=5, method="max"):
        calls.append(chunk.shape)
        return original(chunk, n_quantiles=n_quantiles, method=method)

    monkeypatch.setattr(quantile, "assign_quantiles_batch", bounded_assignments)
    per_date_budget = values.shape[1] * 40 + 20 * 1024
    for budget, expected_rows in ((per_date_budget, 1),
                                  (3 * per_date_budget, 3),
                                  (len(values) * per_date_budget, len(values))):
        calls.clear()
        bins = _assign_daily_quantiles(panel, working_bytes=budget)
        np.testing.assert_array_equal(bins, full_panel_bins)
        actual = layered_decay(panel, bins, (4.,) * 20, allow_research=True)
        np.testing.assert_array_equal(actual, expected)
        assert calls == [(min(expected_rows, len(values) - start), values.shape[1])
                         for start in range(0, len(values), expected_rows)]


def test_layered_quantile_default_respects_working_set_budget(monkeypatch):
    from quant_evaluator.metrics import quantile
    from factor_optimizer.adapters.layered_decay import (
        _assign_daily_quantiles, _QUANTILE_WORKING_BYTES,
    )

    values = np.tile(np.arange(1000, dtype=float), (2000, 1))
    calls = []
    original = quantile.assign_quantiles_batch

    def bounded_assignments(chunk, *, n_quantiles=5, method="max"):
        calls.append(chunk.shape)
        return original(chunk, n_quantiles=n_quantiles, method=method)

    monkeypatch.setattr(quantile, "assign_quantiles_batch", bounded_assignments)
    _assign_daily_quantiles(values)
    max_rows = max(rows for rows, _ in calls)
    assert max_rows * (values.shape[1] * 40 + 20 * 1024) <= _QUANTILE_WORKING_BYTES
    assert len(calls) > 1
    assert max_rows < len(values)
    assert sum(rows for rows, _ in calls) == len(values)


def test_layered_plan_preserves_nonfinite_masks_and_does_not_mutate_input():
    from quant_evaluator.metrics.quantile import assign_quantiles_batch
    from factor_preprocess.transforms.layered_decay import layered_decay
    from factor_optimizer.adapters.layered_decay import LayeredDecayPlan

    rng = np.random.default_rng(2202)
    values = rng.normal(size=(12, 40))
    values[2, :3] = (np.inf, -np.inf, np.nan)
    values[:, -1] = np.nan
    bins = assign_quantiles_batch(values, n_quantiles=20)
    expected = layered_decay(values, bins, (5.,) * 20, allow_research=True)
    frame = pd.DataFrame({"date": np.repeat(np.arange(len(values)), values.shape[1]),
                          "asset_id": np.tile(np.arange(values.shape[1]), len(values)),
                          "value": values.ravel()})
    before = frame.copy(deep=True)

    actual = LayeredDecayPlan((5.,) * 20, "train:nonfinite-parity").execute(
        frame, allow_research=True)

    np.testing.assert_array_equal(actual.to_numpy(), expected.ravel())
    pd.testing.assert_frame_equal(frame, before)
