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
