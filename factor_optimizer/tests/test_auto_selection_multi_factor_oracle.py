"""Multi-factor oracle checks for TRAIN-only automatic candidate selection."""
from dataclasses import replace
import numpy as np


def _case():
    from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
    from quant_evaluator.contracts.label_bundle import LabelBundle
    rng = np.random.default_rng(20261003)
    t, n = 240, 40
    z = np.zeros((t, n))
    shocks = rng.normal(size=z.shape)
    for i in range(1, t):
        z[i] = .9 * z[i - 1] + shocks[i]
    y = z**2
    values = np.stack((-y, z, np.ones_like(z), np.full_like(z, np.nan),
                       np.vstack((np.zeros((1, n)), z[:-1])),
                       rng.normal(size=(t, n))), axis=2)
    ta = AxisRef("time", "int", t, np.arange(t))
    aa = AxisRef("asset", "str", n, np.asarray([f"a{i}" for i in range(n)]))
    batch = FactorBatch(("negative_ic", "u_shape", "constant", "all_missing",
                         "lagged", "noise"), ta, aa, values)
    labels = LabelBundle("multi-factor-oracle", y, 1,
        decision_time=tuple(range(t)), label_start_time=tuple(range(1, t+1)),
        label_end_time=tuple(range(2, t+2)), asset_axis=aa)
    return batch, labels


def _config():
    from factor_optimizer.research_batch import BatchOptimizationConfig
    return BatchOptimizationConfig(
        families=("SIGN_ORIENTATION", "U_SHAPE_REPAIR", "INVERTED_U_REPAIR",
                  "CAUSAL_SMOOTHING", "DECAY_REFINEMENT"),
        selection_objective="rank_ic", minimum_train_days=40,
        minimum_validation_days=20, minimum_test_days=20, minimum_assets=10,
        natural_time_scale=10, bootstrap_draws=99, maximum_candidates=128)


def _train_oracle_identity(result, threshold):
    eligible = [r for r in result.candidates if r.get("status") == "train_evaluated"
                and r.get("train_gain", -np.inf) > threshold]
    if not eligible:
        return None
    return min(eligible, key=lambda r: (-r["train_gain"], r["plan_identity"]))["plan_identity"]


def test_multi_factor_train_oracle_and_validation_no_retry():
    from factor_optimizer.research_batch import optimize_factor_batch
    batch, labels = _case()
    config = _config()
    baseline = optimize_factor_batch(batch, labels, config=config, allow_research=True)
    assert baseline.factors["negative_ic"].selected_family == "SIGN_ORIENTATION"
    assert baseline.factors["u_shape"].selected_family == "U_SHAPE_REPAIR"
    for result in baseline.factors.values():
        assert result.validation_candidate_identity == _train_oracle_identity(
            result, config.minimum_improvement)
    for name in ("constant", "all_missing"):
        assert baseline.factors[name].status == "invalid_raw"
        assert baseline.factors[name].validation_candidate_identity is None

    attacked_values = labels.values.copy()
    attacked_values[baseline.split.validation_start:baseline.split.test_start] = 0.0
    attacked_values[baseline.split.test_start:] = 1e12
    attacked = optimize_factor_batch(batch, replace(labels, values=attacked_values),
                                     config=config, allow_research=True)
    assert baseline.test_evaluated is False
    assert attacked.test_evaluated is False
    for factor_id, original in baseline.factors.items():
        result = attacked.factors[factor_id]
        assert result.validation_candidate_identity == original.validation_candidate_identity
        assert result.train_gain == original.train_gain
        assert _train_oracle_identity(result, config.minimum_improvement) == (
            original.validation_candidate_identity)
        if original.validation_candidate_identity is not None:
            assert result.validation_lower_bound is None
            assert result.selected_family == "NO_OP_RAW"
