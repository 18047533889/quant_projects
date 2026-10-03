"""Deterministic holdout oracle: a VALID runner-up must not be retried."""

import numpy as np
import pandas as pd

from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
import factor_optimizer.research_batch as research_batch
from factor_optimizer.research_batch import (
    BatchOptimizationConfig,
    automatic_time_split,
    optimize_factor_batch,
)


def test_validation_checks_only_frozen_train_winner(monkeypatch):
    time_count, asset_count = 240, 40
    per_date_x = np.concatenate((np.arange(-20, 0), np.arange(1, 21))) / 20.0
    x = np.repeat(per_date_x[None, :], time_count, axis=0)
    u = np.abs(x)
    target = np.empty_like(x)
    target[:144] = -0.8 * x[:144] + 0.6 * u[:144]
    target[144:192] = u[144:192] + 0.2 * x[144:192]
    target[192:] = 0.0

    time_axis = AxisRef("time", "int", time_count, np.arange(time_count))
    asset_axis = AxisRef(
        "asset", "str", asset_count,
        np.array([f"asset-{index}" for index in range(asset_count)]),
    )
    batch = FactorBatch(("x",), time_axis, asset_axis, x[:, :, None])
    labels = LabelBundle(
        "forward", target, 1,
        decision_time=tuple(range(time_count)),
        label_start_time=tuple(range(1, time_count + 1)),
        label_end_time=tuple(range(2, time_count + 2)),
        asset_axis=asset_axis,
    )
    config = BatchOptimizationConfig(
        selection_objective="rank_ic",
        families=("SIGN_ORIENTATION", "U_SHAPE_REPAIR"),
        bootstrap_draws=99,
        minimum_improvement=0.01,
        compose_smoothing_sign=False,
    )

    # Isolate the two predeclared treatments: a linear sign flip and the
    # symmetric rank-distance U(x) = |rank(x) - .5|. The train target remains
    # strictly decreasing in x, so SIGN ranks first; U still beats RAW.
    monkeypatch.setattr(research_batch, "_specs", lambda _config: [
        ("SIGN_ORIENTATION", {"direction": "flip"}),
        ("U_SHAPE_REPAIR", {
            "center": 0.5, "power": 1.0, "asymmetry": False,
        }),
    ])
    split = automatic_time_split(labels, config)
    original_pair_ic = research_batch._pair_ic
    import factor_optimizer.adapters.repair_execution as repair_execution

    original_compile = repair_execution.compile_value_repair
    compiled_plans = []

    def capture_compile(*args, **kwargs):
        plan = original_compile(*args, **kwargs)
        compiled_plans.append(plan)
        return plan

    monkeypatch.setattr(repair_execution, "compile_value_repair", capture_compile)
    validation_candidates = []
    train_candidates = []

    def observe_pair_ic(raw, candidate, batch_arg, labels_arg, indices, config_arg,
                        **kwargs):
        if tuple(indices) == split.validation_indices:
            if np.allclose(candidate, -raw, equal_nan=True):
                validation_candidates.append("SIGN")
            else:
                validation_candidates.append("U")
        elif tuple(indices) == split.train_indices:
            if np.allclose(candidate, -raw, equal_nan=True):
                train_candidates.append("SIGN")
            elif not np.allclose(candidate, raw, equal_nan=True):
                train_candidates.append("U")
        return original_pair_ic(
            raw, candidate, batch_arg, labels_arg, indices, config_arg, **kwargs
        )

    monkeypatch.setattr(research_batch, "_pair_ic", observe_pair_ic)
    result = optimize_factor_batch(
        batch, labels, config=config, allow_research=True
    ).factors["x"]

    train_records = {row["family"]: row for row in result.candidates}
    assert (
        train_records["SIGN_ORIENTATION"]["train_gain"]
        > train_records["U_SHAPE_REPAIR"]["train_gain"]
        > config.minimum_improvement
    )
    assert result.validation_candidate_identity == train_records[
        "SIGN_ORIENTATION"
    ]["plan_identity"]
    assert result.validation_lower_bound <= 0.0
    assert result.selected_family == "NO_OP_RAW"
    assert validation_candidates == ["SIGN"]
    assert set(train_candidates) == {"SIGN", "U"}

    # Independent Spearman oracle: on VALID, U ranks above RAW, while SIGN
    # ranks below RAW. A retry would accept U; the frozen-winner rule retains RAW.
    def spearman(left, right):
        left_rank = pd.Series(left).rank(method="average").to_numpy()
        right_rank = pd.Series(right).rank(method="average").to_numpy()
        return float(np.corrcoef(left_rank, right_rank)[0, 1])

    train_sign_ic = spearman(-per_date_x, target[40])
    train_u_ic = spearman(u[40], target[40])
    train_raw_ic = spearman(per_date_x, target[40])
    valid_sign_ic = spearman(-per_date_x, target[160])
    valid_u_ic = spearman(u[160], target[160])
    valid_raw_ic = spearman(per_date_x, target[160])
    assert train_sign_ic > train_u_ic > train_raw_ic
    assert valid_u_ic > valid_raw_ic > valid_sign_ic

    # Evaluate the exact TRAIN runner-up plan independently after optimization.
    u_plan = next(
        plan for plan in compiled_plans
        if plan.family == "U_SHAPE_REPAIR"
        and dict(plan.parameters) == {
            "center": 0.5, "power": 1.0, "inverted": False,
            "asymmetric": False,
        }
    )
    prefix = x[:split.test_start]
    validation_frame = pd.DataFrame({
        "date": np.repeat(np.arange(split.test_start), asset_count),
        "asset_id": np.tile(asset_axis.values, split.test_start),
        "value": prefix.ravel(),
    })
    u_values = np.asarray(u_plan.execute(
        validation_frame, allow_research=True
    )).reshape(prefix.shape)
    runner_up_delta, runner_up_good, runner_up_coverage = original_pair_ic(
        prefix, u_values, batch, labels, split.validation_indices, config
    )
    runner_up_bound = research_batch._lower_bound(
        runner_up_delta, config, labels.horizon
    )
    assert runner_up_coverage >= config.minimum_coverage
    assert runner_up_good.sum() >= config.minimum_validation_days
    assert runner_up_bound is not None and runner_up_bound > config.minimum_improvement
