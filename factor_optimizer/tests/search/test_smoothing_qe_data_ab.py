"""Research-only data A/B through real FP and QE public seams."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("factor_preprocess.registry.transforms")
from factor_optimizer.adapters.preprocessing import compile_smoothing_repair
from factor_optimizer.search.paired_comparison import ComparisonStatus, ComparisonThresholds, PairedDraws, compare_paired_draws
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle
from quant_evaluator.runtime.evaluator import evaluate

N_ASSETS, N_TIMES, HOLDOUT_START = 40, 220, 100

@dataclass(frozen=True)
class Experiment:
    names: tuple[str, ...]
    values: np.ndarray
    labels: np.ndarray

def _panel(seed, *, instantaneous=False):
    rng = np.random.default_rng(seed)
    latent = np.empty((N_TIMES, N_ASSETS)); latent[0] = rng.normal(size=N_ASSETS)
    for t in range(1, N_TIMES):
        latent[t] = .975 * latent[t - 1] + rng.normal(scale=.20, size=N_ASSETS)
    observed = latent + rng.normal(scale=1.15, size=latent.shape)
    dates = pd.date_range("2024-01-01", periods=N_TIMES, freq="D")
    frame = pd.DataFrame({"date": np.repeat(dates, N_ASSETS), "asset_id": np.tile([f"A{i:02d}" for i in range(N_ASSETS)], N_TIMES), "value": observed.ravel()})
    return frame, observed if instantaneous else latent

def _plans():
    # Confirmatory parameters are declared a priori.  The context reference
    # records that decision; this synthetic test performs no parameter fit.
    common = dict(natural_time_scale=20., training_context_ref="synthetic-prespecified:v1")
    plans = {m: compile_smoothing_repair("CAUSAL_SMOOTHING", {"method": m, "natural_time_scale_relative": .35}, **common) for m in ("SMA", "EWMA", "IIR", "KAMA", "Kalman")}
    plans["DECAY_REL"] = compile_smoothing_repair("DECAY_REFINEMENT", {"decay": .30, "half_life_relative": True}, **common)
    plans["DECAY_ABS"] = compile_smoothing_repair("DECAY_REFINEMENT", {"decay": .70, "half_life_relative": False}, **common)
    return plans

def _experiment(seed, *, instantaneous=False):
    panel, labels = _panel(seed, instantaneous=instantaneous)
    plans = _plans(); names = ("RAW", *plans)
    columns = [panel.value.to_numpy().reshape(N_TIMES, N_ASSETS)]
    columns += [np.asarray(p.execute(panel, allow_research=True), dtype=float).reshape(N_TIMES, N_ASSETS) for p in plans.values()]
    return Experiment(names, np.stack(columns, axis=-1), labels)

def _holdout_rank_ic(experiment):
    values, labels = experiment.values[HOLDOUT_START:], experiment.labels[HOLDOUT_START:]
    times = tuple(range(HOLDOUT_START, N_TIMES))
    ta = AxisRef("time", "int64", len(times), np.asarray(times)); aa = AxisRef("asset", "str", N_ASSETS, np.asarray([f"A{i:02d}" for i in range(N_ASSETS)]))
    factors = FactorBatch(experiment.names, ta, aa, values, validity=np.isfinite(values))
    target = LabelBundle("synthetic-forward", labels, 1, decision_time=times, label_start_time=tuple(t + 1 for t in times), label_end_time=tuple(t + 2 for t in times), asset_axis=aa)
    artifact = evaluate(factors, target, metrics=["rank_ic_series"]).artifacts["rank_ic_series"]
    result = np.asarray(artifact.values, dtype=float)
    # SeriesMetricArtifact represents invalid observations as NaN; unlike
    # vector artifacts it intentionally has no separate validity mask.
    valid = np.isfinite(result)
    return result, valid

def _common_means(ic, valid):
    common = valid.all(axis=1); assert common.sum() >= 100
    return np.mean(ic[common], axis=0)

@pytest.mark.parametrize("seed", [17, 71, 509])
def test_slow_signal_holdout_smoothing_improves_real_qe_rank_ic(seed):
    # Defects caught: current-bar use, wrong long-panel grouping, unequal masks.
    experiment = _experiment(seed)
    ic, valid = _holdout_rank_ic(experiment); means = _common_means(ic, valid); raw = means[0]
    assert .50 < raw < .65
    for name in ("SMA", "EWMA", "IIR", "KAMA", "Kalman", "DECAY_REL", "DECAY_ABS"):
        # Every prespecified treatment has a declared direction; none may
        # silently regress while an aggregate count still passes.
        assert means[experiment.names.index(name)] > raw + .12, name

@pytest.mark.parametrize("seed", [17, 71, 509])
def test_instantaneous_signal_is_a_negative_control_where_raw_wins(seed):
    experiment = _experiment(seed, instantaneous=True)
    ic, valid = _holdout_rank_ic(experiment); means = _common_means(ic, valid)
    assert means[0] > .999
    for name in ("SMA", "EWMA", "IIR", "KAMA", "Kalman", "DECAY_REL", "DECAY_ABS"):
        assert means[experiment.names.index(name)] < .60, name

def test_constant_factor_has_no_valid_ic_and_is_not_zero_filled():
    experiment = _experiment(17); constant = np.ones((*experiment.values.shape[:2], 1))
    ic, valid = _holdout_rank_ic(Experiment(("CONSTANT",), constant, experiment.labels))
    assert not valid.any(); assert np.isnan(ic).all()

@pytest.mark.parametrize("name", ["SMA", "EWMA", "IIR", "KAMA", "Kalman", "DECAY_REL", "DECAY_ABS"])
def test_repair_is_prefix_invariant_when_only_future_values_change(name):
    panel, _ = _panel(71); cutoff = 135 * N_ASSETS; changed = panel.copy(deep=True)
    changed.loc[cutoff:, "value"] += 10_000.
    plan = _plans()[name]
    np.testing.assert_allclose(np.asarray(plan.execute(panel, allow_research=True))[:cutoff], np.asarray(plan.execute(changed, allow_research=True))[:cutoff], equal_nan=True)

def test_common_block_bootstrap_flows_to_fo_paired_comparison():
    experiment = _experiment(509)
    ic, valid = _holdout_rank_ic(experiment)
    # SMA is the a-priori confirmatory treatment.  Never select a winner on
    # this holdout and then reuse the same holdout for its confidence interval.
    candidate = experiment.names.index("SMA")
    common = valid[:, 0] & valid[:, candidate]
    paired = ic[common, candidate] - ic[common, 0]; rng = np.random.default_rng(20260921); block, draws = 10, []
    for _ in range(240):
        starts = rng.integers(0, len(paired) - block + 1, size=int(np.ceil(len(paired) / block)))
        draws.append(float(np.mean(np.concatenate([paired[s:s + block] for s in starts])[:len(paired)])))
    evidence = PairedDraws(experiment.names[candidate], "RAW", tuple(f"block-{i}" for i in range(len(draws))), {"mean_ic_delta": tuple(draws)}, {"mean_ic_delta": (0.,) * len(draws)}, "synthetic-holdout:100-219:seed-509")
    result = compare_paired_draws(evidence, ComparisonThresholds(.05, .01, .01, confidence_level=.95), utility=lambda m: m["mean_ic_delta"], expected_context_identity="synthetic-holdout:100-219:seed-509")
    assert result.status is ComparisonStatus.SUPERIOR
    assert result.mean_difference > .20
    assert result.losses == 0
