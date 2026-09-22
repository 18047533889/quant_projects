import importlib.util
from pathlib import Path
import numpy as np
from factor_optimizer.adapters.preprocessing import compile_admissible_smoothing_grid
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle


def test_method_audit_covers_every_admitted_smoothing_parameterization():
    spec = importlib.util.spec_from_file_location(
        "method_audit", Path(__file__).parents[1] / "examples/method_audit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    expected = compile_admissible_smoothing_grid(
        natural_time_scale=10., training_context_ref="prespecified-method-audit")
    actual = [plan for family, params, plan in audit.method_cases()
              if family == "CAUSAL_SMOOTHING"]
    assert {p.identity for p in actual} == {p.identity for p in expected}


def test_method_audit_executes_every_numeric_family_and_marks_missing_inputs():
    spec = importlib.util.spec_from_file_location(
        "method_audit", Path(__file__).parents[1] / "examples/method_audit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    rng = np.random.default_rng(105)
    t, a = 240, 24
    values = rng.normal(size=(t,a,1))
    values[::11, :3, 0] = np.nan
    ta = AxisRef("time", "int", t, np.arange(t))
    aa = AxisRef("asset", "str", a, np.array([f"a{i}" for i in range(a)]))
    batch = FactorBatch(("probe",), ta, aa, values)
    # Daily returns, not unit-variance scores that can imply loss below -100%.
    labels = LabelBundle("probe", rng.normal(0, .01, size=(t,a)), 1, decision_time=tuple(range(t)),
                         label_start_time=tuple(range(1,t+1)), label_end_time=tuple(range(2,t+2)), asset_axis=aa)
    rows = audit.audit_methods(batch, labels)
    assert not [r for r in rows if r["status"] == "failed"]
    executed = {r["family"] for r in rows if r["status"] == "executed"}
    assert executed == {"NO_OP_RAW", "SIGN_ORIENTATION", "DECAY_REFINEMENT", "CAUSAL_SMOOTHING",
                        "ROBUST_OUTLIER", "MISSINGNESS_FRESHNESS", "U_SHAPE_REPAIR",
                        "INVERTED_U_REPAIR", "TAIL_SATURATION", "TAIL_HINGE",
                        "REPRESENTATION_RANK", "REPRESENTATION_ZSCORE", "ROBUST_SCALE"}
    smooth = {r["transform"] for r in rows if r["family"] == "CAUSAL_SMOOTHING"}
    assert smooth == {"trailing_sma", "ewma", "one_sided_iir_lowpass", "kama", "kalman_local_level"}
    for r in rows:
        if r["status"] == "executed":
            assert r["prefix_invariant"] and r["asset_permutation_invariant"]
        else:
            assert r["reason"]


def test_method_audit_reports_costed_risk_and_keeps_unavailable_metrics_explicit():
    spec = importlib.util.spec_from_file_location(
        "method_audit", Path(__file__).parents[1] / "examples/method_audit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    rng = np.random.default_rng(827)
    t, a = 240, 40
    x = rng.normal(size=(t, a))
    ta = AxisRef("time", "int", t, np.arange(t))
    aa = AxisRef("asset", "str", a, np.array([f"a{i}" for i in range(a)]))
    batch = FactorBatch(("signal",), ta, aa, x[:, :, None])
    labels = LabelBundle("returns", .003*x+rng.normal(0, .01, (t,a)), 1,
        decision_time=tuple(range(t)), label_start_time=tuple(range(1,t+1)),
        label_end_time=tuple(range(2,t+2)), asset_axis=aa)
    rows = audit.audit_methods(batch, labels)
    raw = next(r for r in rows if r["family"] == "NO_OP_RAW")
    assert raw["joint_metrics_status"] == "available"
    assert raw["train_candidate_metrics"]["sharpe"] > 0
    assert raw["train_candidate_metrics"]["turnover"] > 0
    assert raw["joint_utility_delta"] == 0
    flip = next(r for r in rows if r["family"] == "SIGN_ORIENTATION"
                and r["parameters"]["direction"] == "flip")
    assert flip["train_candidate_metrics"]["sharpe"] < 0
    assert not flip["passes_raw_relative_floors"]
    from dataclasses import replace
    unavailable = audit.audit_methods(batch, replace(labels, horizon=2))
    raw = next(r for r in unavailable if r["family"] == "NO_OP_RAW")
    assert raw["status"] == "executed"
    assert raw["joint_metrics_status"] == "unavailable"
    assert "single-bar" in raw["joint_metrics_reason"]
    assert "train_candidate_metrics" not in raw
