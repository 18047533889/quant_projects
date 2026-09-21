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
    labels = LabelBundle("probe", rng.normal(size=(t,a)), 1, decision_time=tuple(range(t)),
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
