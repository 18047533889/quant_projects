"""Audit must not disguise implementation failures as missing economic evidence."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from factor_optimizer.research_fitness import JointMetricsUnavailable
from quant_evaluator.contracts.factor_batch import AxisRef, FactorBatch
from quant_evaluator.contracts.label_bundle import LabelBundle

@pytest.mark.parametrize("expected", [False, True])
def test_method_audit_only_suppresses_typed_evidence_unavailability(monkeypatch, expected):
    spec = importlib.util.spec_from_file_location(
        "method_audit_probe", Path(__file__).parents[1] / "examples/method_audit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    class Identity:
        transform = "identity"
        def execute(self, frame, **kwargs):
            return frame["value"].copy()
    monkeypatch.setattr(audit, "method_cases", lambda: [("NO_OP_RAW", {}, Identity())])
    def broken(*args, **kwargs):
        if expected:
            raise JointMetricsUnavailable("undefined_ratios", "undefined risk", metrics=("sharpe",))
        raise ValueError("broken portfolio implementation")
    monkeypatch.setattr(audit, "summarize", broken)
    rng = np.random.default_rng(20260922)
    t, a = 240, 24
    axis = AxisRef("asset", "str", a, np.asarray([f"a{i}" for i in range(a)]))
    batch = FactorBatch(("probe",), AxisRef("time", "int", t, np.arange(t)), axis,
                        rng.normal(size=(t, a, 1)))
    labels = LabelBundle("returns", rng.normal(0, .01, (t, a)), 1,
        decision_time=tuple(range(t)), label_start_time=tuple(range(1,t+1)),
        label_end_time=tuple(range(2,t+2)), asset_axis=axis)
    row, = audit.audit_methods(batch, labels)
    assert row["status"] == ("executed" if expected else "failed")
    if expected:
        assert row["joint_metrics_status"] == "unavailable"
        assert row["joint_metrics_code"] == "undefined_ratios"
        assert row["joint_metrics_unavailable"] == ("sharpe",)
    else:
        assert "broken portfolio implementation" in row["reason"]


@pytest.mark.parametrize("horizon,end_offset", [(2, 2), (1, 3)])
def test_unsupported_label_accounting_is_typed_not_a_computation_bug(horizon, end_offset):
    from factor_optimizer.research_fitness import paired_series
    t, a = 40, 24
    axis = AxisRef("asset", "str", a, np.asarray([f"a{i}" for i in range(a)]))
    values = np.random.default_rng(92).normal(size=(t, a))
    batch = FactorBatch(("probe",), AxisRef("time", "int", t, np.arange(t)),
                        axis, values[:, :, None])
    labels = LabelBundle("returns", values * .001, horizon,
        decision_time=tuple(range(t)), label_start_time=tuple(range(1,t+1)),
        label_end_time=tuple(range(end_offset,t+end_offset)), asset_axis=axis)
    with pytest.raises(JointMetricsUnavailable) as exc:
        paired_series(values, values, batch, labels, tuple(range(t)))
    assert exc.value.code == "unsupported_label_accounting"
