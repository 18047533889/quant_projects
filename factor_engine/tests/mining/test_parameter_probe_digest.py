import numpy as np
import pytest
from types import SimpleNamespace
from factor_engine.mining.direct_use import _stable_output_hash, _probe_parameter_injectivity

@pytest.mark.parametrize("a,b", [
    (np.nan, -1e30), (-np.inf, -1e30), (np.inf, 1e30), (np.nan, -np.inf),
])
def test_parameter_probe_digest_preserves_nonfinite_identity(a, b):
    assert _stable_output_hash(np.array([[a]])) != _stable_output_hash(np.array([[b]]))

def test_parameter_probe_digest_preserves_shape():
    assert _stable_output_hash(np.arange(4).reshape(2, 2)) != _stable_output_hash(np.arange(4))

def test_nan_payload_does_not_create_spurious_parameter_effect():
    a = np.array([0x7ff8000000000001], dtype=np.uint64).view(np.float64)
    b = np.array([0x7ff8000000000002], dtype=np.uint64).view(np.float64)
    assert _stable_output_hash(a) == _stable_output_hash(b)

def test_probe_detects_finite_to_missing_change_without_disabling_budget(monkeypatch):
    class MissingnessThreshold:
        metadata = SimpleNamespace(param_specs={"window": SimpleNamespace(default=1)})
        def calculate(self, panel, window=1):
            return np.full(panel.shape, -1e30 if window == 1 else np.nan)
    op = MissingnessThreshold()
    monkeypatch.setenv("R64_PROBE_SAMPLE_GATE", "0")
    assert _probe_parameter_injectivity("probe", op, ("window",), ("x",))
    monkeypatch.setenv("R64_PROBE_SAMPLE_GATE", "24000")
    assert not _probe_parameter_injectivity("probe", op, ("window",), ("x",))

@pytest.mark.parametrize("invalid_base", [True, False])
def test_unhashable_probe_output_never_proves_parameter_effect(monkeypatch, invalid_base):
    class UnsupportedOutput:
        metadata = SimpleNamespace(param_specs={"window": SimpleNamespace(default=1)})
        def calculate(self, panel, window=1):
            invalid = (window == 1) if invalid_base else (window != 1)
            return {"not": "numeric"} if invalid else np.zeros(panel.shape)
    monkeypatch.setenv("R64_PROBE_SAMPLE_GATE", "0")
    assert not _probe_parameter_injectivity("probe", UnsupportedOutput(), ("window",), ("x",))
