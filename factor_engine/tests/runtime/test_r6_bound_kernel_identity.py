"""A generic adapter's identity must include its selected numerical kernel."""
from types import SimpleNamespace
from pathlib import Path
import inspect
from factor_engine.runtime.operator_snapshot import _implementation_identity, _live_runtime_digest

def first(x):
    return x + 1

def second(x):
    return x + 2

class Adapter:
    def __init__(self, fn):
        self._fn = fn
        self.metadata = SimpleNamespace(param_names=[], param_specs={})
    def _calculate_series(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

def test_same_adapter_class_different_kernel_has_different_identity():
    left = _implementation_identity(Adapter(first))
    right = _implementation_identity(Adapter(second))
    assert left["qualified_name"] == right["qualified_name"]
    assert left["file_digest"] == right["file_digest"]
    assert left["implementation_digest"] != right["implementation_digest"]
    assert len(left["bound_kernels"]) == 1

def test_kernel_source_edit_invalidates_identity(tmp_path, monkeypatch):
    source = tmp_path / "bounded_test_kernel.py"
    source.write_text("def first(x): return x + 1\n")
    original = inspect.getsourcefile
    monkeypatch.setattr(inspect, "getsourcefile", lambda obj: str(source) if obj is first else original(obj))
    op = Adapter(first)
    before = _implementation_identity(op)
    source.write_text("def first(x): return x + 2\n")
    after = _implementation_identity(op)
    assert before["file_digest"] == after["file_digest"]
    assert before["implementation_digest"] != after["implementation_digest"]

def test_live_snapshot_cache_key_changes_when_kernel_binding_changes(monkeypatch):
    from factor_engine.backend import cleaned_bridge, operator_semantic_version
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    op = Adapter(first)
    monkeypatch.setattr(cleaned_bridge, "ensure_cleaned_loaded", lambda: None)
    monkeypatch.setattr(operator_semantic_version, "semantic_version", lambda _: 1)
    monkeypatch.setattr(OperatorRegistry, "_read_state", classmethod(lambda cls: ({"probe": {"pandas_numpy": op}}, {}, {})))
    before = _live_runtime_digest()
    op._fn = second
    assert before != _live_runtime_digest()
