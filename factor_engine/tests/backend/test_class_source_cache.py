import hashlib
import importlib.util
import inspect
import sys
import pytest
from factor_engine.cleaned_operators.registry import _freeze_value


class SourceExample:
    marker = 1


def test_repeated_class_fingerprint_reads_source_once(monkeypatch):
    original = inspect.getsource
    reads = []
    def measured(value):
        reads.append(value)
        return original(value)
    monkeypatch.setattr(inspect, "getsource", measured)
    first = _freeze_value(SourceExample)
    assert _freeze_value(SourceExample) == first
    assert "class(" in first and "source=" in first
    assert len(reads) == 1


def test_source_edit_changes_class_fingerprint(tmp_path, monkeypatch):
    path = tmp_path / "semantic_class_fixture.py"
    path.write_text("class Example:\n    marker = 1\n")
    spec = importlib.util.spec_from_file_location("semantic_class_fixture", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    before = _freeze_value(module.Example)
    path.write_text("class Example:\n    marker = 12345\n")
    after = _freeze_value(module.Example)
    assert before != after
    assert hashlib.sha256(path.read_text().encode()).hexdigest() in after


def test_failed_source_read_is_not_cached(monkeypatch):
    original = inspect.getsource
    attempts = []
    def transient(value):
        attempts.append(value)
        if len(attempts) == 1:
            raise OSError("transient")
        return original(value)
    monkeypatch.setattr(inspect, "getsource", transient)
    assert _freeze_value(SourceExample) != _freeze_value(SourceExample)


def test_class_qualname_change_is_not_hidden():
    class First:
        pass
    before = _freeze_value(First)
    First.__qualname__ = "ClassNoLongerInSource"
    after = _freeze_value(First)
    assert after != before
    assert hashlib.sha256(b"").hexdigest() in after
