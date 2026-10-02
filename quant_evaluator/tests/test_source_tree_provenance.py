import ast
from pathlib import Path
import pytest

from quant_evaluator.scripts.source_tree_provenance import capture_source_tree, finalize_source_tree


def _tree(tmp_path):
    (tmp_path / "pkg").mkdir(exist_ok=True)
    (tmp_path / "pkg" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "entry.py").write_text("pass\n", encoding="utf-8")
    return capture_source_tree(tmp_path, directories=("pkg",), files=("entry.py",))


def test_source_tree_hash_is_deterministic_and_scoped(tmp_path):
    first = _tree(tmp_path)
    assert first["aggregate_sha256"] == _tree(tmp_path)["aggregate_sha256"]
    assert first["file_count"] == 2
    assert "numpy" in first["versions"] and "pandas" in first["versions"]
    assert "transitive runtime closure" in first["limitations"][-1]


def test_finalize_source_tree_marks_changed_result(tmp_path):
    before = _tree(tmp_path)
    (tmp_path / "pkg" / "a.py").write_text("x = 2\n", encoding="utf-8")
    report = {"status": "complete"}
    assert not finalize_source_tree(
        report, before, tmp_path, directories=before["scope"]["directories"],
        files=before["scope"]["files"])
    assert report["status"] == "source_tree_changed"
    assert report["source_provenance_verification"]["pass"] is False


def test_rejects_scope_escape_and_symlink(tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    try:
        (tmp_path / "linked.py").symlink_to(outside)
    except OSError:
        linked = False
    else:
        linked = True
    with pytest.raises(ValueError, match="relative paths"):
        capture_source_tree(tmp_path, directories=(), files=("../outside.py",))
    if linked:
        with pytest.raises(ValueError, match="symlink"):
            capture_source_tree(tmp_path, directories=(), files=("linked.py",))


def test_per_file_read_is_bounded(tmp_path, monkeypatch):
    import quant_evaluator.scripts.source_tree_provenance as provenance
    (tmp_path / "a.py").write_bytes(b"12345")
    monkeypatch.setattr(provenance, "MAX_FILE_BYTES", 4)
    with pytest.raises(ValueError, match="per-file"):
        provenance.capture_source_tree(tmp_path, directories=(), files=("a.py",))


def test_entry_guard_only_calls_main():
    script = Path(__file__).parents[1] / "scripts" / "benchmark_real_cos_source_batch.py"
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    guards = [node for node in tree.body
              if isinstance(node, ast.If)
              and isinstance(node.test, ast.Compare)
              and isinstance(node.test.left, ast.Name)
              and node.test.left.id == "__name__"]
    assert len(guards) == 1
    calls = []
    module = ast.Module(body=[guards[0]], type_ignores=[])
    exec(compile(module, str(script), "exec"),
         {"__name__": "__main__", "main": lambda: calls.append("main")})
    assert calls == ["main"]


def test_loaded_cupy_is_reported_after_run_without_changing_code_gate(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    import quant_evaluator.scripts.source_tree_provenance as provenance
    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    before = _tree(tmp_path)
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(__version__="test-cupy"))
    report = {"status": "complete"}
    unchanged = provenance.finalize_source_tree(
        report, before, tmp_path, directories=before["scope"]["directories"],
        files=before["scope"]["files"])
    assert unchanged is True
    assert before["versions"]["cupy"] is None
    assert report["source_provenance_verification"]["after_versions"]["cupy"] == "test-cupy"


def test_python_package_version_change_prevents_complete_status(tmp_path, monkeypatch):
    import quant_evaluator.scripts.source_tree_provenance as provenance
    before = _tree(tmp_path)
    monkeypatch.setattr(provenance, "_package_version", lambda name: "changed")
    report = {"status": "complete"}
    unchanged = provenance.finalize_source_tree(
        report, before, tmp_path, directories=before["scope"]["directories"],
        files=before["scope"]["files"])
    assert unchanged is False
    assert report["status"] == "source_tree_changed"

def test_one_shot_scope_iterables_are_recorded(tmp_path):
    _tree(tmp_path)
    receipt = capture_source_tree(
        tmp_path, directories=(item for item in ("pkg",)),
        files=(item for item in ("entry.py",)))
    assert receipt["scope"] == {"directories": ["pkg"], "files": ["entry.py"]}
    assert receipt["file_count"] == 2


@pytest.mark.parametrize("after", ("changed", "removed"))
def test_preloaded_cupy_version_drift_prevents_complete_status(tmp_path, monkeypatch, after):
    import sys
    from types import SimpleNamespace
    import quant_evaluator.scripts.source_tree_provenance as provenance
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(__version__="preloaded"))
    before = _tree(tmp_path)
    if after == "changed":
        monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(__version__="different"))
    else:
        monkeypatch.delitem(sys.modules, "cupy")
    report = {"status": "complete"}
    assert not provenance.finalize_source_tree(
        report, before, tmp_path, directories=before["scope"]["directories"],
        files=before["scope"]["files"])
    assert report["status"] == "source_tree_changed"
