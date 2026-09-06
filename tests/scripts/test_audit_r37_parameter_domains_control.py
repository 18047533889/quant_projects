from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audit_r37_parameter_domains.py"


def _load():
    spec = importlib.util.spec_from_file_location("audit_r37_parameter_domains_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clean_worktree_probe_is_deterministic(tmp_path):
    mod = _load()
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs["cwd"]))
        if argv[1:3] == ["rev-parse", "--show-toplevel"]:
            return SimpleNamespace(stdout=str(tmp_path) + "\n")
        if argv[1:3] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="abc123\n")
        return SimpleNamespace(stdout="")

    assert mod._require_clean_worktree(cwd=tmp_path, runner=runner) == (tmp_path, "abc123")
    assert calls == [
        (["git", "rev-parse", "--show-toplevel"], tmp_path),
        (["git", "rev-parse", "HEAD"], tmp_path),
        (["git", "status", "--porcelain", "--untracked-files=normal"], tmp_path),
    ]


def test_dirty_or_unverifiable_worktree_is_refused(tmp_path):
    mod = _load()

    def dirty_runner(argv, **kwargs):
        if argv[1] == "rev-parse":
            return SimpleNamespace(stdout=str(tmp_path) + "\n")
        return SimpleNamespace(stdout=" M factor_engine/runtime/engine.py\n")

    with pytest.raises(RuntimeError, match="dirty worktree"):
        mod._require_clean_worktree(cwd=tmp_path, runner=dirty_runner)

    def broken_runner(argv, **kwargs):
        raise subprocess.CalledProcessError(128, argv)

    with pytest.raises(RuntimeError, match="cannot verify"):
        mod._require_clean_worktree(cwd=tmp_path, runner=broken_runner)


def test_post_audit_recheck_detects_head_change(tmp_path):
    mod = _load()

    def runner(argv, **kwargs):
        if argv[1:3] == ["rev-parse", "--show-toplevel"]:
            return SimpleNamespace(stdout=str(tmp_path) + "\n")
        if argv[1:3] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="def456\n")
        return SimpleNamespace(stdout="")

    with pytest.raises(RuntimeError, match="HEAD changed"):
        mod._recheck_clean_worktree((tmp_path, "abc123"), runner=runner)


def test_post_audit_recheck_detects_new_dirty_state(tmp_path):
    mod = _load()

    def runner(argv, **kwargs):
        if argv[1:3] == ["rev-parse", "--show-toplevel"]:
            return SimpleNamespace(stdout=str(tmp_path) + "\n")
        if argv[1:3] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="abc123\n")
        return SimpleNamespace(stdout="?? concurrent-output.tmp\n")

    with pytest.raises(RuntimeError, match="dirty worktree"):
        mod._recheck_clean_worktree((tmp_path, "abc123"), runner=runner)


def test_case_audit_rejects_failed_missing_and_duplicate_cases():
    mod = _load()
    expected = {("a", "window=1"), ("a", "window=2"), ("b", "default")}
    rows = [
        {"canonical": "a", "param": "window=1", "pass": True, "reason": ""},
        {"canonical": "a", "param": "window=1", "pass": True, "reason": ""},
        {"canonical": "a", "param": "window=2", "pass": False, "reason": "oracle mismatch"},
    ]
    failures = mod._audit_case_failures(rows, expected=expected)
    assert failures == [
        "a window=2: oracle mismatch",
        "b default: missing/non-executed case",
        "a window=1: executed 2 times",
    ]


def test_main_returns_nonzero_before_writing_failed_audit(monkeypatch, tmp_path):
    mod = _load()
    monkeypatch.setattr(mod, "_require_clean_worktree", lambda: (tmp_path, "abc123"))
    monkeypatch.setattr(mod, "_recheck_clean_worktree", lambda initial: None)
    monkeypatch.setattr(mod, "ensure_cleaned_loaded", lambda: None)
    monkeypatch.setattr(mod, "_REF_BY_CANON", {"broken": "panel"})
    monkeypatch.setattr(
        mod,
        "_certify_panel",
        lambda canonical, store: [
            {"canonical": canonical, "param": "default", "valid": True,
             "pass": False, "reason": "oracle mismatch"}
        ],
    )
    monkeypatch.setattr(mod, "E", tmp_path / "must-not-exist")

    assert mod.main() == 1
    assert not mod.E.exists()


def test_main_dirty_refusal_precedes_operator_loading(monkeypatch):
    mod = _load()
    loaded = []
    monkeypatch.setattr(
        mod, "_require_clean_worktree",
        lambda: (_ for _ in ()).throw(RuntimeError("dirty worktree")),
    )
    monkeypatch.setattr(mod, "ensure_cleaned_loaded", lambda: loaded.append(True))
    assert mod.main() == 2
    assert loaded == []


@pytest.mark.parametrize(
    "message",
    ["dirty worktree after audit", "HEAD changed while R37 audit was running"],
)
def test_main_post_audit_git_change_exits_before_writes(monkeypatch, tmp_path, message):
    mod = _load()
    initial = (tmp_path, "abc123")
    monkeypatch.setattr(mod, "_require_clean_worktree", lambda: initial)
    monkeypatch.setattr(mod, "ensure_cleaned_loaded", lambda: None)
    monkeypatch.setattr(mod, "_REF_BY_CANON", {"ok": "panel"})
    monkeypatch.setattr(
        mod,
        "_certify_panel",
        lambda canonical, store: [
            {"canonical": canonical, "param": "default", "valid": True,
             "pass": True, "reason": ""}
        ],
    )
    monkeypatch.setattr(
        mod, "_recheck_clean_worktree",
        lambda state: (_ for _ in ()).throw(RuntimeError(message)),
    )
    monkeypatch.setattr(mod, "E", tmp_path / "must-not-exist")

    assert mod.main() == 2
    assert not mod.E.exists()
