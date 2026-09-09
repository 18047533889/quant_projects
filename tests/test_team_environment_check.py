import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "team_check", Path(__file__).resolve().parents[1] / "scripts/check_team_environment.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


@pytest.fixture
def receipt(monkeypatch):
    monkeypatch.setattr(check, "git", lambda *args: "b" * 40)
    return {"success": True, "dryRun": False, "rootSHA": "a" * 40,
            "mirrors": [{"name": n, "sourceTree": "b" * 40,
                         "verifiedRemoteTree": "b" * 40, "mirrorCommit": "c" * 40,
                         "status": "pushed"} for n in sorted(check.MIRRORS)]}


def test_valid_receipt_allows_metadata_only_head_commit(receipt):
    assert check.receipt_errors(Path("/repo"), receipt) == []


@pytest.mark.parametrize("field,value", [("dryRun", True), ("success", False), ("rootSHA", None)])
def test_rejects_unpublished_receipt(receipt, field, value):
    receipt[field] = value
    assert check.receipt_errors(Path("/repo"), receipt)


def test_rejects_duplicate_mirror(receipt):
    receipt["mirrors"][-1] = receipt["mirrors"][0]
    assert check.receipt_errors(Path("/repo"), receipt)


@pytest.mark.parametrize("field,value", [("sourceTree", "d" * 40),
                                         ("verifiedRemoteTree", "d" * 40),
                                         ("mirrorCommit", None), ("status", "planned")])
def test_rejects_incomplete_mirror(receipt, field, value):
    receipt["mirrors"][0][field] = value
    assert check.receipt_errors(Path("/repo"), receipt)


def test_rejects_local_source_drift(receipt, monkeypatch):
    monkeypatch.setattr(check, "git", lambda *args: "d" * 40)
    assert len(check.receipt_errors(Path("/repo"), receipt)) == 13


def test_editable_source_identity(tmp_path):
    raw = json.dumps({"url": tmp_path.as_uri(), "dir_info": {"editable": True}})
    assert check.editable_matches(raw, tmp_path)
    assert not check.editable_matches(raw, tmp_path / "different")
    assert not check.editable_matches("{}", tmp_path)
    assert not check.editable_matches("bad json", tmp_path)


def test_full_external_pins(tmp_path):
    p = tmp_path / "pins.txt"
    p.write_text("# version pins\nnumpy==2.2.6\nPyWavelets==1.9.0\n")
    assert check.read_pins(p) == {"numpy": "2.2.6", "pywavelets": "1.9.0"}
    p.write_text("numpy>=2\n")
    with pytest.raises(ValueError):
        check.read_pins(p)


def test_rejects_ambiguous_duplicate_pin(tmp_path):
    p = tmp_path / "pins.txt"
    p.write_text("typing_extensions==4.16.0\ntyping-extensions==4.16.0\n")
    with pytest.raises(ValueError):
        check.read_pins(p)
