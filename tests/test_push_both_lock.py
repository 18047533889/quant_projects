import fcntl
import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "publisher_lock_test", Path(__file__).resolve().parents[1] / "scripts/push_both.py")
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


def test_concurrent_invocation_does_not_fetch_or_replace_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(publisher, "ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(publisher, "_main", lambda argv: calls.append(argv) or 0)
    logs = tmp_path / "logs"
    logs.mkdir()
    receipt = logs / "push_both_result.json"
    receipt.write_text("prior successful receipt")
    with (logs / "push_both.lock").open("a+") as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert publisher.main(["--dry-run"]) == 1
        assert calls == []
        assert receipt.read_text() == "prior successful receipt"
    assert publisher.main(["--dry-run"]) == 0
    assert calls == [["--dry-run"]]
