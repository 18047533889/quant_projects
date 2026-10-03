"""Safety tests for exclusive diagnostic profile progress writing."""
from __future__ import annotations
import json
import pytest
from quant_evaluator.scripts.profile_progress_writer import ExclusiveProgressWriter


def test_late_created_path_is_rejected_without_overwriting(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text("owned by another worker", encoding="utf-8")
    with pytest.raises(FileExistsError):
        with ExclusiveProgressWriter(path):
            pytest.fail("exclusive creation should reject the late file")
    assert path.read_text(encoding="utf-8") == "owned by another worker"


def test_replacement_path_is_not_touched_by_later_update(tmp_path):
    path = tmp_path / "progress.json"
    with ExclusiveProgressWriter(path) as writer:
        path.unlink()
        path.write_text("replacement worker evidence", encoding="utf-8")
        writer.write({"kind": "test"}, [{"run_index": 0}])
    assert path.read_text(encoding="utf-8") == "replacement worker evidence"


def test_oversized_update_is_rejected_before_mutating_owned_file(tmp_path):
    path = tmp_path / "progress.json"
    with ExclusiveProgressWriter(path) as writer:
        writer.write({"kind": "test"}, [])
        original = path.read_bytes()
        with pytest.raises(ValueError, match="exceeds 1 MiB"):
            writer.write({"kind": "test", "large": "x" * (1024**2)}, [])
        assert path.read_bytes() == original


def test_updates_are_valid_diagnostic_json_and_close_on_exit(tmp_path):
    path = tmp_path / "progress.json"
    with ExclusiveProgressWriter(path) as writer:
        writer.write({"kind": "test"}, [{"run_index": 0}], status="running")
        running = json.loads(path.read_text(encoding="utf-8"))
        assert running["status"] == "running"
        assert running["qualification_available"] is False
        writer.write({"kind": "test"}, [{"run_index": 0}],
                     status="failed", error_type="RuntimeError")
    failed = json.loads(path.read_text(encoding="utf-8"))
    assert failed["status"] == "failed" and failed["error_type"] == "RuntimeError"
    with pytest.raises(RuntimeError, match="not open"):
        writer.write({}, [])


def test_writer_creates_missing_parent_directories(tmp_path):
    path = tmp_path / "nested" / "progress.json"
    with ExclusiveProgressWriter(path) as writer:
        writer.write({"kind": "test"}, [])
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "running"
