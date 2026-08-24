import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / ".local" / "lib" / "python3.12" / "site-packages"))
sys.path.insert(0, str(ROOT / "jobs"))

from status_partition_runner import PartitionRunner, cos_cli, is_transient_cos_error


def test_cos_cli_retries_524_then_succeeds_without_real_sleep():
    calls = []
    sleeps = []

    def run(command, **kwargs):
        calls.append(command)
        if len(calls) < 3:
            raise subprocess.CalledProcessError(1, command, stderr="HTTP 524 network timeout")
        return subprocess.CompletedProcess(command, 0, stdout="ok\n")

    assert cos_cli("ls", "cos://bucket/prefix", run=run, sleep=sleeps.append) == "ok\n"
    assert len(calls) == 3
    assert sleeps == [120.0, 120.0]


def test_cos_cli_does_not_retry_non_transient_failure():
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command, stderr="permission denied")

    with pytest.raises(subprocess.CalledProcessError):
        cos_cli("cp", "a", "b", run=run, sleep=lambda _: pytest.fail("unexpected sleep"))
    assert len(calls) == 1


def test_transient_code_detection():
    assert is_transient_cos_error(RuntimeError("request failed with status 400"))
    assert is_transient_cos_error(RuntimeError("HTTP 524"))
    assert not is_transient_cos_error(RuntimeError("HTTP 404"))


def test_partition_runner_checkpoints_failures_and_resumes(tmp_path: Path):
    keys = [
        "clean_data/StockDailyBar/2024-01-01.parquet",
        "clean_data/StockDailyBar/2024-01-02.parquet",
    ]
    downloaded = []
    processed = []
    attempts = {keys[0]: 0, keys[1]: 0}

    def download(key, path):
        downloaded.append(key)
        path.write_text(key)

    def processor(context):
        attempts[context.key] += 1
        processed.append(context.key)
        if context.key == keys[0] and attempts[context.key] == 1:
            raise ValueError("bad partition")
        return {"rows": 1}

    first = PartitionRunner(tmp_path, run_id="run-1").run(
        keys, processor, download=download
    )
    assert first["completed"] == 1
    assert first["failed"] == 1
    state = json.loads(
        (tmp_path / "manifests/partition_status.json").read_text()
    )
    assert state["partitions"][keys[0]]["status"] == "failed"
    assert "ValueError: bad partition" in state["partitions"][keys[0]]["failure_reason"]
    assert state["partitions"][keys[0]]["memory_after"]["rss_bytes"] > 0

    second = PartitionRunner(tmp_path, run_id="run-1", resume=True).run(
        keys, processor, download=download
    )
    assert second["completed"] == 2
    assert second["failed"] == 0
    assert processed == [keys[0], keys[1], keys[0]]
    assert downloaded == keys  # failed input was retained locally


def test_partition_runner_writes_state_after_each_success(tmp_path: Path):
    key = "clean_data/StockDailyBar/2024-01-03.parquet"

    def processor(context):
        assert context.input_path.exists()
        return {"output": "local"}

    runner = PartitionRunner(tmp_path)
    result = runner.run(
        [key], processor, download=lambda _, path: path.write_bytes(b"input")
    )
    assert result["status"] == "completed"
    state = json.loads(
        (tmp_path / "manifests/partition_status.json").read_text()
    )
    assert state["partitions"][key]["status"] == "completed"
    assert state["partitions"][key]["result"] == {"output": "local"}
