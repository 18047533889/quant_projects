import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def test_watchdog_marks_only_its_child_scope(tmp_path):
    watchdog = Path(__file__).with_name("test_watchdog.py")
    log = tmp_path / "child.log"
    scope_names = (
        "FACTOR_CATALOG_EXTERNAL_WATCHDOG",
        "FACTOR_CATALOG_WATCHDOG_PARENT_PID",
        "FACTOR_CATALOG_WATCHDOG_TOKEN",
    )
    parent_scope = {name: os.environ.get(name) for name in scope_names}
    code = (
        "import json,os; print(json.dumps({"
        "'enabled':os.getenv('FACTOR_CATALOG_EXTERNAL_WATCHDOG'),"
        "'pid':os.getenv('FACTOR_CATALOG_WATCHDOG_PARENT_PID'),"
        "'token':os.getenv('FACTOR_CATALOG_WATCHDOG_TOKEN'),"
        "'ppid':os.getppid()}))"
    )
    run = subprocess.run(
        [sys.executable, str(watchdog), "--log", str(log), "--timeout", "10",
         "--", sys.executable, "-c", code],
        check=True, capture_output=True, text=True,
    )
    child = json.loads(log.read_text())
    assert child["enabled"] == "1"
    assert int(child["pid"]) == child["ppid"]
    assert len(child["token"]) >= 32
    int(child["token"], 16)
    assert child["token"] != parent_scope["FACTOR_CATALOG_WATCHDOG_TOKEN"]
    assert {name: os.environ.get(name) for name in scope_names} == parent_scope


def test_watchdog_records_explicit_source_hash_and_sigsegv_stack(tmp_path):
    watchdog = Path(__file__).with_name("test_watchdog.py")
    source = tmp_path / "source.py"
    source.write_text(
        "import os,signal\n"
        "def crash_here():\n"
        "    os.kill(os.getpid(), signal.SIGSEGV)\n"
        "crash_here()\n"
    )
    log = tmp_path / "segv.log"
    run = subprocess.run(
        ["bash", "-c", "ulimit -c 0; exec \"$@\"", "watchdog-test",
         sys.executable, str(watchdog), "--log", str(log), "--timeout", "10",
         "--hash-path", str(source), "--", sys.executable, str(source)],
        capture_output=True, text=True, timeout=15,
    )
    report = json.loads(run.stdout)
    assert run.returncode == 1
    assert report["returncode"] == -signal.SIGSEGV
    assert report["signal_number"] == signal.SIGSEGV
    assert report["signal_name"] == "SIGSEGV"
    assert report["test_guard_reason"] is None
    assert report["source_hashes"][str(source)]["size_bytes"] == source.stat().st_size
    assert len(report["source_hashes"][str(source)]["sha256"]) == 64
    crash_log = log.read_text()
    assert "Fatal Python error: Segmentation fault" in crash_log
    assert "crash_here" in crash_log


def test_watchdog_rejects_directory_hash_path(tmp_path):
    watchdog = Path(__file__).with_name("test_watchdog.py")
    run = subprocess.run(
        [sys.executable, str(watchdog), "--log", str(tmp_path / "unused.log"),
         "--hash-path", str(tmp_path), "--", sys.executable, "-c", "pass"],
        capture_output=True, text=True,
    )
    assert run.returncode == 2
    assert "must be a regular file" in run.stderr


def test_watchdog_rejects_oversized_hash_path(tmp_path):
    watchdog = Path(__file__).with_name("test_watchdog.py")
    source = tmp_path / "oversized.py"
    source.write_bytes(b"x" * (1024 * 1024 + 1))
    run = subprocess.run(
        [sys.executable, str(watchdog), "--log", str(tmp_path / "unused.log"),
         "--hash-path", str(source), "--", sys.executable, "-c", "pass"],
        capture_output=True, text=True,
    )
    assert run.returncode == 2
    assert "exceeds 1048576 bytes" in run.stderr
