"""Bound test wall time and sample only the subprocess family started here.

This is a sampled test guard, not a kernel-enforced memory cap or production
resource acceptance. It never signals unrelated server workloads.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--log", required=True)
parser.add_argument("--max-rss-mib", type=int, default=2048)
parser.add_argument("--timeout", type=int, default=900)
parser.add_argument("--hash-path", action="append", default=[],
                    help="explicit regular file to SHA-256 (repeatable; each file <= 1 MiB)")
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
if not command or args.max_rss_mib <= 0 or args.timeout <= 0:
    parser.error("explicit command and positive bounds required")

MAX_HASH_FILE_BYTES = 1024 * 1024

def source_hashes(paths):
    """Hash only explicit, bounded regular files; never walk directories."""
    hashes = {}
    for raw_path in paths:
        path = Path(raw_path)
        try:
            stat = path.stat()
        except OSError as exc:
            parser.error(f"cannot stat --hash-path {raw_path!r}: {exc}")
        if not path.is_file():
            parser.error(f"--hash-path must be a regular file: {raw_path!r}")
        with path.open("rb") as stream:
            content = stream.read(MAX_HASH_FILE_BYTES + 1)
        if stat.st_size > MAX_HASH_FILE_BYTES or len(content) > MAX_HASH_FILE_BYTES:
            parser.error(f"--hash-path exceeds {MAX_HASH_FILE_BYTES} bytes: {raw_path!r}")
        hashes[str(path)] = {"sha256": hashlib.sha256(content).hexdigest(),
                             "size_bytes": len(content)}
    return hashes

hashed_sources = source_hashes(args.hash_path)


def rss_family(pid, seen=None):
    seen = set() if seen is None else seen
    if pid in seen:
        return 0
    seen.add(pid)
    try:
        status = Path(f"/proc/{pid}/status").read_text()
        rss = next((int(line.split()[1]) * 1024 for line in status.splitlines()
                    if line.startswith("VmRSS:")), 0)
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        return rss + sum(rss_family(int(child), seen) for child in children)
    except (OSError, ValueError):
        return 0


start = time.monotonic()
peak = 0
reason = None
with open(args.log, "wb") as stream:
    child_env = os.environ.copy()
    child_env.update(
        FACTOR_CATALOG_EXTERNAL_WATCHDOG="1",
        FACTOR_CATALOG_WATCHDOG_PARENT_PID=str(os.getpid()),
        FACTOR_CATALOG_WATCHDOG_TOKEN=secrets.token_hex(16),
        PYTHONFAULTHANDLER="1",
    )
    proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                            start_new_session=True, env=child_env)
    try:
        while proc.poll() is None:
            peak = max(peak, rss_family(proc.pid))
            if peak > args.max_rss_mib * 1024 * 1024:
                reason = "SAMPLED_RSS_TEST_LIMIT"
            elif time.monotonic() - start > args.timeout:
                reason = "TEST_WALL_TIME_LIMIT"
            if reason:
                os.killpg(proc.pid, signal.SIGTERM)
                break
            time.sleep(0.1)
        try:
            code = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            code = proc.wait()
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
signal_number = -code if code < 0 else None
signal_name = (signal.Signals(signal_number).name
               if signal_number in signal.Signals._value2member_map_ else None)
print(json.dumps({"command": command, "returncode": code,
                  "signal_number": signal_number, "signal_name": signal_name,
                  "source_hashes": hashed_sources,
                  "sampled_peak_family_rss_bytes": peak,
                  "sampling_seconds": 0.1, "seconds": time.monotonic() - start,
                  "test_guard_reason": reason, "hard_memory_cap": False}))
raise SystemExit(1 if reason else (code if code >= 0 else 1))
