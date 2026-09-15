"""Bound test wall time and sample only the subprocess family started here.

This is a sampled test guard, not a kernel-enforced memory cap or production
resource acceptance. It never signals unrelated server workloads.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--log", required=True)
parser.add_argument("--max-rss-mib", type=int, default=2048)
parser.add_argument("--timeout", type=int, default=900)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
if not command or args.max_rss_mib <= 0 or args.timeout <= 0:
    parser.error("explicit command and positive bounds required")


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
    proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                            start_new_session=True)
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
print(json.dumps({"command": command, "returncode": code,
                  "sampled_peak_family_rss_bytes": peak,
                  "sampling_seconds": 0.1, "seconds": time.monotonic() - start,
                  "test_guard_reason": reason, "hard_memory_cap": False}))
raise SystemExit(1 if reason else (code if code >= 0 else 1))
