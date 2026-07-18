#!/usr/bin/env python3
"""Poll LQTP until reachable, then resume production batch with gentle concurrency."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cogalpha_lqtp.lqtp_client import (  # noqa: E402
    DEFAULT_LQTP_LOGIN_RETRIES,
    DEFAULT_LQTP_LOGIN_WAIT_SEC,
    DEFAULT_SERVER,
    login_with_retry,
)


def _tcp_open(host: str, port: int, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _parse_host_port(server: str) -> tuple[str, int]:
    if ":" in server:
        host, port_s = server.rsplit(":", 1)
        return host, int(port_s)
    return server, 50051


def _clear_transient_failures(progress_path: Path) -> int:
    if not progress_path.exists():
        return 0
    data = json.loads(progress_path.read_text(encoding="utf-8"))
    failed = data.get("failed", {})
    keep: dict[str, str] = {}
    removed = 0
    for name, err in failed.items():
        text = str(err)
        if any(
            x in text
            for x in (
                "Connection refused",
                "UNAVAILABLE",
                "failed to connect",
                "Socket closed",
                "Stream removed",
            )
        ):
            removed += 1
            continue
        keep[name] = err
    if removed:
        data["failed"] = keep
        progress_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for LQTP and resume batch gently")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/cogalpha_lqtp_production")
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--username", default=os.getenv("LQTP_USERNAME", ""))
    parser.add_argument("--password", default=os.getenv("LQTP_PASSWORD", ""))
    parser.add_argument("--poll-sec", type=float, default=60.0, help="seconds between probes")
    parser.add_argument("--max-wait-hours", type=float, default=24.0)
    parser.add_argument("--parallel-lqtp", type=int, default=3, help="LQTP RunFactor concurrency when batch resumes")
    parser.add_argument("--parallel-local", type=int, default=2)
    parser.add_argument("--max-total-parallel", type=int, default=3)
    parser.add_argument("--mem-reserve-gb", type=float, default=10.0)
    args = parser.parse_args()

    host, port = _parse_host_port(args.server)
    deadline = time.time() + args.max_wait_hours * 3600.0
    attempt = 0
    print(f"waiting for LQTP {args.server} (poll={args.poll_sec}s, parallel_lqtp={args.parallel_lqtp})")

    while time.time() < deadline:
        attempt += 1
        tcp_ok = _tcp_open(host, port)
        print(f"[{attempt}] TCP {host}:{port} -> {'open' if tcp_ok else 'closed/refused'}")
        if tcp_ok:
            try:
                login_with_retry(
                    args.server,
                    args.username,
                    args.password,
                    max_attempts=3,
                    wait_sec=10.0,
                )
                print("LQTP gRPC login OK, starting batch ...")
                removed = _clear_transient_failures(args.work_dir / "production_progress.json")
                if removed:
                    print(f"cleared {removed} transient failures from progress")
                cmd = [
                    sys.executable,
                    str(ROOT / "scripts/cogalpha_lqtp/run_production_batch.py"),
                    "--work-dir",
                    str(args.work_dir),
                    "--start",
                    "2019-01-01",
                    "--end",
                    "2026-06-30",
                    "--mem-reserve-gb",
                    str(args.mem_reserve_gb),
                    "--max-total-parallel",
                    str(args.max_total_parallel),
                    "--parallel-lqtp",
                    str(args.parallel_lqtp),
                    "--parallel-local",
                    str(args.parallel_local),
                    "--resume",
                ]
                log_path = args.work_dir / "production.log"
                with log_path.open("a", encoding="utf-8") as log_f:
                    log_f.write(f"\n--- wait_lqtp_and_resume_batch start {time.strftime('%F %T')} ---\n")
                    proc = subprocess.Popen(
                        cmd,
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                        cwd=str(ROOT),
                        env={**os.environ, "PYTHONUNBUFFERED": "1"},
                    )
                print(f"batch pid={proc.pid} log={log_path}")
                return 0
            except Exception as exc:  # noqa: BLE001
                print(f"gRPC still failing after TCP open: {exc}")
        time.sleep(args.poll_sec)

    print(f"gave up after {args.max_wait_hours}h")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
