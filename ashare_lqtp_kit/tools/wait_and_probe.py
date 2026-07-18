#!/usr/bin/env python3
"""Wait until LQTP gRPC port is open, then optionally run probe.

Usage:
  export LQTP_USERNAME=... LQTP_PASSWORD=...
  python3 tools/wait_and_probe.py
  python3 tools/wait_and_probe.py --interval 30 --max-wait 3600
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

KIT_ROOT = Path(__file__).resolve().parents[1]


def tcp_open(host: str, port: int, timeout: float = 3.0) -> bool:
    s = socket.socket()
    s.settimeout(timeout)
    try:
        return s.connect_ex((host, port)) == 0
    finally:
        s.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for LQTP gRPC then probe")
    parser.add_argument("--server", default=os.getenv("LQTP_SERVER", "110.42.223.26:50051"))
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--max-wait", type=float, default=7200.0, help="seconds; 0 = forever")
    parser.add_argument("--skip-probe", action="store_true")
    args = parser.parse_args()

    if ":" not in args.server:
        print(f"bad server: {args.server}", file=sys.stderr)
        return 2
    host, port_s = args.server.rsplit(":", 1)
    port = int(port_s)

    t0 = time.time()
    n = 0
    print(f"waiting for {host}:{port} ...", flush=True)
    while True:
        n += 1
        ok = tcp_open(host, port)
        print(f"[{n}] tcp={'OPEN' if ok else 'refused'} elapsed={time.time()-t0:.0f}s", flush=True)
        if ok:
            break
        if args.max_wait > 0 and (time.time() - t0) >= args.max_wait:
            print("timeout waiting for service", file=sys.stderr)
            return 1
        time.sleep(max(1.0, args.interval))

    if args.skip_probe:
        print("port open; skip probe")
        return 0

    env = os.environ.copy()
    env["LQTP_SERVER"] = args.server
    cmd = [sys.executable, str(KIT_ROOT / "tools" / "probe.py")]
    return subprocess.call(cmd, cwd=str(KIT_ROOT), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
