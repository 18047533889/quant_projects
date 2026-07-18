#!/usr/bin/env python3
"""Minimal LQTP gRPC probe — send this file to platform support.

Usage:
  export LQTP_USERNAME=your@email.com
  export LQTP_PASSWORD=your_password
  python3 lqtp_connectivity_probe.py

Server: 110.42.223.26:50051 (insecure gRPC)

----------------------------------------------------------------------
REPORTED ERRORS (2026-07-12, platform-side)
----------------------------------------------------------------------

[1] gRPC 服务不可达 — 约 17:00 起持续 ~30min，50051 TCP Connection refused

    grpc._channel._InactiveRpcError: <_InactiveRpcError of RPC that terminated with:
        status = StatusCode.UNAVAILABLE
        details = "failed to connect to all addresses; last error: UNKNOWN: ipv4:118.89.191.220:50051: Failed to connect to remote host: Connection refused"
        debug_error_string = "UNAVAILABLE:failed to connect to all addresses; last error: UNKNOWN: ipv4:118.89.191.220:50051: Failed to connect to remote host: Connection refused"
    >

[2] BacktestService 响应超过平台 4MB 解码上限

    <_MultiThreadedRendezvous of RPC that terminated with:
        status = StatusCode.OUT_OF_RANGE
        details = "Error, decoded message length too large: found 4895143 bytes, the limit is: 4194304 bytes"
        debug_error_string = "OUT_OF_RANGE:Error, decoded message length too large: found 4895143 bytes, the limit is: 4194304 bytes"
    >

    回测：BacktestService.Backtest, begin=20190101 end=20260630, price_type=OPEN

----------------------------------------------------------------------
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from textwrap import dedent

import grpc

ROOT = Path(__file__).resolve().parents[2]
_PROTO_CANDIDATES = [
    ROOT / "ashare_lqtp_kit" / "protos",
    ROOT / "lqtp-python-grpc-examples" / "protos",
]
for _p in _PROTO_CANDIDATES:
    if _p.is_dir():
        sys.path.insert(0, str(_p))
        break

import Auth_pb2
import Auth_pb2_grpc
import Factor_pb2
import Factor_pb2_grpc
import Struct_pb2
import Struct_pb2_grpc

SERVER = os.getenv("LQTP_SERVER", "110.42.223.26:50051")
USERNAME = os.getenv("LQTP_USERNAME", "")
PASSWORD = os.getenv("LQTP_PASSWORD", "")

# Historical notes for platform support (old host may still appear in tickets)
REPORTED_ERRORS = dedent("""
    [2026-07-12] LQTP platform errors (old server 118.89.191.220:50051)

    (1) UNAVAILABLE / Connection refused (~17:00 UTC+8, ~30 minutes)
        status = StatusCode.UNAVAILABLE
        details = "failed to connect to all addresses ... Connection refused"

    (2) BacktestService OUT_OF_RANGE — decoded message > 4MB platform limit
        status = StatusCode.OUT_OF_RANGE
        details = "Error, decoded message length too large: found 4895143 bytes, the limit is: 4194304 bytes"

    Current default server: 110.42.223.26:50051 (override with LQTP_SERVER)
""").strip()


def channel(server: str) -> grpc.Channel:
    return grpc.insecure_channel(
        server,
        options=[
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ],
    )


def metadata(token: str) -> tuple[tuple[str, str], ...]:
    return (("authorization", f"Bearer {token}"),)


def probe_login(server: str, username: str, password: str) -> str:
    stub = Auth_pb2_grpc.AuthServiceStub(channel(server))
    resp = stub.Login(
        Auth_pb2.LoginRequest(username=username, password=password),
        timeout=60,
    )
    print(f"[OK] AuthService.Login user={resp.user.username} user_id={resp.user.user_id}")
    return resp.access_token


def probe_run_factor(token: str, server: str) -> None:
    stub = Factor_pb2_grpc.FactorServiceStub(channel(server))
    req = Factor_pb2.RunFactorRequest(
        formula="close / delay(close, 1) - 1",
        begin_date=20240102,
        end_date=20240131,
        warmup=1,
        analyze=True,
        value_return_mode=Factor_pb2.FACTOR_VALUE_RETURN_MODE_ALL,
        factor_name="",
    )
    resp = stub.RunFactor(req, metadata=metadata(token), timeout=600)
    if resp.error:
        raise RuntimeError(resp.error)
    n_days = len(resp.values)
    n_vals = sum(len(d.values) for d in resp.values)
    print(f"[OK] FactorService.RunFactor days={n_days} values={n_vals} mean_ic={resp.analysis.mean_ic:.6f}")


def probe_backtest(token: str, server: str) -> None:
    stub = Struct_pb2_grpc.BacktestServiceStub(channel(server))
    req = Struct_pb2.Request(
        begin_date=20240102,
        end_date=20240131,
        weights=[
            Struct_pb2.Weight(trade_date=20240102, quote_time=0, symbol="000001.SZ", value=1.0),
            Struct_pb2.Weight(trade_date=20240103, quote_time=0, symbol="000001.SZ", value=1.0),
        ],
        cash=10_000_000.0,
        commission_buy=0.01,
        commission_sell=0.01,
        return_orders=False,
        return_targets=False,
        return_positions=False,
        persistence_mode=Struct_pb2.BACKTEST_PERSISTENCE_MODE_NONE,
        price_type=Struct_pb2.OPEN,
        enable_short_selling=False,
    )
    rows = 0
    backtest_id = ""
    for resp in stub.Backtest(req, metadata=metadata(token), timeout=300):
        if resp.error:
            raise RuntimeError(resp.error)
        backtest_id = resp.backtest_id or backtest_id
        rows += 1
    print(f"[OK] BacktestService.Backtest stream_rows={rows} backtest_id={backtest_id}")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in {"--errors", "-e"}:
        print(REPORTED_ERRORS)
        return 0
    if not USERNAME or not PASSWORD:
        print("Set LQTP_USERNAME and LQTP_PASSWORD env vars", file=sys.stderr)
        print("\n--- reported errors (see also file header) ---\n", file=sys.stderr)
        print(REPORTED_ERRORS, file=sys.stderr)
        return 2
    print(f"server={SERVER} user={USERNAME}")
    print("--- reported errors ---")
    print(REPORTED_ERRORS)
    print("--- probe ---")
    try:
        token = probe_login(SERVER, USERNAME, PASSWORD)
        probe_run_factor(token, SERVER)
        probe_backtest(token, SERVER)
        print("all probes passed")
        return 0
    except grpc.RpcError as exc:
        print(f"[FAIL] gRPC {exc.code()}: {exc.details()}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
