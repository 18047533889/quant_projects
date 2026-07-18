#!/usr/bin/env python3
"""LQTP connectivity probe (Login / RunFactor / Backtest).

Usage:
  export LQTP_USERNAME=your@email.com
  export LQTP_PASSWORD=your_password
  # optional: export LQTP_SERVER=110.42.223.26:50051
  python3 tools/probe.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from textwrap import dedent

import grpc

KIT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT_ROOT))
sys.path.insert(0, str(KIT_ROOT / "protos"))

import Auth_pb2  # noqa: E402
import Auth_pb2_grpc  # noqa: E402
import Factor_pb2  # noqa: E402
import Factor_pb2_grpc  # noqa: E402
import Struct_pb2  # noqa: E402
import Struct_pb2_grpc  # noqa: E402

SERVER = os.getenv("LQTP_SERVER", "110.42.223.26:50051")
USERNAME = os.getenv("LQTP_USERNAME", "")
PASSWORD = os.getenv("LQTP_PASSWORD", "")

REPORTED_ERRORS = dedent("""
    Historical notes (old host 118.89.191.220:50051):
    (1) UNAVAILABLE / Connection refused
    (2) BacktestService OUT_OF_RANGE when decoded message > 4MB
        — client already sets 256MB send/recv limits; still fix weights size if needed.
    Current default: 110.42.223.26:50051
""").strip()


def channel(server: str) -> grpc.Channel:
    # Prefer shared options from client when available
    try:
        sys.path.insert(0, str(KIT_ROOT))
        from ashare_lqtp.client import make_channel

        return make_channel(server)
    except Exception:
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
        print("Set LQTP_USERNAME and LQTP_PASSWORD", file=sys.stderr)
        return 2
    print(f"server={SERVER} user={USERNAME}")
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
