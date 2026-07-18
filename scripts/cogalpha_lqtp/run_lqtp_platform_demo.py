#!/usr/bin/env python3
"""Tiny materialize + LQTP platform demo: dump ALL API response fields."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import grpc
import pandas as pd
from google.protobuf.json_format import MessageToDict

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
LQTP_ROOT = ROOT / "lqtp-python-grpc-examples"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LQTP_ROOT / "protos"))

import Factor_pb2  # noqa: E402
import Factor_pb2_grpc  # noqa: E402
import Struct_pb2  # noqa: E402
import Struct_pb2_grpc  # noqa: E402

from scripts.cogalpha_lqtp.lqtp_client import (  # noqa: E402
    DEFAULT_SERVER,
    _channel,
    _metadata,
    fetch_lqtp_universe,
    login,
    long_df_to_daily_values,
    run_backtest_from_weights,
    top_quantile_weights,
)


def _run_py(script: Path, *args: str) -> None:
    subprocess.run([sys.executable, str(script), *args], check=True)


def _yyyymmdd(s: str) -> int:
    return int(s.replace("-", ""))


def _pb(obj: Any) -> dict[str, Any]:
    return MessageToDict(obj, preserving_proto_field_name=True)


def _sample_values(values: list[Factor_pb2.FactorDailyValues], n_days: int = 3, n_syms: int = 5) -> list[dict]:
    out: list[dict] = []
    for pt in values[:n_days]:
        out.append(
            {
                "trade_date": pt.trade_date,
                "quote_time": pt.quote_time,
                "values": [{"symbol": v.symbol, "value": v.value} for v in pt.values[:n_syms]],
                "value_count": len(pt.values),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Tiny LQTP platform response demo")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/cogalpha_lqtp_platform_demo")
    parser.add_argument("--factor", default="factor_persistence")
    parser.add_argument("--start", default="2024-01-02")
    parser.add_argument("--end", default="2024-01-31")
    parser.add_argument("--n-symbols", type=int, default=10)
    parser.add_argument("--server", default=os.getenv("LQTP_SERVER", DEFAULT_SERVER))
    parser.add_argument("--username", default=os.getenv("LQTP_USERNAME", ""))
    parser.add_argument("--password", default=os.getenv("LQTP_PASSWORD", ""))
    args = parser.parse_args()

    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    parsed = work / "parsed_factors.json"
    catalog = work / "dsl_catalog.json"
    smoke_dir = work / "smoke_StockDailyBar"
    lake = work / "factor_lake"
    out_json = work / "lqtp_platform_full_response.json"

    _run_py(SCRIPTS / "parse_factors_md.py", str(ROOT / "factors(1).md"), "--out", str(parsed))
    _run_py(SCRIPTS / "python_to_dsl.py", str(parsed), "--out", str(catalog))
    _run_py(
        SCRIPTS / "smoke_data.py",
        "--out-dir",
        str(smoke_dir),
        "--start",
        args.start,
        "--end",
        args.end,
        "--n-symbols",
        str(args.n_symbols),
    )
    _run_py(
        SCRIPTS / "materialize.py",
        "--catalog",
        str(catalog),
        "--data-root",
        str(smoke_dir),
        "--lake-root",
        str(lake),
        "--start",
        args.start,
        "--end",
        args.end,
        "--only",
        args.factor,
    )

    manifest = json.loads((lake / "materialize_manifest.json").read_text(encoding="utf-8"))
    entry = next(x for x in manifest if x["function_name"] == args.factor)
    long_df = pd.read_parquet(entry["values_path"])
    preview_df = long_df.head(8).copy()
    preview_df["datetime"] = preview_df["datetime"].astype(str)
    materialize_preview = {
        "factor": args.factor,
        "dsl": entry["dsl"],
        "rows": int(long_df.shape[0]),
        "columns": list(long_df.columns),
        "date_range": [str(long_df["datetime"].min()), str(long_df["datetime"].max())],
        "assets": sorted(long_df["asset"].astype(str).unique().tolist())[:15],
        "sample_rows": preview_df.to_dict(orient="records"),
    }

    begin_i = _yyyymmdd(args.start)
    end_i = _yyyymmdd(args.end)
    auth = login(args.server, args.username, args.password)
    token = auth.access_token

    catalog_map = {x["function_name"]: x for x in json.loads(catalog.read_text(encoding="utf-8"))}
    _ = catalog_map[args.factor]  # ensure factor exists in catalog

    daily_values = long_df_to_daily_values(long_df)

    # --- Auth ---
    import Auth_pb2  # noqa: E402
    import Auth_pb2_grpc  # noqa: E402

    auth_stub = Auth_pb2_grpc.AuthServiceStub(_channel(args.server))
    login_resp = auth_stub.Login(
        Auth_pb2.LoginRequest(username=args.username, password=args.password),
        timeout=60,
    )
    current_user = auth_stub.GetCurrentUser(Auth_pb2.EmptyResponse(), metadata=_metadata(token), timeout=60)

    # --- AnalyzeFactor (upload local values) ---
    analyze_factor_resp: dict[str, Any] = {"status": "skipped"}
    try:
        factor_stub = Factor_pb2_grpc.FactorServiceStub(_channel(args.server))
        af_resp = factor_stub.AnalyzeFactor(
            Factor_pb2.FactorAnalysisRequest(
                begin_date=begin_i,
                end_date=end_i,
                daily_values=daily_values,
            ),
            metadata=_metadata(token),
            timeout=300,
        )
        analyze_factor_resp = {
            "rpc": "AnalyzeFactor",
            "error": af_resp.error,
            "analysis": _pb(af_resp.analysis) if af_resp.analysis else None,
        }
    except grpc.RpcError as exc:
        analyze_factor_resp = {"rpc": "AnalyzeFactor", "grpc_error": str(exc), "code": str(exc.code())}

    # --- evaluate uploaded values (factor_engine → LQTP) ---
    from scripts.cogalpha_lqtp.factor_eval import evaluate_uploaded_values  # noqa: E402

    analysis, eval_mode = evaluate_uploaded_values(
        token=token,
        daily_values=daily_values,
        begin_date=begin_i,
        end_date=end_i,
        server=args.server,
    )

    universe = sorted(fetch_lqtp_universe(token=token, begin_date=begin_i, end_date=end_i, server=args.server))

    from scripts.cogalpha_lqtp.lqtp_client import factor_values_to_long_df  # noqa: E402

    lqtp_long = factor_values_to_long_df(daily_values)
    weights = top_quantile_weights(lqtp_long, allowed_symbols=set(universe))

    # --- Backtest stream: ALL fields (orders/targets/positions) ---
    bt_stub = Struct_pb2_grpc.BacktestServiceStub(_channel(args.server))
    bt_request = Struct_pb2.Request(
        name=f"demo_{args.factor}",
        begin_date=begin_i,
        end_date=end_i,
        weights=weights,
        cash=1_000_000.0,
        commission_buy=0.03,
        commission_sell=0.03,
        return_orders=True,
        return_targets=True,
        return_positions=True,
        persistence_mode=Struct_pb2.BACKTEST_PERSISTENCE_MODE_NONE,
        price_type=Struct_pb2.CLOSE,
    )
    backtest_stream: list[dict[str, Any]] = []
    backtest_id = ""
    for resp in bt_stub.Backtest(bt_request, metadata=_metadata(token), timeout=300):
        backtest_id = resp.backtest_id or backtest_id
        day = _pb(resp)
        # keep first 2 days full detail, later days result-only
        if len(backtest_stream) >= 2:
            day = {
                "backtest_id": day.get("backtest_id"),
                "error": day.get("error"),
                "result": day.get("result"),
                "orders_count": len(day.get("orders", [])),
                "targets_count": len(day.get("targets", [])),
                "positions_count": len(day.get("positions", [])),
            }
        backtest_stream.append(day)
        if resp.error:
            break

    # --- ListFactorAnalyses / ListFactors (metadata APIs) ---
    list_analyses = _pb(
        factor_stub.ListFactorAnalyses(Factor_pb2.ListFactorAnalysesRequest(limit=5), metadata=_metadata(token), timeout=60)
    )
    list_factors = _pb(factor_stub.ListFactors(Factor_pb2.ListFactorsRequest(), metadata=_metadata(token), timeout=60))
    if list_factors.get("factors"):
        list_factors["factors"] = list_factors["factors"][:5]

    payload = {
        "server": args.server,
        "date_range": [args.start, args.end],
        "factor": args.factor,
        "materialize_preview": materialize_preview,
        "auth_login": _pb(login_resp),
        "auth_current_user": _pb(current_user),
        "lqtp_universe_sample": universe[:20],
        "lqtp_universe_size": len(universe),
        "analyze_factor_response": analyze_factor_resp,
        "evaluate_uploaded_values": {
            "mode": eval_mode,
            "analysis": analysis,
            "note": "factor_engine DSL computed locally; LQTP evaluates uploaded daily_values only.",
        },
        "backtest": {
            "backtest_id": backtest_id,
            "request": _pb(bt_request),
            "daily_stream": backtest_stream,
            "stream_days": len(backtest_stream),
        },
        "list_factor_analyses": list_analyses,
        "list_factors": list_factors,
        "api_field_reference": {
            "FactorAnalysis": [
                "mean_ic", "std_ic", "icir", "ic_positive_ratio", "coverage",
                "long_short_return", "long_short_sharpe",
                "daily_ic", "daily_ls_returns", "trade_dates", "quote_times",
                "sample_counts", "coverages", "group_mean_returns", "group_pnls",
            ],
            "RunFactorResponse": [
                "values", "analysis", "error", "total_value_rows",
                "total_time_points", "values_truncated", "point_stats",
            ],
            "Backtest_DailyResponse": [
                "backtest_id", "result", "orders", "targets", "positions", "error",
            ],
            "Backtest_Result": [
                "trade_date", "bod_net_asset", "eod_net_asset",
                "bod_market_value", "eod_market_value",
                "bod_long_market_value", "eod_long_market_value",
                "bod_short_market_value", "eod_short_market_value",
                "bod_cash", "eod_cash", "expect_buy_amount", "expect_sell_amount",
                "buy_amount", "sell_amount", "buy_execution_ratio",
                "sell_execution_ratio", "execution_ratio", "turnover_rate",
                "leverage", "commission", "net_asset_diff", "ret",
            ],
        },
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(out_json), "eval_mode": eval_mode, "backtest_days": len(backtest_stream)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
