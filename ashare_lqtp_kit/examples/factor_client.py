"""
Factor gRPC client for RunFactor, AnalyzeFactor, and factor catalog APIs.

Examples:
    python examples/factor_client.py run --formula "close / delay(close, 1) - 1" \
        --begin 20240102 --end 20240329 --warmup 1 --analyze

    python examples/factor_client.py run --definition-id <uuid> --token "$TOKEN" --analyze

    python examples/factor_client.py list --token "$TOKEN"

    python examples/factor_client.py analysis-list --token "$TOKEN"
"""

import argparse
import json
import math
import os
import sys

import grpc
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protos"))

import Factor_pb2
import Factor_pb2_grpc


VALUE_RETURN_MODES = {
    "none": Factor_pb2.FACTOR_VALUE_RETURN_MODE_NONE,
    "sample": Factor_pb2.FACTOR_VALUE_RETURN_MODE_SAMPLE,
    "all": Factor_pb2.FACTOR_VALUE_RETURN_MODE_ALL,
}

PRICE_SOURCES = {
    "unspecified": Factor_pb2.INTRADAY_ANALYSIS_PRICE_SOURCE_UNSPECIFIED,
    "snapshot": Factor_pb2.INTRADAY_ANALYSIS_PRICE_SOURCE_SNAPSHOT,
    "transaction": Factor_pb2.INTRADAY_ANALYSIS_PRICE_SOURCE_TRANSACTION,
}


def add_common_args(parser):
    parser.add_argument("--server", default="localhost:50051", help="gRPC server address")
    parser.add_argument("--token", default=os.getenv("LQTP_TOKEN", ""), help="access token")


def parse_args():
    parser = argparse.ArgumentParser(description="LQTP 因子客户端")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="compute a factor and optionally analyze it")
    add_common_args(run)
    run.add_argument("--begin", type=int, default=20240102, help="开始日期 YYYYMMDD")
    run.add_argument("--end", type=int, default=20240329, help="end date YYYYMMDD")
    run.add_argument("--formula", default="StockDailyBar.Close / StockDailyBar.PreClose - 1")
    run.add_argument("--definition-id", default="", help="已注册因子 UUID")
    run.add_argument("--factor-name", default="", help="随分析结果持久化的展示名称")
    run.add_argument("--description", default="", help="分析描述")
    run.add_argument("--warmup", type=int, default=0, help="warmup trading days")
    run.add_argument("--filter", default="", help="SQL filter condition")
    run.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="factor definition tag; repeatable, e.g. --tag universe=hs300",
    )
    run.add_argument("--interval-minutes", type=int, default=0, help="0 表示日频")
    run.add_argument("--interval-seconds", type=int, default=0, help="大于 0 表示秒级日内")
    run.add_argument("--analyze", action="store_true")
    run.add_argument("--return-mode", choices=VALUE_RETURN_MODES, default="sample")
    run.add_argument("--value-limit", type=int, default=20)
    run.add_argument("--analysis-price-source", choices=PRICE_SOURCES, default="unspecified")
    run.add_argument("--out", default="factor_result.csv", help="output CSV path")

    list_cmd = sub.add_parser("list", help="list registered factors")
    add_common_args(list_cmd)

    get_cmd = sub.add_parser("get", help="get one registered factor")
    add_common_args(get_cmd)
    get_cmd.add_argument("--definition-id", required=True)

    analysis_list = sub.add_parser("analysis-list", help="列出因子分析运行记录")
    add_common_args(analysis_list)
    analysis_list.add_argument("--limit", type=int, default=30)

    analysis_get = sub.add_parser("analysis-get", help="获取单条因子分析运行记录")
    add_common_args(analysis_get)
    analysis_get.add_argument("--analysis-id", required=True)

    update = sub.add_parser("update-published", help="compute values for published factors")
    add_common_args(update)
    update.add_argument("--trade-date", type=int, required=True)
    update.add_argument("--max-frequency", type=int, default=1, help="1=daily, 2=minute, 3=second")
    return parser.parse_args()


def metadata(token):
    return (("authorization", f"Bearer {token}"),) if token else None


def parse_tags(items):
    tags = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--tag must use KEY=VALUE format: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise SystemExit("--tag key cannot be empty")
        tags[key] = value
    return tags


def factor_stub(args):
    channel = grpc.insecure_channel(
        args.server,
        options=[
            ("grpc.max_send_message_length", 256 * 1024 * 1024),
            ("grpc.max_receive_message_length", 256 * 1024 * 1024),
        ],
    )
    try:
        grpc.channel_ready_future(channel).result(timeout=10)
    except grpc.FutureTimeoutError:
        raise SystemExit(f"无法连接服务: {args.server}")
    return Factor_pb2_grpc.FactorServiceStub(channel)


def point_label(trade_date, quote_time):
    if quote_time == 0:
        return str(trade_date)
    width = 6 if quote_time > 9999 else 4
    return f"{trade_date} {quote_time:0{width}d}"


def print_point_summary(point):
    values = [value.value for value in point.values]
    valid = [value for value in values if not math.isnan(value)]
    if not valid:
        print(f"{point_label(point.trade_date, point.quote_time)} 全部为 NaN")
        return
    print(
        f"{point_label(point.trade_date, point.quote_time)} "
        f"n={len(valid):5d} min={min(valid):10.5f} "
        f"max={max(valid):10.5f} mean={sum(valid) / len(valid):10.5f}"
    )


def save_values(points, out_path):
    rows = []
    for point in points:
        for value in point.values:
            rows.append(
                {
                    "trade_date": point.trade_date,
                    "quote_time": point.quote_time,
                    "symbol": value.symbol,
                    "value": value.value,
                }
            )
    if not rows:
        print("未返回因子值")
        return
    df = pd.DataFrame(rows)
    wide = df.pivot(index=["trade_date", "quote_time"], columns="symbol", values="value")
    wide.index.names = ["trade_date", "quote_time"]
    wide.columns.name = None
    wide.to_csv(out_path, encoding="utf-8-sig")
    print(f"因子值已保存: {out_path} shape={wide.shape}")


def save_analysis(analysis, out_path):
    prefix, ext = os.path.splitext(out_path)
    summary_path = f"{prefix}_analysis{ext}"
    pd.DataFrame(
        [
            {
                "mean_ic": analysis.mean_ic,
                "std_ic": analysis.std_ic,
                "icir": analysis.icir,
                "ic_positive_ratio": analysis.ic_positive_ratio,
                "coverage": analysis.coverage,
                "long_short_return": analysis.long_short_return,
                "long_short_sharpe": analysis.long_short_sharpe,
            }
        ]
    ).to_csv(summary_path, index=False, encoding="utf-8-sig")

    quote_times = (
        list(analysis.quote_times)
        if len(analysis.quote_times) == len(analysis.trade_dates)
        else [0] * len(analysis.trade_dates)
    )
    daily_path = f"{prefix}_daily_analysis{ext}"
    pd.DataFrame(
        {
            "trade_date": list(analysis.trade_dates),
            "quote_time": quote_times,
            "daily_ic": list(analysis.daily_ic),
            "daily_ls_return": list(analysis.daily_ls_returns),
        }
    ).to_csv(daily_path, index=False, encoding="utf-8-sig")
    print(f"分析汇总已保存: {summary_path}")
    print(f"每日分析已保存: {daily_path}")


def command_run(args):
    stub = factor_stub(args)
    request = Factor_pb2.RunFactorRequest(
        formula=args.formula,
        begin_date=args.begin,
        end_date=args.end,
        warmup=args.warmup,
        filter=args.filter,
        interval_minutes=args.interval_minutes,
        interval_seconds=args.interval_seconds,
        analyze=args.analyze,
        value_return_mode=VALUE_RETURN_MODES[args.return_mode],
        value_limit=args.value_limit,
        intraday_price_source=PRICE_SOURCES[args.analysis_price_source],
        factor_name=args.factor_name,
        description=args.description,
        definition_id=args.definition_id,
        tags=parse_tags(args.tag),
    )
    try:
        response = stub.RunFactor(request, metadata=metadata(args.token), timeout=300)
    except grpc.RpcError as error:
        raise SystemExit(f"RunFactor 调用失败: {error.code()} - {error.details()}")
    if response.error:
        raise SystemExit(f"RunFactor 返回错误: {response.error}")

    print(f"total_value_rows: {response.total_value_rows}")
    print(f"time_points     : {response.total_time_points}")
    print(f"values_truncated: {response.values_truncated}")
    for point in response.values:
        print_point_summary(point)
    save_values(response.values, args.out)

    if response.analysis:
        analysis = response.analysis
        print(
            "分析: "
            f"mean_ic={analysis.mean_ic:.6f}, "
            f"icir={analysis.icir:.6f}, "
            f"coverage={analysis.coverage:.6f}, "
            f"long_short_sharpe={analysis.long_short_sharpe:.6f}"
        )
        save_analysis(analysis, args.out)


def factor_row(definition):
    return {
        "definition_id": definition.definition_id,
        "factor_name": definition.factor_name,
        "version": definition.version,
        "status": definition.status,
        "frequency": definition.frequency,
        "interval_minutes": definition.interval_minutes,
        "interval_seconds": definition.interval_seconds,
        "warmup": definition.warmup,
        "tags": json.dumps(dict(definition.tags), ensure_ascii=False, sort_keys=True),
        "formula": definition.formula,
        "filter": definition.filter,
        "description": definition.description,
    }


def command_list(args):
    stub = factor_stub(args)
    response = stub.ListFactors(
        Factor_pb2.ListFactorsRequest(),
        metadata=metadata(args.token),
        timeout=60,
    )
    rows = [factor_row(item) for item in response.factors]
    print(pd.DataFrame(rows).to_string(index=False) if rows else "没有因子定义")


def command_get(args):
    stub = factor_stub(args)
    response = stub.GetFactor(
        Factor_pb2.GetFactorRequest(definition_id=args.definition_id),
        metadata=metadata(args.token),
        timeout=60,
    )
    for key, value in factor_row(response).items():
        print(f"{key}: {value}")


def analysis_row(item):
    return {
        "analysis_id": item.analysis_id,
        "definition_id": item.definition_id,
        "factor_name": item.factor_name,
        "begin_date": item.begin_date,
        "end_date": item.end_date,
        "frequency": item.frequency,
        "value_rows": item.value_rows,
        "time_points": item.time_points,
        "created_at": item.created_at,
        "mean_ic": item.analysis.mean_ic if item.analysis else float("nan"),
        "icir": item.analysis.icir if item.analysis else float("nan"),
        "coverage": item.analysis.coverage if item.analysis else float("nan"),
    }


def command_analysis_list(args):
    stub = factor_stub(args)
    response = stub.ListFactorAnalyses(
        Factor_pb2.ListFactorAnalysesRequest(limit=args.limit),
        metadata=metadata(args.token),
        timeout=60,
    )
    rows = [analysis_row(item) for item in response.analyses]
    print(pd.DataFrame(rows).to_string(index=False) if rows else "没有因子分析记录")


def command_analysis_get(args):
    stub = factor_stub(args)
    response = stub.GetFactorAnalysis(
        Factor_pb2.GetFactorAnalysisRequest(analysis_id=args.analysis_id),
        metadata=metadata(args.token),
        timeout=60,
    )
    for key, value in analysis_row(response).items():
        print(f"{key}: {value}")
    if response.analysis:
        save_analysis(response.analysis, "factor_analysis.csv")


def command_update_published(args):
    stub = factor_stub(args)
    response = stub.UpdatePublishedFactors(
        Factor_pb2.UpdatePublishedFactorsRequest(
            trade_date=args.trade_date,
            max_frequency=args.max_frequency,
        ),
        metadata=metadata(args.token),
        timeout=300,
    )
    rows = [
        {
            "definition_id": item.definition_id,
            "factor_name": item.factor_name,
            "value_rows": item.value_rows,
            "error": item.error,
        }
        for item in response.results
    ]
    print(pd.DataFrame(rows).to_string(index=False) if rows else "没有因子更新结果")


def main():
    args = parse_args()
    commands = {
        "run": command_run,
        "list": command_list,
        "get": command_get,
        "analysis-list": command_analysis_list,
        "analysis-get": command_analysis_get,
        "update-published": command_update_published,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
