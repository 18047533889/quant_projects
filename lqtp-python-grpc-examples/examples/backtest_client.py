"""
Backtest gRPC client.

Examples:
    python examples/backtest_client.py --begin 20240102 --end 20240329

    python examples/backtest_client.py --token "$TOKEN" --name demo \
        --return-orders --return-positions --out backtest_result.csv

    python examples/backtest_client.py --quote-time 930 --duration 30 \
        --price-type VWAP --symbols 000001,000002,000063
"""

import argparse
import os
import sys

import grpc
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protos"))

import Struct_pb2
import Struct_pb2_grpc


PRICE_TYPES = ("OPEN", "CLOSE", "PRE_CLOSE", "VWAP")


def parse_args():
    parser = argparse.ArgumentParser(description="LQTP 回测客户端")
    parser.add_argument("--server", default="localhost:50051", help="gRPC server address")
    parser.add_argument("--token", default=os.getenv("LQTP_TOKEN", ""), help="access token")
    parser.add_argument("--begin", type=int, default=20240102, help="开始日期 YYYYMMDD")
    parser.add_argument("--end", type=int, default=20240329, help="end date YYYYMMDD")
    parser.add_argument("--cash", type=float, default=1_000_000.0, help="initial cash")
    parser.add_argument(
        "--commission",
        type=float,
        default=0.03,
        help="one-way commission rate in percent, e.g. 0.03 means 0.03%",
    )
    parser.add_argument("--symbols", default="000001,000002,000003,000004,000005")
    parser.add_argument(
        "--dates",
        default="",
        help="comma-separated rebalance dates; default uses business days in range",
    )
    parser.add_argument("--quote-time", type=int, default=0, help="0=daily, e.g. 930=minute mode")
    parser.add_argument("--duration", type=int, default=1, help="execution duration in minute bars")
    parser.add_argument("--price-type", choices=PRICE_TYPES, default="CLOSE")
    parser.add_argument(
        "--max-turnover-limit",
        type=float,
        default=0.0,
        help="single-stock participation cap in percent; 0 disables the cap",
    )
    parser.add_argument("--price-limit", action="store_true", help="enable limit-up/down checks")
    parser.add_argument("--enable-short-selling", action="store_true", help="enable short selling")
    parser.add_argument("--return-orders", action="store_true")
    parser.add_argument("--return-targets", action="store_true")
    parser.add_argument("--return-positions", action="store_true")
    parser.add_argument("--name", default="", help="持久化任务名称")
    parser.add_argument("--factor-definition-id", default="", help="已注册因子 UUID")
    parser.add_argument("--no-persist", action="store_true", help="不持久化本次运行")
    parser.add_argument("--out", default="backtest_result.csv", help="output CSV path")
    return parser.parse_args()


def metadata(token):
    return (("authorization", f"Bearer {token}"),) if token else None


def business_dates(begin, end):
    return [int(d.strftime("%Y%m%d")) for d in pd.bdate_range(str(begin), str(end))]


def build_weights(args):
    symbols = [symbol.strip() for symbol in args.symbols.split(",") if symbol.strip()]
    if not symbols:
        raise SystemExit("--symbols 不能为空")

    trade_dates = (
        [int(value.strip()) for value in args.dates.split(",") if value.strip()]
        if args.dates
        else business_dates(args.begin, args.end)
    )
    weight = 1.0 / len(symbols)
    return [
        Struct_pb2.Weight(
            trade_date=trade_date,
            quote_time=args.quote_time,
            symbol=symbol,
            value=weight,
        )
        for trade_date in trade_dates
        for symbol in symbols
    ]


def save_detail_rows(base_path, suffix, rows):
    if not rows:
        return
    base, ext = os.path.splitext(base_path)
    path = f"{base}_{suffix}{ext}"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"{suffix} 已保存: {path}")


def main():
    args = parse_args()
    if args.quote_time != 0 and args.price_type == "PRE_CLOSE":
        raise SystemExit("分钟模式不支持 PRE_CLOSE 价格类型")

    request = Struct_pb2.Request(
        name=args.name,
        persistence_mode=(
            Struct_pb2.BACKTEST_PERSISTENCE_MODE_NONE
            if args.no_persist
            else Struct_pb2.BACKTEST_PERSISTENCE_MODE_DAILY
        ),
        factor_definition_id=args.factor_definition_id,
        enable_short_selling=args.enable_short_selling,
        begin_date=args.begin,
        end_date=args.end,
        weights=build_weights(args),
        return_orders=args.return_orders,
        return_targets=args.return_targets,
        return_positions=args.return_positions,
        cash=args.cash,
        commission_buy=args.commission,
        commission_sell=args.commission,
        max_turnover_limit=args.max_turnover_limit,
        price_limit=1.0 if args.price_limit else 0.0,
        duration=args.duration,
        price_type=Struct_pb2.PriceType.Value(args.price_type),
    )

    channel = grpc.insecure_channel(args.server)
    stub = Struct_pb2_grpc.BacktestServiceStub(channel)

    print(f"server: {args.server}")
    print(f"dates : {args.begin} ~ {args.end}")
    print(f"mode  : {'日频' if args.quote_time == 0 else f'分钟 @{args.quote_time}'}")
    print(f"short : {'启用' if args.enable_short_selling else '关闭'}")
    print(f"cash  : {args.cash:,.2f}")
    print("-" * 72)

    rows = []
    orders = []
    targets = []
    positions = []
    backtest_id = ""

    try:
        for resp in stub.Backtest(request, metadata=metadata(args.token), timeout=300):
            if resp.error:
                raise RuntimeError(resp.error)
            backtest_id = resp.backtest_id or backtest_id
            result = resp.result
            rows.append(
                {
                    "trade_date": result.trade_date,
                    "bod_net_asset": result.bod_net_asset,
                    "eod_net_asset": result.eod_net_asset,
                    "ret": result.ret,
                    "eod_market_value": result.eod_market_value,
                    "eod_cash": result.eod_cash,
                    "turnover_rate": result.turnover_rate,
                    "execution_ratio": result.execution_ratio,
                    "commission": result.commission,
                    "buy_amount": result.buy_amount,
                    "sell_amount": result.sell_amount,
                }
            )
            print(
                f"{result.trade_date} nav={result.eod_net_asset:12,.2f} "
                f"ret={result.ret / 10000:+.4%} "
                f"turnover={result.turnover_rate / 100:.4%} "
                f"exec={result.execution_ratio:.4f}"
            )
            orders.extend(
                {
                    "trade_date": item.trade_date,
                    "quote_time": item.quote_time,
                    "symbol": item.symbol,
                    "volume": item.volume,
                    "trader_volume": item.trader_volume,
                    "price": item.price,
                }
                for item in resp.orders
            )
            targets.extend(
                {
                    "trade_date": item.trade_date,
                    "quote_time": item.quote_time,
                    "symbol": item.symbol,
                    "volume": item.volume,
                    "price": item.price,
                }
                for item in resp.targets
            )
            positions.extend(
                {
                    "trade_date": item.trade_date,
                    "quote_time": item.quote_time,
                    "symbol": item.symbol,
                    "volume": item.volume,
                    "price": item.price,
                }
                for item in resp.positions
            )
    except grpc.RpcError as error:
        print(f"gRPC Backtest 调用失败: {error.code()} - {error.details()}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as error:
        print(f"回测失败: {error}", file=sys.stderr)
        sys.exit(1)

    if not rows:
        print("未返回回测结果")
        return

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    total_return = df["eod_net_asset"].iloc[-1] / args.cash - 1
    daily_returns = df["ret"] / 10000
    sharpe = (
        daily_returns.mean() / daily_returns.std() * (252**0.5)
        if daily_returns.std() > 0
        else float("nan")
    )
    max_drawdown = (df["eod_net_asset"] / df["eod_net_asset"].cummax() - 1).min()

    print("-" * 72)
    if backtest_id:
        print(f"backtest_id : {backtest_id}")
    print(f"total_return: {total_return:+.4%}")
    print(f"sharpe      : {sharpe:.4f}")
    print(f"max_drawdown: {max_drawdown:.4%}")
    print(f"saved       : {args.out}")

    save_detail_rows(args.out, "orders", orders)
    save_detail_rows(args.out, "targets", targets)
    save_detail_rows(args.out, "positions", positions)


if __name__ == "__main__":
    main()
