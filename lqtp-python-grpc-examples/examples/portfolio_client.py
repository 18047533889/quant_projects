"""
Paper portfolio gRPC client.

Examples:
    python examples/portfolio_client.py create --token "$TOKEN" --name demo

    python examples/portfolio_client.py submit-weights --token "$TOKEN" \
        --portfolio-id <uuid> --trade-date 20240102 --symbols 000001,000002

    python examples/portfolio_client.py settle --token "$TOKEN" \
        --portfolio-id <uuid> --trade-date 20240103
"""

import argparse
import os
import sys

import grpc
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "protos"))

import Portfolio_pb2
import Portfolio_pb2_grpc
import Struct_pb2


PRICE_TYPES = ("OPEN", "CLOSE", "PRE_CLOSE", "VWAP")


def add_common_args(parser):
    parser.add_argument("--server", default="localhost:50051", help="gRPC server address")
    parser.add_argument("--token", default=os.getenv("LQTP_TOKEN", ""), help="access token")


def parse_args():
    parser = argparse.ArgumentParser(description="LQTP 纸面组合客户端")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="创建纸面组合")
    add_common_args(create)
    create.add_argument("--name", required=True)
    create.add_argument("--initial-cash", type=float, default=1_000_000.0)
    create.add_argument("--commission-buy", type=float, default=0.03)
    create.add_argument("--commission-sell", type=float, default=0.03)
    create.add_argument("--max-turnover-limit", type=float, default=0.0)
    create.add_argument("--price-limit", type=float, default=0.0)
    create.add_argument("--price-type", choices=PRICE_TYPES, default="VWAP")

    list_cmd = sub.add_parser("list", help="list portfolios")
    add_common_args(list_cmd)

    get_cmd = sub.add_parser("get", help="get portfolio detail")
    add_common_args(get_cmd)
    get_cmd.add_argument("--portfolio-id", required=True)

    submit = sub.add_parser("submit-weights", help="submit target weights")
    add_common_args(submit)
    submit.add_argument("--portfolio-id", required=True)
    submit.add_argument("--trade-date", type=int, required=True)
    submit.add_argument("--symbols", required=True, help="逗号分隔的标的列表")
    submit.add_argument("--weights", default="", help="comma-separated weights; default equal")

    settle = sub.add_parser("settle", help="settle one portfolio for one day")
    add_common_args(settle)
    settle.add_argument("--portfolio-id", required=True)
    settle.add_argument("--trade-date", type=int, required=True)

    settle_all = sub.add_parser("settle-active", help="settle all active portfolios, admin only")
    add_common_args(settle_all)
    settle_all.add_argument("--trade-date", type=int, required=True)

    history = sub.add_parser("history", help="list daily results")
    add_common_args(history)
    history.add_argument("--portfolio-id", required=True)
    history.add_argument("--begin", type=int, default=0)
    history.add_argument("--end", type=int, default=0)
    history.add_argument("--details", action="store_true")
    history.add_argument("--out", default="", help="optional CSV output")
    return parser.parse_args()


def metadata(token):
    return (("authorization", f"Bearer {token}"),) if token else None


def stub(args):
    channel = grpc.insecure_channel(args.server)
    return Portfolio_pb2_grpc.PortfolioServiceStub(channel)


def portfolio_row(item):
    return {
        "portfolio_id": item.portfolio_id,
        "name": item.name,
        "status": item.status,
        "initial_cash": item.initial_cash,
        "current_cash": item.current_cash,
        "current_net_asset": item.current_net_asset,
        "last_settled_date": item.last_settled_date,
        "price_type": item.price_type,
        "created_at": item.created_at,
    }


def print_portfolio(item):
    for key, value in portfolio_row(item).items():
        print(f"{key}: {value}")


def parse_weights(args):
    symbols = [item.strip() for item in args.symbols.split(",") if item.strip()]
    if not symbols:
        raise SystemExit("--symbols 不能为空")
    if args.weights:
        weights = [float(item.strip()) for item in args.weights.split(",") if item.strip()]
        if len(weights) != len(symbols):
            raise SystemExit("--weights 长度必须与 --symbols 一致")
    else:
        weights = [1.0 / len(symbols)] * len(symbols)
    return [
        Struct_pb2.Weight(
            trade_date=args.trade_date,
            quote_time=0,
            symbol=symbol,
            value=weight,
        )
        for symbol, weight in zip(symbols, weights)
    ]


def daily_row(item):
    result = item.result
    return {
        "portfolio_id": item.portfolio_id,
        "trade_date": result.trade_date if result else 0,
        "eod_net_asset": result.eod_net_asset if result else 0.0,
        "ret": result.ret if result else 0.0,
        "turnover_rate": result.turnover_rate if result else 0.0,
        "commission": result.commission if result else 0.0,
        "already_settled": item.already_settled,
        "error": item.error,
    }


def command_create(args):
    response = stub(args).CreatePaperPortfolio(
        Portfolio_pb2.CreatePaperPortfolioRequest(
            name=args.name,
            initial_cash=args.initial_cash,
            commission_buy=args.commission_buy,
            commission_sell=args.commission_sell,
            max_turnover_limit=args.max_turnover_limit,
            price_limit=args.price_limit,
            price_type=Struct_pb2.PriceType.Value(args.price_type),
        ),
        metadata=metadata(args.token),
        timeout=60,
    )
    print_portfolio(response)


def command_list(args):
    response = stub(args).ListPaperPortfolios(
        Portfolio_pb2.ListPaperPortfoliosRequest(),
        metadata=metadata(args.token),
        timeout=60,
    )
    rows = [portfolio_row(item) for item in response.portfolios]
    print(pd.DataFrame(rows).to_string(index=False) if rows else "没有纸面组合")


def command_get(args):
    response = stub(args).GetPaperPortfolio(
        Portfolio_pb2.GetPaperPortfolioRequest(portfolio_id=args.portfolio_id),
        metadata=metadata(args.token),
        timeout=60,
    )
    print_portfolio(response.portfolio)
    if response.positions:
        rows = [
            {
                "symbol": item.symbol,
                "volume": item.volume,
                "price": item.price,
            }
            for item in response.positions
        ]
        print(pd.DataFrame(rows).to_string(index=False))


def command_submit_weights(args):
    response = stub(args).SubmitPaperWeights(
        Portfolio_pb2.SubmitPaperWeightsRequest(
            portfolio_id=args.portfolio_id,
            effective_trade_date=args.trade_date,
            weights=parse_weights(args),
        ),
        metadata=metadata(args.token),
        timeout=60,
    )
    print(f"submission_id       : {response.submission_id}")
    print(f"portfolio_id        : {response.portfolio_id}")
    print(f"effective_trade_date: {response.effective_trade_date}")
    print(f"version             : {response.version}")
    print(f"status              : {response.status}")
    print(f"submitted_at        : {response.submitted_at}")


def print_daily_response(item):
    row = daily_row(item)
    print(
        f"{row['trade_date']} nav={row['eod_net_asset']:12,.2f} "
        f"ret={row['ret'] / 10000:+.4%} "
        f"turnover={row['turnover_rate'] / 100:.4%} "
        f"error={row['error']}"
    )


def command_settle(args):
    response = stub(args).SettlePaperPortfolioDay(
        Portfolio_pb2.SettlePaperPortfolioDayRequest(
            portfolio_id=args.portfolio_id,
            trade_date=args.trade_date,
        ),
        metadata=metadata(args.token),
        timeout=300,
    )
    print_daily_response(response)


def command_settle_active(args):
    response = stub(args).SettleActivePaperPortfolios(
        Portfolio_pb2.SettleActivePaperPortfoliosRequest(trade_date=args.trade_date),
        metadata=metadata(args.token),
        timeout=300,
    )
    for item in response.results:
        print_daily_response(item)


def command_history(args):
    response = stub(args).ListPaperPortfolioDailyResults(
        Portfolio_pb2.ListPaperPortfolioDailyResultsRequest(
            portfolio_id=args.portfolio_id,
            begin_date=args.begin,
            end_date=args.end,
            include_details=args.details,
        ),
        metadata=metadata(args.token),
        timeout=60,
    )
    rows = [daily_row(item) for item in response.results]
    if not rows:
        print("没有每日结果")
        return
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"已保存: {args.out}")


def main():
    args = parse_args()
    commands = {
        "create": command_create,
        "list": command_list,
        "get": command_get,
        "submit-weights": command_submit_weights,
        "settle": command_settle,
        "settle-active": command_settle_active,
        "history": command_history,
    }
    try:
        commands[args.command](args)
    except grpc.RpcError as error:
        raise SystemExit(f"gRPC 调用失败: {error.code()} - {error.details()}")


if __name__ == "__main__":
    main()
