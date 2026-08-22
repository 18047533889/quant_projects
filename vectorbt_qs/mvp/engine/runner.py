"""
统一回测入口

封装 vectorbt.Portfolio.from_orders，串联 数据加载 → 约束 → 回测 → 结果 全流程。
支持 A 股和美股，统一接口。
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, Optional, List

# 确保项目内的 vectorbt 优先被导入
# vectorbt 源码在 vectorbt_qs/vectorbt/vectorbt/ 下（上游仓库嵌套一层）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "vectorbt"))
import vectorbt as vbt

from ..data.adapter import load_ashare_calendar, load_benchmark_close, load_market_data
from ..constraints.ashare import apply_ashare_constraints, resolve_limit_check_prices
from ..constraints.us_stock import apply_us_constraints
from .execution import ExecutionCosts, plan_ashare_orders, schedule_target_signals


# ============================================================
# 默认回测参数
# ============================================================

DEFAULT_CONFIG = {
    "init_cash": 1_000_000.0,     # 初始资金 ¥1,000,000 / $1,000,000
    "freq": "1D",                  # 日频
    "slippage": 0.001,             # 滑点 0.1%
    "allow_partial": True,         # 允许部分成交
    "cash_sharing": True,          # 共享资金池
    "call_seq": "auto",            # 自动排序（先卖后买）
    "execution_mode": "accurate",  # accurate | fast
    "planner_engine": "numba",     # accurate: numba | python
    "limit_check_mode": "execution",  # execution | strict
    "limit_price_rtol": 1e-5,
    "limit_price_atol": 1e-8,
    "signal_time": "close",        # close | preopen
    "fast_tradeability": "ignore", # ignore | approximate
    # A 股 accurate/fast 均按真实交易日历计数。
    "execution_lag": 1,
    "price_type": "open_adj",      # close_adj | open_adj | vwap_adj
    "strict_symbols": True,        # 行情缺失标的时报错
    "performance_year_days": 252,  # A股交易日年化口径
    "risk_free_rate": 0.0,         # 年化无风险收益率
    "max_participation_rate": None,  # 单股订单占当日成交量上限；None=关闭
}


def validate_target_weights(target_weights: pd.DataFrame) -> None:
    """Validate the public target-weight contract before loading market data."""
    if not isinstance(target_weights, pd.DataFrame) or target_weights.empty:
        raise ValueError("target_weights 必须是非空 pandas.DataFrame")
    if not isinstance(target_weights.index, pd.DatetimeIndex):
        raise TypeError("target_weights.index 必须是 DatetimeIndex")
    if not target_weights.index.is_monotonic_increasing:
        raise ValueError("target_weights 日期必须单调递增")
    if not target_weights.index.is_unique:
        raise ValueError("target_weights 日期不能重复")
    if not target_weights.columns.is_unique:
        raise ValueError("target_weights 标的列不能重复")
    if not all(isinstance(symbol, str) and symbol.strip() for symbol in target_weights.columns):
        raise TypeError("target_weights 标的列必须是非空字符串")
    if target_weights.index.tz is not None:
        raise ValueError("target_weights 日期必须是不带时区的交易日")
    if not target_weights.index.equals(target_weights.index.normalize()):
        raise ValueError("target_weights 日期不能包含日内时间")
    try:
        values = target_weights.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("target_weights 必须全部为数值") from exc
    if np.isinf(values).any():
        raise ValueError("target_weights 不能包含 inf")
    if np.isnan(values).all():
        raise ValueError("target_weights 不能全部为 NaN")


def resample_target_weights(target_weights: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Legacy vectorized path: select the last actual row of each week.

    Both A-share modes use :func:`schedule_target_signals`; this helper remains
    for the US adapter's legacy vectorized path.
    """
    if str(freq).upper() != "1W":
        return target_weights
    periods = target_weights.index.to_period("W-FRI")
    return target_weights.groupby(periods, sort=True).tail(1)


def prepare_target_weights(target_weights: pd.DataFrame, freq: str, execution_lag: int) -> pd.DataFrame:
    """Legacy vectorized path: validate, resample, and lag in row space."""
    validate_target_weights(target_weights)
    if not isinstance(execution_lag, (int, np.integer)) or execution_lag < 0:
        raise ValueError("execution_lag 必须是非负整数")
    prepared = resample_target_weights(target_weights.astype(np.float64), freq)
    if execution_lag:
        prepared = prepared.shift(execution_lag)
    if prepared.isna().all(axis=None):
        raise ValueError("执行滞后后没有可执行权重，请增加输入期数或设置 execution_lag=0")
    return prepared


def _resolve_price(data: Dict[str, pd.DataFrame], price_type: str) -> pd.DataFrame:
    normalized = str(price_type).lower().replace("-", "_")
    key_by_name = {
        "close": "close",
        "close_adj": "close",
        "open": "open",
        "open_adj": "open",
        "vwap": "vwap",
        "vwap_adj": "vwap",
    }
    if normalized not in key_by_name:
        raise ValueError("price_type 仅支持 close_adj、open_adj 或 vwap_adj")
    key = key_by_name[normalized]
    if key not in data:
        raise ValueError(f"行情数据不包含 price_type={price_type!r} 所需的 {key!r} 字段")
    return data[key]


def _resolve_raw_price(data: Dict[str, pd.DataFrame], price_type: str) -> pd.DataFrame:
    """Resolve an exchange price in real currency per real share.

    Legacy synthetic tests may only provide adjusted keys; Factor=1 keeps that
    path exactly equivalent. Production A-share adapters expose ``raw_*``.
    """
    normalized = str(price_type).lower().replace("-", "_")
    key_by_name = {
        "close": "raw_close",
        "close_adj": "raw_close",
        "open": "raw_open",
        "open_adj": "raw_open",
        "vwap": "raw_vwap",
        "vwap_adj": "raw_vwap",
    }
    adjusted_fallback = {
        "raw_close": "close",
        "raw_open": "open",
        "raw_vwap": "vwap",
    }
    if normalized not in key_by_name:
        raise ValueError(
            "price_type 仅支持 open、vwap、close（兼容 *_adj 别名）"
        )
    key = key_by_name[normalized]
    if key in data:
        return data[key]
    fallback = adjusted_fallback[key]
    if fallback in data:
        return data[fallback]
    raise ValueError(f"行情数据不包含 price_type={price_type!r} 所需的 {key!r} 字段")


def _attach_benchmark(
    pf: vbt.Portfolio,
    market: str,
    benchmark_index: Optional[str],
    start: str,
    end: str,
    portfolio_index: pd.DatetimeIndex,
) -> None:
    if not benchmark_index:
        return
    benchmark_close = load_benchmark_close(market, benchmark_index, start=start, end=end)
    benchmark_close = benchmark_close.reindex(portfolio_index).ffill()
    pf._qs_benchmark_close = benchmark_close
    pf._qs_benchmark_returns = benchmark_close.pct_change(fill_method=None)
    pf._qs_benchmark_symbol = benchmark_index


def _attach_performance_settings(pf: vbt.Portfolio, cfg: Dict) -> None:
    pf._qs_performance_year_days = int(
        cfg.get("performance_year_days", 252)
    )
    pf._qs_risk_free_rate = float(cfg.get("risk_free_rate", 0.0))


def _run_ashare_accurate(
    target_weights: pd.DataFrame,
    symbols: Optional[List[str]],
    start: Optional[str],
    end: Optional[str],
    cfg: Dict,
    config: Dict,
    kwargs: Dict,
) -> vbt.Portfolio:
    """Run the basic calendar-driven, stateful A-share execution model."""
    validate_target_weights(target_weights)
    target_weights = target_weights.astype(np.float64)
    if (target_weights.fillna(0.0) < 0.0).any(axis=None):
        raise ValueError("A 股现货回测不支持负权重；请移除空头")
    row_sums = target_weights.fillna(0.0).sum(axis=1)
    if (row_sums > 1.0 + 1e-8).any():
        first_date = row_sums.index[row_sums > 1.0 + 1e-8][0]
        raise ValueError(f"{first_date.date()} A 股目标权重之和超过 1")

    if start is not None:
        target_weights = target_weights.loc[target_weights.index >= pd.Timestamp(start)]
    if end is not None:
        target_weights = target_weights.loc[target_weights.index <= pd.Timestamp(end)]
    if target_weights.empty:
        raise ValueError("指定日期范围内没有目标权重")

    if symbols is None:
        selected_symbols = sorted(map(str, target_weights.columns))
    else:
        selected_symbols = list(dict.fromkeys(map(str, symbols)))
        missing_targets = pd.Index(selected_symbols).difference(target_weights.columns)
        if len(missing_targets) > 0:
            raise ValueError(f"目标权重缺少指定标的: {', '.join(map(str, missing_targets[:10]))}")
        selected_symbols = sorted(selected_symbols)
    target_weights = target_weights.reindex(columns=selected_symbols)

    signal_start = pd.Timestamp(target_weights.index.min()).normalize()
    signal_end = pd.Timestamp(target_weights.index.max()).normalize()
    calendar_extension_days = max(31, int(cfg["execution_lag"]) * 7 + 14)
    calendar_end = signal_end + pd.Timedelta(days=calendar_extension_days)
    calendar = load_ashare_calendar(
        start=str(signal_start.date()),
        end=str(calendar_end.date()),
    )
    scheduled_targets = schedule_target_signals(
        target_weights,
        calendar,
        freq=cfg["freq"],
        delay_sessions=cfg["execution_lag"],
        signal_time=cfg["signal_time"],
    )

    market_end = max(signal_end, pd.Timestamp(scheduled_targets.index.max()))
    print(
        f"[ashare] 加载日频行情: {signal_start.date()} ~ {market_end.date()}, "
        f"{len(selected_symbols)} 个标的"
    )
    data = load_market_data(
        "ashare",
        symbols=selected_symbols,
        start=str(signal_start.date()),
        end=str(market_end.date()),
    )
    close = data.get("raw_close", data["close"])
    raw_close = close
    raw_order_price = _resolve_raw_price(data, cfg["price_type"])
    missing_symbols = pd.Index(selected_symbols).difference(close.columns)
    if cfg["strict_symbols"] and len(missing_symbols) > 0:
        active_missing = missing_symbols[
            target_weights.reindex(columns=missing_symbols).fillna(0.0).abs().gt(1e-12).any(axis=0)
        ]
        if len(active_missing) > 0:
            raise ValueError(
                f"行情缺少有非零目标权重的标的: {', '.join(map(str, active_missing[:10]))}"
            )

    trading_index = pd.DatetimeIndex(calendar)
    trading_index = trading_index[(trading_index >= signal_start) & (trading_index <= market_end)]
    available_symbols = pd.Index(selected_symbols).intersection(close.columns)
    if len(available_symbols) == 0 or len(trading_index) == 0:
        raise ValueError("目标权重与行情没有可执行交集")
    close = close.reindex(index=trading_index, columns=available_symbols)
    raw_close = raw_close.reindex_like(close)
    raw_order_price = raw_order_price.reindex_like(close)
    scheduled_targets = scheduled_targets.reindex(columns=available_symbols)

    def required_frame(key: str) -> pd.DataFrame:
        frame = data.get(key)
        if frame is None:
            raise ValueError(f"准确模式要求行情字段: {key}")
        return frame.reindex_like(close)

    def raw_frame(raw_key: str, adjusted_key: str) -> pd.DataFrame:
        frame = data.get(raw_key)
        if frame is None:
            frame = data.get(adjusted_key)
        if frame is None:
            raise ValueError(f"准确模式要求行情字段: {raw_key}")
        return frame.reindex_like(close)

    factor = data.get("factor")
    if factor is None:
        factor = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    else:
        factor = factor.reindex_like(close)
    cash_dividend = data.get("cash_dividend")
    if cash_dividend is not None:
        cash_dividend = cash_dividend.reindex_like(close).fillna(0.0)
    share_multiplier = data.get("share_multiplier")
    if share_multiplier is not None:
        share_multiplier = share_multiplier.reindex_like(close).fillna(1.0)
    max_participation_rate = cfg.get("max_participation_rate")
    volume = (
        required_frame("volume")
        if max_participation_rate is not None
        else None
    )
    is_suspend = required_frame("is_suspend")
    high_limit = raw_frame("raw_high_limit", "high_limit")
    low_limit = raw_frame("raw_low_limit", "low_limit")
    raw_high_source = data.get("raw_high", data.get("high"))
    raw_low_source = data.get("raw_low", data.get("low"))
    if str(cfg["limit_check_mode"]).lower() == "strict" and (
        raw_high_source is None or raw_low_source is None
    ):
        raise ValueError("limit_check_mode=strict 要求原始 High 和 Low 行情")
    # Legacy synthetic tests did not expose intraday extremes. In execution
    # mode, exchange limits remain a conservative outer price boundary.
    raw_high = (
        high_limit
        if raw_high_source is None
        else raw_high_source.reindex_like(close)
    )
    raw_low = (
        low_limit
        if raw_low_source is None
        else raw_low_source.reindex_like(close)
    )
    high_limit, low_limit = resolve_limit_check_prices(
        raw_order_price,
        high_limit,
        low_limit,
        mode=cfg["limit_check_mode"],
        high=raw_high if str(cfg["limit_check_mode"]).lower() == "strict" else None,
        low=raw_low if str(cfg["limit_check_mode"]).lower() == "strict" else None,
        rtol=float(cfg["limit_price_rtol"]),
        atol=float(cfg["limit_price_atol"]),
    )
    costs = ExecutionCosts.from_config(config)
    lot_size = int(config.get("lot_size", 100))
    plan = plan_ashare_orders(
        close=raw_close,
        order_price=raw_order_price,
        scheduled_targets=scheduled_targets,
        is_suspend=is_suspend,
        high_limit=high_limit,
        low_limit=low_limit,
        init_cash=float(cfg["init_cash"]),
        costs=costs,
        slippage=float(cfg["slippage"]),
        lot_size=lot_size,
        planner_engine=str(cfg["planner_engine"]),
        adjustment_factor=factor,
        cash_dividend=cash_dividend,
        share_multiplier=share_multiplier,
        volume=volume,
        max_participation_rate=max_participation_rate,
        high=raw_high,
        low=raw_low,
    )
    simulated_real_holdings = (
        plan.order_size.fillna(0.0) + plan.asset_deposits
    ).cumsum()
    missing_held_close = (
        simulated_real_holdings.abs().gt(1e-12) & raw_close.isna()
    )
    if missing_held_close.any(axis=None):
        date, symbol = (
            missing_held_close.stack()
            .loc[lambda values: values]
            .index[0]
        )
        raise ValueError(
            f"{date.date()} {symbol} 持仓缺少收盘估值价；"
            "accurate 模式禁止静默沿用退市或异常行情"
        )

    protected_kwargs = {
        "close", "size", "size_type", "price", "fees", "fixed_fees",
        "direction", "slippage", "init_cash", "freq", "cash_sharing",
        "call_seq", "size_granularity", "allow_partial", "raise_reject",
        "cash_deposits", "asset_deposits",
    }
    conflicts = protected_kwargs.intersection(kwargs)
    if conflicts:
        raise ValueError(f"accurate 模式不允许覆盖执行参数: {', '.join(sorted(conflicts))}")
    if not cfg["cash_sharing"]:
        raise ValueError("accurate 模式要求 cash_sharing=True")

    print(
        f"[ashare] 开始准确回测: {close.shape[0]} 个交易日 × {close.shape[1]} 个标的, "
        f"{len(scheduled_targets)} 个执行日"
    )
    pf = vbt.Portfolio.from_orders(
        close=raw_close,
        size=plan.order_size,
        size_type="amount",
        price=raw_order_price,
        fees=plan.fees,
        fixed_fees=plan.fixed_fees,
        direction="longonly",
        slippage=cfg["slippage"],
        cash_deposits=plan.cash_deposits,
        asset_deposits=plan.asset_deposits,
        init_cash=cfg["init_cash"],
        freq="1D",
        cash_sharing=True,
        call_seq=plan.call_seq,
        size_granularity=np.nan,
        allow_partial=False,
        raise_reject=True,
        **kwargs,
    )
    pf._qs_execution_log = plan.log
    pf._qs_real_order_size = plan.order_size
    pf._qs_real_holdings = plan.real_holdings
    pf._qs_planner_final_cash = plan.final_cash
    pf._qs_cash_dividend_deposits = plan.cash_deposits
    pf._qs_share_distribution_deposits = plan.asset_deposits
    pf._qs_adjustment_factor = factor
    pf._qs_execution_price_domain = "raw_real_share"
    pf._qs_scheduled_targets = scheduled_targets
    pf._qs_execution_mode = "accurate"
    pf._qs_planner_engine = str(cfg["planner_engine"]).lower()
    pf._qs_limit_check_mode = str(cfg["limit_check_mode"]).lower()
    _attach_performance_settings(pf, cfg)
    _attach_benchmark(
        pf,
        "ashare",
        cfg.get("benchmark_index"),
        str(signal_start.date()),
        str(market_end.date()),
        close.index,
    )
    print("[ashare] 准确回测完成")
    return pf


def _run_ashare_factor_fast(
    target_weights: pd.DataFrame,
    symbols: Optional[List[str]],
    start: Optional[str],
    end: Optional[str],
    cfg: Dict,
    config: Dict,
    kwargs: Dict,
) -> vbt.Portfolio:
    """Run the frictionless A-share factor-research backtest.

    The contract is deliberately narrow and comparable across portfolios:
    close signal, next-session adjusted open, fractional shares, zero costs,
    zero slippage, shared cash, and a complete daily valuation timeline.
    """
    validate_target_weights(target_weights)
    target_weights = target_weights.astype(np.float64)
    if (target_weights.fillna(0.0) < 0.0).any(axis=None):
        raise ValueError("A 股 factor research 模式不支持负权重；请分别回测各分组")
    row_sums = target_weights.fillna(0.0).sum(axis=1)
    if (row_sums > 1.0 + 1e-8).any():
        first_date = row_sums.index[row_sums > 1.0 + 1e-8][0]
        raise ValueError(f"{first_date.date()} A 股目标权重之和超过 1")

    if str(cfg["signal_time"]).lower() != "close":
        raise ValueError("fast factor research 固定使用 signal_time=close")
    if int(cfg["execution_lag"]) != 1:
        raise ValueError("fast factor research 固定使用 execution_lag=1（下一交易日）")
    if str(cfg["price_type"]).lower().replace("-", "_") != "open_adj":
        raise ValueError("fast factor research 固定使用 price_type=Open_adj")
    if float(config.get("fees", 0.0)) != 0.0 or float(config.get("fixed_fees", 0.0)) != 0.0:
        raise ValueError("fast factor research 固定使用 fees=0、fixed_fees=0")
    if float(config.get("slippage", 0.0)) != 0.0:
        raise ValueError("fast factor research 固定使用 slippage=0")
    costs_config = config.get("costs")
    if costs_config is not None:
        if not isinstance(costs_config, dict):
            raise TypeError("costs 必须是字典")
        if any(float(value) != 0.0 for value in costs_config.values()):
            raise ValueError("fast factor research 不收取 costs；请删除非零 costs 配置")
    if "size_granularity" in config and not pd.isna(config["size_granularity"]):
        raise ValueError("fast factor research 固定使用分数股，不支持 size_granularity")
    if not cfg["cash_sharing"]:
        raise ValueError("fast factor research 要求 cash_sharing=True")
    if not cfg["allow_partial"]:
        raise ValueError("fast factor research 要求 allow_partial=True")
    if str(cfg["call_seq"]).lower() != "auto":
        raise ValueError("fast factor research 固定使用 call_seq=auto")
    if not np.isfinite(float(cfg["init_cash"])) or float(cfg["init_cash"]) <= 0:
        raise ValueError("init_cash 必须是正有限数")

    tradeability = str(cfg.get("fast_tradeability", "ignore")).lower()
    if tradeability not in {"ignore", "approximate"}:
        raise ValueError("fast_tradeability 仅支持 ignore 或 approximate")

    if start is not None:
        target_weights = target_weights.loc[target_weights.index >= pd.Timestamp(start)]
    if end is not None:
        target_weights = target_weights.loc[target_weights.index <= pd.Timestamp(end)]
    if target_weights.empty:
        raise ValueError("指定日期范围内没有目标权重")

    if symbols is None:
        selected_symbols = sorted(map(str, target_weights.columns))
    else:
        selected_symbols = list(dict.fromkeys(map(str, symbols)))
        missing_targets = pd.Index(selected_symbols).difference(target_weights.columns)
        if len(missing_targets) > 0:
            raise ValueError(f"目标权重缺少指定标的: {', '.join(map(str, missing_targets[:10]))}")
        selected_symbols = sorted(selected_symbols)
    target_weights = target_weights.reindex(columns=selected_symbols)

    signal_start = pd.Timestamp(target_weights.index.min()).normalize()
    signal_end = pd.Timestamp(target_weights.index.max()).normalize()
    calendar = load_ashare_calendar(
        start=str(signal_start.date()),
        end=str((signal_end + pd.Timedelta(days=31)).date()),
    )
    scheduled_targets = schedule_target_signals(
        target_weights,
        calendar,
        freq=cfg["freq"],
        delay_sessions=1,
        signal_time="close",
    )
    market_end = pd.Timestamp(scheduled_targets.index.max())

    print(
        f"[ashare/fast] 加载日频行情: {signal_start.date()} ~ {market_end.date()}, "
        f"{len(selected_symbols)} 个标的"
    )
    data = load_market_data(
        "ashare",
        symbols=selected_symbols,
        start=str(signal_start.date()),
        end=str(market_end.date()),
    )
    close = data["close"]
    order_price = _resolve_price(data, "open_adj")
    missing_symbols = pd.Index(selected_symbols).difference(close.columns)
    if cfg["strict_symbols"] and len(missing_symbols) > 0:
        active_missing = missing_symbols[
            target_weights.reindex(columns=missing_symbols).fillna(0.0).abs().gt(1e-12).any(axis=0)
        ]
        if len(active_missing) > 0:
            raise ValueError(
                f"行情缺少有非零目标权重的标的: {', '.join(map(str, active_missing[:10]))}"
            )

    trading_index = pd.DatetimeIndex(calendar)
    trading_index = trading_index[(trading_index >= signal_start) & (trading_index <= market_end)]
    available_symbols = pd.Index(selected_symbols).intersection(close.columns)
    if len(available_symbols) == 0 or len(trading_index) == 0:
        raise ValueError("目标权重与行情没有可执行交集")
    close = close.reindex(index=trading_index, columns=available_symbols)
    order_price = order_price.reindex_like(close)
    scheduled_targets = scheduled_targets.reindex(columns=available_symbols)

    if tradeability == "ignore":
        execution_prices = order_price.reindex(index=scheduled_targets.index)
        active_targets = scheduled_targets.fillna(0.0).abs().gt(1e-12)
        missing_execution_price = active_targets & execution_prices.isna()
        if missing_execution_price.any(axis=None):
            date, symbol = missing_execution_price.stack().loc[lambda values: values].index[0]
            raise ValueError(f"{date.date()} {symbol} 非零目标缺少复权开盘价")

    execution_targets = scheduled_targets
    if tradeability == "approximate":
        def required_frame(key: str) -> pd.DataFrame:
            frame = data.get(key)
            if frame is None:
                raise ValueError(f"fast_tradeability=approximate 要求行情字段: {key}")
            return frame.reindex(index=scheduled_targets.index, columns=available_symbols)

        check_high_limit, check_low_limit = resolve_limit_check_prices(
            order_price.reindex(index=scheduled_targets.index),
            required_frame("high_limit"),
            required_frame("low_limit"),
            mode=cfg["limit_check_mode"],
            high=required_frame("high") if str(cfg["limit_check_mode"]).lower() == "strict" else None,
            low=required_frame("low") if str(cfg["limit_check_mode"]).lower() == "strict" else None,
            rtol=float(cfg["limit_price_rtol"]),
            atol=float(cfg["limit_price_atol"]),
        )
        constrained = apply_ashare_constraints(
            close=close.reindex(index=scheduled_targets.index),
            target_weights=scheduled_targets,
            is_suspend=required_frame("is_suspend"),
            high_limit=check_high_limit,
            low_limit=check_low_limit,
            order_price=order_price.reindex(index=scheduled_targets.index),
        )
        execution_targets = constrained["target_weights"]

    sizes = execution_targets.reindex(index=trading_index, columns=available_symbols)
    protected_kwargs = {
        "close", "size", "size_type", "price", "fees", "fixed_fees",
        "direction", "slippage", "init_cash", "freq", "cash_sharing",
        "call_seq", "size_granularity", "allow_partial", "raise_reject",
    }
    conflicts = protected_kwargs.intersection(kwargs)
    if conflicts:
        raise ValueError(
            f"fast factor research 不允许覆盖执行参数: {', '.join(sorted(conflicts))}"
        )

    print(
        f"[ashare/fast] 开始无摩擦因子回测: {close.shape[0]} 个交易日 × "
        f"{close.shape[1]} 个标的, {len(execution_targets)} 个调仓日, "
        f"tradeability={tradeability}"
    )
    pf = vbt.Portfolio.from_orders(
        close=close,
        size=sizes,
        size_type="targetpercent",
        price=order_price,
        fees=0.0,
        fixed_fees=0.0,
        direction="longonly",
        slippage=0.0,
        init_cash=cfg["init_cash"],
        freq="1D",
        cash_sharing=True,
        call_seq="auto",
        size_granularity=np.nan,
        allow_partial=True,
        # vectorbt marks a repeated target of zero on an already-flat asset as
        # ``NoOpenPosition`` and extremely small target changes as
        # ``MinSizeNotReached``.  Both are expected no-ops in this frictionless
        # target-percent contract.
        raise_reject=False,
        **kwargs,
    )
    pf._qs_scheduled_targets = scheduled_targets
    pf._qs_execution_targets = execution_targets
    pf._qs_execution_mode = "fast"
    pf._qs_fast_contract = "factor_research"
    pf._qs_fast_tradeability = tradeability
    pf._qs_limit_check_mode = str(cfg["limit_check_mode"]).lower()
    _attach_performance_settings(pf, cfg)
    _attach_benchmark(
        pf,
        "ashare",
        cfg.get("benchmark_index"),
        str(signal_start.date()),
        str(market_end.date()),
        close.index,
    )
    print("[ashare/fast] 因子回测完成")
    return pf


# ============================================================
# 核心：单次回测
# ============================================================

def run_backtest(
    market: str,
    target_weights: pd.DataFrame,
    symbols: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    config: Optional[Dict] = None,
    **kwargs,
) -> vbt.Portfolio:
    """
    统一回测入口

    参数
    ----
    market : 'ashare' | 'us'
    target_weights : (n_days, n_assets) 目标权重矩阵
        - 正值 = 做多权重，负值 = 做空权重，0 = 不持
        - 例如 0.05 = 组合净值的 5%
    symbols : 标的列表（None = 使用 target_weights 的所有列）
    start / end : 日期范围（None = 使用 target_weights 的索引范围）
    config : 回测配置 dict（覆盖默认值）
    **kwargs : 传递给 Portfolio.from_orders 的额外参数

    返回
    ----
    vbt.Portfolio 对象

    用法
    ----
    >>> pf = run_backtest('ashare', target_weights, config={'init_cash': 5e6})
    >>> pf.stats()
    >>> pf.plot_value().show()
    """
    if config is None:
        config = {}
    cfg = {**DEFAULT_CONFIG, **config}
    market = market.lower()
    if market not in {"ashare", "us"}:
        raise ValueError(f"不支持的市场: {market}")

    execution_mode = str(cfg.get("execution_mode", "accurate")).lower()
    if execution_mode not in {"accurate", "fast"}:
        raise ValueError("execution_mode 仅支持 accurate 或 fast")
    if market == "ashare" and execution_mode == "accurate":
        return _run_ashare_accurate(
            target_weights=target_weights,
            symbols=symbols,
            start=start,
            end=end,
            cfg=cfg,
            config=config,
            kwargs=kwargs,
        )
    if market == "ashare" and execution_mode == "fast":
        return _run_ashare_factor_fast(
            target_weights=target_weights,
            symbols=symbols,
            start=start,
            end=end,
            cfg=cfg,
            config=config,
            kwargs=kwargs,
        )

    # The remaining vectorized path is retained for the US adapter.  A-share
    # fast mode above has its own explicit factor-research contract.
    target_weights = prepare_target_weights(
        target_weights,
        freq=cfg["freq"],
        execution_lag=cfg["execution_lag"],
    )
    if market == "ashare" and (target_weights.fillna(0.0) < 0.0).any(axis=None):
        raise ValueError("A 股现货回测不支持负权重；请移除空头或使用支持融券的专用模型")

    # 1. 确定日期范围
    if start is None:
        start = str(target_weights.index.min().date())
    if end is None:
        end = str(target_weights.index.max().date())

    # 2. 确定标的列表
    if symbols is None:
        symbols = list(target_weights.columns)

    # 3. 加载行情数据
    print(f"[{market}] 加载行情数据: {start} ~ {end}, {len(symbols)} 个标的")
    data = load_market_data(market, symbols=symbols, start=start, end=end)
    close = data["close"]
    order_price = _resolve_price(data, cfg["price_type"])

    # 对齐 close 和 target_weights 的 index/columns
    common_dates = close.index.intersection(target_weights.index)
    common_symbols = close.columns.intersection(target_weights.columns)
    missing_symbols = target_weights.columns.difference(close.columns)
    if cfg["strict_symbols"] and len(missing_symbols) > 0:
        preview = ", ".join(map(str, missing_symbols[:10]))
        raise ValueError(f"行情缺少 {len(missing_symbols)} 个目标标的: {preview}")
    close = close.loc[common_dates, common_symbols]
    order_price = order_price.reindex(index=common_dates, columns=common_symbols)
    target_weights = target_weights.loc[common_dates, common_symbols]

    if close.empty:
        raise ValueError("close 和 target_weights 无交集！请检查日期范围和标的列表")

    # 4. 应用市场约束
    if market == "ashare":
        is_suspend = data.get("is_suspend", pd.DataFrame(False, index=close.index, columns=close.columns))
        high_limit = data.get("high_limit", close * 1.10)
        low_limit = data.get("low_limit", close * 0.90)

        constraints = apply_ashare_constraints(
            close, target_weights,
            is_suspend=is_suspend.reindex_like(close) if not is_suspend.empty else pd.DataFrame(False, index=close.index, columns=close.columns),
            high_limit=high_limit.reindex_like(close) if not high_limit.empty else close * 1.10,
            low_limit=low_limit.reindex_like(close) if not low_limit.empty else close * 0.90,
            order_price=order_price,
        )

        size_granularity = config.get("size_granularity", 100)
        direction = "longonly"
        print(f"[{market}] 已应用目标级近似约束: 停牌 + 涨跌停 + 分方向费率")

    elif market == "us":
        constraints = apply_us_constraints(
            close,
            target_weights,
            is_adr=data.get("is_adr"),
            delist_info=data.get("delist_info"),
        )
        constraints["price"] = order_price
        size_granularity = config.get("size_granularity", np.nan)
        direction = "both"
        applied = ["SSR"]
        if data.get("is_adr") is not None:
            applied.append("ADR排除")
        if data.get("delist_info") is not None:
            applied.append("退市")
        print(f"[{market}] 已应用约束: {' + '.join(applied)}")

    # 5. 执行回测
    print(f"[{market}] 开始回测: {close.shape[0]} 天 × {close.shape[1]} 个标的")

    # 确保约束输出与 close 完全对齐（index + columns）
    for key in ["target_weights", "price", "fees", "fixed_fees"]:
        val = constraints.get(key)
        if val is not None and isinstance(val, pd.DataFrame):
            constraints[key] = val.reindex_like(close)

    pf = vbt.Portfolio.from_orders(
        close=close,
        size=constraints["target_weights"],
        size_type="targetpercent",
        price=constraints["price"],
        fees=config["fees"] if "fees" in config else constraints.get("fees", 0.001),
        fixed_fees=config["fixed_fees"] if "fixed_fees" in config else constraints.get("fixed_fees", 0.0),
        direction=direction,
        slippage=cfg["slippage"],
        init_cash=cfg["init_cash"],
        freq=cfg["freq"],
        cash_sharing=cfg["cash_sharing"],
        call_seq=cfg["call_seq"],
        size_granularity=size_granularity,
        allow_partial=cfg["allow_partial"],
        **kwargs,
    )

    _attach_performance_settings(pf, cfg)
    _attach_benchmark(pf, market, cfg.get("benchmark_index"), start, end, close.index)

    print(f"[{market}] 回测完成")
    return pf


# ============================================================
# 批量回测（多组参数）
# ============================================================

def run_backtest_grid(
    market: str,
    target_weights_list: List[pd.DataFrame],
    labels: List[str],
    **kwargs,
) -> Dict[str, vbt.Portfolio]:
    """
    批量回测（多组目标权重对比）

    返回 {label: Portfolio} 字典
    """
    if len(target_weights_list) != len(labels):
        raise ValueError("target_weights_list 与 labels 长度必须一致")
    results = {}
    for label, tw in zip(labels, target_weights_list):
        print(f"\n{'='*50}\n  批量回测: {label}\n{'='*50}")
        results[label] = run_backtest(market, tw, **kwargs)
    return results


# ============================================================
# 结果报告
# ============================================================

def portfolio_report(pf: vbt.Portfolio) -> pd.Series:
    """Generate a frozen trading-day performance report.

    vectorbt defaults to a 365-day annualization because ``freq='1D'`` means a
    calendar-day timedelta. A-share portfolios contain trading-day rows, so
    the framework explicitly uses 252 observations per year.
    """
    benchmark_returns = getattr(pf, "_qs_benchmark_returns", None)
    year_days = int(getattr(pf, "_qs_performance_year_days", 252))
    annual_risk_free = float(getattr(pf, "_qs_risk_free_rate", 0.0))
    if year_days <= 0:
        raise ValueError("performance_year_days 必须为正整数")
    if annual_risk_free <= -1.0 or not np.isfinite(annual_risk_free):
        raise ValueError("risk_free_rate 必须是大于 -1 的有限年化收益率")
    settings = {
        "year_freq": f"{year_days} days",
        "risk_free": (1.0 + annual_risk_free) ** (1.0 / year_days) - 1.0,
    }
    if benchmark_returns is not None:
        settings["benchmark_rets"] = benchmark_returns
    stats = pf.stats(settings=settings)

    # Lightweight fake portfolios used by downstream callers may expose only
    # stats. Keep that protocol backward compatible.
    if not hasattr(pf, "returns"):
        return stats
    returns = pf.returns()
    if isinstance(returns, pd.DataFrame):
        if returns.shape[1] != 1:
            returns = returns.sum(axis=1)
        else:
            returns = returns.iloc[:, 0]
    returns = pd.Series(returns, dtype=float).dropna()
    if returns.empty:
        return stats

    daily_rf = settings["risk_free"]
    excess = returns - daily_rf
    sample_std = excess.std(ddof=1)
    stats["Annualized Return [%]"] = (
        (1.0 + returns).prod() ** (year_days / len(returns)) - 1.0
    ) * 100.0
    stats["Annualized Volatility [%]"] = (
        returns.std(ddof=1) * np.sqrt(year_days) * 100.0
    )
    stats["Sharpe Ratio"] = (
        np.nan
        if pd.isna(sample_std)
        else np.inf
        if sample_std == 0.0 and excess.mean() > 0.0
        else np.nan
        if sample_std == 0.0
        else excess.mean() / sample_std * np.sqrt(year_days)
    )

    if benchmark_returns is not None:
        benchmark = pd.Series(benchmark_returns, dtype=float).reindex(
            returns.index
        )
        paired = pd.concat(
            [returns.rename("strategy"), benchmark.rename("benchmark")],
            axis=1,
        ).dropna()
        if not paired.empty:
            active = paired["strategy"] - paired["benchmark"]
            tracking_error = active.std(ddof=1)
            stats["Tracking Error [%]"] = (
                tracking_error * np.sqrt(year_days) * 100.0
            )
            stats["Information Ratio"] = (
                np.nan
                if pd.isna(tracking_error) or tracking_error == 0.0
                else active.mean() / tracking_error * np.sqrt(year_days)
            )
            relative_returns = (
                (1.0 + paired["strategy"]) / (1.0 + paired["benchmark"]) - 1.0
            )
            stats["Annualized Active Return [%]"] = (
                (1.0 + relative_returns).prod()
                ** (year_days / len(relative_returns))
                - 1.0
            ) * 100.0
    return stats


def compare_reports(results: Dict[str, vbt.Portfolio]) -> pd.DataFrame:
    """多组回测绩效对比"""
    reports = {}
    for label, pf in results.items():
        reports[label] = portfolio_report(pf)
    return pd.DataFrame(reports).T
