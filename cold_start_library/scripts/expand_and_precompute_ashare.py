#!/usr/bin/env python3
"""大幅扩充 A 股冷启动库：补齐稀有/缺失算子，加入估值字段因子，并增量预计算 IC。

数据：本地 COS 镜像 ~/quant_projects/data/a_share/lqtp_data
  - StockDailyBar（价量）
  - StockValuationDaily（pe/pb/turnover/market_cap…）

输出更新：
  data/ashare/backend_v9_core.yaml
  data/ashare/backend_v9_metrics.jsonl
  data/ashare/backend_v9_precompute_summary.json
  data/ashare/expand_generated.json  # 新增因子底稿
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
FE = ROOT.parent / "factor_engine"
DA = ROOT.parent / "data_access"
for p in (str(SRC), str(FE), str(DA)):
    if p not in sys.path:
        sys.path.insert(0, p)

_LOCAL_ASHARE = Path.home() / "quant_projects" / "data" / "a_share" / "lqtp_data"
RET = "(safe_div_null(close, pre_close) - 1.0)"
LOGV = "log(add(volume, 1.0))"

# 覆盖短→长；略扩窗口以抬高种子密度
W_ALL = (3, 5, 7, 10, 14, 20, 30, 40, 60, 90, 120, 180, 250, 500)
W_MID = (5, 10, 20, 40, 60, 120, 250)
PAIRS = (
    (5, 20), (5, 60), (5, 120), (10, 40), (10, 60), (10, 120),
    (20, 60), (20, 120), (20, 250), (40, 120), (60, 250), (20, 500),
)


def _add(
    bag: list[dict[str, Any]],
    *,
    fid: str,
    expr: str,
    topic: str,
    description: str,
    family: str,
) -> None:
    bag.append(
        {
            "factor_id": fid,
            "expr": expr,
            "topic": topic,
            "description": description,
            "family": family,
            "source_library": "ashare_expand_v1",
        }
    )


def generate_expand_factors() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    n = 0

    def nid(prefix: str) -> str:
        nonlocal n
        n += 1
        return f"pv_exp_{prefix}_{n:05d}"

    # ---- missing / alias ops ----
    for w in W_ALL:
        _add(out, fid=nid("ema"), expr=f"rank(ema(close, {w}))", topic="V9/expand_ema", description=f"ema({w}) 截面排名", family="expand_ema")
        _add(out, fid=nid("ema"), expr=f"rank(ts_ema({RET}, {w}))", topic="V9/expand_ema", description=f"收益 ts_ema({w})", family="expand_ema")
        _add(out, fid=nid("pct"), expr=f"rank(pct_change(close, {w}))", topic="V9/expand_pct", description=f"pct_change({w})", family="expand_pct")
        _add(out, fid=nid("ret"), expr=f"rank(ts_sum(returns(close), {w}))", topic="V9/expand_returns", description=f"returns 累计{w}", family="expand_returns")
        _add(out, fid=nid("mm"), expr=f"rank(safe_div_null(subtract(close, min(ts_min(close, {w}), open)), add(max(ts_max(close, {w}), high), 0.001)))", topic="V9/expand_minmax", description=f"max/min 位置{w}", family="expand_minmax")

    for w in W_MID:
        _add(out, fid=nid("ewm"), expr=f"rank(ewm(close, {w}, 0.1))", topic="V9/expand_ewm", description=f"ewm alpha0.1 w={w}", family="expand_ewm")
        _add(out, fid=nid("ewm"), expr=f"rank(subtract(ewm(close, {w}, 0.2), ewm(close, {w}, 0.05)))", topic="V9/expand_ewm", description=f"双 ewm 差{w}", family="expand_ewm")

    for w1, w2 in PAIRS:
        _add(
            out,
            fid=nid("logic"),
            expr=f"if_else(gt(ts_mean(close, {w1}), ts_mean(close, {w2})), rank(pct_change(volume, {w1})), neg(rank(pct_change(volume, {w1}))))",
            topic="V9/expand_logic",
            description=f"if_else 短{w1}>长{w2} 量能门控",
            family="expand_logic",
        )
        _add(
            out,
            fid=nid("logic"),
            expr=f"where(or_(gt(close, ts_mean(close, {w2})), gt(volume, ts_mean(volume, {w1}))), rank(ts_delta(close, {w1})), rank(ts_delta(close, {w2})))",
            topic="V9/expand_logic",
            description=f"where+or_ 价量门控 {w1}/{w2}",
            family="expand_logic",
        )
        _add(
            out,
            fid=nid("logic"),
            expr=f"rank(if_else(and_(gt(close, open), eq(sign(ts_delta(close, 1)), 1.0)), ts_rank(volume, {w1}), ts_rank(volume, {w2})))",
            topic="V9/expand_logic",
            description=f"and_/eq 阳线量能 {w1}/{w2}",
            family="expand_logic",
        )

    # ---- technical indicators dense ----
    for n_ in (7, 14, 20, 28):
        _add(out, fid=nid("rsi"), expr=f"rank(RSI_WILDER(close, {n_}))", topic="V9/expand_tech", description=f"RSI({n_})", family="expand_tech")
        _add(out, fid=nid("adx"), expr=f"rank(ADX(high, low, close, {n_}))", topic="V9/expand_tech", description=f"ADX({n_})", family="expand_tech")
        _add(out, fid=nid("atr"), expr=f"rank(safe_div_null(ATR_WILDER(high, low, close, {n_}), close))", topic="V9/expand_tech", description=f"ATR/close({n_})", family="expand_tech")
        _add(out, fid=nid("tr"), expr=f"rank(safe_div_null(true_range(high, low, close), close))", topic="V9/expand_tech", description="true_range/close", family="expand_tech")
    for fast, slow, sig in ((12, 26, 9), (8, 17, 9), (5, 35, 5), (12, 26, 5)):
        _add(out, fid=nid("macd"), expr=f"rank(MACD_hist(close, {fast}, {slow}, {sig}))", topic="V9/expand_tech", description=f"MACD_hist({fast},{slow},{sig})", family="expand_tech")
        _add(out, fid=nid("macd"), expr=f"rank(subtract(MACD_line(close, {fast}, {slow}, {sig}), MACD_signal(close, {fast}, {slow}, {sig})))", topic="V9/expand_tech", description=f"MACD line-signal", family="expand_tech")

    # ---- rare ts ops dense + multi-window ----
    for w in W_ALL:
        _add(out, fid=nid("risk"), expr=f"rank(ts_sharpe({RET}, {w}))", topic="V9/expand_risk", description=f"ts_sharpe({w})", family="expand_risk")
        _add(out, fid=nid("risk"), expr=f"rank(ts_max_drawdown(close, {w}))", topic="V9/expand_risk", description=f"max_drawdown({w})", family="expand_risk")
        _add(out, fid=nid("risk"), expr=f"rank(ts_skew({RET}, {w}))", topic="V9/expand_risk", description=f"ts_skew({w})", family="expand_risk")
        _add(out, fid=nid("risk"), expr=f"rank(ts_kurt({RET}, {w}))", topic="V9/expand_risk", description=f"ts_kurt({w})", family="expand_risk")
        _add(out, fid=nid("stat"), expr=f"rank(ts_zscore({RET}, {w}))", topic="V9/expand_stat", description=f"ts_zscore ret({w})", family="expand_stat")
        _add(out, fid=nid("stat"), expr=f"rank(ts_mad({RET}, {w}))", topic="V9/expand_stat", description=f"ts_mad({w})", family="expand_stat")
        _add(out, fid=nid("stat"), expr=f"rank(ts_median(close, {w}))", topic="V9/expand_stat", description=f"ts_median({w})", family="expand_stat")
        _add(out, fid=nid("stat"), expr=f"rank(ts_time_slope(log(close), {w}))", topic="V9/expand_stat", description=f"time_slope({w})", family="expand_stat")
        _add(out, fid=nid("stat"), expr=f"rank(ts_trend_tstat(close, {w}))", topic="V9/expand_stat", description=f"trend_tstat({w})", family="expand_stat")
        _add(out, fid=nid("stat"), expr=f"rank(ts_tail_mean({RET}, {w}))", topic="V9/expand_stat", description=f"tail_mean({w})", family="expand_stat")
        _add(out, fid=nid("stat"), expr=f"rank(ts_decay_linear({RET}, {w}))", topic="V9/expand_stat", description=f"decay_linear({w})", family="expand_stat")
        _add(out, fid=nid("stat"), expr=f"rank(ts_product(add(1.0, clip({RET}, -0.2, 0.2)), {w}))", topic="V9/expand_stat", description=f"ts_product({w})", family="expand_stat")
        _add(out, fid=nid("arg"), expr=f"rank(safe_div_null(ts_argmax(close, {w}), {w}.0))", topic="V9/expand_arg", description=f"argmax/close({w})", family="expand_arg")
        _add(out, fid=nid("arg"), expr=f"rank(safe_div_null(ts_argmin(close, {w}), {w}.0))", topic="V9/expand_arg", description=f"argmin/close({w})", family="expand_arg")
        _add(out, fid=nid("ac"), expr=f"rank(ts_autocorr({RET}, {w}, 1))", topic="V9/expand_ac", description=f"autocorr({w})", family="expand_ac")
        _add(out, fid=nid("q"), expr=f"rank(ts_quantile({RET}, {w}, 0.1))", topic="V9/expand_q", description=f"q10({w})", family="expand_q")
        _add(out, fid=nid("q"), expr=f"rank(ts_quantile({RET}, {w}, 0.9))", topic="V9/expand_q", description=f"q90({w})", family="expand_q")

    for w in W_MID:
        for k in (2, 3, 5):
            if k >= w:
                continue
            _add(out, fid=nid("topk"), expr=f"rank(ts_topk_mean({RET}, {w}, {k}))", topic="V9/expand_topk", description=f"topk_mean({w},{k})", family="expand_topk")
            _add(out, fid=nid("topk"), expr=f"rank(ts_topk_std({RET}, {w}, {k}))", topic="V9/expand_topk", description=f"topk_std({w},{k})", family="expand_topk")
            _add(out, fid=nid("topk"), expr=f"rank(ts_topk_sum({RET}, {w}, {k}))", topic="V9/expand_topk", description=f"topk_sum({w},{k})", family="expand_topk")
            _add(out, fid=nid("bot"), expr=f"rank(ts_bottomk_mean({RET}, {w}, {k}))", topic="V9/expand_bot", description=f"bottomk_mean({w},{k})", family="expand_bot")
            _add(out, fid=nid("bot"), expr=f"rank(ts_bottomk_std({RET}, {w}, {k}))", topic="V9/expand_bot", description=f"bottomk_std({w},{k})", family="expand_bot")

    for w in W_ALL:
        _add(out, fid=nid("cond"), expr=f"rank(ts_count_if(gt(close, ts_delay(close, 1)), {w}))", topic="V9/expand_cond", description=f"upday count({w})", family="expand_cond")
        _add(out, fid=nid("cond"), expr=f"rank(ts_sum_if(gt(close, open), volume, {w}))", topic="V9/expand_cond", description=f"sum_if 阳线量({w})", family="expand_cond")
        _add(out, fid=nid("cond"), expr=f"rank(ts_days_since(gt(close, ts_mean(close, {w}))))", topic="V9/expand_cond", description=f"days_since>ma({w})", family="expand_cond")
        _add(out, fid=nid("cond"), expr=f"rank(ts_true_streak(gt(close, open), {w}))", topic="V9/expand_cond", description=f"true_streak({w})", family="expand_cond")

    for w in W_MID:
        _add(out, fid=nid("corr"), expr=f"rank(ts_corr(ts_pct(close, 1), ts_pct(volume, 1), {w}))", topic="V9/expand_corr", description=f"价量 corr({w})", family="expand_corr")
        _add(out, fid=nid("corr"), expr=f"rank(ts_cov(ts_pct(close, 1), ts_pct(volume, 1), {w}))", topic="V9/expand_corr", description=f"价量 cov({w})", family="expand_corr")
        _add(out, fid=nid("corr"), expr=f"rank(ts_beta(ts_pct(close, 1), ts_pct(volume, 1), {w}))", topic="V9/expand_corr", description=f"beta({w})", family="expand_corr")
        _add(out, fid=nid("corr"), expr=f"rank(ts_partial_corr(close, volume, ts_mean(volume, {w}), {w}))", topic="V9/expand_corr", description=f"partial_corr({w})", family="expand_corr")
        _add(out, fid=nid("reg"), expr=f"rank(ts_regression_slope(close, {LOGV}, {w}))", topic="V9/expand_reg", description=f"reg_slope({w})", family="expand_reg")
        _add(out, fid=nid("reg"), expr=f"rank(ts_regression_r2(close, {LOGV}, {w}))", topic="V9/expand_reg", description=f"reg_r2({w})", family="expand_reg")
        _add(out, fid=nid("reg"), expr=f"cs_mad_zscore(ts_regression_resid(close, {LOGV}, {w}))", topic="V9/expand_reg", description=f"reg_resid({w})", family="expand_reg")

    # ---- multi-window spreads ----
    for w1, w2 in PAIRS:
        _add(out, fid=nid("mw"), expr=f"rank(subtract(ts_mean(close, {w1}), ts_mean(close, {w2})))", topic="V9/expand_mw", description=f"均线差 {w1}-{w2}", family="expand_mw")
        _add(out, fid=nid("mw"), expr=f"rank(safe_div_null(ts_std(close, {w1}), ts_mean(close, {w2})))", topic="V9/expand_mw", description=f"短波动/长均 {w1}/{w2}", family="expand_mw")
        _add(out, fid=nid("mw"), expr=f"rank(subtract(ts_rank(close, {w1}), ts_rank(close, {w2})))", topic="V9/expand_mw", description=f"双 ts_rank {w1}-{w2}", family="expand_mw")
        _add(out, fid=nid("mw"), expr=f"rank(subtract(RSI_WILDER(close, {w1 if w1!=5 else 7}), RSI_WILDER(close, {14 if w2<30 else 28})))", topic="V9/expand_mw", description="双 RSI 差", family="expand_mw")
        _add(out, fid=nid("mw"), expr=f"rank(subtract(ts_sharpe({RET}, {w1}), ts_sharpe({RET}, {w2})))", topic="V9/expand_mw", description=f"双 sharpe {w1}-{w2}", family="expand_mw")

    # ---- cross-section / transform rare ----
    for w in W_MID:
        _add(out, fid=nid("cs"), expr=f"cs_mad_zscore(ts_delta(close, {w}))", topic="V9/expand_cs", description=f"cs_mad_zscore delta{w}", family="expand_cs")
        _add(out, fid=nid("cs"), expr=f"cs_pct_rank(ts_pct(close, {w}))", topic="V9/expand_cs", description=f"cs_pct_rank pct{w}", family="expand_cs")
        _add(out, fid=nid("cs"), expr=f"cs_rank_gaussian(ts_mean({RET}, {w}))", topic="V9/expand_cs", description=f"cs_rank_gaussian{w}", family="expand_cs")
        _add(out, fid=nid("cs"), expr=f"cs_demean(ts_pct(close, {w}))", topic="V9/expand_cs", description=f"cs_demean{w}", family="expand_cs")
        _add(out, fid=nid("cs"), expr=f"safe_div_null(ts_pct(close, {w}), add(cs_std(ts_pct(close, {w})), 0.0001))", topic="V9/expand_cs", description=f"x / cs_std{w}", family="expand_cs")
        _add(out, fid=nid("cs"), expr=f"subtract(ts_pct(close, {w}), cs_mean(ts_pct(close, {w})))", topic="V9/expand_cs", description=f"minus cs_mean{w}", family="expand_cs")
        _add(out, fid=nid("cs"), expr=f"rank(safe_div_null(ts_pct(close, {w}), add(cs_mad(ts_pct(close, {w})), 0.0001)))", topic="V9/expand_cs", description=f"/cs_mad{w}", family="expand_cs")
        _add(out, fid=nid("cs"), expr=f"cs_weighted_demean(ts_pct(close, {w}), volume)", topic="V9/expand_cs", description=f"cs_w_demean{w}", family="expand_cs")
        _add(out, fid=nid("cs"), expr=f"zscore(ts_sum({RET}, {w}))", topic="V9/expand_cs", description=f"zscore sum{w}", family="expand_cs")
        _add(out, fid=nid("cs"), expr=f"winsorize(ts_pct(close, {w}), 0.01)", topic="V9/expand_cs", description=f"winsorize{w}", family="expand_cs")
        _add(out, fid=nid("cs"), expr=f"normalize(ts_pct(close, {w}))", topic="V9/expand_cs", description=f"normalize{w}", family="expand_cs")
        _add(out, fid=nid("cs"), expr=f"scale(ts_pct(close, {w}), 1.0)", topic="V9/expand_cs", description=f"scale{w}", family="expand_cs")
        _add(out, fid=nid("cs"), expr=f"rank(cs_count(is_finite(ts_pct(close, {w}))))", topic="V9/expand_cs", description="cs_count/is_finite", family="expand_cs")
        _add(out, fid=nid("cs"), expr=f"tanh(ts_corr({RET}, ts_log_return(add(volume, 1.0), 1), {w}))", topic="V9/expand_xf", description=f"tanh corr{w}", family="expand_xf")
        _add(out, fid=nid("cs"), expr=f"signed_sqrt(ts_corr({RET}, ts_pct(volume, 1), {w}))", topic="V9/expand_xf", description=f"signed_sqrt corr{w}", family="expand_xf")
        _add(out, fid=nid("cs"), expr=f"signed_log(ts_delta({LOGV}, {w}))", topic="V9/expand_xf", description=f"signed_log dvol{w}", family="expand_xf")
        _add(out, fid=nid("cs"), expr=f"log_abs(ts_zscore({RET}, {w}))", topic="V9/expand_xf", description=f"log_abs z{w}", family="expand_xf")
        _add(out, fid=nid("cs"), expr=f"rank(exp(neg(abs(ts_zscore({RET}, {w})))))", topic="V9/expand_xf", description=f"exp -|z|{w}", family="expand_xf")
        _add(out, fid=nid("cs"), expr=f"rank(inverse(add(abs(ts_pct(close, {w})), 0.0001)))", topic="V9/expand_xf", description=f"inverse abs pct{w}", family="expand_xf")
        _add(out, fid=nid("cs"), expr=f"rank(sqrt(abs(ts_pct(close, {w}))))", topic="V9/expand_xf", description=f"sqrt abs pct{w}", family="expand_xf")
        _add(out, fid=nid("cs"), expr=f"rank(power(add(abs({RET}), 0.0001), 0.5))", topic="V9/expand_xf", description="power 0.5 |ret|", family="expand_xf")
        _add(out, fid=nid("cs"), expr=f"fillna_const(ts_pct(close, {w}), 0.0)", topic="V9/expand_xf", description=f"fillna pct{w}", family="expand_xf")
        _add(out, fid=nid("cs"), expr=f"clip(ts_pct(close, {w}), -0.2, 0.2)", topic="V9/expand_xf", description=f"clip pct{w}", family="expand_xf")
        _add(out, fid=nid("cs"), expr=f"coalesce(ts_pct(close, {w}), 0.0)", topic="V9/expand_xf", description=f"coalesce pct{w}", family="expand_xf")
        _add(out, fid=nid("cs"), expr=f"rank(maximum(ts_pct(close, {w}), ts_pct(open, {w})))", topic="V9/expand_xf", description=f"maximum pct{w}", family="expand_xf")
        _add(out, fid=nid("cs"), expr=f"rank(minimum(ts_pct(close, {w}), ts_pct(vwap, {w})))", topic="V9/expand_xf", description=f"minimum pct{w}", family="expand_xf")
        _add(out, fid=nid("cs"), expr=f"where(ne(volume, 0.0), safe_div_null(amount, volume), vwap)", topic="V9/expand_xf", description="where ne amount/vol", family="expand_xf")
        _add(out, fid=nid("cs"), expr=f"rank(if_else(and_(ge(close, ts_mean(close, {w})), le(open, close)), ts_rank(volume, {w}), neg(ts_rank(volume, {w}))))", topic="V9/expand_logic", description=f"ge/le 阳线量能门控{w}", family="expand_logic")
        _add(out, fid=nid("cs"), expr=f"rank(divide(ts_sum({RET}, {w}), add(ts_std({RET}, {w}), 0.0001)))", topic="V9/expand_stat", description=f"divide sum/std{w}", family="expand_stat")
        _add(out, fid=nid("cs"), expr=f"rank(safe_div_null(cs_sum(ts_pct(close, {w})), add(cs_count(is_finite(ts_pct(close, {w}))), 1.0)))", topic="V9/expand_cs", description=f"cs_sum/cs_count{w}", family="expand_cs")
        _add(out, fid=nid("cs"), expr=f"subtract(ts_pct(close, {w}), cs_mean(ts_pct(vwap, {w})))", topic="V9/expand_cs", description=f"ret - cs_mean vwap{w}", family="expand_cs")
        _add(out, fid=nid("pv"), expr=f"rank(safe_div_null(subtract(high, low), add(pre_close, 0.001)))", topic="V9/expand_pv", description="amplitude/pre_close", family="expand_pv")
        _add(out, fid=nid("pv"), expr=f"rank(ts_corr(ts_pct(close, 1), ts_pct(amount, 1), {w}))", topic="V9/expand_pv", description=f"close-amount corr{w}", family="expand_pv")
        _add(out, fid=nid("pv"), expr=f"rank(ts_mean(safe_div_null(subtract(close, vwap), vwap), {w}))", topic="V9/expand_pv", description=f"vwap 偏离 ma{w}", family="expand_pv")
        _add(out, fid=nid("pv"), expr=f"cs_wls_resid(ts_pct(close, {w}), log(add(amount, 1.0)), volume)", topic="V9/expand_cs", description=f"cs_wls_resid{w}", family="expand_cs")
        _add(out, fid=nid("pv"), expr=f"cs_multi_resid(ts_pct(close, {w}), log(add(volume, 1.0)), log(add(amount, 1.0)))", topic="V9/expand_cs", description=f"cs_multi_resid{w}", family="expand_cs")

    # ---- valuation (COS StockValuationDaily via FE composite) ----
    for w in W_ALL:
        _add(out, fid=nid("val"), expr=f"rank(inverse(add(pe, 0.001)))", topic="V9/expand_val", description="earnings yield ~1/pe", family="expand_val")
        _add(out, fid=nid("val"), expr=f"rank(inverse(add(pb, 0.001)))", topic="V9/expand_val", description="1/pb", family="expand_val")
        _add(out, fid=nid("val"), expr=f"rank(turnover_ratio)", topic="V9/expand_val", description="turnover_ratio", family="expand_val")
        _add(out, fid=nid("val"), expr=f"cs_mad_zscore(log(add(market_cap, 1.0)))", topic="V9/expand_val", description="log mcap z", family="expand_val")
        _add(out, fid=nid("val"), expr=f"rank(ts_pct(pe, {w}))", topic="V9/expand_val", description=f"pe pct{w}", family="expand_val")
        _add(out, fid=nid("val"), expr=f"rank(ts_pct(pb, {w}))", topic="V9/expand_val", description=f"pb pct{w}", family="expand_val")
        _add(out, fid=nid("val"), expr=f"rank(ts_zscore(turnover_ratio, {w}))", topic="V9/expand_val", description=f"turnover z{w}", family="expand_val")
        _add(out, fid=nid("val"), expr=f"rank(ts_mean(turnover_ratio, {w}))", topic="V9/expand_val", description=f"turnover ma{w}", family="expand_val")
        _add(out, fid=nid("val"), expr=f"rank(multiply(ts_pct(close, {w}), turnover_ratio))", topic="V9/expand_val", description=f"ret*turnover{w}", family="expand_val")
        _add(out, fid=nid("val"), expr=f"rank(safe_div_null(volume, add(circulating_market_cap, 1.0)))", topic="V9/expand_val", description="vol/circ_mcap", family="expand_val")
        _add(out, fid=nid("val"), expr=f"cs_mad_zscore(subtract(ts_pct(close, {w}), ts_pct(pb, {w})))", topic="V9/expand_val", description=f"price vs pb {w}", family="expand_val")
        _add(out, fid=nid("val"), expr=f"rank(ts_corr(ts_pct(close, 1), ts_pct(turnover_ratio, 1), {w}))", topic="V9/expand_val", description=f"ret-turnover corr{w}", family="expand_val")
        _add(out, fid=nid("val"), expr=f"rank(ts_std(inverse(add(pe, 0.001)), {w}))", topic="V9/expand_val", description=f"ey vol{w}", family="expand_val")
        _add(out, fid=nid("val"), expr=f"rank(ts_delta(log(add(market_cap, 1.0)), {w}))", topic="V9/expand_val", description=f"dlog mcap{w}", family="expand_val")
        _add(out, fid=nid("val"), expr=f"zscore(safe_div_null(ts_mean(turnover_ratio, {w}), add(ts_std(turnover_ratio, {w}), 0.0001)))", topic="V9/expand_val", description=f"turnover sharpe-like{w}", family="expand_val")

    for w1, w2 in PAIRS[:6]:
        _add(out, fid=nid("val"), expr=f"rank(subtract(ts_mean(turnover_ratio, {w1}), ts_mean(turnover_ratio, {w2})))", topic="V9/expand_val", description=f"turnover 短长差{w1}-{w2}", family="expand_val")
        _add(out, fid=nid("val"), expr=f"if_else(gt(pe, ts_mean(pe, {w2})), rank(ts_pct(close, {w1})), neg(rank(ts_pct(close, {w1}))))", topic="V9/expand_val", description=f"pe 门控动量{w1}/{w2}", family="expand_val")

    # dedupe by expr
    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for item in out:
        e = " ".join(item["expr"].split())
        if e in seen:
            continue
        seen.add(e)
        item["expr"] = e
        uniq.append(item)
    return uniq


class _RetDecimalProxy:
    def __init__(self, inner: Any, scale: float = 1.0 / 10000.0) -> None:
        self._inner = inner
        self._scale = float(scale)

    def load_column(self, name: str):
        series = self._inner.load_column(name)
        if name in {"ret", "Return"}:
            return series * self._scale
        return series

    def load_columns(self, names: list[str]):
        out = self._inner.load_columns(names)
        for name in names:
            if name in {"ret", "Return"} and name in out:
                out[name] = out[name] * self._scale
        return out

    def __getattr__(self, item: str):
        return getattr(self._inner, item)


def build_engine(start: str, end: str, *, use_valuation: bool = True):
    """本地 parquet 直连（比 data_access DESCRIBE 快很多），估值 asof 对齐。"""
    from backend.factory import build_backend
    from runtime.engine import FactorEngine
    from storage.factory import build_data_source
    from api.mining_integration import _VALUATION_FIELD_ALIASES

    os.environ.setdefault("ASHARE_PARQUET_ROOT", str(_LOCAL_ASHARE))
    pv_fields = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "vwap": "Vwap",
        "amount": "Amount",
        "ret": "Return",
        "pre_close": "PreClose",
        "preclose": "PreClose",
        "factor": "Factor",
    }
    pv_cfg: dict[str, Any] = {
        "type": "parquet",
        "root": str(_LOCAL_ASHARE / "StockDailyBar"),
        "timestamp_col": "TradeDate",
        "instrument_col": "Symbol",
        "fields": pv_fields,
        "start_date": start,
        "end_date": end,
    }
    if not use_valuation:
        raw_ds = build_data_source(pv_cfg)
    else:
        val_cfg: dict[str, Any] = {
            "type": "parquet",
            "root": str(_LOCAL_ASHARE / "StockValuationDaily"),
            "timestamp_col": "TradeDate",
            "instrument_col": "Symbol",
            "fields": {
                "pe": "PeRatio",
                "pb": "PbRatio",
                "turnover_ratio": "TurnoverRatio",
                "market_cap": "MarketCap",
                "circulating_market_cap": "CirculatingMarketCap",
            },
            "start_date": start,
            "end_date": end,
        }
        cfg = {
            "type": "composite",
            "anchor": "pv",
            "anchor_column": "close",
            "sources": {"pv": pv_cfg, "valuation": val_cfg},
            "joins": {"valuation": "asof_backward"},
            "aliases": dict(_VALUATION_FIELD_ALIASES),
        }
        raw_ds = build_data_source(cfg)
    ds = _RetDecimalProxy(raw_ds)
    eng = FactorEngine(backend=build_backend("pandas"), data_source=ds, run_mode="research")
    return eng, ds


def _panel_pearson_ic(f_mat: np.ndarray, y_mat: np.ndarray, *, min_cs: int = 30) -> tuple[float | None, float | None]:
    ics: list[float] = []
    n_days = min(f_mat.shape[0], y_mat.shape[0])
    for i in range(n_days):
        a = f_mat[i]
        b = y_mat[i]
        mask = np.isfinite(a) & np.isfinite(b)
        if int(mask.sum()) < min_cs:
            continue
        aa = a[mask] - a[mask].mean()
        bb = b[mask] - b[mask].mean()
        da = float(np.dot(aa, aa))
        db = float(np.dot(bb, bb))
        if da < 1e-18 or db < 1e-18:
            continue
        c = float(np.dot(aa, bb) / math.sqrt(da * db))
        if np.isfinite(c):
            ics.append(c)
    if not ics:
        return None, None
    arr = np.asarray(ics, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
    icir = float(mean / std) if std and np.isfinite(std) and std > 1e-12 else None
    return mean, icir


def make_fwd_matrix(ds, label_days: int):
    vwap = ds.load_column("vwap")
    fwd = vwap.groupby(level=1).shift(-label_days) / vwap - 1.0
    wide = fwd.unstack(level=1)
    return wide.index, wide.columns, wide.to_numpy(dtype=float)


def precompute_one(eng, formula: str, *, dates, stocks, y_mat, label_days: int) -> dict[str, Any]:
    from api.dsl_parser import parse_factor

    t0 = time.time()
    fac = parse_factor(formula, name="cs", surface="compat")
    result = eng.run(fac)["result"]
    s = pd.to_numeric(result, errors="coerce")
    vals = s.to_numpy(dtype=float)
    finite = int(np.isfinite(vals).sum())
    total = int(len(vals))
    wide = s.unstack(level=1).reindex(index=dates, columns=stocks)
    ic, icir = _panel_pearson_ic(wide.to_numpy(dtype=float), y_mat)
    uniq = int(np.unique(vals[np.isfinite(vals)]).size) if finite else 0
    return {
        "status": "pass",
        "finite_ratio": finite / total if total else 0.0,
        "n_finite": finite,
        "n_total": total,
        "n_unique": uniq,
        "ic": ic,
        "icir": icir,
        "abs_ic": abs(ic) if ic is not None else None,
        "label_days": label_days,
        "elapsed_sec": round(time.time() - t0, 4),
    }


def extract_ops(expr: str) -> list[str]:
    from cold_start_library.runtime.dsl import extract_dsl_operator_names

    return list(extract_dsl_operator_names(expr))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-07-01")
    ap.add_argument("--end", default="2021-09-30")
    ap.add_argument("--label-days", type=int, default=20)
    ap.add_argument("--min-finite-ratio", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-precompute", action="store_true")
    ap.add_argument("--no-valuation", action="store_true")
    ap.add_argument("--generate-only", action="store_true")
    args = ap.parse_args()

    from api.dsl_parser import parse_expr

    gen_path = ROOT / "data" / "ashare" / "expand_generated.json"
    yaml_path = ROOT / "data" / "ashare" / "backend_v9_core.yaml"
    metrics_path = ROOT / "data" / "ashare" / "backend_v9_metrics.jsonl"
    summary_path = ROOT / "data" / "ashare" / "backend_v9_precompute_summary.json"

    factors = generate_expand_factors()
    if args.limit > 0:
        factors = factors[: args.limit]
    print(f"[gen] unique expand factors: {len(factors)}", flush=True)

    parse_ok: list[dict[str, Any]] = []
    parse_fail: list[dict[str, Any]] = []
    for item in factors:
        try:
            parse_expr(item["expr"], surface="compat")
            item = dict(item)
            item["operators"] = extract_ops(item["expr"])
            parse_ok.append(item)
        except Exception as exc:
            parse_fail.append({"factor_id": item["factor_id"], "expr": item["expr"], "error": f"{type(exc).__name__}: {exc}"})
    print(f"[parse] ok={len(parse_ok)} fail={len(parse_fail)}", flush=True)
    if parse_fail[:5]:
        print("[parse] fail samples:", json.dumps(parse_fail[:5], ensure_ascii=False), flush=True)

    gen_path.parent.mkdir(parents=True, exist_ok=True)
    gen_path.write_text(json.dumps({"count": len(parse_ok), "items": parse_ok, "parse_fail": parse_fail}, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.generate_only:
        print(f"[done] generate-only -> {gen_path}")
        return 0 if parse_ok else 1

    # load existing yaml
    existing = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    old_entries: list[dict[str, Any]] = list(existing.get("entries") or [])
    old_exprs = {" ".join(str(e.get("expr") or "").split()) for e in old_entries}
    old_ids = {str(e.get("factor_id")) for e in old_entries}

    new_items = [x for x in parse_ok if x["expr"] not in old_exprs and x["factor_id"] not in old_ids]
    print(f"[merge] old={len(old_entries)} new_candidates={len(new_items)}", flush=True)

    metrics_by_id: dict[str, dict[str, Any]] = {}
    # keep old metrics from yaml when present
    for e in old_entries:
        fid = str(e.get("factor_id"))
        m = e.get("metrics")
        if isinstance(m, dict):
            metrics_by_id[fid] = {"status": "pass", **m, "factor_id": fid, "formula": e.get("expr")}

    exec_fail: list[dict[str, Any]] = []
    if not args.skip_precompute and new_items:
        use_val = not args.no_valuation
        print(f"[precompute] {args.start}..{args.end} valuation={use_val} n={len(new_items)}", flush=True)
        eng, ds = build_engine(args.start, args.end, use_valuation=use_val)
        dates, stocks, y_mat = make_fwd_matrix(ds, args.label_days)
        print(f"[precompute] panel days={len(dates)} stocks={len(stocks)}", flush=True)
        # append metrics jsonl
        with metrics_path.open("a", encoding="utf-8") as mf:
            mf.write(f"\n# expand_batch {time.strftime('%Y-%m-%dT%H:%M:%S')} n={len(new_items)}\n")
            for i, item in enumerate(new_items):
                fid = item["factor_id"]
                formula = item["expr"]
                try:
                    m = precompute_one(eng, formula, dates=dates, stocks=stocks, y_mat=y_mat, label_days=args.label_days)
                    m["factor_id"] = fid
                    m["formula"] = formula
                    m["batch"] = "expand_v1"
                    metrics_by_id[fid] = m
                    mf.write(json.dumps(m, ensure_ascii=False) + "\n")
                except Exception as exc:
                    err = {"factor_id": fid, "formula": formula, "status": "fail", "error": f"{type(exc).__name__}: {exc}", "batch": "expand_v1"}
                    exec_fail.append(err)
                    mf.write(json.dumps(err, ensure_ascii=False) + "\n")
                if (i + 1) % 25 == 0 or i + 1 == len(new_items):
                    print(f"[precompute] {i+1}/{len(new_items)} pass={len(metrics_by_id)-len(old_entries)} fail={len(exec_fail)}", flush=True)

    # rebuild entries
    merged: list[dict[str, Any]] = list(old_entries)
    added = 0
    for item in new_items:
        fid = item["factor_id"]
        m = metrics_by_id.get(fid)
        if not args.skip_precompute:
            if m is None or m.get("status") == "fail":
                continue
            if float(m.get("finite_ratio") or 0) < args.min_finite_ratio:
                continue
        entry = {
            "factor_id": fid,
            "expr": item["expr"],
            "topic": item["topic"],
            "description": item["description"],
            "operators": item.get("operators") or extract_ops(item["expr"]),
            "family": item.get("family"),
            "source_library": item.get("source_library"),
        }
        if m and m.get("ic") is not None or (m and "finite_ratio" in m):
            entry["metrics"] = {
                "ic": m.get("ic"),
                "icir": m.get("icir"),
                "abs_ic": m.get("abs_ic"),
                "finite_ratio": m.get("finite_ratio"),
                "n_unique": m.get("n_unique"),
                "label_days": m.get("label_days", args.label_days),
                "window": {"start": args.start, "end": args.end},
            }
        merged.append(entry)
        added += 1

    payload = {
        "schema_version": "cold_start.backend_v9.ashare",
        "quality_policy": "factorengine_backend_audited_v9_precomputed_expand_v1",
        "market": "ashare",
        "frequency": "daily",
        "domain": "price_volume_valuation",
        "expression_syntax": "factor_engine_dsl",
        "operator_policy": "lqtp_pv_daily",
        "dsl_validated": True,
        "values_precomputed": not args.skip_precompute,
        "source": f"V9 core + expand_v1; valuation={not args.no_valuation}; precompute={not args.skip_precompute}",
        "entry_count": len(merged),
        "entries": merged,
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False, width=120)

    abs_ics = [float(e["metrics"]["abs_ic"]) for e in merged if isinstance(e.get("metrics"), dict) and e["metrics"].get("abs_ic") is not None]
    summary = {
        "old_entries": len(old_entries),
        "generated": len(factors),
        "parse_ok": len(parse_ok),
        "parse_fail": len(parse_fail),
        "new_precompute_pass": added,
        "new_exec_fail": len(exec_fail),
        "exported_total": len(merged),
        "out_yaml": str(yaml_path),
        "precompute_window": {"start": args.start, "end": args.end, "label_days": args.label_days},
        "use_valuation": not args.no_valuation,
        "abs_ic_quantiles": None,
        "parse_fail_samples": parse_fail[:20],
        "exec_fail_samples": exec_fail[:20],
    }
    if abs_ics:
        arr = np.asarray(sorted(abs_ics))
        summary["abs_ic_quantiles"] = {
            "p50": float(np.quantile(arr, 0.5)),
            "p90": float(np.quantile(arr, 0.9)),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
        }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("old_entries", "parse_ok", "new_precompute_pass", "new_exec_fail", "exported_total")}, ensure_ascii=False), flush=True)
    print(f"[done] yaml -> {yaml_path} (+{added})", flush=True)
    return 0 if merged else 1


if __name__ == "__main__":
    raise SystemExit(main())
