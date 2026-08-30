#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""后复权口径统一入口（2026-08-28 硬性）：
所有因子评估的收益一律用 后复权 AdjVwap（StockDailyBarAdj.AdjVwap），
禁止使用未复权 StockDailyBar.Vwap。
按需缓存单例以免重复读盘。
"""
import duckdb
import pandas as pd
from pathlib import Path

START, END = "2019-01-02", "2026-08-24"
_HAS_ADJ = None

def load_adj_vwap(start=START, end=END):
    """返回 date×symbol 的 后复权 AdjVwap 矩阵（StockDailyBarAdj）。"""
    global _HAS_ADJ
    if _HAS_ADJ is not None:
        df = _HAS_ADJ
    else:
        files = sorted(Path.home().glob("cos_data/StockDailyBarAdj/*.parquet"))
        fs = "[" + ",".join(f"'{f}'" for f in files) + "]"
        con = duckdb.connect()
        df = con.execute(f"""
            SELECT TradeDate as date, Symbol as symbol, AdjVwap as vwap
            FROM read_parquet({fs})
            WHERE TradeDate >= DATE '{start}' AND TradeDate <= DATE '{end}'
        """).df()
        df = df.pivot_table(index="date", columns="symbol", values="vwap", aggfunc="first")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
    # 返回副本以免被外部 reindex 污染缓存
    return df.copy()

def adj_fwd_return(vwap=None, start=None, end=None):
    """vwap-to-vwap 后复权收益 = AdjVwap(t+2)/AdjVwap(t+1) - 1（shift(-2)）。
    企业级口径（用户 2026-08-28 确认）：t 日收盘算因子 → t+1 日用 t+1 的 AdjVwap 下单 → 持有到 t+2 卖出。
    对齐 StockDailyBarAdj.TargetVwapReturnH01（实测 = v(t+2)/v(t+1)-1）。"""
    if vwap is None:
        vwap = load_adj_vwap()
    return vwap.pct_change().shift(-2)
