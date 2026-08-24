#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新计算所有 61 个因子在 2019-01~2025-12 期间的因子值，生成新 parquet。
用 8 进程并行，每因子 ~30s。
"""
import sys, os, json, warnings, pickle, tempfile
warnings.filterwarnings("ignore")
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import duckdb

PROJECT = Path("/home/sunhaiwei/quant_projects")
CONV_DIR = PROJECT / "factor_delivery_converted" / "factors_combined"
BACKTEST_OUT = PROJECT / "weekly_backtest_output"
LOCAL = Path.home() / "cos_data" / "StockDailyBar"
FV_PATH = BACKTEST_OUT / "factor_values.parquet"

# 加载所有 61 个因子名
FACTOR_NAMES = sorted([p.stem.replace("factor_", "") for p in (Path("/home/sunhaiwei/quant_projects/docs/reports/2026-08-23/factors")).glob("factor_*.html")])
# 用的是 docs/reports/2026-08-23/factors 下面的 html 文件名作为因子名清单
print(f"[init] 因子总数: {len(FACTOR_NAMES)}")


def load_market_via_duckdb(files):
    files_str = "[" + ",".join(f"'{f}'" for f in files) + "]"
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT TradeDate as TradingDay,
               Symbol as OrderBookId,
               Open as open, High as high, Low as low,
               Close as close_price, Close as close,
               PreClose as pre_close,
               Volume as volume,
               Amount as amount,
               Vwap as vwap
        FROM read_parquet({files_str})
    """).df()
    df = df.sort_values(['OrderBookId','TradingDay']).reset_index(drop=True)
    return df


def load_market():
    files = sorted(LOCAL.glob("*.parquet"))
    print(f"[mkt] 加载行情 {len(files)} 天...")
    return load_market_via_duckdb(files)


def compute_factor_for_name(task):
    """单因子计算：按 symbol 逐个调用 (限制 top 200 股票)，再 concat 到 wide 矩阵"""
    name, code, mkt_pkl, n_stocks = task
    try:
        import sys
        _here = Path(__file__).resolve().parent.parent
        if str(_here) not in sys.path:
            sys.path.insert(0, str(_here))
        with open(mkt_pkl, "rb") as f:
            mkt = pickle.load(f)
        from jobs.weekly_factor_backtest import alpha_tools, talib, minute_tools
        ns = {
            "np": np,
            "pd": pd,
            "alpha_tools": alpha_tools,
            "talib": talib,
            "minute_tools": minute_tools,
        }
        exec(code, ns)
        fn = ns[f"factor_{name}"]

        # 选 n_stocks 个 full-history 股票
        sym_counts = mkt.groupby('OrderBookId').size()
        sym_counts = sym_counts[sym_counts >= 1000].sort_values(ascending=False)
        symbols = sym_counts.head(n_stocks).index.tolist()
        mkt_sub = mkt[mkt['OrderBookId'].isin(symbols)].copy()

        results = {}
        for sym in symbols:
            sub = mkt_sub[mkt_sub['OrderBookId'] == sym].copy()
            if len(sub) < 50:
                continue
            try:
                r = fn(sub)
                if isinstance(r, pd.Series) and len(r) == len(sub):
                    sub_dates = pd.to_datetime(sub['TradingDay'].values)
                    results[sym] = pd.Series(r.values, index=sub_dates)
            except Exception:
                continue
        if not results:
            return name, None
        matrix = pd.concat(results, axis=1)
        matrix.columns = list(results.keys())
        return name, matrix
    except Exception as e:
        return name, f"ERR: {str(e)[:200]}"


def get_factor_code(name: str) -> str | None:
    """从 json 文件取 code。'foo_flipped' 找 'foo'"""
    base = name.replace("_flipped", "")
    candidates = [
        CONV_DIR / f"factor_{name}.json",
        CONV_DIR / f"factor_{base}.json",
    ]
    # 尝试带数字后缀
    for suf in ["", "_109", "_110", "_116", "_118", "_190", "_246", "_315", "_363", "_118", "_340", "_209", "_001", "_5_2", "_002", "_330", "_418", "_223", "_359", "_251", "_145", "_308", "_030"]:
        candidates.append(CONV_DIR / f"factor_{base}{suf}.json")
    for path in candidates:
        if path.exists():
            with open(path) as f:
                d = json.load(f)
            code = d.get("code", "")
            # 翻转: 在外面包一层取负
            if name.endswith("_flipped") and code:
                # 替换函数名为 _flipped 名字 (若还没有)
                base = name.replace("_flipped", "")
                if f"def factor_{base}(" in code and f"def factor_{name}(" not in code:
                    wrapped = code.replace(f"def factor_{base}(", f"def factor_{name}(")
                else:
                    wrapped = code
                # 在 return 前加取负 (假设原 fn 返回 Series)
                inject = f"\n    if isinstance(result, pd.Series):\n        return -result\n    return result\n"
                wrapped = wrapped + inject
                return wrapped
            return code
    return None


def main():
    print("=" * 60)
    print("重算所有 61 个因子 (2019-2025, parquet)")
    print("=" * 60)

    # 加载行情
    files = sorted(LOCAL.glob("*.parquet"))
    print(f"[mkt] 加载行情 {len(files)} 天...")
    mkt = load_market_via_duckdb(files)
    print(f"  行情: {mkt.shape}, symbols={mkt['OrderBookId'].nunique()}")

    import pickle
    tmp = Path(tempfile.gettempdir()) / "mkt_all61.pkl"
    with open(tmp, "wb") as f:
        pickle.dump(mkt, f)
    print(f"  行情缓存: {tmp} ({tmp.stat().st_size/1024/1024:.1f} MB)")

    # 任务列表：取每个因子的 code
    N_STOCKS = 300  # top 300 stocks (流动性好)
    tasks = []
    for name in FACTOR_NAMES:
        code = get_factor_code(name)
        if code:
            tasks.append((name, code, str(tmp), N_STOCKS))
        else:
            print(f"  [warn] {name}: no code found")

    print(f"\n并行计算 {len(tasks)} 个因子 (top {N_STOCKS} stocks)...")
    results = {}
    errors = {}

    n_workers = min(8, max(2, os.cpu_count() or 4))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(compute_factor_for_name, t): t[0] for t in tasks}
        for future in as_completed(futures):
            name, res = future.result()
            if isinstance(res, str):
                errors[name] = res
                print(f"  {name}: ERROR {res[:80]}")
            elif res is None:
                errors[name] = "empty"
                print(f"  {name}: empty")
            else:
                results[name] = res
                print(f"  {name}: matrix {res.shape}, NaN%={res.isna().sum().sum()/res.size:.1%}")

    print(f"\n完成: ok={len(results)}, err={len(errors)}")
    for k, v in errors.items():
        print(f"  ERR {k}: {v[:80]}")

    # 拼成 MultiIndex 矩阵 (date, factor)
    print("\n拼装 MultiIndex 矩阵...")
    all_matrices = {}
    for name, mat in results.items():
        mat.index = pd.to_datetime(mat.index)
        # 过滤掉 shape too small (single-stock daily-agg 等异常矩阵)
        if mat.shape[0] < 100 or mat.shape[1] < 50:
            errors[name] = f"shape too small: {mat.shape}"
            print(f"  skip {name}: shape too small {mat.shape}")
            continue
        all_matrices[f"factor_{name}"] = mat

    if not all_matrices:
        print("[ERR] 没有可用的矩阵!")
        return

    # 共有的标的与日期
    common_dates = sorted(set.intersection(*[set(m.index) for m in all_matrices.values()]))
    common_symbols = sorted(set.intersection(*[set(m.columns) for m in all_matrices.values()]))
    print(f"  common: {len(common_dates)} 天 × {len(common_symbols)} 标的")

    # 拼 (date, factor, symbol) long form
    rows = []
    for factor_label, mat in all_matrices.items():
        sub = mat.loc[common_dates, common_symbols]
        # stack 到 long
        s = sub.stack(future_stack=True).rename("value").reset_index()
        s["factor"] = factor_label
        s.columns = ['TradingDay', 'OrderBookId', 'value', 'factor']
        rows.append(s[['TradingDay', 'OrderBookId', 'factor', 'value']])

    big = pd.concat(rows, ignore_index=True)
    print(f"  long shape: {big.shape}")

    # 写 wide (date x (factor, symbol))
    print("\n写 wide 矩阵到 parquet...")
    pivot = big.pivot_table(index='TradingDay', columns=['factor', 'OrderBookId'], values='value', aggfunc='first')
    print(f"  pivot shape: {pivot.shape}")

    # 删除旧文件
    if FV_PATH.exists():
        FV_PATH.unlink()
        print(f"  删除旧 {FV_PATH}")

    # 用 pyarrow 引擎避免 thrift 问题
    pivot.to_parquet(str(FV_PATH), engine='pyarrow', compression='snappy')
    print(f"  ✅ 保存 {FV_PATH} ({FV_PATH.stat().st_size/1024/1024:.1f} MB)")

    # 验证
    import pyarrow.parquet as pq
    f = pq.ParquetFile(str(FV_PATH))
    print(f"  schema: {f.schema_arrow}")


if __name__ == "__main__":
    main()