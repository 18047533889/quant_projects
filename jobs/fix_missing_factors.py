#!/usr/bin/env python3
"""补齐 parquet 里 16 个因子（值为 0）的真实落值。

策略: 直接算矩阵, 写入 parquet 中对应 (factor, *) 列的 NaN 单元格
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import duckdb

PROJECT = Path("/home/sunhaiwei/quant_projects")
BACKTEST_OUT = PROJECT / "weekly_backtest_output"
LOCAL = Path.home() / "cos_data" / "StockDailyBar"
FV_PATH = BACKTEST_OUT / "factor_values.parquet"


# =============== 16 个缺失因子的代码 ===============
REWROTE_CODE = {
    'volume_weighted_impact': '''
def factor_volume_weighted_impact(df):
    import numpy as np
    df = df.sort_values(['OrderBookId','TradingDay']).reset_index(drop=True)
    ret = df['close_price'].pct_change().fillna(0.0)
    abs_ret = ret.abs()
    df['ar'] = abs_ret
    grouped = df.groupby('TradingDay')
    wm = grouped.apply(lambda g: (g['ar'] * g['volume']).sum() / max(g['volume'].sum(), 1))
    res = df.merge(wm.rename('vwi'), left_on='TradingDay', right_index=True)
    return res.set_index(['TradingDay','OrderBookId'])['vwi']
''',
    'amount_weighted_squared_impact': '''
def factor_amount_weighted_squared_impact(df):
    import numpy as np
    df = df.sort_values(['OrderBookId','TradingDay']).reset_index(drop=True)
    ret = df['close_price'].pct_change().fillna(0.0)
    sq_ret = ret ** 2
    df['s'] = sq_ret
    grouped = df.groupby('TradingDay')
    wm = grouped.apply(lambda g: (g['s'] * g['amount']).sum() / max(g['amount'].sum(), 1))
    res = df.merge(wm.rename('awsi'), left_on='TradingDay', right_index=True)
    return res.set_index(['TradingDay','OrderBookId'])['awsi']
''',
    'volume_weighted_squared_impact': '''
def factor_volume_weighted_squared_impact(df):
    import numpy as np
    df = df.sort_values(['OrderBookId','TradingDay']).reset_index(drop=True)
    ret = df['close_price'].pct_change().fillna(0.0)
    sq_ret = ret ** 2
    df['s'] = sq_ret
    grouped = df.groupby('TradingDay')
    wm = grouped.apply(lambda g: (g['s'] * g['volume']).sum() / max(g['volume'].sum(), 1))
    res = df.merge(wm.rename('vwsi'), left_on='TradingDay', right_index=True)
    return res.set_index(['TradingDay','OrderBookId'])['vwsi']
''',
    'pressure_ema_mutation': '''
def factor_pressure_ema_mutation(df):
    import numpy as np
    df = df.sort_values(['OrderBookId','TradingDay']).reset_index(drop=True)
    dollar_imb = (df['close'] - df['open']) * df['volume']
    df['di'] = dollar_imb
    df['ema21'] = df.groupby('OrderBookId')['di'].transform(lambda x: x.ewm(span=21, adjust=False).mean())
    df['tr'] = np.maximum(df['high']-df['low'], np.abs(df['high']-df['close'].shift(1)))
    df['tr'] = df['tr'].fillna(df['high']-df['low'])
    df['atr21'] = df.groupby('OrderBookId')['tr'].transform(lambda x: x.ewm(span=21, adjust=False).mean())
    df['prev_close'] = df.groupby('OrderBookId')['close'].shift(1)
    df['fc'] = (df['prev_close'] - df['close']) / df['prev_close'].replace(0, np.nan)
    df['factor_pressure_ema_mutation'] = df['ema21'] * (1 + df['fc'].fillna(0)) / df['atr21'].replace(0, np.nan)
    return df.set_index(['TradingDay','OrderBookId'])['factor_pressure_ema_mutation']
''',
    'volume_adaptive_momentum_smoothed_v3': '''
def factor_volume_adaptive_momentum_smoothed_v3(df):
    import numpy as np
    df = df.sort_values(['OrderBookId','TradingDay']).reset_index(drop=True)
    df['ret5'] = df['close'] / df.groupby('OrderBookId')['close'].shift(5) - 1
    df['vol_ema30'] = df.groupby('OrderBookId')['volume'].transform(lambda x: x.ewm(span=30, adjust=False).mean())
    df['vol_ratio'] = df['volume'] / df['vol_ema30'].replace(0, np.nan)
    df['lr'] = np.log(df['vol_ratio'].replace(0, np.nan))
    df['signal'] = df['ret5'] * df['lr']
    df['vam'] = df.groupby('OrderBookId')['signal'].transform(lambda x: x.ewm(span=15, adjust=False).mean())
    return df.set_index(['TradingDay','OrderBookId'])['vam']
''',
    'volume_adaptive_momentum_fast_vol': '''
def factor_volume_adaptive_momentum_fast_vol(df):
    import numpy as np
    df = df.sort_values(['OrderBookId','TradingDay']).reset_index(drop=True)
    df['ret5'] = df['close'] / df.groupby('OrderBookId')['close'].shift(5) - 1
    df['vol_ema10'] = df.groupby('OrderBookId')['volume'].transform(lambda x: x.ewm(span=10, adjust=False).mean())
    df['vol_ratio'] = df['volume'] / df['vol_ema10'].replace(0, np.nan)
    df['lr'] = np.log(df['vol_ratio'].replace(0, np.nan))
    df['signal'] = df['ret5'] * df['lr']
    df['vafv'] = df.groupby('OrderBookId')['signal'].transform(lambda x: x.ewm(span=10, adjust=False).mean())
    return df.set_index(['TradingDay','OrderBookId'])['vafv']
''',
    'fear_adjusted_dollar_pressure_short_ema': '''
def factor_fear_adjusted_dollar_pressure_short_ema(df):
    import numpy as np
    df = df.sort_values(['OrderBookId','TradingDay']).reset_index(drop=True)
    df['di'] = (df['close'] - df['open']) * df['volume']
    df['ema10'] = df.groupby('OrderBookId')['di'].transform(lambda x: x.ewm(span=10, adjust=False).mean())
    df['tr'] = np.maximum(df['high']-df['low'], np.abs(df['high']-df['close'].shift(1)))
    df['tr'] = df['tr'].fillna(df['high']-df['low'])
    df['atr21'] = df.groupby('OrderBookId')['tr'].transform(lambda x: x.ewm(span=21, adjust=False).mean())
    df['prev_close'] = df.groupby('OrderBookId')['close'].shift(1)
    df['fc'] = (df['prev_close'] - df['close']) / df['prev_close'].replace(0, np.nan)
    df['fadp'] = df['ema10'] * (1 + df['fc'].fillna(0)) / df['atr21'].replace(0, np.nan)
    return df.set_index(['TradingDay','OrderBookId'])['fadp']
''',
    'volatility_adjusted_dollar_pressure': '''
def factor_volatility_adjusted_dollar_pressure(df):
    import numpy as np
    df = df.sort_values(['OrderBookId','TradingDay']).reset_index(drop=True)
    df['di'] = (df['close'] - df['open']) * df['volume']
    df['ema21'] = df.groupby('OrderBookId')['di'].transform(lambda x: x.ewm(span=21, adjust=False).mean())
    df['tr'] = np.maximum(df['high']-df['low'], np.abs(df['high']-df['close'].shift(1)))
    df['tr'] = df['tr'].fillna(df['high']-df['low'])
    df['atr21'] = df.groupby('OrderBookId')['tr'].transform(lambda x: x.ewm(span=21, adjust=False).mean())
    df['vadp'] = df['ema21'] / df['atr21'].replace(0, np.nan)
    return df.set_index(['TradingDay','OrderBookId'])['vadp']
''',
    'drawdown_depth_atr_gated': '''
def factor_drawdown_depth_atr_gated(df):
    import numpy as np
    df = df.sort_values(['OrderBookId','TradingDay']).reset_index(drop=True)
    df['rmax'] = df.groupby('OrderBookId')['close'].transform(lambda x: x.rolling(63, min_periods=10).max())
    df['depth'] = -(df['close'] / df['rmax'].replace(0, np.nan) - 1)
    df['tr'] = np.maximum(df['high']-df['low'], np.abs(df['high']-df['close'].shift(1)))
    df['tr'] = df['tr'].fillna(df['high']-df['low'])
    df['atr63'] = df.groupby('OrderBookId')['tr'].transform(lambda x: x.rolling(63, min_periods=10).mean())
    df['ddag'] = df['depth'] / df['atr63'].replace(0, np.nan)
    return df.set_index(['TradingDay','OrderBookId'])['ddag']
''',
    'asym_intraday_sma': '''
def factor_asym_intraday_sma(df):
    import numpy as np
    df = df.sort_values(['OrderBookId','TradingDay']).reset_index(drop=True)
    df['sma_down'] = df.groupby('OrderBookId').apply(lambda x: (x['open']-x['low']).rolling(40, min_periods=5).mean()).reset_index(level=0, drop=True)
    df['sma_up'] = df.groupby('OrderBookId').apply(lambda x: (x['high']-x['open']).rolling(40, min_periods=5).mean()).reset_index(level=0, drop=True)
    df['ais'] = df['sma_down'] / df['sma_up'].replace(0, np.nan)
    return df.set_index(['TradingDay','OrderBookId'])['ais']
''',
    'lag_vol_volatility_adjusted_gate': '''
def factor_lag_vol_volatility_adjusted_gate(df):
    import numpy as np
    df = df.sort_values(['OrderBookId','TradingDay']).reset_index(drop=True)
    df['ret5'] = df['close'] / df.groupby('OrderBookId')['close'].shift(5) - 1
    df['vol_ema30'] = df.groupby('OrderBookId')['volume'].transform(lambda x: x.ewm(span=30, adjust=False).mean())
    df['lv_ratio'] = np.log(df['volume'] / df['vol_ema30'].replace(0, np.nan))
    df['tr'] = np.maximum(df['high']-df['low'], np.abs(df['high']-df['close'].shift(1)))
    df['tr'] = df['tr'].fillna(df['high']-df['low'])
    df['atr20'] = df.groupby('OrderBookId')['tr'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df['atr_ratio'] = df['atr20'] / df['close']
    df['signal'] = df['ret5'] * df['lv_ratio'] * df['atr_ratio']
    df['lvvg'] = df.groupby('OrderBookId')['signal'].transform(lambda x: x.ewm(span=10, adjust=False).mean())
    return df.set_index(['TradingDay','OrderBookId'])['lvvg']
''',
    'lag_response_vol_tool_adaptive': '''
def factor_lag_response_vol_tool_adaptive(df):
    import numpy as np
    df = df.sort_values(['OrderBookId','TradingDay']).reset_index(drop=True)
    df['ret5'] = df['close'] / df.groupby('OrderBookId')['close'].shift(5) - 1
    df['vol_ema30'] = df.groupby('OrderBookId')['volume'].transform(lambda x: x.ewm(span=30, adjust=False).mean())
    df['vol_ratio'] = df['volume'] / df['vol_ema30'].replace(0, np.nan)
    df['lr'] = np.log(df['vol_ratio'].replace(0, np.nan))
    df['tr'] = np.maximum(df['high']-df['low'], np.abs(df['high']-df['close'].shift(1)))
    df['tr'] = df['tr'].fillna(df['high']-df['low'])
    df['atr20'] = df.groupby('OrderBookId')['tr'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df['atr_ratio'] = df['atr20'] / df['close']
    df['signal'] = df['ret5'] * df['lr'] * df['atr_ratio']
    df['lrvta'] = df.groupby('OrderBookId')['signal'].transform(lambda x: x.ewm(span=10, adjust=False).mean())
    return df.set_index(['TradingDay','OrderBookId'])['lrvta']
''',
    'volume_adjusted_price_range': '''
def factor_volume_adjusted_price_range(df):
    import numpy as np
    df = df.sort_values(['OrderBookId','TradingDay']).reset_index(drop=True)
    df['hl'] = df['high'] - df['low']
    df['vol_ema10'] = df.groupby('OrderBookId')['volume'].transform(lambda x: x.ewm(span=10, adjust=False).mean())
    df['vol_adj'] = df['hl'] * (df['volume'] / df['vol_ema10'].replace(0, np.nan))
    df['vapr'] = df.groupby('OrderBookId')['vol_adj'].transform(lambda x: x.rolling(5, min_periods=2).mean())
    return df.set_index(['TradingDay','OrderBookId'])['vapr']
''',
    'volume_adjusted_price_range_ema_gated': '''
def factor_volume_adjusted_price_range_ema_gated(df):
    import numpy as np
    df = df.sort_values(['OrderBookId','TradingDay']).reset_index(drop=True)
    df['hl'] = df['high'] - df['low']
    df['vol_ema10'] = df.groupby('OrderBookId')['volume'].transform(lambda x: x.ewm(span=10, adjust=False).mean())
    df['vol_adj'] = df['hl'] * (df['volume'] / df['vol_ema10'].replace(0, np.nan))
    df['vaprg'] = df.groupby('OrderBookId')['vol_adj'].transform(lambda x: x.ewm(span=10, adjust=False).mean())
    return df.set_index(['TradingDay','OrderBookId'])['vaprg']
''',
    'ts_size_adaptive_earnings_book_smooth': '''
def factor_ts_size_adaptive_earnings_book_smooth(df):
    import numpy as np
    df = df.sort_values(['OrderBookId','TradingDay']).reset_index(drop=True)
    df['log_volume'] = np.log(df['volume'].replace(0, np.nan))
    df['size_mean'] = df.groupby('OrderBookId')['log_volume'].transform(lambda x: x.rolling(60, min_periods=20).mean())
    df['size_std'] = df.groupby('OrderBookId')['log_volume'].transform(lambda x: x.rolling(60, min_periods=20).std())
    df['size_z'] = ((df['log_volume'] - df['size_mean']) / df['size_std'].replace(0, np.nan)).clip(-10, 10)
    df['weight'] = 1.0 / (1 + np.exp(-df['size_z']))
    df['book_proxy'] = df['vwap'] / df.groupby('OrderBookId')['vwap'].transform(lambda x: x.rolling(60, min_periods=20).mean())
    df['tsabes'] = df['weight'] * df['book_proxy'] + (1 - df['weight']) * 1.0
    return df.set_index(['TradingDay','OrderBookId'])['tsabes']
''',
    'book_attention': '''
def factor_book_attention(df):
    import numpy as np
    df = df.sort_values(['OrderBookId','TradingDay']).reset_index(drop=True)
    df['book_proxy'] = df['vwap'] / df.groupby('OrderBookId')['vwap'].transform(lambda x: x.rolling(60, min_periods=20).mean())
    df['turn_proxy'] = df['volume'] / df.groupby('OrderBookId')['volume'].transform(lambda x: x.rolling(60, min_periods=20).mean())
    df['turn_rank'] = df.groupby('TradingDay')['turn_proxy'].rank(method='dense', pct=True)
    df['ba'] = (1.0 / df['book_proxy'].replace(0, np.nan)) * (1 - df['turn_rank'])
    return df.set_index(['TradingDay','OrderBookId'])['ba']
''',
}


# 16 个 0 因子名（按 parquet 检测结果）
MISSING_FACTOR_NAMES = [
    'volume_adaptive_momentum_smoothed_v3',
    'fear_adjusted_dollar_pressure_short_ema',
    'volume_adaptive_momentum_fast_vol',
    'drawdown_depth_atr_gated',
    'asym_intraday_sma',
    'lag_vol_volatility_adjusted_gate',
    'lag_response_vol_tool_adaptive',
    'volume_adjusted_price_range',
    'volume_adjusted_price_range_ema_gated',
    'volatility_adjusted_dollar_pressure',
    'ts_size_adaptive_earnings_book_smooth',
    'book_attention',
    'amount_weighted_squared_impact',
    'volume_weighted_squared_impact',
    'volume_adjusted_price_range',
    'volume_weighted_impact',
    'pressure_ema_mutation',
]
# de-dup, keep order
MISSING_FACTOR_NAMES = list(dict.fromkeys(MISSING_FACTOR_NAMES))


def load_market_via_duckdb(files):
    files_str = "[" + ",".join(f"'{f}'" for f in files) + "]"
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT TradeDate as TradingDay,
               Symbol as OrderBookId,
               Open as open, High as high, Low as low, Close as close_price, Close as close,
               PreClose, Volume as volume, Amount as amount, Vwap as vwap
        FROM read_parquet({files_str})
        ORDER BY TradingDay, OrderBookId
    """).df()
    return df


def compute_factor_for_name(args):
    name, mkt_pkl_path, code = args
    import pickle
    import pandas as pd
    import numpy as np

    with open(mkt_pkl_path, "rb") as f:
        mkt = pickle.load(f)

    try:
        ns = {"np": np, "pd": pd}
        exec(code, ns)
        fn = ns[f"factor_{name}"]
        result = fn(mkt)
        if result is None or len(result) == 0:
            return name, None
        if isinstance(result, pd.Series):
            df = result.reset_index()
            df.columns = ['TradingDay', 'OrderBookId', 'value']
            # pivot to date x symbol
            matrix = df.pivot(index='TradingDay', columns='OrderBookId', values='value')
            return name, matrix
        return name, None
    except Exception as e:
        return name, f"ERR: {str(e)[:200]}"


def main():
    print("=" * 60)
    print("补齐16个核心因子的真值, 写入 parquet")
    print("=" * 60)

    # 1. Load parquet
    print(f"加载 {FV_PATH} ...")
    df_full = pd.read_parquet(FV_PATH)
    # ensure index is DatetimeIndex
    if not isinstance(df_full.index, pd.DatetimeIndex):
        df_full.index = pd.to_datetime(df_full.index)
    print(f"  原始 shape: {df_full.shape}, {len(df_full.columns.get_level_values(0).unique())} 个因子")

    # 2. Get existing symbols/columns for the missing factors
    # columns structure: ('factor_xxx', 'STOCK')
    # missing factors just have all-NaN values; we just need to overwrite them

    # 3. Load market
    files = sorted(LOCAL.glob("*.parquet"))
    print(f"加载行情 {len(files)} 天 ...")
    mkt = load_market_via_duckdb(files)

    import pickle, tempfile
    tmp = Path(tempfile.gettempdir()) / "mkt_fix.pkl"
    with open(tmp, "wb") as f:
        pickle.dump(mkt, f)
    print(f"行情已缓存到 {tmp} ({tmp.stat().st_size / 1024 / 1024:.1f} MB)")

    # 4. Parallel compute
    tasks = [(name, str(tmp), REWROTE_CODE[name]) for name in MISSING_FACTOR_NAMES if name in REWROTE_CODE]
    print(f"\n并行计算 {len(tasks)} 个因子 ...")
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
                print(f"  {name}: matrix {res.shape}, NaN% = {res.isna().sum().sum() / res.size:.1%}")

    # 5. Write back into parquet
    # Parquet columns are tuples (factor, symbol). Pandas doesn't always preserve MultiIndex cols on roundtrip with parquet.
    # Workaround: save current df as long form, append new, save back.
    print("\n写入 parquet (用 long form)...")

    # Convert to long format with existing data
    existing = df_full.copy()
    existing.columns = [f"{c[0]}__{c[1]}" for c in existing.columns]
    existing.index.name = "TradingDay"
    # 重塑
    existing_long = existing.reset_index().melt(id_vars="TradingDay", var_name="fact_sym", value_name="value")
    existing_long[["factor", "OrderBookId"]] = existing_long["fact_sym"].str.split("__", n=1, expand=True)
    existing_long = existing_long.drop(columns=["fact_sym"])

    print(f"  existing long: {existing_long.shape}, factors={existing_long['factor'].nunique()}, symbols={existing_long['OrderBookId'].nunique()}")

    # Build new long form for missing factors
    new_rows = []
    for name, mat in results.items():
        if mat is None or mat.empty:
            continue
        factor_label = f"factor_{name}"
        # mat 索引是 TradingDay(str), 列是 OrderBookId(str)
        mat.index = pd.to_datetime(mat.index)
        stacked = mat.stack().reset_index()
        stacked.columns = ['TradingDay', 'OrderBookId', 'value']
        stacked['factor'] = factor_label
        new_rows.append(stacked[['TradingDay', 'OrderBookId', 'factor', 'value']])

    if new_rows:
        new_long = pd.concat(new_rows, ignore_index=True)
        print(f"  new long: {new_long.shape}, factors={new_long['factor'].nunique()}, symbols={new_long['OrderBookId'].nunique()}")

        # Drop existing rows for the missing factors, then append new
        missing_labels = {f"factor_{n}" for n in results.keys()}
        keep_mask = ~existing_long["factor"].isin(missing_labels)
        existing_long = existing_long[keep_mask]
        print(f"  dropped {~keep_mask.sum()} existing rows for missing factors")

        combined = pd.concat([existing_long, new_long], ignore_index=True)
        print(f"  combined: {combined.shape}")

        # Reshape back to wide
        pivot = combined.pivot_table(index="TradingDay", columns=["factor", "OrderBookId"], values="value", aggfunc="first")
        pivot.columns = pd.MultiIndex.from_tuples(pivot.columns, names=["factor", "OrderBookId"])
        pivot = pivot.sort_index(axis=1)
        print(f"  pivot wide: {pivot.shape}")

        # Save
        pivot.to_parquet(str(FV_PATH))
        print(f"\n  ✅ 保存到 {FV_PATH}")

    print("\n完成！")
    if errors:
        print(f"失败: {len(errors)}")
        for k, v in errors.items():
            print(f"  {k}: {v[:80]}")


if __name__ == "__main__":
    main()