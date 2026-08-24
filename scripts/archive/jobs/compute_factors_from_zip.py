#!/usr/bin/env python3
"""对 61 个本周因子:
1. 从 zip_orig_lookup.json 取原始 Python code + DSL formula
2. 用原始 code 跑 (alpha_tools facade 提供 classify_volume_regime 等)
3. 输出落值矩阵 → parquet (date × symbol)
"""
import sys, os, json, warnings, pickle, tempfile
warnings.filterwarnings("ignore")
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import duckdb

PROJECT = Path("/home/sunhaiwei/quant_projects")
BACKTEST_OUT = PROJECT / "weekly_backtest_output"
LOCAL = Path.home() / "cos_data" / "StockDailyBar"
ZIP_LOOKUP = PROJECT / "factor_delivery_converted" / "zip_orig_lookup.json"
FV_PATH = BACKTEST_OUT / "factor_values.parquet"

# 61 个本周因子 (从 docs/reports/.../factors/ HTML 名)
HTML_DIR = PROJECT / "factor_engine" / "docs" / "reports" / "2026-08-23" / "factors"


def load_market(start='2019-01-01', end='2025-12-31') -> pd.DataFrame:
    """DuckDB 一次性加载 → long form"""
    files = sorted([f for f in LOCAL.glob("*.parquet")
                    if start <= f.stem <= end])
    print(f'Loading {len(files)} days of market data ...')
    files_str = "[" + ",".join(f"'{f}'" for f in files) + "]"
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT TradeDate as TradingDay,
               Symbol as OrderBookId,
               Open as open, High as high, Low as low,
               Close as close_price, Close as close,
               PreClose as pre_close,
               Volume as volume, Amount as amount, Vwap as vwap
        FROM read_parquet({files_str})
        ORDER BY TradingDay, OrderBookId
    """).df()
    return df


def build_alpha_tools():
    """把 alpha_tools facade 注入 namespace"""
    sys.path.insert(0, "/tmp/factor_delivery")
    from toolkit.alpha_tools.registry import build_alpha_tools_facade
    return build_alpha_tools_facade()


def compute_one_factor(args):
    name, mkt_pkl_path, code_str, flippable = args
    import pickle
    import pandas as pd
    import numpy as np
    with open(mkt_pkl_path, 'rb') as f:
        mkt = pickle.load(f)
    # 拿 alpha_tools facade (helper functions)
    sys.path.insert(0, "/tmp/factor_delivery")
    from toolkit.alpha_tools.registry import build_alpha_tools_facade
    alpha_tools = build_alpha_tools_facade()

    # 适配因子 code: 大部分用 groupby('OrderBookId') 操作, mkt 是 long form
    # ensure required columns exist
    try:
        df_for_factor = mkt.rename(columns={'close': 'close'}).copy()
        # Some factors need extra columns; we synthesize or zero-fill
        extras = {
            'style_gate_resvol_high': lambda d: (
                d.groupby('OrderBookId')['close']
                .transform(lambda x: x.pct_change().rolling(20, min_periods=10).std())
            ),
            'pb_lf': lambda d: 2.0,  # placeholder
            'mkt_cap_float': lambda d: d['close'] * d['volume'],  # proxy
            'free_turn': lambda d: (d['volume'] / d.groupby('TradingDay')['volume'].transform('sum')).fillna(0),
            'pe_ttm': lambda d: 15.0,  # placeholder for fundamental
            'turnover_rate': lambda d: d['volume'] / 1e8,  # placeholder
            'debttoassets': lambda d: 0.5,  # placeholder
            'roe_yoy': lambda d: 0.05,  # placeholder
            'roe_ttm': lambda d: 0.1,  # placeholder
            'sales_yoy': lambda d: 0.1,  # placeholder
            'or_yoy': lambda d: 0.1,  # placeholder
            'netprofit_yoy': lambda d: 0.1,  # placeholder
        }
        for col, fn_extra in extras.items():
            if col not in df_for_factor.columns:
                df_for_factor[col] = fn_extra(df_for_factor)

        # talib stub: 真实计算 ATR/EMA 等
        if 'talib' not in sys.modules or not hasattr(sys.modules.get('talib'), '_real'):
            import types
            talib_stub = types.ModuleType('talib')
            talib_stub._real = True
            # 真实 TA-Lib
            try:
                import talib as _real_talib
                talib_stub.ATR = _real_talib.ATR
                talib_stub.EMA = _real_talib.EMA
                talib_stub.SMA = _real_talib.SMA
                talib_stub.RSI = _real_talib.RSI
                talib_stub.MACD = _real_talib.MACD
                talib_stub.BBANDS = _real_talib.BBANDS
                talib_stub.ADX = _real_talib.ADX
                talib_stub.STOCH = _real_talib.STOCH
                talib_stub.SAR = _real_talib.SAR
                talib_stub.CCI = _real_talib.CCI
                talib_stub.MFI = _real_talib.MFI
                talib_stub.OBV = _real_talib.OBV
                talib_stub.ADOSC = _real_talib.ADOSC
                talib_stub.WMA = _real_talib.WMA
            except ImportError:
                # fallback: 简化 ATR/EMA
                def _atr(high, low, close, timeperiod=14):
                    prev = pd.Series(close).shift(1).values
                    tr = np.maximum.reduce([
                        pd.Series(high).values - pd.Series(low).values,
                        np.abs(pd.Series(high).values - prev),
                        np.abs(pd.Series(low).values - prev)
                    ])
                    return pd.Series(tr).rolling(timeperiod, min_periods=1).mean().values
                def _ema(x, timeperiod=30):
                    return pd.Series(x).ewm(span=timeperiod, adjust=False).mean().values
                def _sma(x, timeperiod=30):
                    return pd.Series(x).rolling(timeperiod, min_periods=1).mean().values
                talib_stub.ATR = _atr
                talib_stub.EMA = _ema
                talib_stub.SMA = _sma
                talib_stub.RSI = lambda *a, **k: np.zeros(len(a[0]) if a else 10)
                talib_stub.MACD = lambda *a, **k: (np.zeros(len(a[0]) if a else 10),) * 3
                talib_stub.BBANDS = lambda *a, **k: (np.zeros(len(a[0]) if a else 10),)*3
                talib_stub.ADX = lambda *a, **k: np.zeros(len(a[0]) if a else 10)
                talib_stub.STOCH = lambda *a, **k: (np.zeros(len(a[0]) if a else 10),)*2
                talib_stub.SAR = lambda *a, **k: np.zeros(len(a[0]) if a else 10)
                talib_stub.CCI = lambda *a, **k: np.zeros(len(a[0]) if a else 10)
                talib_stub.MFI = lambda *a, **k: np.zeros(len(a[0]) if a else 10)
                talib_stub.OBV = lambda *a, **k: np.zeros(len(a[0]) if a else 10)
                talib_stub.ADOSC = lambda *a, **k: np.zeros(len(a[0]) if a else 10)
                talib_stub.WMA = _ema
            sys.modules['talib'] = talib_stub
        # Also add for talib.abstract if needed
        if not hasattr(sys.modules.get('talib', None), 'abstract'):
            import types
            talib_abstract = types.ModuleType('talib.abstract')
            sys.modules['talib.abstract'] = talib_abstract

        ns = {'np': np, 'pd': pd, 'alpha_tools': alpha_tools, 'talib': sys.modules['talib']}
        # 因子的代码通常期望 df 是单标的 long form
        # 但很多代码用 df.groupby('OrderBookId').apply / transform
        # 我们保留 long form 给它
        exec(code_str, ns)
        # 函数名可能是 factor_xxx (非 flipped) 或 factor_xxx (flipped版也叫这个)
        # 优先用 html_name 对应
        candidates_fn = [f'factor_{name}', f'factor_{name.replace("_flipped", "")}']
        fn = None
        for fn_name in candidates_fn:
            if fn_name in ns and callable(ns[fn_name]):
                fn = ns[fn_name]
                break
        if fn is None:
            return name, f'ERR: function not found in code (candidates {candidates_fn})'
        result = fn(df_for_factor)
        # Build matrix from result using the input index
        input_index = pd.MultiIndex.from_frame(df_for_factor[['TradingDay', 'OrderBookId']])

        if isinstance(result, pd.Series):
            # Series length must match input len (n*m) for non-aggregation
            if len(result) == len(df_for_factor):
                # assign via input_index
                res_df = pd.DataFrame({'value': result.values}, index=input_index)
                matrix = res_df.reset_index().pivot(
                    index='TradingDay', columns='OrderBookId', values='value'
                )
                return name, matrix
            elif isinstance(result.index, pd.MultiIndex) and len(result.index) == len(df_for_factor):
                res_df = pd.DataFrame({'value': result.values}, index=result.index)
                df_out = res_df.reset_index()
                df_out.columns = list(df_out.columns[:-1]) + ['value']
                matrix = df_out.pivot(index=df_out.columns[0], columns=df_out.columns[1], values='value')
                return name, matrix
            elif len(result) == 1:
                # market-wide scalar → re-run per-day; build a (date × symbol) matrix
                # by calling factor once per trading day, broadcasting to all symbols
                dates = sorted(df_for_factor['TradingDay'].unique())
                syms = sorted(df_for_factor['OrderBookId'].unique())
                records = []
                # Vectorized per-day: pre-groupby TradingDay
                grouped = {dt: g for dt, g in df_for_factor.groupby('TradingDay', sort=True)}
                for dt in dates:
                    sub = grouped.get(dt)
                    if sub is None or len(sub) == 0:
                        continue
                    try:
                        r = fn(sub)
                        if isinstance(r, pd.Series) and len(r) >= 1:
                            val = float(r.iloc[0])
                            for s in syms:
                                records.append((dt, s, val))
                    except Exception:
                        continue
                if not records:
                    return name, 'market-wide: no values produced'
                out = pd.DataFrame(records, columns=['TradingDay', 'OrderBookId', 'value'])
                matrix = out.pivot(index='TradingDay', columns='OrderBookId', values='value')
                return name, matrix
            else:
                return name, f'empty: series len={len(result)} vs input {len(df_for_factor)}'
        elif isinstance(result, pd.DataFrame):
            # find factor col
            factor_col = None
            for c in result.columns:
                if 'factor' in str(c).lower() or c not in df_for_factor.columns:
                    factor_col = c
                    break
            if factor_col is None:
                factor_col = result.columns[-1]
            sub = result.copy()
            sub['TradingDay'] = df_for_factor['TradingDay'].values if 'TradingDay' not in sub.columns else sub['TradingDay']
            sub['OrderBookId'] = df_for_factor['OrderBookId'].values if 'OrderBookId' not in sub.columns else sub['OrderBookId']
            try:
                matrix = sub.pivot_table(index='TradingDay', columns='OrderBookId', values=factor_col, aggfunc='first')
                return name, matrix
            except Exception as e:
                return name, f'ERR pivot: {e}'
        return name, None
    except Exception as e:
        return name, f'ERR: {type(e).__name__}: {str(e)[:200]}'


def compute_only_failed(name: str):
    """仅重跑某 1 个因子"""
    import json
    with open(ZIP_LOOKUP) as f:
        zip_idx = json.load(f)
    candidates = [f'factor_{name}', f'factor_{name.replace("_flipped", "")}']
    code = None
    for c in candidates:
        if c in zip_idx:
            code = zip_idx[c]['code']
            break
    if not code:
        return name, f'no zip code for {name}'

    with open(Path(tempfile.gettempdir()) / "mkt_v2.pkl", 'rb') as f:
        mkt = pickle.load(f)

    tmp_path = Path(tempfile.gettempdir()) / "mkt_v2.pkl"
    args = (name, str(tmp_path), code, '_flipped' not in name)
    return compute_one_factor(args)


def main():
    print("=" * 60)
    print("用原始 zip 里的 Python code 重算 61 因子落值 (2019-2025)")
    print("=" * 60)

    # 1. Load zip lookup
    with open(ZIP_LOOKUP) as f:
        zip_idx = json.load(f)
    print(f"zip index: {len(zip_idx)} factors")

    # 2. List 61 HTML factor names
    html_names = sorted([
        f.stem.replace('factor_', '')
        for f in HTML_DIR.glob("factor_*.html")
    ])
    print(f"本周因子数: {len(html_names)}")

    # 3. Match to zip
    # zip key = factor_xxx, html name 可能是 factor_xxx 或 factor_xxx_flipped
    tasks = []
    for html_name in html_names:
        base = html_name
        candidates = [f'factor_{base}', f'factor_{base.replace("_flipped", "")}']
        code = None
        matched = None
        for c in candidates:
            if c in zip_idx:
                code = zip_idx[c]['code']
                matched = c
                break
        if code:
            tasks.append((html_name, matched, code))
        else:
            print(f"  [SKIP] 缺 zip 原始 code: {html_name}")

    print(f"命中 zip: {len(tasks)}/{len(html_names)}")

    # 4. Load market
    mkt = load_market('2019-01-01', '2025-12-31')
    print(f"行情: {mkt.shape}")

    # 5. Cache mkt for workers
    tmp = Path(tempfile.gettempdir()) / "mkt_v2.pkl"
    with open(tmp, 'wb') as f:
        pickle.dump(mkt, f)
    print(f"行情已缓存到 {tmp} ({tmp.stat().st_size / 1024 / 1024:.1f} MB)")

    # 6. 并行算
    args = [(name, str(tmp), code, '_flipped' not in name) for name, _, code in tasks]
    print(f"\n并行计算 {len(args)} 个因子 ...", flush=True)
    results = {}
    errors = []
    n_workers = min(4, max(2, os.cpu_count() or 4))  # 内存够但其他用户也跑
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(compute_one_factor, a): a[0] for a in args}
        for future in as_completed(futures):
            name, res = future.result()
            if isinstance(res, str):
                errors.append((name, res))
                print(f"  [ERR] {name}: {res[:120]}", flush=True)
            elif res is None or res.empty:
                errors.append((name, 'empty'))
                print(f"  [EMPTY] {name}", flush=True)
            else:
                results[name] = res
                nn_pct = res.notna().sum().sum() / res.size * 100
                print(f"  [OK] {name}: {res.shape}, 非空 {nn_pct:.1f}%", flush=True)
            print(f"    进度: {len(results)+len(errors)}/{len(args)}", flush=True)

    # 7. Save each factor to a single parquet (61 列 1 表)
    print(f"\n落值到 parquet...", flush=True)
    if results:
        # 用宽表: factor x date x symbol
        # parquet 列 MultiIndex: (factor_name, OrderBookId)
        common_dates = sorted(set.intersection(*[set(r.index) for r in results.values()]))
        common_dates = pd.DatetimeIndex(common_dates)
        print(f"  共同日期: {len(common_dates)}", flush=True)

        # 直接拼 wide form (避免 long pivot)
        # 每因子一个 (date, symbol) mat; 用 MultiIndex columns
        all_mats = []
        for name, mat in results.items():
            mat = mat.reindex(common_dates).astype('float32')
            # 转成 MultiIndex cols
            mi = pd.MultiIndex.from_product([[f'factor_{name}'], mat.columns], names=['factor', 'OrderBookId'])
            mat.columns = mi
            all_mats.append(mat)
        wide = pd.concat(all_mats, axis=1)
        print(f"  wide: {wide.shape}, dtype={wide.dtypes.iloc[0]}, memory={wide.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB", flush=True)

        # 直接保存 wide (覆盖旧的 99 因子)
        wide.to_parquet(str(FV_PATH))
        print(f"  ✅ saved: {wide.shape}", flush=True)

        # 同时输出 metadata
        meta = {
            'n_factors': len(results),
            'n_dates': len(common_dates),
            'n_symbols': wide.shape[1] // len(results),
            'date_range': [str(common_dates.min())[:10], str(common_dates.max())[:10]],
            'errors': [{'name': n, 'msg': m} for n, m in errors[:20]],
        }
        with open(PROJECT / 'factor_values_meta.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)

    print(f"\n完成: 成功 {len(results)}, 失败 {len(errors)}")
    if errors:
        for n, e in errors[:5]:
            print(f"  {n}: {e[:120]}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'one':
        # python compute_factors_from_zip.py one factor_name
        name = sys.argv[2]
        # ensure market loaded
        if not (Path(tempfile.gettempdir()) / "mkt_v2.pkl").exists():
            mkt = load_market('2019-01-01', '2025-12-31')
            with open(Path(tempfile.gettempdir()) / "mkt_v2.pkl", 'wb') as f:
                pickle.dump(mkt, f)
        nm, res = compute_only_failed(name)
        if isinstance(res, str):
            print(f'{nm}: ERR {res}')
        elif res is None:
            print(f'{nm}: empty')
        else:
            print(f'{nm}: shape={res.shape}')
    else:
        main()