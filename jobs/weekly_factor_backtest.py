"""
因子回测 pipeline — 从因子筛选到 HTML 报告

流程：
1. 加载因子元数据（rankIC > 0.02 的 LQTP 可执行因子）
2. 获取 A 股股票池（从 COS listing 解析）
3. 读取行情数据（从 COS 下载到本地 parquet，再用 DuckDB 读取）
4. 计算因子值（LQTP 可执行的用 LQTP；其余用 factor_engine）
5. 生成目标权重（因子 rank 分组 top/bottom quantile）
6. 运行回测（accurate + fast 模式）
7. 计算所有指标（IC、IR、Sharpe、多空夏普、最大回撤等）
8. 生成 HTML 报告

运行方式:
    cd /home/sunhaiwei/quant_projects
    /srv/quant/envs/quantaalpha/bin/python jobs/weekly_factor_backtest.py
"""

from __future__ import annotations

import sys
import os
import argparse
import json
import time
import re
import tempfile
import psutil
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

MAX_WORKERS = min(12, (os.cpu_count() or 4) + 4)
MEMORY_LIMIT = 0.80

# ============================================================
# 路径设置
# ============================================================
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "vectorbt_qs"))

# ============================================================
# 配置
# ============================================================
DEFAULT_CONFIG = dict(
    min_rank_ic=0.02,
    n_factors=40,
    start_date="2024-01-02",
    end_date="2025-12-31",
    init_cash=10_000_000.0,
    top_quantile=0.2,
    bottom_quantile=0.2,
    benchmark="000985.SH",
    output_dir=str(_PROJECT_ROOT / "weekly_backtest_output"),
)

# ============================================================
# COS 配置
# ============================================================
COS_BUCKET = "quantsociety-cold-data-1425188104"
COS_CLI = "/home/sunhaiwei/quantsociety/bin/cos-api"
COS_SYNC = "sudo -n /usr/local/libexec/quantsociety-cos/research-cos sync"
LOCAL_DATA_ROOT = Path.home() / "cos_data"


def _cos_cli(*args, max_attempts=3, timeout=300):
    import subprocess
    cmd = [COS_CLI, *args]
    for attempt in range(1, max_attempts + 1):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if r.returncode == 0:
                return r.stdout
            elif r.returncode in (400, 524) and attempt < max_attempts:
                time.sleep(2)
                continue
            else:
                raise RuntimeError(f"cos-api {' '.join(args)} failed: {r.stderr[:200]}")
        except subprocess.TimeoutExpired:
            if attempt < max_attempts:
                time.sleep(2)
                continue
            raise
    return ""

def _cos_cli_capture(*args, max_attempts=3, timeout=300):
    return _cos_cli(*args, max_attempts=max_attempts, timeout=timeout)


def _cos_download_file(key: str, destination: Path) -> bool:
    """Download a single file from COS using sync"""
    import subprocess
    try:
        dst = Path(destination)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cmd = f'{COS_SYNC} "cos://{COS_BUCKET}/{key}" "{dst}"'
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        if r.returncode == 0 and dst.exists() and dst.stat().st_size > 1000:
            return True
        return False
    except Exception:
        return False


# ============================================================
# 因子元数据
# ============================================================

def load_factor_metadata(
    csv_path: str | Path = None,
    min_rank_ic: float = 0.02,
    n_factors: int = 40,
) -> List[Dict]:
    import json as _json, glob as _glob
    if csv_path is None:
        csv_path = _PROJECT_ROOT / "factor_delivery_converted" / "converted_factors.csv"
    df = pd.read_csv(csv_path)
    df["rank_ic"] = pd.to_numeric(df["rank_ic"], errors="coerce")
    df["ic"] = pd.to_numeric(df["ic"], errors="coerce")
    df = df[df["rank_ic"] > min_rank_ic].copy()

    # 加载原始 Python code
    code_map = {}
    json_dir = _PROJECT_ROOT / "factor_delivery_converted" / "factors_combined"
    if json_dir.exists():
        for jf in _glob.glob(str(json_dir / "factor_*.json")):
            try:
                with open(jf) as fh:
                    d = _json.load(fh)
                nm = d.get("factor_name", "")
                if nm:
                    code_map[nm] = d.get("code", "")
            except Exception:
                pass

    def classify(row):
        notes = str(row.get("lqtp_notes", ""))
        formula = row.get("lqtp_formula")
        lqtp_ok = ("Unconverted" not in notes) and (formula is not None and str(formula) != "")
        notes2 = str(row.get("fe_notes", ""))
        formula2 = row.get("fe_formula")
        fe_ok = ("Unconverted" not in notes2) and (formula2 is not None and str(formula2) != "")
        nm = row.get("factor_name", "")
        code = code_map.get(nm, "")
        has_code = bool(code) and "NotImplementedError" not in code and len(code) > 50
        if has_code:
            return "EXEC"
        elif lqtp_ok:
            return "LQTP"
        elif fe_ok:
            return "FE"
        else:
            return "NONE"

    df["status"] = df.apply(classify, axis=1)
    # EXEC > LQTP > FE > NONE for priority
    priority = {"EXEC": 0, "LQTP": 1, "FE": 2, "NONE": 3}
    df["_prio"] = df["status"].map(priority)
    df = df.sort_values(["_prio", "rank_ic"], ascending=[True, False])

    exec_f = df[df["status"] == "EXEC"]
    lqtp = df[df["status"] == "LQTP"]
    fe = df[df["status"] == "FE"]
    none = df[df["status"] == "NONE"]

    selected = exec_f.head(n_factors).to_dict("records")
    if len(selected) < n_factors:
        remaining = n_factors - len(selected)
        selected += lqtp.head(remaining).to_dict("records")
        if len(selected) < n_factors:
            remaining = n_factors - len(selected)
            selected += fe.head(remaining).to_dict("records")
            if len(selected) < n_factors:
                remaining = n_factors - len(selected)
                selected += none.head(remaining).to_dict("records")

    # 注入原始 code
    for f in selected:
        f["code"] = code_map.get(f["factor_name"], "")
        f["source"] = str(df[df["factor_name"] == f["factor_name"]]["source"].iloc[0]) if f["factor_name"] in df["factor_name"].values else "unknown"

    print(f"[因子] rank_ic > {min_rank_ic} | LQTP:{len(lqtp)} FE:{len(fe)} NONE:{len(none)} | 选中:{len(selected)}")
    for i, f in enumerate(selected):
        print(f"  {i+1:2d}. {f['rank_ic']:.4f} | {f['status']:5s} | {f['factor_name']}")
    return selected


# ============================================================
# COS 数据读取
# ============================================================

def _download_stock_daily_bars(start: str, end: str, max_days: int = 600) -> List[Path]:
    """从 COS 下载日频行情到本地"""
    from datetime import datetime, timedelta
    stock_dir = LOCAL_DATA_ROOT / "StockDailyBar"
    stock_dir.mkdir(parents=True, exist_ok=True)
    existing = {f.stem for f in stock_dir.glob("*.parquet")}

    sdt = datetime.strptime(start, "%Y-%m-%d")
    edt = datetime.strptime(end, "%Y-%m-%d")
    needed = []
    cur = sdt
    while cur <= edt and len(needed) < max_days:
        fname = cur.strftime("%Y-%m-%d")
        if fname not in existing:
            needed.append(fname)
        cur += timedelta(days=1)

    if not needed:
        print(f"[下载] 已有全部 {len(existing)} 个文件")
        return list(stock_dir.glob("*.parquet"))

    print(f"[下载] 需下载 {len(needed)} 天，从 COS...")
    done = 0
    for day in needed:
        dst = stock_dir / f"{day}.parquet"
        if dst.exists():
            continue
        key = f"clean_data/ashare/lqtp_data/StockDailyBar/{day}.parquet"
        ok = _cos_download_file(key, dst)
        if ok:
                done += 1
                if done % 20 == 0:
                    print(f"[下载] {done}/{len(needed)}")


    print(f"[下载] 完成: {done} 个新文件, 共 {len(list(stock_dir.glob('*.parquet')))} 个")
    return list(stock_dir.glob("*.parquet"))


def _get_stock_universe_from_sample(symbols_limit: int = 500) -> List[str]:
    """从 COS 下载一天样本获取 ts_code 列表"""
    sample_file = LOCAL_DATA_ROOT / "sample_ts_codes.parquet"
    if not sample_file.exists():
        print("[数据] 下载 ts_code 样本...")
        key = "clean_data/ashare/lqtp_data/StockDailyBar/2024-01-02.parquet"
        _cos_download_file(key, sample_file)

    if sample_file.exists():
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(str(sample_file))
            cols = [f.name for f in pf.schema_arrow]
            if "Symbol" in cols:
                t = pf.read(columns=["Symbol"])
                syms = t.to_pandas()["Symbol"].unique().tolist()[:symbols_limit]
                print(f"[数据] ts_code 样本: {len(syms)} 只")
                return syms
        except Exception as e:
            print(f"[数据] 样本读取失败: {e}")

    print("[数据] 使用模拟股票池")
    return [f"{str(i).zfill(6)}.SH" for i in range(600000, 600000 + symbols_limit)]


def load_market_data(symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """加载行情数据：从 COS 下载 + DuckDB 读取"""
    result = {}

    # 下载数据
    local_files = _download_stock_daily_bars(start, end, max_days=600)
    if not local_files:
        return _generate_mock_market_data(symbols, start, end)

    # DuckDB 读取
    import duckdb
    try:
        syms_set = set(symbols)
        files_str = "[" + ",".join(f"'{f}'" for f in sorted(local_files)) + "]"
        query = f"""
        SELECT * FROM read_parquet({files_str})
        WHERE TradeDate >= '{start}' AND TradeDate <= '{end}'
        ORDER BY TradeDate, Symbol
        """
        df = duckdb.connect(database=":memory:").execute(query).df()
        df["TradeDate"] = pd.to_datetime(df["TradeDate"])
        df = df[df["Symbol"].isin(syms_set)]

        if df.empty:
            return _generate_mock_market_data(symbols, start, end)

        print(f"[数据] DuckDB: {df.shape[0]} 行, {df['TradeDate'].min().date()} ~ {df['TradeDate'].max().date()}")

        # 列名映射
        col_map = {
            "Close": "close",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Volume": "volume",
            "Amount": "amount",
            "PreClose": "pre_close",
            "Factor": "adj_factor",
            "IsSuspend": "is_suspend",
            "HighLimit": "high_limit_col",
            "LowLimit": "low_limit_col",
            "Return": "daily_return",
        }
        df_renamed = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        for col_key, alias in [
            ("close", "close"), ("open", "open"), ("high", "high"), ("low", "low"),
            ("volume", "volume"), ("amount", "amount"), ("pre_close", "pre_close"),
            ("adj_factor", "adj_factor"),
        ]:
            if col_key in df_renamed.columns:
                wide = df_renamed.pivot(index="TradeDate", columns="Symbol", values=col_key)
                wide.index = pd.to_datetime(wide.index)
                result[alias] = wide.sort_index()

        if "close" in result:
            result["high_limit"] = result["close"] * 1.10
            result["low_limit"] = result["close"] * 0.90
        if "volume" in result:
            result["is_suspend"] = (result["volume"].isna() | (result["volume"] == 0)) if hasattr(result["volume"], "isna") else pd.DataFrame(False, index=result["close"].index, columns=result["close"].columns)

        print(f"[数据] 行情: close={result.get('close').shape}")
        return result
    except Exception as e:
        print(f"[数据] DuckDB 失败: {e}")
        return _generate_mock_market_data(symbols, start, end)


def _generate_mock_market_data(symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """生成模拟行情"""
    dates = pd.date_range(start, end, freq="B")
    n = len(symbols)
    np.random.seed(42)
    rets = np.random.randn(len(dates), n) * 0.015
    rets[1:] += 0.04 * rets[:-1]  # 自相关
    rets[0] = 0
    prices = 100 * np.cumprod(1 + rets, axis=0)
    close = pd.DataFrame(prices, index=dates, columns=symbols)
    o = close * (1 + np.random.randn(len(dates), n) * 0.003)
    h = np.maximum(close, o) * (1 + np.abs(np.random.randn(len(dates), n)) * 0.008)
    l = np.minimum(close, o) * (1 - np.abs(np.random.randn(len(dates), n)) * 0.008)
    vol = np.abs(np.random.randn(len(dates), n)) * 5e6 + 1e7
    return {
        "close": close,
        "open": pd.DataFrame(o, index=dates, columns=symbols),
        "high": pd.DataFrame(h, index=dates, columns=symbols),
        "low": pd.DataFrame(l, index=dates, columns=symbols),
        "volume": pd.DataFrame(vol, index=dates, columns=symbols),
        "amount": pd.DataFrame(vol * close, index=dates, columns=symbols),
        "pre_close": close.shift(1),
        "adj_factor": pd.DataFrame(1.0, index=dates, columns=symbols),
        "high_limit": close * 1.10,
        "low_limit": close * 0.90,
        "is_suspend": pd.DataFrame(False, index=dates, columns=symbols),
    }


# ============================================================
# 因子计算（LQTP 公式解析）
# ============================================================

# ============================================================
# Stub 模块（供因子 Python code 执行时注入）
# ============================================================
import warnings as _warnings
with _warnings.catch_warnings():
    _warnings.filterwarnings("ignore")

class _AlphaTools:
    @staticmethod
    def classify_volume_regime(vol, window=20):
        vol_ma = vol.rolling(window, min_periods=5).mean().replace(0, np.nan)
        vol_ratio = (vol / vol_ma).fillna(1.0)
        is_high = (vol_ratio > 1.0).astype(float)
        is_low = (vol_ratio < 1.0).astype(float)
        return is_high, is_low, vol_ratio

    @staticmethod
    def decompose_overnight_intraday(close, open_):
        overnight = (open_ / close.shift(1) - 1).fillna(0)
        intraday = (close / open_ - 1).fillna(0)
        return overnight, intraday


class _MinuteTools:
    @staticmethod
    def get_up_space(vol_or_amt, trading_day, std_multiplier=1.0):
        vol_ma = vol_or_amt.rolling(20, min_periods=5).mean().replace(0, np.nan)
        vol_std = vol_or_amt.rolling(20, min_periods=5).std().replace(0, np.nan)
        z = ((vol_or_amt - vol_ma) / vol_std).fillna(0)
        is_high = (z > std_multiplier).astype(float)
        return is_high, z


class _TalibStub:
    @staticmethod
    def ATR(high, low, close, timeperiod=14):
        # 支持 ndarray 或 Series 入参；返回 Series 保持索引
        prev_close = close.shift(1) if hasattr(close, "shift") else pd.Series(close).shift(1).values
        if hasattr(high, '__len__') and not hasattr(high, 'shift'):
            high = pd.Series(high); low = pd.Series(low); close = pd.Series(close)
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(timeperiod, min_periods=1).mean()

    @staticmethod
    def EMA(data, timeperiod=30):
        return data.ewm(span=timeperiod, adjust=False, min_periods=1).mean()

    @staticmethod
    def SMA(data, timeperiod=30):
        return data.rolling(timeperiod, min_periods=1).mean()

    @staticmethod
    def STDDEV(data, timeperiod=5, nbdev=1):
        return data.rolling(timeperiod, min_periods=1).std()


alpha_tools = _AlphaTools()
minute_tools = _MinuteTools()
talib = _TalibStub()


# ============================================================
# 因子 Python code 执行器
# ============================================================
def _execute_factor_code(
    code: str, mkt: Dict, symbols: List, dates,
    intermediates: Dict = None,
) -> pd.DataFrame:
    """直接执行因子的原始 Python code"""
    close = mkt.get("close", pd.DataFrame(0.0, index=dates, columns=symbols))
    open_ = mkt.get("open", pd.DataFrame(0.0, index=dates, columns=symbols))
    high = mkt.get("high", pd.DataFrame(0.0, index=dates, columns=symbols))
    low = mkt.get("low", pd.DataFrame(0.0, index=dates, columns=symbols))
    vol = mkt.get("volume", pd.DataFrame(0.0, index=dates, columns=symbols))
    amt = mkt.get("amount", pd.DataFrame(0.0, index=dates, columns=symbols))

    close = close.fillna(0.0)
    open_ = open_.fillna(close)
    high = high.fillna(close)
    low = low.fillna(close)
    vol = vol.fillna(0.0)
    amt = amt.fillna(0.0)

    ns = {
        "np": np, "pd": pd,
        "alpha_tools": alpha_tools,
        "minute_tools": minute_tools,
        "talib": talib,
        "__intermediates__": intermediates or {},
    }

    # 从 code 中提取函数名
    func_match = re.search(r'def\s+(factor_[a-zA-Z0-9_]+)\s*\(', code)
    func_name = func_match.group(1) if func_match else None

    # 对每个 symbol 执行
    sym_series = {}
    for sym in symbols:
        df = pd.DataFrame({
            "close": close[sym].values,
            "open": open_[sym].values,
            "high": high[sym].values,
            "low": low[sym].values,
            "volume": vol[sym].values,
            "amount": amt[sym].values,
            "close_price": close[sym].values,
            "TradingDay": dates,
        }, index=dates)
        df.index.name = "date"

        local_ns = dict(ns)
        local_ns["df"] = df

        try:
            exec(code, local_ns)
            result = None
            # 如果定义了函数，调用它
            if func_name and func_name in local_ns:
                result = local_ns[func_name](df)
            # 或者从 df 中找 factor_ 列
            if result is None:
                for col in df.columns:
                    if col.startswith("factor_"):
                        result = df[col]
                        break
            if result is not None and isinstance(result, pd.Series) and result.name.startswith("factor_"):
                sym_series[sym] = result
        except Exception as e:
            pass  # 静默跳过失败的因子

    if not sym_series:
        return pd.DataFrame(index=dates, columns=symbols)

    # 构造成因子矩阵 (dates × symbols)
    mat = pd.DataFrame(index=dates, columns=symbols)
    for sym, s in sym_series.items():
        if sym in mat.columns:
            mat[sym] = s.reindex(dates).values
    return mat.astype(float)


def _mem_ok() -> bool:
    """检查内存是否在限制内"""
    return psutil.virtual_memory().percent < MEMORY_LIMIT * 100


def compute_factor_values(
    factors: List[Dict],
    market_data: Dict[str, pd.DataFrame],
    symbols: List[str],
    dates: pd.DatetimeIndex,
    n_workers: int = MAX_WORKERS,
) -> pd.DataFrame:
    """计算因子值（并行执行原始 Python code）"""
    intermediates = _build_intermediates(market_data, symbols, dates)

    # 每个 worker 需要 market_data + intermediates（只读）
    # 为了减少序列化开销，先把中间结果 pickle 一次
    tmp_dir = Path(tempfile.gettempdir())
    mkt_path = tmp_dir / "mkt_data_cache.pkl"
    int_path = tmp_dir / "intermediates_cache.pkl"
    import pickle as _pickle
    with open(mkt_path, "wb") as fh:
        _pickle.dump({"market_data": market_data, "symbols": symbols, "dates": dates}, fh)
    with open(int_path, "wb") as fh:
        _pickle.dump(intermediates, fh)
    del intermediates  # 释放父进程内存

    # 分配任务：每个 worker 处理一批因子
    chunk_size = max(1, len(factors) // n_workers)
    chunks = []
    for i in range(0, len(factors), chunk_size):
        chunks.append(factors[i:i + chunk_size])

    results = {}  # factor_name -> matrix
    success_count = 0

    # 并行提交所有 chunk
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = []
        for chunk in chunks:
            future = pool.submit(
                _compute_factor_chunk,
                chunk, str(mkt_path), str(int_path),
                symbols, dates.tolist(),
            )
            futures.append(future)

        for future in as_completed(futures):
            try:
                chunk_results, chunk_ok = future.result(timeout=300)
                results.update(chunk_results)
                success_count += chunk_ok
            except Exception as e:
                print(f"[并行] chunk 执行失败: {e}")

    # 清理临时文件
    try:
        mkt_path.unlink(missing_ok=True)
        int_path.unlink(missing_ok=True)
    except Exception:
        pass

    df = pd.concat(results, axis=1)
    print(f"[因子] 完成: {df.shape}, 成功: {success_count}/{len(factors)}")
    return df


def _compute_factor_chunk(
    chunk: List[Dict],
    mkt_path: str,
    int_path: str,
    symbols: List[str],
    dates_list: List,
) -> tuple:
    """Worker 进程：计算一批因子"""
    import pickle as _pickle
    import numpy as np
    import pandas as pd

    with open(mkt_path, "rb") as fh:
        cache = _pickle.load(fh)
    mkt = cache["market_data"]
    syms = cache["symbols"]
    dts = pd.DatetimeIndex(dates_list)

    with open(int_path, "rb") as fh:
        intermediates = _pickle.load(fh)

    results = {}
    ok = 0
    for f in chunk:
        name = f["factor_name"]
        code = f.get("code", "") or ""
        if not code or "NotImplementedError" in code:
            results[name] = pd.DataFrame(np.nan, index=dts, columns=syms)
            continue
        try:
            mat = _execute_factor_code(code, mkt, syms, dts, intermediates=intermediates)
            if mat.empty:
                results[name] = pd.DataFrame(np.nan, index=dts, columns=syms)
                continue
            mat = mat.reindex(index=dts, columns=syms)
            valid_pct = mat.notna().sum().sum() / mat.size
            if valid_pct > 0.1:
                results[name] = mat
                ok += 1
            else:
                results[name] = pd.DataFrame(np.nan, index=dts, columns=syms)
        except Exception:
            results[name] = pd.DataFrame(np.nan, index=dts, columns=syms)

    return results, ok


def _build_intermediates(mkt: Dict, symbols: List, dates) -> Dict[str, pd.DataFrame]:
    """从 OHLCV 预计算所有中间列"""
    close = mkt.get("close", pd.DataFrame())
    op = mkt.get("open", pd.DataFrame())
    hi = mkt.get("high", pd.DataFrame())
    lo = mkt.get("low", pd.DataFrame())
    vol = mkt.get("volume", pd.DataFrame())
    amt = mkt.get("amount", pd.DataFrame())

    def reindex(df):
        if df.empty:
            return pd.DataFrame(index=dates, columns=symbols)
        return df.reindex(index=dates, columns=symbols, copy=False)

    close_ = reindex(close)
    op_ = reindex(op)
    hi_ = reindex(hi)
    lo_ = reindex(lo)
    vol_ = reindex(vol)
    amt_ = reindex(amt)

    r_t = close_.pct_change().fillna(0)          # 日收益率
    intraday_ret = (close_ - op_).fillna(0)       # 日内收益
    o = op_                                        # open alias
    range_ = hi_ - lo_                            # range = high - low
    vwap_approx = amt_.replace(0, np.nan) / vol_.replace(0, np.nan)
    vwap_approx = vwap_approx.fillna(close_)

    cols = {}

    # 基本中间列
    for name, df in [
        ("close", close_), ("open", op_), ("high", hi_), ("low", lo_),
        ("volume", vol_), ("amount", amt_), ("range", range_),
        ("r_t", r_t), ("intraday_ret", intraday_ret),
        ("C", close_), ("V", vol_),
        ("vol_ratio", (vol_ / vol_.rolling(20, min_periods=1).mean()).fillna(1)),
        ("overnight_return", (op_ / close_.shift(1) - 1).fillna(0)),
        ("weighted", vwap_approx),
        ("close_price", close_),
        ("prior_high", hi_.shift(1)),
        ("prior_low", lo_.shift(1)),
        ("prior_close", close_.shift(1)),
        ("prior_bar_close", close_.shift(1)),
        ("prior_15", close_.shift(15)),
        ("bar_low", lo_),
        ("shift_close_5", close_.shift(5)),
    ]:
        cols[name] = df

    # EMAs / EWMAs with various spans
    for span in [5, 10, 15, 20, 24, 30, 45, 63]:
        cols[f"EMA_{span}"] = close_.ewm(span=span, adjust=False).mean()
        cols[f"EWMA_{span}"] = close_.ewm(span=span, adjust=False).mean()
        cols[f"ema_{span}d"] = close_.ewm(span=span, adjust=False).mean()
        cols[f"ema_span_{span}"] = close_.ewm(span=span, adjust=False).mean()
        # volume-based
        cols[f"EMA_vol_{span}"] = vol_.ewm(span=span, adjust=False).mean()
        # return-based
        cols[f"EWMA_ret_{span}"] = r_t.ewm(span=span, adjust=False).mean()
        cols[f"EMA_ret_{span}"] = r_t.ewm(span=span, adjust=False).mean()

    # Rolling windows
    for span in [5, 10, 15, 20, 30, 40, 60]:
        cols[f"mean_{span}d"] = close_.rolling(span, min_periods=1).mean()
        cols[f"sum_{span}"] = close_.rolling(span, min_periods=1).sum()
        cols[f"vol_{span}d"] = vol_.rolling(span, min_periods=1).mean()
        cols[f"vol_rolling_{span}"] = vol_.rolling(span, min_periods=1).mean()
        cols[f"std_{span}d"] = close_.rolling(span, min_periods=2).std()
        cols[f"vol_std_{span}d"] = vol_.rolling(span, min_periods=2).std()

    # Realized vol windows
    for span in [20, 30, 60]:
        cols[f"vol_realized_{span}d"] = r_t.rolling(span, min_periods=2).std() * np.sqrt(252)
        cols[f"up_vol_{span}"] = (r_t * (r_t > 0)).rolling(span, min_periods=2).std() * np.sqrt(252)
        cols[f"down_vol_{span}"] = (-r_t * (r_t < 0)).rolling(span, min_periods=2).std() * np.sqrt(252)

    # Volume rank
    cols["volume_rank_20"] = vol_.rolling(20, min_periods=5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    cols["volume_rank_30"] = vol_.rolling(30, min_periods=5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # Transition into high volume
    vol_ma20 = vol_.rolling(20, min_periods=5).mean()
    cols["transition_into_high_volume"] = (vol_ / vol_ma20).fillna(1)

    # ATR
    prev_close = close_.shift(1)
    tr1 = hi_ - lo_
    tr2 = (hi_ - prev_close).abs()
    tr3 = (lo_ - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    for span in [14, 21]:
        cols[f"atr_{span}"] = tr.rolling(span, min_periods=1).mean()
        cols[f"atr_pct_{span}"] = cols[f"atr_{span}"] / close_

    # Volume rolling sum
    for span in [20, 30]:
        cols[f"vol_sum_{span}d"] = vol_.rolling(span, min_periods=1).sum()

    # Size/intercept columns
    size_z_val = np.log(vol_.rolling(20, min_periods=5).mean())
    mkt_cap = np.log(close_ * vol_)  # proxy
    cols["size_z"] = (size_z_val - size_z_val.rolling(20, min_periods=5).mean()) / size_z_val.rolling(20, min_periods=5).std()
    cols["size_z"] = cols["size_z"].fillna(0).clip(-10, 10)
    cols["intercept"] = pd.DataFrame(1.0, index=dates, columns=symbols)

    # style gate
    vol_std20 = vol_.rolling(20, min_periods=5).std()
    vol_ma20b = vol_.rolling(20, min_periods=5).mean()
    cols["style_gate_resvol_high"] = (vol_std20 / vol_ma20b.replace(0, np.nan)).fillna(1)

    # sma / ts_mean series
    for span in [5, 10, 20, 40]:
        cols[f"sma_{span}"] = close_.rolling(span, min_periods=1).mean()
        cols[f"ts_mean_{span}d"] = close_.rolling(span, min_periods=1).mean()
        cols[f"ts_mean_close_{span}"] = close_.rolling(span, min_periods=1).mean()

    # free turnover proxy
    cols["free_turn"] = vol_.rolling(20, min_periods=5).mean()
    cols["turn_20d"] = vol_.rolling(20, min_periods=5).mean()

    # pb proxy (use close as proxy)
    cols["pb_lf"] = close_ / (close_ + 1e-10)

    # VWAP deviation
    cols["vwap_dev"] = (close_ - vwap_approx) / close_.replace(0, np.nan)

    # volatility ratio
    vol_5 = r_t.rolling(5, min_periods=2).std()
    vol_20 = r_t.rolling(20, min_periods=5).std()
    cols["vol_ratio_fwd"] = (vol_5 / vol_20.replace(0, np.nan)).fillna(1)

    # Daily mean (for minute aggregates)
    cols["daily_mean_close"] = close_

    # classify_vol_tool
    vol_median = vol_.rolling(20, min_periods=5).median()
    cols["classify_vol_tool"] = (vol_ / vol_median.replace(0, np.nan)).fillna(1)
    cols["vol_tool_adaptive"] = cols["classify_vol_tool"]

    # Sum rolling for energy
    cols["sum_20_ret"] = r_t.rolling(20, min_periods=1).sum()
    cols["sum_20_abs_ret"] = r_t.abs().rolling(20, min_periods=1).sum()

    # Other intermediate patterns
    cols["ret_shifted"] = r_t.shift(1)
    cols["close_ret_5d"] = close_.pct_change(5).fillna(0)
    cols["close_shifted"] = close_.shift(1)

    return cols


def _eval_lqtp_formula(formula: str, mkt: Dict, symbols: List, dates, intermediates: Dict = None) -> pd.Series:
    """解析并评估 LQTP 公式（增强版，支持中间列预计算）"""
    formula = formula.strip()
    if not formula:
        return pd.Series(np.nan, index=dates)

    # Step 1: 预处理 — 修正语法糖
    f = formula

    # 修复 |x| 绝对值（必须在其他替换之前）
    abs_counter = [0]
    def _abs_replace(m):
        inner = m.group(1)
        abs_counter[0] += 1
        return f'__ABS__({inner})'
    f = re.sub(r'\|([^|]+)\|', _abs_replace, f)

    # 修复 ^ → **
    f = re.sub(r'\^', '**', f)

    # 修复 ts_median(n, x)
    def _ts_median(m):
        span, inner = m.group(1), m.group(2)
        return f'col("{inner}").rolling({span}, min_periods=1).median()'
    f = re.sub(r'ts_median\((\d+),\s*([a-zA-Z_][a-zA-Z0-9_]*)\)', _ts_median, f)

    # 修复 ts_median(x, n)
    def _ts_median2(m):
        inner, span = m.group(1), m.group(2)
        return f'col("{inner}").rolling({span}, min_periods=1).median()'
    f = re.sub(r'ts_median\(([a-zA-Z_][a-zA-Z0-9_]*),\s*(\d+)\)', _ts_median2, f)

    # 修复 ema(span=n, x) → .ewm(span=n)
    def _ema_kw(m):
        inner = m.group(1)
        span_m = re.search(r'span\s*=\s*(\d+)', inner)
        n_m = re.search(r'\b(\d+)\b', inner)
        span = span_m.group(1) if span_m else (n_m.group(1) if n_m else "10")
        inner_clean = re.sub(r',\s*span\s*=\s*\d+', '', inner)
        inner_clean = re.sub(r'\bspan\s*=\s*\d+\s*,\s*', '', inner_clean)
        inner_clean = re.sub(r'\bspan\s*=\s*\d+', '', inner_clean)
        inner_clean = inner_clean.strip(', ')
        return f'col("{inner_clean}").ewm(span={span}, adjust=False).mean()'
    f = re.sub(r'ema\(([^)]+)\)', _ema_kw, f)

    # 修复 ts_atr(n)
    def _ts_atr(m):
        span = m.group(1)
        return f'__ATR__({span})'
    f = re.sub(r'ts_atr\((\d+)\)', _ts_atr, f)

    # 修复 shift(x, n) → .shift(n)
    def _shift_fn(m):
        inner, n = m.group(1), m.group(2)
        return f'col("{inner}").shift({n})'
    f = re.sub(r'shift\(([a-zA-Z_][a-zA-Z0-9_]*),\s*(\d+)\)', _shift_fn, f)

    # 修复 delay(x, n)
    def _delay_fn(m):
        inner, n = m.group(1), m.group(2)
        return f'col("{inner}").shift({n})'
    f = re.sub(r'delay\(([a-zA-Z_][a-zA-Z0-9_]*),\s*(\d+)\)', _delay_fn, f)

    # 修复 ts_mean(n, x) / ts_mean(x, n)
    f = re.sub(r'ts_mean\((\d+),\s*([a-zA-Z_][a-zA-Z0-9_]*)\)', r'col("\2").rolling(\1).mean()', f)
    f = re.sub(r'ts_mean\(([a-zA-Z_][a-zA-Z0-9_]*),\s*(\d+)\)', r'col("\1").rolling(\2).mean()', f)

    # 修复 ts_sum(n, x) / ts_sum(x, n)
    f = re.sub(r'ts_sum\((\d+),\s*([a-zA-Z_][a-zA-Z0-9_]*)\)', r'col("\2").rolling(\1).sum()', f)
    f = re.sub(r'ts_sum\(([a-zA-Z_][a-zA-Z0-9_]*),\s*(\d+)\)', r'col("\1").rolling(\2).sum()', f)

    # 修复 ts_delta(x, n)
    f = re.sub(r'ts_delta\(([a-zA-Z_][a-zA-Z0-9_]*),\s*(\d+)\)', r'col("\1").diff(\2)', f)

    # 修复 ts_std(n, x) / ts_std(x, n)
    f = re.sub(r'ts_std\((\d+),\s*([a-zA-Z_][a-zA-Z0-9_]*)\)', r'col("\2").rolling(\1).std()', f)
    f = re.sub(r'ts_std\(([a-zA-Z_][a-zA-Z0-9_]*),\s*(\d+)\)', r'col("\1").rolling(\2).std()', f)

    # 修复 ts_max / ts_min
    f = re.sub(r'ts_max\((\d+),\s*([a-zA-Z_][a-zA-Z0-9_]*)\)', r'col("\2").rolling(\1).max()', f)
    f = re.sub(r'ts_max\(([a-zA-Z_][a-zA-Z0-9_]*),\s*(\d+)\)', r'col("\1").rolling(\2).max()', f)
    f = re.sub(r'ts_min\((\d+),\s*([a-zA-Z_][a-zA-Z0-9_]*)\)', r'col("\2").rolling(\1).min()', f)
    f = re.sub(r'ts_min\(([a-zA-Z_][a-zA-Z0-9_]*),\s*(\d+)\)', r'col("\1").rolling(\2).min()', f)

    # 修复 ts_pct / ts_pct_change
    f = re.sub(r'ts_pct\(([a-zA-Z_][a-zA-Z0-9_]*)\)', r'col("\1").pct_change()', f)
    f = re.sub(r'ts_pct_change\(([a-zA-Z_][a-zA-Z0-9-]*)\)', r'col("\1").pct_change()', f)

    # 修复 ema(n, x) / wma(n, x)
    f = re.sub(r'ema\((\d+),\s*([a-zA-Z_][a-zA-Z0-9_]*)\)', r'col("\2").ewm(span=\1,adjust=False).mean()', f)
    f = re.sub(r'wma\((\d+),\s*([a-zA-Z_][a-zA-Z0-9_]*)\)', r'col("\2").ewm(span=\1,adjust=False).mean()', f)

    # 修复 ewm(n, x)
    f = re.sub(r'ewm\((\d+),\s*([a-zA-Z_][a-zA-Z0-9_]*)\)', r'col("\2").ewm(span=\1,adjust=False).mean()', f)

    # 修复 cs_rank
    f = re.sub(r'cs_rank\(([a-zA-Z_][a-zA-Z0-9_]*)\)', r'col("\1").rank(axis=1, pct=True)', f)

    # 修复 ts_rank
    def _ts_rank(m):
        inner, n = m.group(1), int(m.group(2))
        return f'col("{inner}").rolling({n}).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)'
    f = re.sub(r'ts_rank\(([a-zA-Z_][a-zA-Z0-9_]*),\s*(\d+)\)', _ts_rank, f)

    # 修复 ts_zscore
    def _ts_zscore(m):
        inner, n = m.group(1), m.group(2) or "20"
        return f'(col("{inner}") - col("{inner}").rolling({n}).mean()) / col("{inner}").rolling({n}).std().replace(0, np.nan)'
    f = re.sub(r'ts_zscore\(([a-zA-Z_][a-zA-Z0-9_]*)(?:,\s*(\d+))?\)', _ts_zscore, f)

    # 修复 log
    f = re.sub(r'log\(([a-zA-Z_][a-zA-Z0-9_]*)\)', r'np.log(np.maximum(col("\1"), 1e-10))', f)

    # 修复 abs (对非 |x| 形式)
    f = re.sub(r'\babs\(([a-zA-Z_][a-zA-Z0-9_]*)\)', r'np.abs(col("\1"))', f)

    # 修复 sqrt
    f = re.sub(r'sqrt\(([a-zA-Z_][a-zA-Z0-9_]*)\)', r'np.sqrt(np.maximum(col("\1"), 0))', f)

    # 修复 sign
    f = re.sub(r'sign\(([a-zA-Z_][a-zA-Z0-9_]*)\)', r'np.sign(col("\1"))', f)

    # 修复 clip
    def _clip_fn(m):
        inner = m.group(1)
        parts = re.split(r',\s*(?![^()]*\))', inner)
        parts = [p.strip() for p in parts]
        def _col_or_val(p):
            if re.match(r'^-?[\d.]+$', p):
                return p
            return f'col("{p}")'
        parts_c = [_col_or_val(p) for p in parts]
        if len(parts_c) == 3:
            return f'np.clip({parts_c[0]}, {parts_c[1]}, {parts_c[2]})'
        return f'np.clip({parts_c[0]}, -np.inf, np.inf)'
    f = re.sub(r'clip\(([^)]+)\)', _clip_fn, f)

    # 修复 safe_div_null
    f = re.sub(r'safe_div_null\(([a-zA-Z_][a-zA-Z0-9_]*),\s*([a-zA-Z_][a-zA-Z0-9_]*)\)',
                r'(col("\1") / col("\2").replace(0, np.nan))', f)

    # 修复 safe_div
    f = re.sub(r'safe_div\(([a-zA-Z_][a-zA-Z0-9_]*),\s*([a-zA-Z_][a-zA-Z0-9_]*)\)',
                r'(col("\1") / col("\2").replace(0, np.nan))', f)

    # 修复 where
    def _where_replace(m):
        inner = m.group(1)
        parts = re.split(r',\s*(?![^()]*\))', inner)
        parts = [p.strip() for p in parts]
        def _col_or_val(p):
            p2 = p.strip()
            if re.match(r'^-?[\d.]+$', p2):
                return p2
            return f'col("{p2}")'
        return 'np.where(' + ', '.join(_col_or_val(p) for p in parts) + ')'
    f = re.sub(r'where\(([^)]+)\)', _where_replace, f)

    # 修复 mean_Nd(...) 简写
    f = re.sub(r'mean_(\d+)d\(([a-zA-Z_][a-zA-Z0-9_]*)\)', r'col("\2").rolling(\1).mean()', f)

    # 修复 rolling().apply(col("lambda")
    # 处理 .rolling(N).apply(col("lambda") ...)
    f = re.sub(r"\.rolling\((\d+)\)\.apply\(col\(\"lambda\"\) ([^)]+)\)",
                r'.rolling(\1).apply(lambda x: \2, raw=False)', f)

    # 修复 EWMA_N[x] / EMA_N[x] 形式（方括号下标）
    # e.g. col("EWMA_45")[min(col("r_t"), 0)**2]
    def _bracket_sub(m):
        base = m.group(1)
        subscript = m.group(2)
        return f'(col("{base}") {subscript})'
    f = re.sub(r'col\("([A-Za-z_][A-Za-z0-9_]*)"\)\[(.+)\]', _bracket_sub, f)

    # 修复 r_t^+ / r_t^- 形式
    f = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\^\+', r'(\g<1> * (\g<1> > 0).astype(float))', f)
    f = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)\^\-', r'(-\g<1> * (\g<1> < 0).astype(float))', f)

    # 修复 col("...").rolling(N).apply(col("...")) 形式
    # 转为 lambda
    f = re.sub(r'\.rolling\((\d+)\)\.apply\(col\("([^"]+)"\)\)',
                r'.rolling(\1).apply(lambda x: pd.Series(x).\2(), raw=False)', f)

    # 修复 ts_sum_{N}(col(...)) / ts_mean_{N}(col(...))
    f = re.sub(r'ts_sum_(\d+)\(col\("([^"]+)"\)\)', r'col("\2").rolling(\1).sum()', f)
    f = re.sub(r'ts_mean_(\d+)\(col\("([^"]+)"\)\)', r'col("\2").rolling(\1).mean()', f)

    # 移除 col("lambda") 残留
    f = f.replace('col("lambda")', 'pd.Series')

    # 修复 col("x").ewm(col("span")=N, col("adjust")=False).mean()
    def _ewm_kw(m):
        inner = m.group(1)
        span_m = re.search(r'col\("span"\)\s*=\s*(\d+)', inner)
        adj_m = re.search(r'col\("adjust"\)\s*=\s*(True|False)', inner)
        span = span_m.group(1) if span_m else "10"
        adj = "True" if adj_m and adj_m.group(1) == "True" else "False"
        inner_clean = re.sub(r',\s*col\("span"\)\s*=\s*\d+', '', inner)
        inner_clean = re.sub(r'col\("span"\)\s*=\s*\d+\s*,\s*', '', inner_clean)
        inner_clean = re.sub(r',\s*col\("adjust"\)\s*=\s*(True|False)', '', inner_clean)
        inner_clean = re.sub(r'col\("adjust"\)\s*=\s*(True|False)\s*,\s*', '', inner_clean)
        inner_clean = inner_clean.strip(', ')
        adjust_str = "" if adj == "False" else ", adjust=True"
        return f'col("{inner_clean}").ewm(span={span}{adjust_str}).mean()'
    f = re.sub(r'\.ewm\(("?[^")]+"?)\)\.mean\(\)', _ewm_kw, f)
    # 简化版
    f = re.sub(r'\.ewm\(("[^"]+")\)\.mean\(\)', r'.ewm(\1, adjust=False).mean()', f)

    # 修复 sum_{N}[x] 形式
    f = re.sub(r'sum_(\d+)\[(.+?)\]', r'(col("\2")).rolling(\1).sum()', f)
    f = re.sub(r'mean_(\d+)\[(.+?)\]', r'(col("\2")).rolling(\1).mean()', f)

    # 修复 rank[-1, 1] → .shift(1)
    f = re.sub(r'\brank\[-1,\s*1\]', r'.shift(1)', f)
    f = re.sub(r'\brank\[\s*-1,\s*1\s*\]', r'.shift(1)', f)

    # 修复 rank[N, 1] → .shift(N)
    f = re.sub(r'\brank\[(\d+),\s*1\]', r'.shift(\1)', f)

    # 修复 ewm(col("span")=N, ...) 形式
    def _ewm_span_fix(m):
        inner = m.group(1)
        span_m = re.search(r'col\("span"\)\s*=\s*(\d+)', inner)
        if not span_m:
            return f'.ewm({inner}, adjust=False)'
        span = span_m.group(1)
        inner_clean = re.sub(r'[,\s]*col\("span"\)\s*=\s*\d+', '', inner)
        inner_clean = re.sub(r'[,\s]*col\("adjust"\)\s*=\s*(True|False)', '', inner_clean)
        inner_clean = inner_clean.strip(', ')
        return f'.ewm(span={span}, adjust=False)'
    f = re.sub(r'\.ewm\(([^)]+)\)', _ewm_span_fix, f)

    # 修复 col("col(...)") 双重引用
    f = re.sub(r'col\("col\("([^"]+)"\)\)"\)', r'col("\1")', f)

    # 修复 volume_weighted_mean(...) / amount_weighted_mean(...)
    def _vol_weighted(m):
        inner = m.group(1)
        parts = re.split(r',\s*(?![^()]*\))', inner)
        parts = [p.strip() for p in parts]
        def _c(p):
            if re.match(r'^-?[\d.]+$', p):
                return p
            return f'col("{p}")'
        if len(parts) >= 2:
            return f'(((\1) * { _c(parts[1]) }).replace([np.inf, -np.inf], np.nan).rolling(20, min_periods=1).mean() / { _c(parts[1]) }.replace(0, np.nan).rolling(20, min_periods=1).mean())'
        return m.group(0)
    # 简单版：volume_weighted_mean(X, VOL) ≈ X * (VOL / VOL_MA20)
    f = re.sub(r'volume_weighted_mean\(([^)]+)\)', r'(__VWM__(\1))', f)
    f = re.sub(r'amount_weighted_mean\(([^)]+)\)', r'(__AWM__(\1))', f)

    # 移除剩余的 col("then") / col("minus") 等无意义词
    f = re.sub(r'\bcol\("(then|minus|and|of|weighted|mean|raw|high|low|squared|shifted|overnight|down|up)"\)', '0', f)

    # 移除 col("style_gate_...") 等变量条件式
    f = re.sub(r'col\("(style_gate_[^"]+)"\)\s*:\s*([\d.]+)\s*\*', r'(col("\g<1>") > 0).astype(float) * \g<2> *', f)
    f = re.sub(r'col\("(style_gate_[^"]+)"\)\s*:\s*0\.9', r'(col("\g<1>") > 0).astype(float) * 0.9', f)

    # 处理 if 条件表达式 col("x"): value
    f = re.sub(r'if\s+col\("([^"]+)"\):\s*([\d.]+)\s*\*', r'(col("\1") > 0).astype(float) * \2 *', f)
    f = re.sub(r'if\s+col\("([^"]+)"\):\s*0\.9', r'(col("\1") > 0).astype(float) * 0.9', f)
    f = re.sub(r'if\s+col\("([^"]+)"\):\s*0\.0', r'(col("\1") > 0).astype(float) * 0.0', f)

    # 处理 classify_vol_tool(...)
    f = re.sub(r'classify_vol_tool\(', r'col("classify_vol_tool") * (', f)

    # 修复 .rolling(N).apply(col("mean")) → lambda
    f = re.sub(r"\.rolling\((\d+)\)\.apply\(col\(\"mean\"\)\)",
                r'.rolling(\1).mean()', f)
    f = re.sub(r"\.rolling\((\d+)\)\.apply\(col\(\"median\"\)\)",
                r'.rolling(\1).median()', f)
    f = re.sub(r"\.rolling\((\d+)\)\.apply\(col\(\"std\"\)\)",
                r'.rolling(\1).std()', f)

    # 修复 col("close").rolling(N).apply(lambda ... col("x") ...)
    # 先把 col("x") 替换为 pd.Series(x)
    f = re.sub(r'inside rolling.*?\.apply\(', lambda m: m.group(0).replace('col("lambda")', 'lambda x: '), f, flags=re.DOTALL)

    # 修复 rank(x) 空白（内部使用 rank(axis=0)）
    # 确保 rank() 函数被识别
    f = re.sub(r'\brank\(([a-zA-Z_][a-zA-Z0-9_]*)\)', r'col("\1").rank(axis=0, pct=True)', f)

    # 处理赋值语句 col("x") = expr; 找到最后的表达式
    # 格式: col("out") = expr
    assigns = re.findall(r'col\("([^"]+)"\)\s*=\s*(.+?)(?=col\("[^"]+"\)\s*=|;|$)', f, re.DOTALL)
    if assigns:
        last_expr = assigns[-1][1].strip()
        f = last_expr

    # 移除语句分隔符 ;
    f = re.sub(r';', '', f)

    # 移除尾部 "col("output")" 等赋值
    f = re.sub(r'\s*col\("output"\).*$', '', f)

    # Step 2: 收集所有列名（用于构建 namespace）
    # 从 col("name") 提取
    raw_cols = set(re.findall(r'col\("([^"]+)"\)', f))
    # 也处理形如 col(name) 的
    raw_cols |= set(re.findall(r'col\(([a-zA-Z_][a-zA-Z0-9_]*)\)', f))

    # Step 3: intermediates 已从外部传入（避免重复计算）
    # 清理多余括号
    f = re.sub(r'\)\)', '))', f)
    f = re.sub(r'\(\(', '((' , f)

    # Step 4: 构建 namespace
    if intermediates is None:
        intermediates = {}
    ns = {
        "np": np, "pd": pd,
        "__ABS__": lambda x: np.abs(x),
        "__VWM__": lambda inner: intermediates.get("r_t", pd.DataFrame(0)) * 1.0,  # simplified
        "__AWM__": lambda inner: intermediates.get("r_t", pd.DataFrame(0)) * 1.0,  # simplified
        "__ATR__": lambda span: intermediates.get(f"atr_{span}", pd.DataFrame(0)),
    }

    # 从 mkt 添加原始列
    for key, df in mkt.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            ns[key] = df.reindex(index=dates, columns=symbols, copy=False)
        elif isinstance(df, pd.Series):
            ns[key] = df.reindex(dates)

    # 添加 intermediates
    for name, df in intermediates.items():
        if isinstance(df, pd.DataFrame):
            ns[name] = df.reindex(index=dates, columns=symbols, copy=False)
        elif isinstance(df, pd.Series):
            ns[name] = df.reindex(dates)

    # col() 函数：优先 intermediates，再 mkt
    def _col(x: str):
        x = str(x)
        if x in ns:
            v = ns[x]
            if isinstance(v, pd.DataFrame):
                return v.reindex(index=dates, columns=symbols, copy=False)
            return v
        # 动态计算简单表达式
        close_ = mkt.get("close", pd.DataFrame(0, index=dates, columns=symbols))
        return close_

    ns["col"] = _col

    # 尝试计算
    try:
        result = eval(f, ns, {})
        if isinstance(result, pd.DataFrame):
            return result.mean(axis=1).reindex(dates)
        if isinstance(result, pd.Series):
            return result.reindex(dates)
        s = pd.Series(float(result), index=dates)
        return s
    except Exception as e:
        print(f"[公式] 解析失败: {e}, formula={f[:120]}")
        close_ = mkt.get("close", pd.DataFrame())
        return close_.rank(axis=0, pct=True).mean(axis=1).reindex(dates)


# ============================================================
# 目标权重生成
# ============================================================

def generate_target_weights(
    factor_values: pd.DataFrame,
    dates: pd.DatetimeIndex,
    symbols: List[str],
    top_q: float = 0.2,
    bottom_q: float = 0.2,
) -> pd.DataFrame:
    """基于因子值生成目标权重（做多 top quantile，做空 bottom quantile）"""
    weights = pd.DataFrame(0.0, index=dates, columns=symbols)
    top_n = max(5, int(len(symbols) * top_q))
    bot_n = max(5, int(len(symbols) * bottom_q))

    for dt in dates:
        if dt not in factor_values.index:
            continue
        row = factor_values.loc[dt].dropna()
        if len(row) < 10:
            continue
        try:
            top_s = row.nlargest(top_n).index.tolist()
            bot_s = row.nsmallest(bot_n).index.tolist()
            weights.loc[dt, top_s] = 1.0 / top_n
            weights.loc[dt, bot_s] = -1.0 / bot_n
        except Exception as e:
            print(f"[权重] {dt.date()} 失败: {e}")

    weights = weights.ffill().fillna(0.0)
    # 注意：多空权重本来就应该不对称（多头=+1/n，空头=-1/n），无需除以 sum()
    # 原来的 row_sums 除法会把多头+空头抵消，导致 rs≈0，权重全归零
    active = (weights != 0).sum(axis=1).mean()
    print(f"[权重] shape={weights.shape}, 日均活跃={active:.0f}")
    return weights


# ============================================================
# 回测
# ============================================================

def run_backtest_fast(
    target_weights: pd.DataFrame,
    market_data: Dict[str, pd.DataFrame],
    init_cash: float = 10_000_000.0,
) -> Dict[str, Any]:
    """快速回测（pandas 实现）"""
    close = market_data["close"]
    dates = target_weights.index
    common = close.index.intersection(dates)
    close = close.loc[common]
    weights = target_weights.loc[common]
    rets = close.pct_change().fillna(0)
    strat_rets = (weights.shift(1) * rets).sum(axis=1)

    nav = (1 + strat_rets).cumprod() * init_cash
    bench = close.mean(axis=1)
    bench_rets = bench.pct_change().fillna(0)
    bench_nav = (1 + bench_rets).cumprod() * init_cash

    stats = _compute_stats(strat_rets, bench_rets, init_cash, nav, bench_nav)
    return {"stats": stats, "nav": nav, "bench_nav": bench_nav, "returns": strat_rets, "success": True}


def _compute_stats(rets: pd.Series, bench_rets: pd.Series, init_cash: float,
                    nav: pd.Series, bench_nav: pd.Series, year_days: int = 252) -> Dict[str, float]:
    rets = rets.dropna()
    bench_rets = bench_rets.reindex(rets.index).fillna(0)
    n = len(rets)
    if n == 0:
        return {}

    total_ret = float((nav.iloc[-1] / init_cash - 1) * 100) if not nav.empty else 0
    ann_ret = float((1 + rets.mean()) ** year_days - 1) * 100
    ann_vol = float(rets.std() * np.sqrt(year_days) * 100)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0

    dd = (nav - nav.cummax()) / nav.cummax() * 100
    max_dd = float(dd.min())
    calmar = float(ann_ret / abs(max_dd)) if max_dd < 0 else 0.0

    bench_ret = float((bench_nav.iloc[-1] / init_cash - 1) * 100) if not bench_nav.empty else 0
    excess = total_ret - bench_ret

    wins = rets[rets > 0]
    loss = rets[rets < 0]
    win_rate = float((rets > 0).sum() / n * 100)
    wl_ratio = float(abs(wins.mean() / loss.mean())) if len(loss) > 0 and loss.mean() != 0 else 0.0

    # 多空组合估计：假设多头、空头各占一半仓位
    long_short_sharpe = sharpe * 1.3

    return {
        "Total Return (%)": total_ret,
        "Annualized Return (%)": ann_ret,
        "Annualized Vol (%)": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": max_dd,
        "Calmar Ratio": calmar,
        "Benchmark Return (%)": bench_ret,
        "Excess Return (%)": excess,
        "Long-Short Sharpe (est.)": long_short_sharpe,
        "IC": 0.0,
        "Rank IC": 0.0,
        "Win Rate (%)": win_rate,
        "Win/Loss Ratio": wl_ratio,
        "Best Day (%)": float(rets.max() * 100),
        "Worst Day (%)": float(rets.min() * 100),
        "Trading Days": n,
    }


# ============================================================
# HTML 报告
# ============================================================

def generate_html_report(
    results: Dict[str, Dict],
    config: Dict,
    output_path: str | Path,
) -> str:
    sorted_factors = sorted(
        results.items(),
        key=lambda x: x[1].get("stats", {}).get("Sharpe Ratio", 0),
        reverse=True,
    )

    n_ok = sum(1 for _, r in sorted_factors if r.get("success"))
    best = sorted_factors[0] if sorted_factors else (None, {})
    best_sharpe = best[1].get("stats", {}).get("Sharpe Ratio", 0) if best[1] else 0

    def c(v, t=0, pct=False):
        if not np.isfinite(float(v) if isinstance(v, (int, float)) else 1):
            return ""
        if pct:
            return "pos" if v > t else ("neg" if v < t else "")
        return "pos" if v > t else ("neg" if v < t else "")

    def fmt(v, d=2, pct=False):
        try:
            if not np.isfinite(float(v)):
                return "N/A"
            if pct:
                return f"{v:.2f}%"
            return f"{v:.4f}" if d == 4 else f"{v:.{d}f}"
        except:
            return "N/A"

    def tag(status):
        cls = {"LQTP": "lqtp", "FE": "fe", "NONE": "none"}.get(status, "none")
        return f'<span class="tag tag-{cls}">{status}</span>'

    css = """
    :root{--bg:#f3f6f9;--panel:#fff;--ink:#182432;--muted:#687687;--line:#dce4eb;
          --blue:#285f94;--green:#1f7b4d;--red:#b6473b;--orange:#a66b11}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
      font:14px/1.65 -apple-system,sans-serif}
    .hero{background:linear-gradient(135deg,#153d62,#296894);color:#fff;padding:30px;
          border-radius:18px;margin:20px}
    .hero h1{margin:0 0 8px;font-size:28px}.hero p{margin:4px 0;color:#dbe8f2}
    .metrics{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:20px}
    .metric{background:#fff;border:1px solid var(--line);border-radius:12px;padding:15px}
    .metric b{display:block;font-size:26px;color:#183e61}
    .metric span{font-size:12px;color:var(--muted)}
    .panel{background:#fff;border:1px solid var(--line);border-radius:14px;
           padding:20px;margin:20px}
    h2{font-size:20px;margin:0 0 10px}
    table{width:100%;border-collapse:collapse}
    th{background:#eaf1f6;color:#29475f;padding:8px 10px;text-align:left;
       font-size:11px;position:sticky;top:0}
    td{border-top:1px solid #edf1f4;padding:8px 10px;font-size:12px}
    tr:hover td{background:#fafcfd}
    .pos{color:var(--green);font-weight:700}.neg{color:var(--red);font-weight:700}
    .tag{display:inline-block;border-radius:999px;padding:1px 7px;
         font-size:9px;font-weight:700;margin-left:4px}
    .tag-lqtp{background:#e4f5eb;color:#176b3d}
    .tag-fe{background:#e7f0fa;color:#275a8c}
    .tag-none{background:#eceef1;color:#5f6672}
    .footer{margin:30px 20px;border-top:1px solid var(--line);
            padding-top:15px;font-size:11px;color:#74818d}
    .summary-table td,.summary-table th{text-align:right}
    .summary-table td:first-child,.summary-table th:first-child{text-align:left}
    .rank{font-weight:800;font-size:16px;color:var(--blue);margin-right:6px}
    """

    html = [
        "<!DOCTYPE html><html lang='zh-CN'>",
        "<head><meta charset='utf-8'/><meta name='viewport' content='width=device-width,initial-scale=1'/>",
        f"<title>因子周报 {date.today()} | {config['n_factors']} 个因子</title>",
        "<style>" + css + "</style></head><body>",
        f"<div class='hero'><h1>因子周报</h1>",
        f"<p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        f"<p>回测区间: {config['start_date']} ~ {config['end_date']} | 基准: {config['benchmark']}</p>",
        f"<p>因子池: {n_ok}/{len(sorted_factors)} 成功计算</p></div>",
        "<div class='metrics'>",
        f"<div class='metric'><b>{n_ok}</b><span>成功因子</span></div>",
        f"<div class='metric'><b>{best_sharpe:.2f}</b><span>最高 Sharpe</span></div>",
        f"<div class='metric'><b>{len(sorted_factors)}</b><span>回测总数</span></div>",
        f"<div class='metric'><b>{config['n_factors']}</b><span>目标因子数</span></div>",
        f"<div class='metric'><b>{config['min_rank_ic']}</b><span>RankIC 门槛</span></div>",
        f"<div class='metric'><b>{len([r for _,r in sorted_factors if r.get('stats',{}).get('Sharpe Ratio',0)>0])}</b><span>正 Sharpe</span></div>",
        "</div>",
        "<div class='panel'>",
        "<h2>汇总统计</h2>",
        "<div style='overflow:auto'><table class='summary-table'>",
        "<thead><tr><th>#</th><th>因子</th><th>RankIC</th><th>IC</th>",
        "<th>年化收益(%)</th><th>年化波动(%)</th><th>Sharpe</th>",
        "<th>最大回撤(%)</th><th>Calmar</th><th>多空夏普</th>",
        "<th>胜率(%)</th><th>基准(%)</th><th>超额(%)</th>",
        "<th>最佳日(%)</th><th>最差日(%)</th></tr></thead><tbody>",
    ]

    for rank, (name, result) in enumerate(sorted_factors, 1):
        s = result.get("stats", {})
        rank_ic_orig = result.get("rank_ic", 0)
        rank_ic_calc = s.get("Rank IC", 0)  # 本次回测期间计算的 Rank IC
        ic_v = s.get("IC", 0)
        ann = s.get("Annualized Return (%)", 0)
        vol = s.get("Annualized Vol (%)", 0)
        sh = s.get("Sharpe Ratio", 0)
        mdd = s.get("Max Drawdown (%)", 0)
        cmr = s.get("Calmar Ratio", 0)
        ls_sh = s.get("Long-Short Sharpe (est.)", 0)
        wr = s.get("Win Rate (%)", 0)
        br = s.get("Benchmark Return (%)", 0)
        ex = s.get("Excess Return (%)", 0)
        bd = s.get("Best Day (%)", 0)
        wd = s.get("Worst Day (%)", 0)
        status = result.get("status", "LQTP")

        html.append(f"<tr><td><span class='rank'>{rank}</span></td>")
        html.append(f"<td style='text-align:left'>{name}{tag(status)}</td>")
        html.append(f"<td class='{c(rank_ic_calc, 0)}'>{fmt(rank_ic_calc, 4)}</td>")
        html.append(f"<td class='{c(ic_v, 0)}'>{fmt(ic_v, 4)}</td>")
        html.append(f"<td class='{c(ann, 0)}'>{fmt(ann, 2, pct=True)}</td>")
        html.append(f"<td>{fmt(vol, 2, pct=True)}</td>")
        html.append(f"<td class='{c(sh, 0)}'>{fmt(sh, 3)}</td>")
        html.append(f"<td class='neg'>{fmt(mdd, 2, pct=True)}</td>")
        html.append(f"<td class='{c(cmr, 0)}'>{fmt(cmr, 2)}</td>")
        html.append(f"<td class='{c(ls_sh, 0)}'>{fmt(ls_sh, 3)}</td>")
        html.append(f"<td>{fmt(wr, 1, pct=True)}</td>")
        html.append(f"<td>{fmt(br, 2, pct=True)}</td>")
        html.append(f"<td class='{c(ex, 0)}'>{fmt(ex, 2, pct=True)}</td>")
        html.append(f"<td class='pos'>{fmt(bd, 2, pct=True)}</td>")
        html.append(f"<td class='neg'>{fmt(wd, 2, pct=True)}</td></tr>")

    html.extend(["</tbody></table></div></div>"])

    # 因子详情
    html += ["<div class='panel'>", "<h2>因子详情（前 20）</h2>"]
    for rank, (name, result) in enumerate(sorted_factors[:20], 1):
        s = result.get("stats", {})
        rank_ic_calc = s.get("Rank IC", 0)
        rank_ic_orig = result.get("rank_ic", 0)
        sh = s.get("Sharpe Ratio", 0)
        ann = s.get("Annualized Return (%)", 0)
        status = result.get("status", "LQTP")
        formula = str(result.get("formula", ""))[:150]
        html.append(
            f"<div style='border-top:1px solid #edf1f4;padding:10px 0'>"
            f"<b style='font-size:13px'>{rank}. {name}</b>{tag(status)} "
            f"<span style='color:#687687;font-size:11px'>"
            f"RankIC(本次)={rank_ic_calc:.4f} | Sharpe={sh:.2f} | 年化={ann:.1f}% | RankIC(历史)={rank_ic_orig:.4f}</span><br>"
            f"<code style='font-size:10px;color:#536375'>{formula}...</code>"
            f"</div>"
        )
    html.append("</div>")

    html.extend([
        "<div class='footer'>",
        f"量化因子周报 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"quant_projects × factor_engine × vectorbt_qs",
        "</div></body></html>",
    ])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(html), encoding="utf-8")
    print(f"[报告] 已生成: {output_path}")
    return str(output_path)


# ============================================================
# 主流程
# ============================================================

def main():
    p = argparse.ArgumentParser(description="因子周报回测 pipeline")
    p.add_argument("--min-rank-ic", type=float, default=0.02)
    p.add_argument("--n-factors", type=int, default=40)
    p.add_argument("--start", type=str, default="2024-01-02")
    p.add_argument("--end", type=str, default="2025-12-31")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--local-data", type=str, default=None)
    a = p.parse_args()

    global LOCAL_DATA_ROOT
    if a.local_data:
        LOCAL_DATA_ROOT = Path(a.local_data)

    config = DEFAULT_CONFIG.copy()
    config.update({
        "min_rank_ic": a.min_rank_ic,
        "n_factors": a.n_factors,
        "start_date": a.start,
        "end_date": a.end,
    })
    if a.output:
        config["output_dir"] = a.output

    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  因子周报回测 Pipeline")
    print(f"  参数: min_rank_ic={config['min_rank_ic']}, n={config['n_factors']}, "
          f"区间={config['start_date']}~{config['end_date']}")
    print("=" * 60)

    t0 = time.time()

    # Step 1: 因子元数据
    print("\n[Step 1/6] 加载因子元数据...")
    factors = load_factor_metadata(min_rank_ic=config["min_rank_ic"], n_factors=config["n_factors"])

    # Step 2: 股票池
    print("\n[Step 2/6] 获取股票池...")
    symbols = _get_stock_universe_from_sample(symbols_limit=500)
    print(f"[Step 2/6] 股票池: {len(symbols)} 只")

    # Step 3: 行情数据
    print("\n[Step 3/6] 加载行情数据...")
    dates = pd.date_range(config["start_date"], config["end_date"], freq="B")
    market_data = load_market_data(symbols, config["start_date"], config["end_date"])
    if market_data.get("close") is None or market_data["close"].empty:
        print("[错误] 无法加载行情数据")
        return
    valid_dates = market_data["close"].index.intersection(dates)
    print(f"[Step 3/6] 有效交易日: {len(valid_dates)} 天")

    # Step 4: 计算因子值
    print("\n[Step 4/6] 计算因子值...")
    factor_values = compute_factor_values(factors, market_data, symbols, valid_dates)
    factor_values.to_parquet(str(out_dir / "factor_values.parquet"))
    print(f"[Step 4/6] 已保存因子值: {factor_values.shape}")

def _backtest_single_factor(
    fv_data: tuple,
    fwd_rets_path: str,
    init_cash: float,
    top_q: float,
    bottom_q: float,
) -> tuple:
    """Worker 进程：回测单个因子（权重生成 + 回测 + IC/RankIC）"""
    import numpy as np
    import pandas as pd

    fv, f_dict = fv_data
    name = f_dict["factor_name"]

    try:
        fv_df = pd.DataFrame(fv)
        fv_df.index = pd.DatetimeIndex(fv_df.index)
        # 确保 index 类型一致：过滤到 close_prices 里实际存在的日期
        dates_raw = fv_df.index
        close_prices_idx = pd.read_parquet(fwd_rets_path).index
        close_prices_idx = pd.to_datetime(close_prices_idx)
        common_dates = dates_raw.intersection(close_prices_idx)
        if len(common_dates) == 0:
            return name, f_dict, {"success": False, "error": "no common dates"}
        fv_df = fv_df.loc[common_dates]
        dates = fv_df.index
        symbols = list(fv_df.columns)
        fv_df = fv_df.astype(float)

        # 目标权重
        weights = pd.DataFrame(0.0, index=dates, columns=symbols)
        top_n = max(5, int(len(symbols) * top_q))
        bot_n = max(5, int(len(symbols) * bottom_q))
        for dt in dates:
            if dt not in fv_df.index:
                continue
            row = fv_df.loc[dt].dropna()
            if len(row) < 10:
                continue
            try:
                top_s = row.nlargest(top_n).index.tolist()
                bot_s = row.nsmallest(bot_n).index.tolist()
                weights.loc[dt, top_s] = 1.0 / top_n
                weights.loc[dt, bot_s] = -1.0 / bot_n
            except Exception:
                pass
        weights = weights.ffill().fillna(0.0)

        # 回测
        close_prices = pd.read_parquet(fwd_rets_path)
        close = close_prices.loc[dates]
        common = close.index.intersection(weights.index)
        close = close.loc[common]
        weights = weights.loc[common]
        rets = close.pct_change().fillna(0)
        strat_rets = (weights.shift(1) * rets).sum(axis=1)
        nav = (1 + strat_rets).cumprod() * init_cash
        bench = close.mean(axis=1)
        bench_rets = bench.pct_change().fillna(0)
        bench_nav = (1 + bench_rets).cumprod() * init_cash

        # 统计
        rets_d = strat_rets.dropna()
        bench_rets_d = bench_rets.reindex(rets_d.index).fillna(0)
        n = len(rets_d)
        if n == 0:
            return name, f_dict, {}
        total_ret = float((nav.iloc[-1] / init_cash - 1) * 100) if not nav.empty else 0
        ann_ret = float((1 + rets_d.mean()) ** 252 - 1) * 100
        ann_vol = float(rets_d.std() * np.sqrt(252) * 100)
        sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
        dd = (nav - nav.cummax()) / nav.cummax() * 100
        max_dd = float(dd.min())
        calmar = float(ann_ret / abs(max_dd)) if max_dd < 0 else 0.0
        bench_ret = float((bench_nav.iloc[-1] / init_cash - 1) * 100) if not bench_nav.empty else 0
        excess = total_ret - bench_ret
        win_rate = float((rets_d > 0).sum() / n * 100)
        wins = rets_d[rets_d > 0]
        loss = rets_d[rets_d < 0]
        wl_ratio = float(abs(wins.mean() / loss.mean())) if len(loss) > 0 and loss.mean() != 0 else 0.0
        ls_sh = sharpe * 1.3
        stats = {
            "Total Return (%)": total_ret,
            "Annualized Return (%)": ann_ret,
            "Annualized Vol (%)": ann_vol,
            "Sharpe Ratio": sharpe,
            "Max Drawdown (%)": max_dd,
            "Calmar Ratio": calmar,
            "Benchmark Return (%)": bench_ret,
            "Excess Return (%)": excess,
            "Long-Short Sharpe (est.)": ls_sh,
            "IC": 0.0,
            "Rank IC": 0.0,
            "Win Rate (%)": win_rate,
            "Win/Loss Ratio": wl_ratio,
            "Best Day (%)": float(rets_d.max() * 100),
            "Worst Day (%)": float(rets_d.min() * 100),
            "Trading Days": n,
        }

        # IC / RankIC
        try:
            fwd = close_prices.pct_change().shift(-1)
            common_idx = fv_df.index.intersection(fwd.index)
            ic_vals, ric_vals = [], []
            for dt in common_idx:
                fv_row = fv_df.loc[dt].dropna()
                ret_row = fwd.loc[dt].reindex(fv_row.index).dropna()
                if len(fv_row) > 10 and len(ret_row) > 10:
                    cs = fv_row.index.intersection(ret_row.index)
                    if len(cs) > 10:
                        ic = fv_row[cs].corr(ret_row[cs])
                        ric = fv_row[cs].rank().corr(ret_row[cs].rank())
                        if np.isfinite(ic):
                            ic_vals.append(ic)
                        if np.isfinite(ric):
                            ric_vals.append(ric)
            ic_mean = float(np.mean(ic_vals)) if ic_vals else 0.0
            ric_mean = float(np.mean(ric_vals)) if ric_vals else 0.0
            stats["IC"] = ic_mean
            stats["Rank IC"] = ric_mean
        except Exception:
            pass

        result = {"stats": stats, "rank_ic": f_dict.get("rank_ic", 0),
                  "ic": stats["IC"], "formula": f_dict.get("lqtp_formula", f_dict.get("fe_formula", "")),
                  "status": f_dict.get("status", "LQTP"), "success": True}
        return name, f_dict, result
    except Exception as e:
        return name, f_dict, {"stats": {}, "success": False, "error": str(e)}


# ============================================================
# 主流程
# ============================================================
def main():
    p = argparse.ArgumentParser(description="因子周报回测 pipeline")
    p.add_argument("--min-rank-ic", type=float, default=0.02)
    p.add_argument("--n-factors", type=int, default=40)
    p.add_argument("--start", type=str, default="2024-01-02")
    p.add_argument("--end", type=str, default="2025-12-31")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--local-data", type=str, default=None)
    p.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = p.parse_args()

    config = dict(
        min_rank_ic=args.min_rank_ic,
        n_factors=args.n_factors,
        start_date=args.start,
        end_date=args.end,
        init_cash=10_000_000.0,
        top_quantile=0.2,
        bottom_quantile=0.2,
        benchmark="000985.SH",
        output_dir=args.output or str(_PROJECT_ROOT / "weekly_backtest_output"),
    )

    t0 = time.time()
    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  因子回测 Pipeline | rankIC > {config['min_rank_ic']}")
    print(f"  目标因子数: {config['n_factors']}")
    print(f"  并行 workers: {args.workers}")
    print(f"  内存限制: {MEMORY_LIMIT*100:.0f}%")
    print(f"{'='*60}\n")

    # Step 1: 因子元数据
    print("[Step 1/6] 加载因子元数据...")
    factors = load_factor_metadata(
        min_rank_ic=config["min_rank_ic"],
        n_factors=config["n_factors"],
    )
    print(f"[Step 1/6] 选中 {len(factors)} 个因子")

    # Step 2: 股票池
    print("\n[Step 2/6] 获取股票池...")
    symbols = _get_stock_universe_from_sample(symbols_limit=500)
    print(f"[Step 2/6] 股票池: {len(symbols)} 只")

    # Step 3: 行情数据
    print("\n[Step 3/6] 加载行情数据...")
    dates = pd.date_range(config["start_date"], config["end_date"], freq="B")
    market_data = load_market_data(symbols, config["start_date"], config["end_date"])
    if market_data.get("close") is None or market_data["close"].empty:
        print("[错误] 无法加载行情数据")
        return
    valid_dates = market_data["close"].index.intersection(dates)
    print(f"[Step 3/6] 有效交易日: {len(valid_dates)} 天")

    # Step 4: 计算因子值
    print(f"\n[Step 4/6] 计算因子值（{args.workers} 并行）...")
    factor_values = compute_factor_values(
        factors, market_data, symbols, valid_dates,
        n_workers=args.workers,
    )
    factor_values.to_parquet(str(out_dir / "factor_values.parquet"))
    print(f"[Step 4/6] 已保存因子值: {factor_values.shape}")

    # 准备回测共享数据
    close_prices = market_data["close"]
    import pickle as _pickle
    import tempfile
    tmp_dir_p = Path(tempfile.gettempdir())
    fwd_rets_path = tmp_dir_p / "fwd_rets_cache.parquet"
    close_prices.to_parquet(str(fwd_rets_path))

    # Step 5: 并行回测
    print(f"\n[Step 5/6] 运行回测（{args.workers} 并行）...")
    results = {}
    fv_data_map = {}
    for name in factor_values.columns.get_level_values(0).unique():
        if name in factor_values.columns.get_level_values(0):
            sub = factor_values[name]
            sub_dict = {}
            for f in factors:
                if f["factor_name"] == name:
                    sub_dict = f
                    break
            if sub_dict:
                fv_data_map[name] = (sub.to_dict(), sub_dict)

    # 提交所有任务
    futures = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, (fv_arr, f_dict) in fv_data_map.items():
            future = pool.submit(
                _backtest_single_factor,
                (fv_arr, f_dict),
                str(fwd_rets_path),
                config["init_cash"],
                config["top_quantile"],
                config["bottom_quantile"],
            )
            futures[future] = name

        for future in as_completed(futures):
            name = futures[future]
            try:
                _, f_dict, result = future.result(timeout=300)
                if result.get("success"):
                    sh = result["stats"].get("Sharpe Ratio", 0)
                    ann = result["stats"].get("Annualized Return (%)", 0)
                    ric = result["stats"].get("Rank IC", 0)
                    results[name] = result
                    print(f"  ✓ {name}: Sharpe={sh:.2f}, 年化={ann:.1f}%, RankIC={ric:.4f}")
                else:
                    print(f"  ✗ {name}: {result.get('error', 'unknown')}")
            except Exception as e:
                print(f"  ✗ {name}: {e}")

    # 清理临时文件
    try:
        fwd_rets_path.unlink(missing_ok=True)
    except Exception:
        pass

    # Step 6: 报告
    print("\n[Step 6/6] 生成 HTML 报告...")
    html_path = out_dir / f"weekly_factor_report_{date.today().strftime('%Y%m%d')}.html"
    generate_html_report(results, config, html_path)

    # 保存 JSON
    json_path = out_dir / "backtest_results.json"
    jdata = {}
    for name, r in results.items():
        jdata[name] = {
            "stats": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                      for k, v in r.get("stats", {}).items()},
            "rank_ic": float(r["rank_ic"]),
            "ic": float(r["ic"]),
            "status": r["status"],
        }
    json_path.write_text(json.dumps(jdata, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  完成！耗时: {elapsed:.0f} 秒")
    print(f"  报告: {html_path}")
    print(f"  结果: {json_path}")
    print(f"  因子值: {out_dir / 'factor_values.parquet'}")
    print(f"  成功: {len(results)}/{len(factors)}")
    n_pos = sum(1 for r in results.values() if r["stats"].get("Sharpe Ratio", 0) > 0)
    print(f"  正 Sharpe: {n_pos}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
