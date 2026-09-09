"""
数据适配器：COS parquet → vectorbt 可用的 pandas 宽表

支持两种数据源:
- 本地文件系统（开发调试）
- COS 对象存储（生产环境）

输出格式：
  (index=TradeDate, columns=Symbol/Ticker, values=复权价格/信号/约束)
"""

import os
import posixpath
from warnings import warn
import pandas as pd
from typing import Optional, List, Dict, Tuple

from .cos_reader import get_cos_reader

# === data_access 一体化数据入口（替代 COS 逐文件读取） ===
_DATA_ACCESS_AVAILABLE = True
try:
    from data_access import get_store as _da_get_store
except ImportError:
    _DATA_ACCESS_AVAILABLE = False


# ============================================================
# 配置：数据路径映射
# ============================================================

# 默认使用项目内的本地 data/ 目录（适合开发调试）
# 切换到 COS 路径即可直接从 COS 读取:
#   set_data_root("ashare", "cos://qs-cold/clean_data/ashare/lqtp_data")
_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data"
)

DATA_ROOTS = {
    "ashare": os.path.join(_DATA_DIR, "ashare"),
    "us_stock": os.path.join(_DATA_DIR, "us_stock"),
    "us_raw": os.path.join(_DATA_DIR, "us_raw"),
    # COS 路径参考（切换时使用）:
    # "ashare": "cos://qs-cold/clean_data/ashare/lqtp_data",
    # "us_stock": "cos://qs-cold/clean_data/us_stock/massive_data",
}


def _join_data_path(root: str, *parts: str) -> str:
    """Join local paths and ``cos://`` URIs without mixing separators."""
    if root.startswith("cos://"):
        normalized_parts = [str(part).replace("\\", "/").strip("/") for part in parts]
        return posixpath.join(root.rstrip("/"), *normalized_parts)
    return os.path.join(root, *parts)


def _coerce_bool_series(values: pd.Series, field_name: str) -> pd.Series:
    """Parse common bool encodings without treating ``"False"`` as true."""
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(values.dtype):
        return values.fillna(0).ne(0)

    normalized = values.astype("string").str.strip().str.lower()
    truthy = {"1", "true", "t", "yes", "y"}
    falsy = {"0", "false", "f", "no", "n", "", "none", "null", "nan"}
    unknown = normalized.notna() & ~normalized.isin(truthy | falsy)
    if unknown.any():
        examples = ", ".join(map(str, normalized[unknown].drop_duplicates().head(5)))
        warn(f"{field_name} 包含无法识别的布尔值，将按 False 处理: {examples}")
    return normalized.isin(truthy)

# 子路径映射：market → {表名: 相对路径}
TABLE_PATHS = {
    "ashare": {
        "StockDailyBar": "StockDailyBar",
        "Calendar": "Calendar",
    },
    "us_stock": {
        "PanelDaily": "PanelDaily",
        "StockDailyBar": "StockDailyBar",
        "DimSecurityMaster": "DimSecurityMaster",
        "DimCalendar": "DimCalendar",
        "DimTickerMap": "DimTickerMap",
        "DimTickerAlias": "DimTickerAlias",
    },
}


def set_data_root(market: str, path: str):
    """修改数据根路径（用于本地调试）"""
    DATA_ROOTS[market] = path


# ============================================================
# 核心：parquet 目录 → pivot 宽表
# ============================================================

def _read_parquet_dir(
    parquet_dir: str,
    date_col: str = "TradeDate",
    symbol_col: str = "Symbol",
    value_cols: Optional[List[str]] = None,
    extra_cols: Optional[List[str]] = None,
    date_range: Optional[Tuple[str, str]] = None,
    post_process: Optional[callable] = None,
) -> pd.DataFrame:
    """
    通用按日分区 parquet → 拼接 → pivot 宽表

    支持本地路径和 cos:// 路径。

    参数
    ----
    parquet_dir : 目录路径（本地或 cos://bucket/prefix/）
    date_col : 日期列名
    symbol_col : 标识别名
    value_cols : 需要 pivot 的数值列（None = 全部数值列）
    extra_cols : 额外保留的列
    date_range : (start, end) 日期过滤
    post_process : 在 pivot 前对 DataFrame 的后处理函数

    返回
    ----
    如果 value_cols 为单列 → 返回 (n_days, n_assets) DataFrame
    如果 value_cols 为多列 → 返回 Dict[str, DataFrame]
    """
    reader = get_cos_reader()
    files = sorted(reader.glob(_join_data_path(parquet_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"目录下无 parquet 文件: {parquet_dir}")

    frames = []
    for f in files:
        # 按文件名过滤日期范围（格式 YYYY-MM-DD.parquet）
        fname = os.path.basename(f)
        if date_range and fname != "full.parquet":
            file_date = fname.replace(".parquet", "")
            if file_date < date_range[0] or file_date > date_range[1]:
                continue

        df = reader.read_parquet(f)

        # 确保有日期列
        if date_col not in df.columns:
            continue

        keep_cols = [date_col, symbol_col]
        if value_cols:
            keep_cols += [c for c in value_cols if c in df.columns]
        if extra_cols:
            keep_cols += [c for c in extra_cols if c in df.columns]
        df = df[keep_cols]

        if post_process:
            df = post_process(df)

        frames.append(df)

    if not frames:
        raise ValueError(f"日期范围内无数据: {date_range}")

    raw = pd.concat(frames, ignore_index=True)
    raw[date_col] = pd.to_datetime(raw[date_col])
    if date_range:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        raw = raw.loc[raw[date_col].between(start, end)]
        if raw.empty:
            raise ValueError(f"日期范围内无数据: {date_range}")

    # pivot 每个 value_col 独立
    if not value_cols:
        value_cols = [c for c in raw.columns if c not in [date_col, symbol_col]]

    if len(value_cols) == 1:
        result = raw.pivot(index=date_col, columns=symbol_col, values=value_cols[0])
        result = result.sort_index().sort_index(axis=1)
        return result
    else:
        results = {}
        for vc in value_cols:
            pivot_df = raw.pivot(index=date_col, columns=symbol_col, values=vc)
            results[vc] = pivot_df.sort_index().sort_index(axis=1)
        return results


# ============================================================
# A 股数据加载
# ============================================================

def load_ashare_daily_bar(
    symbols: Optional[List[str]] = None,
    start: str = "2016-01-01",
    end: str = "2026-06-01",
) -> Dict[str, pd.DataFrame]:
    """
    加载 A 股 StockDailyBar → 多张宽表（通过 data_access）

    优先使用 data_access（DuckDB 列裁剪 + 谓词下推，比逐文件遍历快 5-10×）；
    data_access 不可用时 fallback 到 COS 逐文件 read_parquet。

    返回
    ----
    {
        "close":      pd.DataFrame,  # 复权收盘价 (n_days, n_assets)
        "open":       pd.DataFrame,  # 不复权开盘价
        "high":       pd.DataFrame,  # 最高价
        "low":        pd.DataFrame,  # 最低价
        "volume":     pd.DataFrame,  # 成交量
        "is_suspend": pd.DataFrame,  # 停牌标记
        "high_limit": pd.DataFrame,  # 涨停价
        "low_limit":  pd.DataFrame,  # 跌停价
        "factor":     pd.DataFrame,  # 复权因子
    }
    """
    value_cols = ["Open", "High", "Low", "Close", "Vwap", "Volume",
                  "IsSuspend", "HighLimit", "LowLimit", "Factor"]

    # === 优先路径：data_access（DuckDB 下推） ===
    if _DATA_ACCESS_AVAILABLE:
        try:
            store = _da_get_store()
            tbl = store.read_arrow(
                "ashare_stock_daily",
                columns=["TradeDate", "Symbol"] + [
                    c for c in value_cols
                    if c in {"Open", "High", "Low", "Close", "Vwap", "Volume",
                             "IsSuspend", "HighLimit", "LowLimit", "Factor"}
                ],
                time_range=(start, end),
                instrument_filter=list(symbols) if symbols else None,
            )
            df = tbl.to_pandas()
            df["TradeDate"] = pd.to_datetime(df["TradeDate"])

            # pivot 每个 value_col → 宽表（与原 _read_parquet_dir 输出一致）
            results = {}
            available_cols = [c for c in value_cols if c in df.columns]
            for vc in available_cols:
                pivot_df = df.pivot(index="TradeDate", columns="Symbol", values=vc)
                results[vc] = pivot_df.sort_index().sort_index(axis=1)

            # 计算复权价（OHLCV + 涨跌停）
            for col in ["Open", "High", "Low", "Close", "Vwap", "HighLimit", "LowLimit"]:
                if col in results and "Factor" in results:
                    results[f"{col}_adj"] = results[col] * results["Factor"]

            return results
        except Exception as e:
            from warnings import warn
            warn(f"data_access 读取失败 ({e})，fallback 到 cos_reader")

    # === Fallback：COS 逐文件读取（保留兼容） ===
    parquet_dir = _join_data_path(DATA_ROOTS["ashare"], "StockDailyBar")

    def _filter_symbols(df):
        if symbols:
            return df[df["Symbol"].isin(symbols)]
        return df

    results = _read_parquet_dir(
        parquet_dir,
        date_col="TradeDate",
        symbol_col="Symbol",
        value_cols=value_cols,
        date_range=(start, end),
        post_process=_filter_symbols,
    )

    for price_col in ["Open", "High", "Low", "Close", "Vwap", "HighLimit", "LowLimit"]:
        if price_col in results:
            results[f"{price_col}_adj"] = results[price_col] * results["Factor"]

    return results


def load_ashare_calendar(
    start: str = "2016-01-01",
    end: str = "2026-06-01",
) -> pd.DatetimeIndex:
    """加载 A 股交易日历（通过 data_access）"""
    # 优先 data_access
    if _DATA_ACCESS_AVAILABLE:
        try:
            store = _da_get_store()
            tbl = store.read_arrow(
                "ashare_calendar",
                columns=["TradeDate", "IsTradeDay"],
                time_range=(start, end),
            )
            df = tbl.to_pandas()
            df["TradeDate"] = pd.to_datetime(df["TradeDate"])
            calendar = df[df["IsTradeDay"]]["TradeDate"]
            return pd.DatetimeIndex(sorted(calendar))
        except Exception:
            pass

    # Fallback: COS 本地文件
    reader = get_cos_reader()
    parquet_dir = _join_data_path(DATA_ROOTS["ashare"], "Calendar")
    fpath = _join_data_path(parquet_dir, "full.parquet")
    df = reader.read_parquet(fpath)
    df["TradeDate"] = pd.to_datetime(df["TradeDate"])
    calendar = df[df["IsTradeDay"]]["TradeDate"]
    if start:
        calendar = calendar[calendar >= start]
    if end:
        calendar = calendar[calendar <= end]
    return pd.DatetimeIndex(sorted(calendar))


def load_ashare_corporate_actions(
    symbols: Optional[List[str]] = None,
    start: str = "2016-01-01",
    end: str = "2026-06-01",
) -> Dict[str, pd.DataFrame]:
    """Load ex-date cash and share distributions from ``StockDividend``."""
    try:
        actions = _read_parquet_dir(
            _join_data_path(DATA_ROOTS["ashare"], "StockDividend"),
            value_cols=["CashDividend", "StockDividend", "StockTransfer"],
            date_range=(start, end),
            post_process=(
                (lambda frame: frame[frame["Symbol"].isin(symbols)])
                if symbols
                else None
            ),
        )
    except (FileNotFoundError, ValueError):
        return {}
    return {
        "cash_dividend": actions["CashDividend"],
        "share_multiplier": (
            1.0
            + actions["StockDividend"].fillna(0.0)
            + actions["StockTransfer"].fillna(0.0)
        ),
    }


# ============================================================
# 美股数据加载
# ============================================================

def load_us_daily_bar(
    symbols: Optional[List[str]] = None,
    start: str = "2010-01-01",
    end: str = "2026-06-01",
    use_panel: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    加载美股日线行情

    use_panel=True → 从 PanelDaily 加载（2010 起有基本面）
    use_panel=False → 从 StockDailyBar 加载（2003 起，需自行复权）

    返回
    ----
    {
        "close":  pd.DataFrame,  # 复权收盘价
        "open":   pd.DataFrame,
        "high":   pd.DataFrame,
        "low":    pd.DataFrame,
        "volume": pd.DataFrame,
        "returns": pd.DataFrame, # 日收益率（PanelDaily 才有）
        ...
    }
    """
    if use_panel:
        parquet_dir = _join_data_path(DATA_ROOTS["us_stock"], "PanelDaily")
        date_col, symbol_col = "TradeDate", "Ticker"
        # PanelDaily 核心 OHLCV 列（所有年份都有）
        value_cols = ["Open", "High", "Low", "Close", "Volume", "AdjFactor"]
    else:
        parquet_dir = _join_data_path(DATA_ROOTS["us_stock"], "StockDailyBar")
        date_col, symbol_col = "TradeDate", "Ticker"
        value_cols = ["Open", "High", "Low", "Close", "Volume", "AdjFactor"]

    def _filter_symbols(df):
        if symbols:
            return df[df[symbol_col].isin(symbols)]
        return df

    results = _read_parquet_dir(
        parquet_dir,
        date_col=date_col,
        symbol_col=symbol_col,
        value_cols=value_cols,
        date_range=(start, end),
        post_process=_filter_symbols,
    )

    if "AdjFactor" in results:
        for price_col in ["Open", "High", "Low", "Close"]:
            if price_col in results:
                results[f"{price_col}_adj"] = results[price_col] * results["AdjFactor"]

    # 统一 key 命名
    key_map = {
        "Close_adj": "close",
        "Open_adj": "open",
        "High_adj": "high",
        "Low_adj": "low",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Volume": "volume",
        "RetPrice": "returns",
        "AdjFactor": "adj_factor",
    }
    normalized = {key_map.get(k, k): v for k, v in results.items()}
    if "close" not in normalized and "Close" in results:
        normalized["close"] = results["Close"]
    return normalized


def load_us_dims() -> Dict[str, pd.DataFrame]:
    """加载美股维度表（截面数据）"""
    reader = get_cos_reader()
    dims = {}
    for dim_name in ["DimSecurityMaster", "DimCalendar", "DimTickerMap", "DimTickerAlias"]:
        fpath = _join_data_path(DATA_ROOTS["us_stock"], dim_name, "full.parquet")
        try:
            dims[dim_name] = reader.read_parquet(fpath)
        except (FileNotFoundError, Exception):
            pass  # 维度表可选
    return dims


def load_us_universe(
    year: int,
) -> pd.DataFrame:
    """加载美股每日股票池"""
    fpath = _join_data_path(
        DATA_ROOTS["us_raw"].replace("raw_data", "clean_data"),
        "universe_daily",
        str(year),
    )
    reader = get_cos_reader()
    files = reader.glob(_join_data_path(fpath, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"无股票池数据: {fpath}")
    return pd.concat([reader.read_parquet(f) for f in files], ignore_index=True)


def _extract_us_constraints(dims: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    """Extract optional ADR and delisting metadata from available dimensions."""
    result: Dict[str, object] = {}
    security_master = dims.get("DimSecurityMaster")
    if security_master is None or security_master.empty:
        return result

    ticker_col = next((c for c in ["Ticker", "ticker", "Symbol", "symbol"] if c in security_master), None)
    adr_col = next((c for c in ["IsADR", "IsAdr", "is_adr", "is_adr_asset"] if c in security_master), None)
    if ticker_col and adr_col:
        raw_is_adr = security_master.drop_duplicates(ticker_col).set_index(ticker_col)[adr_col]
        result["is_adr"] = _coerce_bool_series(raw_is_adr, adr_col)

    delist_col = next((c for c in ["DelistDate", "delist_date"] if c in security_master), None)
    if ticker_col and delist_col:
        delist_info = security_master.loc[security_master[delist_col].notna(), [ticker_col, delist_col]].copy()
        delist_info.columns = ["ticker", "delist_date"]
        result["delist_info"] = delist_info
    return result


def load_benchmark_close(
    market: str,
    symbol: str,
    start: str,
    end: str,
) -> pd.Series:
    """Load one benchmark close series for report metrics."""
    if market == "ashare":
        if _DATA_ACCESS_AVAILABLE:
            try:
                store = _da_get_store()
                table = store.read_arrow(
                    "ashare_index_daily",
                    columns=["TradeDate", "Symbol", "Close"],
                    time_range=(start, end),
                    instrument_filter=[symbol],
                )
                frame = table.to_pandas()
                if not frame.empty:
                    frame["TradeDate"] = pd.to_datetime(frame["TradeDate"])
                    return frame.set_index("TradeDate")["Close"].sort_index().rename(symbol)
            except Exception:
                pass
        benchmark = _read_parquet_dir(
            _join_data_path(DATA_ROOTS["ashare"], "IndexDailyBar"),
            value_cols=["Close"],
            date_range=(start, end),
            post_process=lambda frame: frame[frame["Symbol"] == symbol],
        )
        if symbol not in benchmark.columns:
            raise ValueError(f"基准指数无数据: {symbol}")
        return benchmark[symbol].rename(symbol)
    if market == "us":
        benchmark = load_us_daily_bar(symbols=[symbol], start=start, end=end)
        if symbol not in benchmark["close"].columns:
            raise ValueError(f"美股基准无数据: {symbol}")
        return benchmark["close"][symbol].rename(symbol)
    raise ValueError(f"不支持的市场: {market}")


# ============================================================
# 便捷函数：一键加载
# ============================================================

def load_market_data(
    market: str,
    symbols: Optional[List[str]] = None,
    start: str = "2016-01-01",
    end: str = "2026-06-01",
) -> Dict[str, pd.DataFrame]:
    """
    一键加载市场数据

    market='ashare'  → A 股 StockDailyBar（2016 起）
    market='us'      → 美股 PanelDaily（2010 起）
    """
    if market == "ashare":
        data = load_ashare_daily_bar(symbols=symbols, start=start, end=end)
        data.update(
            load_ashare_corporate_actions(
                symbols=symbols,
                start=start,
                end=end,
            )
        )
        # Accurate execution needs two explicit price domains:
        #
        # - raw_*: exchange prices and real-share transaction notionals
        # - lower-case adjusted keys: total-return valuation inside vectorbt
        #
        # Never apply a real 100-share board lot directly to adjusted prices.
        _ashare_raw_key_map = {
            "Close": "raw_close",
            "Open": "raw_open",
            "High": "raw_high",
            "Low": "raw_low",
            "Vwap": "raw_vwap",
            "HighLimit": "raw_high_limit",
            "LowLimit": "raw_low_limit",
        }
        for source_key, target_key in _ashare_raw_key_map.items():
            if source_key in data:
                data[target_key] = data[source_key]

        # 统一复权 key 命名为小写（runner.py 预期 snake_case）
        _ashare_key_map = {
            "Close_adj": "close",
            "Open_adj": "open",
            "High_adj": "high",
            "Low_adj": "low",
            "Vwap_adj": "vwap",
            "Volume": "volume",
            "IsSuspend": "is_suspend",
            "HighLimit_adj": "high_limit",
            "LowLimit_adj": "low_limit",
            "Factor": "factor",
        }
        for old_key, new_key in _ashare_key_map.items():
            if old_key in data:
                data[new_key] = data.pop(old_key)
        return data
    elif market == "us":
        data = load_us_daily_bar(symbols=symbols, start=start, end=end, use_panel=True)
        data.update(_extract_us_constraints(load_us_dims()))
        return data
    else:
        raise ValueError(f"不支持的市场: {market}")
