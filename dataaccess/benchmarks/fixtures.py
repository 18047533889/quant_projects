"""R30-P0-001 —— 固定可复现 synthetic fixtures。

构造"看起来真实"的 A 股风格数据（确定性 seed，同一 tmp 重建 → 同一数据）：

    - ``ashare_stock_daily``   日频面板：TradeDate/Symbol/Open/High/Low/Close/
                                 Volume/Amount/VWAP
    - ``ashare_stock_minute``   分钟数据：datetime(UTC)/Symbol/open/high/low/
                                 close/volume/amount
    - ``ashare_stock_income``   基本面（ashare_stock_income 风格）：
                                 PubDate/Symbol/period_end/net_profit/revenue
    - ``ashare_universe``       时变 universe（INNER JOIN 成员过滤）

store 通过 ``DataAccessStore(load_registry(cfg_yaml), DuckDBEngine(threads=2))``
构造（与 tests/unit 的 fixture 同路径），数据集全部注册为 ``staging`` 以允许
``store.write_arrow`` 写入，写完 ``build_dataset_manifest``。

Scale 预设：
    - tiny   ：daily 8 符号 × 1 年, minute 2 符号 × 1 天（测试用, <30s）
    - small  ：daily 300 符号 × 3 年, minute 20 符号 × 1 月
    - full   ：daily 4000 符号 × 10 年, minute 100 符号 × 1 年
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pyarrow as pa

DEFAULT_SEED = 42
_END_YEAR = 2024  # 固定锚点，不依赖当前时钟 → 完全可复现
_EPOCH = dt.date(1970, 1, 1)

_DAILY_COLUMNS = (
    "TradeDate",
    "Symbol",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Amount",
    "VWAP",
)
_DAILY_SCHEMA = {
    "TradeDate": "date",
    "Symbol": "string",
    "Open": "double",
    "High": "double",
    "Low": "double",
    "Close": "double",
    "Volume": "int",
    "Amount": "double",
    "VWAP": "double",
}
_MINUTE_SCHEMA = {
    "datetime": "timestamp",
    "Symbol": "string",
    "open": "double",
    "high": "double",
    "low": "double",
    "close": "double",
    "volume": "int",
    "amount": "double",
}
_INCOME_SCHEMA = {
    "PubDate": "date",
    "Symbol": "string",
    "period_end": "date",
    "net_profit": "double",
    "revenue": "double",
}
_UNIVERSE_SCHEMA = {"TradeDate": "date", "Symbol": "string"}


# ---------------------------------------------------------------------------
# 确定性序列生成
# ---------------------------------------------------------------------------

def symbol_list(n: int) -> list[str]:
    """确定性 pseudo-A 股代码：600000..600000+n-1。"""
    return [f"{600000 + i:06d}" for i in range(int(n))]


def trading_dates(start_year: int, end_year: int) -> list[dt.date]:
    """[start_year-01-01, end_year-12-31] 全工作日（确定性 bdate_range）。"""
    import pandas as pd

    idx = pd.bdate_range(
        dt.date(start_year, 1, 1), dt.date(end_year, 12, 31)
    )
    return [d.date() for d in idx]


def _minute_timestamps_utc(day: dt.date) -> list[dt.datetime]:
    """A 股交易时段（北京 09:30-11:30 / 13:00-15:00）的 naive UTC 表示。

    北京 09:30 = UTC 01:30，因此存 UTC 01:30-03:29 与 05:00-06:59。
    与 read/aggregation 的约定一致：naive TIMESTAMP 视为 UTC，
    market='ashare' 聚合时再 timezone() 转回北京 → 09:30-11:30 精确命中。
    """
    base = dt.datetime(day.year, day.month, day.day)
    out: list[dt.datetime] = []
    for hour, minute_start, count in ((1, 30, 120), (5, 0, 120)):
        for i in range(count):
            out.append(base + dt.timedelta(hours=hour, minutes=minute_start + i))
    return out


# ---------------------------------------------------------------------------
# 表格生成器（返回 chunk 迭代器，避免一次性物化全量大表）
# ---------------------------------------------------------------------------

def _gen_daily_chunks(
    symbols: list[str],
    dates: list[dt.date],
    *,
    seed: int,
    chunk_symbols: int = 200,
) -> Iterator[pa.Table]:
    n_sym = len(symbols)
    n_days = len(dates)
    rng = np.random.default_rng(seed)
    # 每标的一次性随机游走（先算满 (n_sym, n_days)，再按 chunk 切）。
    base = rng.lognormal(mean=4.2, sigma=0.7, size=n_sym)  # ~50-300 元
    rets = rng.normal(0.0, 0.02, size=(n_sym, n_days))
    shock = rng.normal(0.0, 0.003, size=(n_sym, n_days))
    hi_lo = rng.uniform(0.002, 0.018, size=(n_sym, n_days))
    vw_noise = rng.normal(0.0, 0.001, size=(n_sym, n_days))
    volume = rng.integers(50_000, 5_000_000, size=(n_sym, n_days))

    close = base[:, None] * np.exp(np.cumsum(rets, axis=1))
    open_ = close * (1.0 + shock)
    high = np.maximum(open_, close) * (1.0 + hi_lo)
    low = np.minimum(open_, close) * (1.0 - hi_lo)
    amount = volume * close * 100.0
    vwap = close * (1.0 + vw_noise)

    dates_np = np.array(dates, dtype="datetime64[D]")
    symbols_arr = np.array(symbols, dtype=object)

    for start in range(0, n_sym, chunk_symbols):
        sl = slice(start, min(start + chunk_symbols, n_sym))
        sym_block = np.repeat(symbols_arr[sl], n_days)
        date_block = np.tile(dates_np, sym_block.shape[0] // n_days)
        yield pa.table(
            {
                "TradeDate": pa.array(date_block, type=pa.date32()),
                "Symbol": pa.array(sym_block),
                "Open": _flat(open_[sl]),
                "High": _flat(high[sl]),
                "Low": _flat(low[sl]),
                "Close": _flat(close[sl]),
                "Volume": _flat(volume[sl]),
                "Amount": _flat(amount[sl]),
                "VWAP": _flat(vwap[sl]),
            }
        )


def _gen_universe_chunks(
    symbols: list[str],
    dates: list[dt.date],
    *,
    seed: int,
    chunk_symbols: int = 1000,
) -> Iterator[pa.Table]:
    """universe 成员：确定性 ``i % 2 == 0``（约 50% 标的）。

    与 daily 同日期轴，INNER JOIN 把输出行空间裁剪到 in-universe 标的。
    """
    n_days = len(dates)
    keep = [i for i in range(len(symbols)) if i % 2 == 0]
    sym_in = np.array([symbols[i] for i in keep], dtype=object)
    dates_np = np.array(dates, dtype="datetime64[D]")
    for start in range(0, len(sym_in), chunk_symbols):
        sl = slice(start, min(start + chunk_symbols, len(sym_in)))
        sub = sym_in[sl]
        sym_block = np.repeat(sub, n_days)
        date_block = np.tile(dates_np, sym_block.shape[0] // n_days)
        yield pa.table(
            {
                "TradeDate": pa.array(date_block, type=pa.date32()),
                "Symbol": pa.array(sym_block),
            }
        )


def _gen_fundamental_table(symbols: list[str], start_year: int, end_year: int, *, seed: int) -> pa.Table:
    """季度财务：period_end=季度末, PubDate≈period_end+25 天（公告滞后）。"""
    rng = np.random.default_rng(seed + 1)
    periods: list[dt.date] = []
    for y in range(start_year, end_year + 1):
        for m, d in ((3, 31), (6, 30), (9, 30), (12, 31)):
            periods.append(dt.date(y, m, d))
    pubdate: list[dt.date] = []
    sym_col: list[str] = []
    period_col: list[dt.date] = []
    profit_col: list[float] = []
    rev_col: list[float] = []
    for sym in symbols:
        profit = rng.lognormal(mean=18.0, sigma=1.0)  # ~ 数千万
        for pe in periods:
            pub = pe + dt.timedelta(days=25)
            if pub.year > end_year:
                pub = dt.date(end_year, 12, 20)
            pubdate.append(pub)
            sym_col.append(sym)
            period_col.append(pe)
            profit = profit * (1.0 + rng.normal(0.0, 0.1))
            profit_col.append(profit)
            rev_col.append(profit * rng.uniform(3.0, 6.0))
    return pa.table(
        {
            "PubDate": pa.array(np.array(pubdate, dtype="datetime64[D]"), type=pa.date32()),
            "Symbol": pa.array(sym_col),
            "period_end": pa.array(np.array(period_col, dtype="datetime64[D]"), type=pa.date32()),
            "net_profit": pa.array(profit_col),
            "revenue": pa.array(rev_col),
        }
    )


def _gen_minute_chunks(
    symbols: list[str],
    days: list[dt.date],
    *,
    seed: int,
    chunk_symbols: int = 50,
) -> Iterator[pa.Table]:
    """分钟数据：每 (symbol, day) 240 根，日内随机游走；UTC 时间戳。"""
    n_sym = len(symbols)
    n_min = 240
    rng = np.random.default_rng(seed + 3)
    bases = 10.0 + (np.arange(n_sym) % 7) * 5.0
    symbols_arr = np.array(symbols, dtype=object)
    times_np = np.array(_minute_timestamps_utc(days[0]), dtype="datetime64[ns]")

    for start in range(0, n_sym, chunk_symbols):
        sym_idx = np.arange(start, min(start + chunk_symbols, n_sym))
        m = len(sym_idx)
        parts: dict[str, list[np.ndarray]] = {
            "datetime": [],
            "Symbol": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
            "amount": [],
        }
        for day in days:
            # 每 (symbol, day) 独立日内游走。
            rets = rng.normal(0.0, 0.0006, size=(m, n_min))
            close = bases[sym_idx, None] * np.exp(np.cumsum(rets, axis=1))
            open_ = np.concatenate([bases[sym_idx, None], close[:, :-1]], axis=1)
            hi = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.002, size=(m, n_min)))
            lo = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.002, size=(m, n_min)))
            vol = rng.integers(100, 5000, size=(m, n_min))
            amt = vol * close * 100.0
            parts["datetime"].append(np.tile(times_np, m))
            parts["Symbol"].append(np.repeat(symbols_arr[sym_idx], n_min))
            parts["open"].append(_flat(open_))
            parts["high"].append(_flat(hi))
            parts["low"].append(_flat(lo))
            parts["close"].append(_flat(close))
            parts["volume"].append(_flat(vol))
            parts["amount"].append(_flat(amt))
        yield pa.table(
            {
                "datetime": pa.array(np.concatenate(parts["datetime"]), type=pa.timestamp("ns")),
                "Symbol": pa.array(np.concatenate(parts["Symbol"])),
                "open": pa.array(np.concatenate(parts["open"])),
                "high": pa.array(np.concatenate(parts["high"])),
                "low": pa.array(np.concatenate(parts["low"])),
                "close": pa.array(np.concatenate(parts["close"])),
                "volume": pa.array(np.concatenate(parts["volume"])),
                "amount": pa.array(np.concatenate(parts["amount"])),
            }
        )


def _flat(a: np.ndarray) -> np.ndarray:
    """(m, n) → 行优先展平 (m*n,)；row-major = 每标的连续 n 行。"""
    return np.ascontiguousarray(a).ravel()


# ---------------------------------------------------------------------------
# Store 构造
# ---------------------------------------------------------------------------

def _registry_cfg(tmp_dir: Path, build_minute: bool) -> dict[str, dict[str, Any]]:
    daily_root = tmp_dir / "daily"
    minute_root = tmp_dir / "minute"
    fin_root = tmp_dir / "fundamental"
    uni_root = tmp_dir / "universe"
    for r in (daily_root, minute_root, fin_root, uni_root):
        r.mkdir(parents=True, exist_ok=True)
    cfg: dict[str, dict[str, Any]] = {
        "ashare_stock_daily": {
            "kind": "static",
            "access_mode": "staging",
            "layout": "plain",
            "root": str(daily_root),
            "glob": "part-*.parquet",
            "time_column": "TradeDate",
            "instrument_column": "Symbol",
            "schema": dict(_DAILY_SCHEMA),
        },
        "ashare_universe": {
            "kind": "static",
            "access_mode": "staging",
            "layout": "plain",
            "root": str(uni_root),
            "glob": "part-*.parquet",
            "time_column": "TradeDate",
            "instrument_column": "Symbol",
            "schema": dict(_UNIVERSE_SCHEMA),
        },
        "ashare_stock_income": {
            "kind": "static",
            "access_mode": "staging",
            "layout": "plain",
            "root": str(fin_root),
            "glob": "part-*.parquet",
            "time_column": "PubDate",
            "instrument_column": "Symbol",
            "schema": dict(_INCOME_SCHEMA),
        },
    }
    if build_minute:
        cfg["ashare_stock_minute"] = {
            "kind": "static",
            "access_mode": "staging",
            "layout": "plain",
            "root": str(minute_root),
            "glob": "part-*.parquet",
            "time_column": "datetime",
            "instrument_column": "Symbol",
            "schema": dict(_MINUTE_SCHEMA),
        }
    return cfg


def _make_store(tmp_dir: Path, cfg: dict[str, dict[str, Any]]):
    import yaml

    cfg_path = tmp_dir / "datasets.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    from data_access.core.engine import DuckDBEngine
    from data_access.registry import load_registry
    from data_access.store import DataAccessStore

    return DataAccessStore(load_registry(cfg_path), DuckDBEngine(threads=2))


def _write_chunked(store: Any, dataset: str, chunks: Iterator[pa.Table]) -> int:
    """首块 overwrite（清目录）→ 后续 append；最后 build manifest。"""
    first = True
    total = 0
    for tbl in chunks:
        mode = "overwrite" if first else "append"
        res = store.write_arrow(dataset, tbl, mode=mode)
        total += int(res.get("rows", tbl.num_rows))
        first = False
    store.build_dataset_manifest(dataset)
    return total


def _scale_params(scale: str | None, symbols: int | None, years: int | None) -> dict[str, int]:
    s = (scale or "small").strip().lower()
    if s == "full":
        p = {"daily_symbols": 4000, "daily_years": 10, "minute_symbols": 100, "minute_days": 244}
    elif s == "tiny":
        p = {"daily_symbols": 8, "daily_years": 1, "minute_symbols": 2, "minute_days": 1}
    else:
        p = {"daily_symbols": 300, "daily_years": 3, "minute_symbols": 20, "minute_days": 22}
    if symbols is not None:
        p["daily_symbols"] = int(symbols)
    if years is not None:
        p["daily_years"] = int(years)
    p["minute_symbols"] = min(p["minute_symbols"], p["daily_symbols"])
    return p


# ---------------------------------------------------------------------------
# 共享执行工具（benchmark_*.py 用）
# ---------------------------------------------------------------------------

def time_ms(fn: Callable[[], Any]) -> tuple[float, Any]:
    """执行 fn 并返回 (耗时 ms, 结果)。"""
    import time

    t0 = time.perf_counter()
    result = fn()
    return (time.perf_counter() - t0) * 1000.0, result


class ScanProbe:
    """包装 engine.execute_arrow，统计物理扫描次数与返回字节（单线程 benchmark）。"""

    def __init__(self, store: Any) -> None:
        self.store = store
        self.scans = 0
        self.bytes = 0
        self.rows = 0
        self._orig = None

    def __enter__(self) -> "ScanProbe":
        engine = getattr(self.store, "_engine", None)
        self._orig = getattr(engine, "execute_arrow", None)
        if self._orig is None:
            return self

        def _wrapped(sql, params=None, **kw):
            self.scans += 1
            table = self._orig(sql, params, **kw)
            if table is not None:
                self.rows += int(getattr(table, "num_rows", 0) or 0)
                self.bytes += int(getattr(table, "nbytes", 0) or 0)
            return table

        engine.execute_arrow = _wrapped  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc: Any) -> None:
        engine = getattr(self.store, "_engine", None)
        if engine is not None and self._orig is not None:
            engine.execute_arrow = self._orig  # type: ignore[method-assign]

    def snapshot(self) -> dict[str, int]:
        return {"physical_scan_count": self.scans, "bytes_scanned": self.bytes, "rows": self.rows}


def read_full(store: Any, dataset: str, **kw: Any) -> pa.Table:
    """直接读全量 → Arrow Table（benchmark 热路径）。"""
    return store.read(dataset, **kw).to_arrow()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_fixture_store(
    tmp_dir: str | Path,
    *,
    symbols: int | None = None,
    years: int | None = None,
    freq: str = "daily",
    scale: str | None = None,
    seed: int = DEFAULT_SEED,
) -> tuple[Any, dict[str, Path]]:
    """构造完整 synthetic store。

    参数:
        tmp_dir: 数据落盘目录
        symbols/years: 显式覆盖日频规模（scale 仍决定 minute/fundamental 规模）
        freq: 'daily' | 'minute' | 'both' —— 是否构造分钟数据
        scale: 'tiny' | 'small' | 'full'
        seed: 确定性随机种子

    返回:
        (store, paths) —— paths 含各数据集根目录与 datasets.yaml 路径
    """
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    build_minute = freq in ("both", "minute")
    cfg = _registry_cfg(tmp_dir, build_minute=build_minute)
    store = _make_store(tmp_dir, cfg)

    p = _scale_params(scale, symbols, years)
    dates = trading_dates(_END_YEAR - p["daily_years"] + 1, _END_YEAR)
    daily_symbols = symbol_list(p["daily_symbols"])

    _write_chunked(
        store, "ashare_stock_daily", _gen_daily_chunks(daily_symbols, dates, seed=seed)
    )
    _write_chunked(
        store, "ashare_universe", _gen_universe_chunks(daily_symbols, dates, seed=seed)
    )
    _write_chunked(
        store,
        "ashare_stock_income",
        iter(
            [
                _gen_fundamental_table(
                    daily_symbols,
                    _END_YEAR - p["daily_years"] + 1,
                    _END_YEAR,
                    seed=seed,
                )
            ]
        ),
    )
    if build_minute:
        minute_symbols = symbol_list(p["minute_symbols"])
        minute_days = dates[-p["minute_days"] :]
        _write_chunked(
            store,
            "ashare_stock_minute",
            _gen_minute_chunks(minute_symbols, minute_days, seed=seed),
        )

    paths = {name: Path(v["root"]) for name, v in cfg.items()}
    paths["cfg"] = tmp_dir / "datasets.yaml"
    return store, paths


def tiny_fixture(tmp_dir: str | Path) -> tuple[Any, dict[str, Path]]:
    """测试用极小 fixture（<30s 建完）。"""
    return build_fixture_store(tmp_dir, scale="tiny", freq="both")


__all__ = [
    "DEFAULT_SEED",
    "build_fixture_store",
    "symbol_list",
    "tiny_fixture",
    "trading_dates",
]
