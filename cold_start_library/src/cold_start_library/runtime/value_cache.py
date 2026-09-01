"""冷启动因子值落盘缓存：按 DSL hash 存 panel，供 AlphaPROBE 免重算。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def default_cache_root() -> Path:
    env = os.getenv("COLD_START_VALUE_CACHE")
    if env:
        return Path(os.path.expanduser(env))
    return Path(__file__).resolve().parents[3] / "data" / "ashare" / "value_cache"


def dsl_hash(expr: str) -> str:
    norm = " ".join(str(expr).split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:16]


def factor_dir(cache_root: Path, expr: str) -> Path:
    return cache_root / "factors" / dsl_hash(expr)


def save_factor_panel(
    cache_root: Path,
    *,
    factor_id: str,
    expr: str,
    dates: pd.Index,
    stocks: pd.Index,
    values: np.ndarray,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """values: [n_days, n_stocks] float."""
    out = factor_dir(cache_root, expr)
    out.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(values, dtype=np.float16)
    np.savez_compressed(
        out / "panel.npz",
        values=arr,
        dates=np.asarray(pd.Index(dates).astype(str).to_numpy()),
        stocks=np.asarray(pd.Index(stocks).astype(str).to_numpy()),
    )
    meta = {
        "factor_id": factor_id,
        "expr": " ".join(str(expr).split()),
        "expr_hash": dsl_hash(expr),
        "n_days": int(arr.shape[0]),
        "n_stocks": int(arr.shape[1]),
        "dtype": "float16",
        "metrics": metrics or {},
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_factor_meta(cache_root: Path, expr: str) -> dict[str, Any] | None:
    path = factor_dir(cache_root, expr) / "meta.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_factor_panel(cache_root: Path, expr: str) -> tuple[pd.Index, pd.Index, np.ndarray] | None:
    path = factor_dir(cache_root, expr) / "panel.npz"
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=False)
    dates = pd.Index(data["dates"].astype(str))
    stocks = pd.Index(data["stocks"].astype(str))
    values = np.asarray(data["values"], dtype=np.float32)
    return dates, stocks, values


def panel_to_aligned_tensor(
    dates: pd.Index,
    stocks: pd.Index,
    values: np.ndarray,
    target_dates: pd.Index,
    target_stocks: pd.Index,
) -> np.ndarray:
    """把缓存 panel 对齐到目标日历/股票宇宙，返回 float32 [n_days, n_stocks]。"""
    df = pd.DataFrame(values, index=pd.Index(dates.astype(str)), columns=pd.Index(stocks.astype(str)))
    aligned = df.reindex(index=pd.Index(target_dates.astype(str)), columns=pd.Index(target_stocks.astype(str)))
    return aligned.to_numpy(dtype=np.float32)
