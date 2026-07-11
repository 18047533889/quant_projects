"""
AutoFactorEvaluation 缓存工具模块

提供全局共享的行情数据缓存、forward return 缓存等功能，
避免多 Worker 重复加载同一份数据。

设计原理（利用 Linux fork 写时复制）：
    主进程通过 preload_market_data() 预加载行情数据到模块级全局变量。
    multiprocessing.Process 使用 fork 创建子进程时，子进程继承父进程的
    完整内存空间（含已加载的 DataFrame）。由于 COW 机制，子进程读取
    该数据不产生额外内存拷贝，速度等同于直接访问内存。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("cache_utils")

# 进程级全局缓存（Linux fork 继承给子进程）
_global_market_data: pd.DataFrame | None = None
_global_cache_key: str | None = None


def preload_market_data(
    market_data_path: str | Path,
    cache_dir: str | Path | None = None,
    *,
    rename_columns: dict[str, str] | None = None,
    lowercase: bool = False,
) -> pd.DataFrame:
    """预加载行情数据到进程内存（供 fork 子进程继承）。

    应在主进程创建 Worker 池之前调用。子进程通过 fork 继承已加载的
    DataFrame，读取时不产生额外 I/O 或内存拷贝（Linux COW）。

    首次调用会从磁盘加载并构建缓存；后续调用直接返回内存中的副本引用。
    """
    global _global_market_data
    if _global_market_data is not None:
        logger.info("  命中进程内存缓存: %d 行", len(_global_market_data))
        return _global_market_data

    df = _load_and_cache(market_data_path, cache_dir, rename_columns=rename_columns, lowercase=lowercase)
    _global_market_data = df
    return df


def load_market_data_cached(
    market_data_path: str | Path,
    cache_dir: str | Path | None = None,
    *,
    force_reload: bool = False,
    rename_columns: dict[str, str] | None = None,
    lowercase: bool = False,
) -> pd.DataFrame:
    """加载行情数据（优先进程内存 → 磁盘缓存 → 原始加载）。

    优先级：
        1. 进程内存缓存（由 preload_market_data 或本函数预先设置）
           → 零 I/O，纳秒级返回
        2. 磁盘缓存（单合并 parquet 文件）
           → ~2.6 秒
        3. 原始 2536 个日频文件
           → ~18 秒

    在 multiprocessing.Process(fork) 场景下，子进程首次调用本函数时，
    若主进程已 preload，第 1 级直接命中，无需任何磁盘读取。
    """
    global _global_market_data
    if _global_market_data is not None and not force_reload:
        return _global_market_data

    df = _load_and_cache(market_data_path, cache_dir, rename_columns=rename_columns, lowercase=lowercase, force_reload=force_reload)
    if not force_reload:
        _global_market_data = df
    return df


def _load_and_cache(
    market_data_path: str | Path,
    cache_dir: str | Path | None = None,
    *,
    rename_columns: dict[str, str] | None = None,
    lowercase: bool = False,
    force_reload: bool = False,
) -> pd.DataFrame:
    """底层加载逻辑：磁盘缓存 → 原始 2536 文件。"""
    market_data_path = Path(market_data_path)

    # 确定缓存路径（优先外部传参，次选 ConfigManager，拒绝硬编码回退）
    if cache_dir is None:
        from config_manager import ConfigManager
        try:
            cache_dir = ConfigManager().path("market_data_cache")
        except Exception:
            raise ValueError(
                "cache_dir 必须从外部传入或通过 ConfigManager.path('market_data_cache') 配置"
            )
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(
            f"行情数据缓存目录不存在: {cache_dir}\n"
            "  → 请确保该路径已在配置文件中声明且已被创建"
        )

    import hashlib
    _path_str = str(market_data_path.resolve())
    _cache_key = hashlib.sha256(_path_str.encode()).hexdigest()[:16]
    cache_file = cache_dir / f"merged_{_cache_key}.parquet"

    # 检查磁盘缓存
    if cache_file.exists() and not force_reload:
        logger.info("  命中磁盘缓存: %s", cache_file)
        df = pd.read_parquet(cache_file, engine="fastparquet")
        # 后处理列名
        if rename_columns:
            df = df.rename(columns=rename_columns)
        if lowercase:
            df.columns = [c.lower() for c in df.columns]
        return df

    # 重新加载 2536 个文件
    logger.info("  磁盘缓存未命中，加载原始行情: %s", market_data_path)
    if not market_data_path.exists():
        raise FileNotFoundError(f"行情数据目录不存在: {market_data_path}")

    parquet_files = sorted(market_data_path.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"行情目录下无 parquet 文件: {market_data_path}")

    dfs = []
    for f in parquet_files:
        try:
            df = pd.read_parquet(f, engine="fastparquet")
            dfs.append(df)
        except Exception as e:
            logger.debug("读取 %s 失败: %s", f, e)

    if not dfs:
        raise ValueError(f"无法读取任何行情 parquet 文件: {market_data_path}")

    merged = pd.concat(dfs, ignore_index=True)

    # 列名处理
    if rename_columns:
        merged = merged.rename(columns=rename_columns)
    if lowercase:
        merged.columns = [c.lower() for c in merged.columns]

    # 写入磁盘缓存
    logger.info("  写入磁盘缓存: %s (%d 行)", cache_file, len(merged))
    merged.to_parquet(cache_file, index=False)

    return merged


def clear_market_data_cache() -> None:
    """清理进程内存中的行情缓存。"""
    global _global_market_data, _global_cache_key
    _global_market_data = None
    _global_cache_key = None


def get_forward_return_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """获取 forward return 共享缓存目录。

    Args:
        cache_dir: 外部指定的缓存目录。为 None 时从 ConfigManager 读取。

    Raises:
        FileNotFoundError: 目录不存在时抛出。
    """
    if cache_dir is not None:
        _dir = Path(cache_dir)
    else:
        from config_manager import ConfigManager
        try:
            _dir = ConfigManager().path("timeseries_forward_return_cache")
        except Exception:
            raise ValueError(
                "forward return cache_dir 必须从外部传入"
                "或通过 ConfigManager.path('timeseries_forward_return_cache') 配置"
            )
    if not _dir.exists():
        raise FileNotFoundError(
            f"前向收益缓存目录不存在: {_dir}\n"
            "  → 请确保该路径已在配置文件中声明且已被创建"
        )
    return _dir
