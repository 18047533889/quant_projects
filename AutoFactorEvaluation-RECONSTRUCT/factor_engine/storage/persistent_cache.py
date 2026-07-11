"""持久化磁盘缓存 — 进程间共享的因子计算缓存。

设计目标:
  - 列数据缓存：DataSource 读取的 parquet 列持久化到本地磁盘，
    后续任何进程可直接读取缓存，避免重复 I/O。
  - 中间结果缓存：PlanNode 子树的计算结果持久化到本地磁盘，
    不同因子的评估可复用重叠的计算子树。

缓存目录结构::

    {cache_root}/factor_cache/
    ├── columns/{hash}.parquet       # 已加载的列数据
    ├── plans/{hash}.parquet         # 中间计算结果
    └── metadata.json                 # LRU 驱逐元数据
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PersistentCache:
    """持久化磁盘缓存。

    Args:
        cache_root: 缓存根目录。所有缓存文件写入 ``{cache_root}/factor_cache/``。
        max_mb: 缓存大小上限（MB）。超过时驱逐最久未访问的条目。
        ttl_seconds: 缓存条目 TTL（秒）。超过此时间的条目将被视为过期。
    """

    def __init__(
        self,
        cache_root: str | Path,
        max_mb: int = 1024,
        ttl_seconds: int = 86400,
    ):
        self._base = Path(cache_root) / "factor_cache"
        self._columns_dir = self._base / "columns"
        self._plans_dir = self._base / "plans"
        self._metadata_path = self._base / "metadata.json"
        self._max_mb = max_mb
        self._ttl_seconds = ttl_seconds
        self._in_memory: dict[str, Any] = {}  # 进程内热缓存
        self._hit_count = 0
        self._miss_count = 0

        self._columns_dir.mkdir(parents=True, exist_ok=True)
        self._plans_dir.mkdir(parents=True, exist_ok=True)
        self._metadata = self._load_metadata()

    # ── 公共 API ──

    def get_column(self, key: str) -> pd.Series | None:
        """从缓存获取列数据。"""
        return self._get(f"columns/{self._hash(key)}")

    def set_column(self, key: str, series: pd.Series) -> None:
        """将列数据写入缓存。"""
        self._set(f"columns/{self._hash(key)}", series)

    def get_plan(self, key: str) -> Any | None:
        """从缓存获取中间计算结果。"""
        return self._get(f"plans/{self._hash(key)}")

    def set_plan(self, key: str, result: Any) -> None:
        """将中间计算结果写入缓存。"""
        self._set(f"plans/{self._hash(key)}", result)

    @property
    def hit_rate(self) -> float:
        total = self._hit_count + self._miss_count
        return self._hit_count / total if total > 0 else 0.0

    def stats(self) -> dict:
        return {
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": round(self.hit_rate, 3),
            "cache_size_mb": self._estimate_size_mb(),
            "column_files": len(list(self._columns_dir.glob("*.parquet"))),
            "plan_files": len(list(self._plans_dir.glob("*.parquet"))),
        }

    # ── 内部 ──

    def _get(self, rel_path: str) -> Any | None:
        """尝试从缓存读取。"""
        # 1. 内存热缓存
        if rel_path in self._in_memory:
            self._hit_count += 1
            return self._in_memory[rel_path]

        # 2. 磁盘缓存
        cache_file = self._base / rel_path
        if not cache_file.exists():
            self._miss_count += 1
            return None

        # 检查 TTL
        mtime = cache_file.stat().st_mtime
        if time.time() - mtime > self._ttl_seconds:
            self._miss_count += 1
            cache_file.unlink(missing_ok=True)
            return None

        try:
            result = pd.read_parquet(cache_file)
            # 如果是单列 DataFrame 转回 Series
            if isinstance(result, pd.DataFrame) and len(result.columns) == 1:
                result = result.iloc[:, 0]
            self._in_memory[rel_path] = result  # 提升至内存
            self._hit_count += 1
            self._touch_metadata(rel_path)
            return result
        except Exception:
            self._miss_count += 1
            return None

    def _set(self, rel_path: str, value: Any) -> None:
        """将值写入缓存。"""
        # 1. 写入内存
        self._in_memory[rel_path] = value

        # 2. 写入磁盘
        cache_file = self._base / rel_path
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            if isinstance(value, pd.Series):
                df = value.to_frame("value") if not isinstance(value, pd.DataFrame) else value
                df.to_parquet(cache_file, index=True, engine="pyarrow")
            elif isinstance(value, pd.DataFrame):
                value.to_parquet(cache_file, index=True, engine="pyarrow")
            else:
                # 非 DataFrame 类型暂不持久化
                return
        except Exception:
            pass  # 写入失败不阻塞

        self._touch_metadata(rel_path)

        # 3. 检查是否需要驱逐
        if self._estimate_size_mb() > self._max_mb:
            self._evict_lru()

    def _hash(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _touch_metadata(self, rel_path: str) -> None:
        self._metadata[rel_path] = time.time()
        self._flush_metadata()

    def _load_metadata(self) -> dict[str, float]:
        try:
            if self._metadata_path.exists():
                return json.loads(self._metadata_path.read_text())
        except Exception:
            pass
        return {}

    def _flush_metadata(self) -> None:
        try:
            self._metadata_path.write_text(
                json.dumps(self._metadata, ensure_ascii=False)
            )
        except Exception:
            pass

    def _estimate_size_mb(self) -> int:
        total = 0
        for f in self._base.rglob("*.parquet"):
            try:
                total += f.stat().st_size
            except OSError:
                pass
        return total // (1024 * 1024)

    def _evict_lru(self) -> None:
        """LRU 驱逐：删除最久未访问的条目直到低于上限。"""
        if not self._metadata:
            return
        sorted_items = sorted(self._metadata.items(), key=lambda x: x[1])
        for rel_path, _ in sorted_items:
            if self._estimate_size_mb() <= self._max_mb * 0.8:
                break
            cache_file = self._base / rel_path
            if cache_file.exists():
                cache_file.unlink(missing_ok=True)
            self._in_memory.pop(rel_path, None)
            self._metadata.pop(rel_path, None)
        self._flush_metadata()


# ── 兼容 CacheManager 接口的适配器 ──


class PersistentCacheAdapter:
    """将 ``PersistentCache`` 适配为 ``CacheManager`` 兼容的接口。

    这样 ``PandasBackend`` 等现有代码无需修改即可使用持久化缓存。
    """

    def __init__(self, persistent_cache: PersistentCache):
        self._pc = persistent_cache

    def get(self, key: str):
        return self._pc.get_plan(key)

    def set(self, key: str, value) -> None:
        self._pc.set_plan(key, value)
