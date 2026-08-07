"""DuckDB 版本能力探测：避免假设 PRAGMA / API 语义跨版本不变。"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import duckdb

logger = logging.getLogger("data_access.duckdb_capabilities")


@dataclass(frozen=True)
class DuckDBCapabilities:
    """按安装的 DuckDB 版本探测出的能力集。"""

    version: tuple[int, ...]
    supports_create_secret: bool      # >= 1.1：CREATE OR REPLACE SECRET
    object_cache_deprecated: bool     # 官方把 enable_object_cache 标为 legacy/no-op 的版本
    supports_hive_in_read_csv: bool   # read_csv 支持 hive_partitioning 命名参数
    supports_filename_in_read_parquet: bool  # read_parquet 支持 filename 参数

    @property
    def version_str(self) -> str:
        return ".".join(str(x) for x in self.version)


def detect_duckdb_capabilities() -> DuckDBCapabilities:
    """探测当前安装的 DuckDB 能力。结果进程内缓存。"""
    try:
        raw = duckdb.__version__.split("-")[0].split(".")
        version = tuple(int(x) for x in raw[:2])
    except (AttributeError, ValueError):
        version = (0, 0)
    return DuckDBCapabilities(
        version=version,
        supports_create_secret=version >= (1, 1),
        # 1.3 起官方文档把 enable_object_cache 标为 legacy（"does nothing"）；
        # 1.5.4 实测仍接受该 PRAGMA 但不再作为主性能开关。
        object_cache_deprecated=version >= (1, 3),
        supports_hive_in_read_csv=version >= (0, 10),
        supports_filename_in_read_parquet=version >= (0, 10),
    )


_caps: DuckDBCapabilities | None = None
_caps_lock = threading.Lock()
_object_cache_warned = False


def get_duckdb_capabilities() -> DuckDBCapabilities:
    global _caps
    if _caps is not None:
        return _caps
    with _caps_lock:
        if _caps is None:
            _caps = detect_duckdb_capabilities()
        return _caps


def reset_duckdb_capabilities() -> None:
    """测试用：清空能力缓存。"""
    global _caps, _object_cache_warned
    with _caps_lock:
        _caps = None
        _object_cache_warned = False


def warn_object_cache_once() -> None:
    """对 deprecated 的 enable_object_cache 只告警一次。"""
    global _object_cache_warned
    if _object_cache_warned:
        return
    _object_cache_warned = True
    logger.warning(
        "当前 DuckDB %s 已把 PRAGMA enable_object_cache 标为 legacy/no-op；"
        "建议改用 external file cache / HTTP metadata cache / connection reuse。",
        get_duckdb_capabilities().version_str,
    )
