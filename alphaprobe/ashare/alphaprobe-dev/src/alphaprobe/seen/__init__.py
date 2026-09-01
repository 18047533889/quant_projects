"""AlphaPROBE Seen Index（Phase B）—— 有状态持久化去重包。

独立 SQLite 数据库（默认 ``artifacts/global_seen_index.sqlite3``，
env ``ALPHAPROBE_SEEN_DB`` 可覆盖），与 GlobalMemoryStore 的 §39 全表分离。
"""

from __future__ import annotations

from alphaprobe.seen.store import SeenStore
from alphaprobe.seen.dedup_service import DedupConfig, DedupService


def build_seen_index(
    db_path: str | None = None,
    *,
    config: DedupConfig | None = None,
) -> DedupService:
    """构建有状态持久化 GlobalSeenIndex。

    Parameters
    ----------
    db_path : str, optional
        SQLite 数据库路径。默认 ``artifacts/global_seen_index.sqlite3`` 或
        env ``ALPHAPROBE_SEEN_DB``。
    config : DedupConfig, optional
        去重配置（阈值等）。

    Returns
    -------
    DedupService
        高层去重服务 API。
    """
    store = SeenStore(db_path)
    return DedupService(store, config=config)


__all__ = [
    "SeenStore",
    "DedupConfig",
    "DedupService",
    "build_seen_index",
]