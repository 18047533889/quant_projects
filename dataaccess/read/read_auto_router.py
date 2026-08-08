"""read_auto / read_auto_stream 共享路由逻辑。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from .query_budget import QueryBudget
from data_access.registry import StaticDataset
from .stats import DatasetReadStats, dataset_read_stats, load_stats_sidecar

if TYPE_CHECKING:
    from .store import DataAccessStore

logger = logging.getLogger("data_access.read_auto_router")

ReadAutoMode = Literal["auto", "arrow", "stream", "polars"]


def resolve_read_auto_mode(
    store: "DataAccessStore",
    dataset: str,
    *,
    columns: list[str] | tuple[str, ...] | None = None,
    time_range: tuple[Any, Any] | None = None,
    query_budget: QueryBudget | None = None,
    mode: str = "auto",
    prefer_polars: bool = False,
    budget: QueryBudget | None = None,
    **params: Any,
) -> tuple[str, DatasetReadStats | None]:
    """解析 read_auto 有效读路径；返回 (resolved_mode, stats)。"""
    ds = store._registry.get(dataset)
    effective_budget = budget if budget is not None else store._resolve_read_budget(
        ds, query_budget
    )
    resolved_mode = str(mode or "auto").lower()
    stats: DatasetReadStats | None = None

    if resolved_mode == "auto":
        sidecar = None
        if isinstance(ds, StaticDataset):
            sidecar = load_stats_sidecar(ds.root)
            if sidecar is not None and not _stats_sidecar_fresh(store, dataset, sidecar):
                sidecar = None  # #P1-71 数据已变 → 旧 sidecar 不参与路由
        stats = dataset_read_stats(
            store,
            dataset,
            columns=list(columns) if columns else None,
            time_range=time_range,
            prefer_polars=prefer_polars,
            sidecar=sidecar,
            **params,
        )
        resolved_mode = stats.suggested_mode
        if (
            effective_budget.max_rows is not None
            and stats.estimated_rows > effective_budget.max_rows
            and resolved_mode == "arrow"
        ):
            resolved_mode = "stream"
        logger.info(
            "read_auto dataset=%s mode=%s estimated_rows=%d files=%d budget_max_rows=%s",
            dataset,
            resolved_mode,
            stats.estimated_rows,
            stats.parquet_files,
            effective_budget.max_rows,
        )

    return resolved_mode, stats


def _stats_sidecar_fresh(store: "DataAccessStore", dataset: str, sidecar: Any) -> bool:
    """#P1-71/#P0-C10 sidecar 的 source identity 必须与当前数据集一致才参与 CBO。

    无 source_epoch 的 legacy sidecar：
        - production/strict → **stale**（fail-closed，重新统计）——旧统计没有源
          身份、无法判过期，可能把实际上很大的表错误路由到 Arrow materialization；
        - research → 视为 fresh（向后兼容）。
    有 source_epoch 但当前 manifest 不一致 → stale。``manifest_version()``
    检查失败 → **stale**（fail-closed：宁可重统计，也不能拿可能过期的统计路由）。
    """
    epoch = getattr(sidecar, "source_epoch", None)
    if epoch is None:
        from data_access.read.query_budget import is_strict_semantics

        if is_strict_semantics():
            return False
        return True
    try:
        token = store.manifest_version(dataset)
    except Exception:
        return False
    if not isinstance(token, dict):
        return False
    cur = token.get("source_epoch") or token.get("manifest_epoch")
    return cur == epoch
