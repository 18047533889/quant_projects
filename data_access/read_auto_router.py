"""read_auto / read_auto_stream 共享路由逻辑。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from .query_budget import QueryBudget
from .registry import StaticDataset
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
