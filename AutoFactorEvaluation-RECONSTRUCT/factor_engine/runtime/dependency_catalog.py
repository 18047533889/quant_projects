# -*- coding: utf-8 -*-
"""因子依赖 catalog 查询层：data event → 受影响因子。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from runtime.incremental_scheduler import DataEvent, FactorUpdatePlan, plan_updates_from_data_event
from storage.catalog import FactorCatalog


@dataclass(frozen=True)
class DependencySummary:
    """列 → 因子反向索引摘要。"""

    column: str
    factor_ids: tuple[str, ...]
    source_datasets: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "factor_ids": list(self.factor_ids),
            "source_datasets": list(self.source_datasets),
            "factor_count": len(self.factor_ids),
        }


class DependencyCatalog:
    """``FactorCatalog.factor_dependency`` 表的只读/写入门面。"""

    def __init__(self, catalog: FactorCatalog) -> None:
        self._catalog = catalog

    @classmethod
    def from_lake(cls, lake_root: str | Path) -> "DependencyCatalog":
        from storage.materializer import ParquetMaterializer

        cat = ParquetMaterializer(lake_root=lake_root).catalog
        return cls(cat)

    @property
    def catalog(self) -> FactorCatalog:
        return self._catalog

    def factors_for_column(
        self,
        column: str,
        *,
        dataset: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = self._catalog.list_factors_for_column(column)
        if dataset is None:
            return rows
        return [
            r
            for r in rows
            if not r.get("source_dataset") or str(r["source_dataset"]) == str(dataset)
        ]

    def factors_for_dataset(self, dataset: str) -> list[dict[str, Any]]:
        if hasattr(self._catalog, "list_factors_for_dataset"):
            return self._catalog.list_factors_for_dataset(dataset)
        rows = self._catalog.list_factors()
        out: list[dict[str, Any]] = []
        for row in rows:
            dep = self._catalog.get_factor_dependency(str(row.get("factor_id", "")))
            if dep is None:
                continue
            if dep.get("source_dataset") == dataset:
                out.append(dep)
        return out

    def reverse_index(self, columns: Iterable[str] | None = None) -> list[DependencySummary]:
        """构建列 → 因子列表摘要。"""
        if columns is None:
            cols = self._catalog.list_dependency_columns()
        else:
            cols = sorted(set(columns))
        summaries: list[DependencySummary] = []
        for col in cols:
            deps = self._catalog.list_factors_for_column(col)
            factor_ids = tuple(sorted(d["factor_id"] for d in deps))
            datasets = tuple(
                sorted(
                    {
                        str(d["source_dataset"])
                        for d in deps
                        if d.get("source_dataset")
                    }
                )
            )
            summaries.append(
                DependencySummary(column=col, factor_ids=factor_ids, source_datasets=datasets)
            )
        return summaries

    def plan_for_event(
        self,
        event: DataEvent | dict[str, Any],
        *,
        end_date: str | None = None,
        lookback_extra: int = 5,
        market: str | None = None,
    ) -> list[FactorUpdatePlan]:
        if not isinstance(event, DataEvent):
            event = DataEvent(
                dataset=str(event["dataset"]),
                column=str(event["column"]),
                updated_date=str(event["updated_date"]),
            )
        return plan_updates_from_data_event(
            self._catalog,
            event,
            end_date=end_date,
            lookback_extra=lookback_extra,
            market=market,
        )
