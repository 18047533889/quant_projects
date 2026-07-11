# -*- coding: utf-8
"""受控 Polars 扫描句柄：production 下 collect 必经 budget + snapshot。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from .exceptions import ValidationError
from .query_budget import QueryBudget, collect_polars_with_budget
from .read_contract import DataSnapshot, ReadLineage, ReadResult, ReadStats


@dataclass
class ScanHandle:
    """包装 LazyFrame；``collect()`` 强制 budget 与 snapshot 审计。"""

    _lf: Any
    snapshot: DataSnapshot
    budget: QueryBudget
    lineage: ReadLineage
    _store: Any = None

    def lazyframe(self) -> Any:
        """返回底层 LazyFrame（development 用；production 建议只用 collect）。"""
        from .query_budget import _production_mode

        if _production_mode():
            raise ValidationError(
                "production 模式禁止直接获取裸 LazyFrame；请使用 ScanHandle.collect()。"
            )
        return self._lf

    def collect(self) -> ReadResult:
        import time

        start = time.perf_counter()
        table = collect_polars_with_budget(self._lf, query_budget=self.budget)
        elapsed_ms = (time.perf_counter() - start) * 1000
        stats = ReadStats(
            rows=table.num_rows,
            bytes=table.nbytes,
            elapsed_ms=elapsed_ms,
            paths=tuple(f.path for f in self.snapshot.files[:20]),
        )
        return ReadResult(
            table=table,
            snapshot=self.snapshot,
            stats=stats,
            lineage=self.lineage,
        )

    def collect_table(self) -> pa.Table:
        return self.collect().table

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        attr = getattr(self._lf, name)
        if not callable(attr):
            return attr

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            result = attr(*args, **kwargs)
            try:
                import polars as pl

                if isinstance(result, pl.LazyFrame):
                    return ScanHandle(
                        _lf=result,
                        snapshot=self.snapshot,
                        budget=self.budget,
                        lineage=self.lineage,
                        _store=self._store,
                    )
            except ImportError:
                pass
            return result

        return _wrapped
