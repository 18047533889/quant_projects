# -*- coding: utf-8
"""受控 Polars 扫描句柄：production 下 collect 必经 budget + snapshot。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from data_access.core import audit
from data_access.core.exceptions import ValidationError
from .query_budget import QueryBudget, collect_polars_with_budget, _production_mode
from .read_contract import DataSnapshot, ReadLineage, ReadResult, ReadStats

# 这些 LazyFrame 方法会执行计划、产生物化结果或写出数据，若通过 __getattr__
# 直接透传就能绕过 QueryBudget / lineage / audit。
_MATERIALIZING_METHODS = {
    "collect",
    "collect_async",
    "fetch",
    "profile",
    "sink_batches",
    "sink_csv",
    "sink_ipc",
    "sink_ndjson",
    "sink_parquet",
}


@dataclass
class ScanHandle:
    """包装 LazyFrame；``collect()`` 强制 budget、snapshot 与真实执行审计。"""

    _lf: Any
    snapshot: DataSnapshot
    budget: QueryBudget
    lineage: ReadLineage
    _store: Any = None

    def lazyframe(self) -> Any:
        """返回底层 LazyFrame（development 用；production 只能使用受控方法）。"""
        if _production_mode():
            raise ValidationError(
                "production 模式禁止直接获取裸 LazyFrame；请使用 ScanHandle.collect()。"
            )
        return self._lf

    def collect(self) -> ReadResult:
        """执行 Polars 计划并返回绑定 snapshot/lineage 的 ``ReadResult``。"""
        import time

        start = time.perf_counter()
        table: pa.Table | None = None
        ok = False
        err_msg: str | None = None
        try:
            table = collect_polars_with_budget(self._lf, query_budget=self.budget)
            elapsed_ms = (time.perf_counter() - start) * 1000
            stats = ReadStats(
                rows=table.num_rows,
                bytes=table.nbytes,
                elapsed_ms=elapsed_ms,
                paths=tuple(f.path for f in self.snapshot.files[:20]),
            )
            ok = True
            return ReadResult(
                table=table,
                snapshot=self.snapshot,
                stats=stats,
                lineage=self.lineage,
            )
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            audit.record(
                op="scan_collect",
                dataset=self.lineage.dataset,
                ok=ok,
                rows=table.num_rows if table is not None else None,
                paths=[f.path for f in self.snapshot.files[:5]] or None,
                params=dict(self.lineage.params) or None,
                elapsed_ms=(time.perf_counter() - start) * 1000,
                error=err_msg,
                extra={
                    "snapshot_id": self.snapshot.snapshot_id,
                    "columns": list(self.lineage.columns) or None,
                    "time_range": self.lineage.time_range,
                    "instrument_count": len(self.lineage.instrument_filter),
                },
            )

    def collect_table(self) -> pa.Table:
        return self.collect().table

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if _production_mode() and (
            name in _MATERIALIZING_METHODS or name.startswith("sink_")
        ):
            raise ValidationError(
                f"production 模式禁止通过 ScanHandle.{name}() 绕过受控 collect；"
                "请使用 ScanHandle.collect()，写出数据请走 data_access.write_arrow/publish。"
            )

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
