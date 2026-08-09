# -*- coding: utf-8
"""受控 Polars 扫描句柄：production 下 collect 必经 budget + snapshot。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from data_access.core import audit
from data_access.core.exceptions import ValidationError
from .query_budget import QueryBudget, collect_polars_with_budget, is_strict_semantics
from .read_contract import (
    DataSnapshot,
    FileVersion,
    ReadLineage,
    ReadResult,
    ReadStats,
    _remote_object_meta,
    _remote_snapshot_meta_enabled,
    rebuild_snapshot_files,
)

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


def _remote_identity_changed(fv: Any, meta: dict[str, Any]) -> bool:
    """#P0 收官（0.9.5）：对比重新 HEAD 到的远程对象身份与 snapshot 记录值。

    只比真实身份字段（etag/content_length/version_id）——last_modified 在
    manifest/parquet/内存三种表示之间往返会丢精度，不比。snapshot 侧**没有记录
    任何可比较身份**（research 模式建的 snapshot）时，strict 下视为无法证明未变
    → fail-closed；research 下视为未变（宽容放行）。
    """
    etag = meta.get("etag")
    if fv.etag is not None and etag is not None and fv.etag != etag:
        return True
    length = meta.get("content_length")
    if (
        fv.content_length is not None
        and length is not None
        and int(fv.content_length) != int(length)
    ):
        return True
    version = meta.get("version_id")
    if fv.version_id is not None and version is not None and fv.version_id != version:
        return True
    if is_strict_semantics() and (
        fv.etag is None and fv.content_length is None and fv.version_id is None
    ):
        return True  # 无法证明未变 → fail-closed
    return False


@dataclass
class ScanHandle:
    """包装 LazyFrame；``collect()`` 强制 budget、snapshot 与真实执行审计。"""

    _lf: Any
    snapshot: DataSnapshot
    budget: QueryBudget
    lineage: ReadLineage
    _store: Any = None

    def lazyframe(self) -> Any:
        """返回底层 LazyFrame（research 用；production/strict 只能使用受控方法）。"""
        if is_strict_semantics():
            raise ValidationError(
                "production/strict 模式禁止直接获取裸 LazyFrame；请使用 "
                "ScanHandle.collect() / collect_table()。"
            )
        return self._lf

    def native_lazyframe(self) -> Any:
        """composition-only 返回底层 LazyFrame（Phase 5 正式接口）。

        允许在扫描之上链式组合 Polars 原生表达式（select/filter/join/rename…），
        但**物化**仍只能走 ``ScanHandle.collect()``——那里强制 QueryBudget /
        snapshot / audit，因此这里不构成绕过预算的逃生口。跨包消费方禁止直接
        访问 ``_lf`` 字段。

        #28：production/strict 下即使 composition 也不能放裸 LazyFrame——
        拿到它就能 ``lf.collect()`` / ``lf.sink_parquet()`` 绕过治理。research
        放行（composition 需要），但禁止 sink_* 与直接物化（__getattr__ 拦截）。
        """
        if is_strict_semantics():
            raise ValidationError(
                "production/strict 模式禁止 native_lazyframe()（可绕过 budget/audit "
                "直接物化）；请用 ScanHandle.collect() 受控终点。"
            )
        return self._lf

    def _revalidate_snapshot(self) -> DataSnapshot:
        """#7 collect 前 revalidate：scan() 生成 LazyFrame 后可能很久才 collect，
        底层文件已被覆盖/删除——不能把 scan 时刻的旧 snapshot 当成刚读的数据。

        本地文件逐文件 stat 对比 (size, mtime_ns)：
            - 无变化 → 原 snapshot
            - 有变化且 strict → 抛 ValidationError（fail-closed，要求重新 scan）
            - 有变化且 research → 重建 snapshot 文件版本（lineage 不再撒谎）
        远程对象（s3:// / cos://，#P0 收官 0.9.5）不能只靠 manifest hash 里记录
        的旧 etag：**collect 前必须强制重新 HEAD**（``fresh=True`` 绕过 TTL memo）
        并对比 etag/content_length/version_id——scan(t0) 后对象同 key 被覆盖，
        collect(t1) 会读到新数据，若还声称旧 ETag，训练/回测 lineage 直接失真。
        strict/pin 下变化或无法验证（无凭证/网络/未记录身份）→ fail-closed。
        无 _store（无法 stat）→ 原样返回。
        """
        if self._store is None:
            return self.snapshot
        changed: list[str] = []
        restat: list[FileVersion] = []
        for fv in self.snapshot.files:
            path = str(fv.path)
            if path.startswith("s3://") or path.startswith("cos://"):
                if _remote_snapshot_meta_enabled():
                    meta = _remote_object_meta(path, fresh=True)
                    if meta is None:
                        if is_strict_semantics():
                            changed.append(f"{path} (remote identity unverifiable)")
                            continue
                        restat.append(fv)
                        continue
                    if _remote_identity_changed(fv, meta):
                        changed.append(f"{path} (remote etag/size changed)")
                        continue
                restat.append(fv)
                continue
            from pathlib import Path

            try:
                st = Path(path).stat()
            except OSError:
                changed.append(f"{path} (gone)")
                continue
            if (st.st_size, st.st_mtime_ns) != (fv.size, fv.mtime_ns):
                changed.append(f"{path} (mtime/size changed)")
                continue
            restat.append(fv)
        if not changed:
            return self.snapshot
        if is_strict_semantics():
            raise ValidationError(
                "ScanHandle 绑定 snapshot 已失效：collect 前底层文件发生变化 "
                f"（{len(changed)} 个）。请重新 scan() 绑定最新文件清单，避免 "
                "lineage 与实际读到数据不一致。"
            )
        return rebuild_snapshot_files(self.snapshot, restat)

    def collect(self) -> ReadResult:
        """执行 Polars 计划并返回绑定 snapshot/lineage 的 ``ReadResult``。"""
        import time

        start = time.perf_counter()
        table: pa.Table | None = None
        ok = False
        err_msg: str | None = None
        try:
            snapshot = self._revalidate_snapshot()
            table = collect_polars_with_budget(self._lf, query_budget=self.budget)
            elapsed_ms = (time.perf_counter() - start) * 1000
            stats = ReadStats(
                rows=table.num_rows,
                bytes=table.nbytes,
                elapsed_ms=elapsed_ms,
                paths=tuple(f.path for f in snapshot.files[:20]),
            )
            ok = True
            return ReadResult(
                table=table,
                snapshot=snapshot,
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
                    "instrument_count": len(self.lineage.instrument_filter or ()),
                },
            )

    def collect_table(self) -> pa.Table:
        return self.collect().table

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if is_strict_semantics() and (
            name in _MATERIALIZING_METHODS or name.startswith("sink_")
        ):
            raise ValidationError(
                f"production/strict 模式禁止通过 ScanHandle.{name}() 绕过受控 collect；"
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
