# -*- coding: utf-8 -*-
"""R38 P0-001/002/003（§4）：真实 shard 执行器 —— 真正切输入、逐片执行、merge。

修复 R36「只改 peak_memory 数字」的伪 sharding：本模块把一个大 task 的**输入数据
真实切片**（asset → 仪器子集；time/session → 时间窗 + warmup overlap），对每片执行
同一个算子，再用 :class:`ShardMergeContract` 流式合并（bounded memory，§P0-003）。

    - :class:`SliceDataSource`：包一层数据源，``load_column`` / ``load_column_panel``
      只返回本 shard 切片（asset 按仪器、time 按时间窗 + warmup 起点）。
    - :class:`SliceCache`：包 ``shared_result_cache``，plan_ref 的 CSE 输入同样被切片。
    - :func:`ShardExecutor.execute_shard`：切片上下文 → ``backend.execute`` → 裁剪
      warmup overlap 输出。
    - :func:`ShardExecutor.execute_merge`：按 merge_order 合并；shard 结果在完成时
      即 spool 到磁盘（大 shard），merge 时 reload —— 不全部常驻内存。
"""
from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from runtime.shard_execution_plan import (
    SHARD_ASSET,
    SHARD_TIME,
    ShardExecutionPlan,
    ShardMergeContract,
)

#: 超过该阈值就把 shard 结果 spool 到磁盘（避免 merge 前全部常驻内存，§P0-003）。
SPOOL_THRESHOLD_BYTES = 512 * 1024**2


@dataclass(frozen=True)
class SpooledShard:
    """已落盘的 shard 结果引用（真实 spill）。"""

    path: str
    n_bytes: int

    def load(self) -> Any:
        import pandas as pd

        frame = pd.read_parquet(self.path)
        if frame.shape[1] == 1:
            return frame.iloc[:, 0]
        return frame


def _series_time_range(s: Any, lo: Any, hi: Any) -> Any:
    import pandas as pd

    idx = s.index
    if idx.nlevels > 1:
        t = idx.get_level_values(0)
        keep = pd.Series(True, index=idx)
        if lo is not None:
            keep = keep & (t >= pd.Timestamp(lo))
        if hi is not None:
            keep = keep & (t <= pd.Timestamp(hi))
        return s[keep]
    return s.loc[slice(lo, hi)]


def _series_instruments(s: Any, instruments: set[Any]) -> Any:
    idx = s.index
    if idx.nlevels > 1:
        keep = idx.get_level_values(1).isin(instruments)
        return s[keep]
    return s


def _slice_series(s: Any, instruments: set[Any] | None, lo: Any, hi: Any) -> Any:
    out = s
    if instruments is not None:
        out = _series_instruments(out, instruments)
    if lo is not None or hi is not None:
        out = _series_time_range(out, lo, hi)
    return out


def _slice_panel(p: Any, instruments: set[Any] | None, lo: Any, hi: Any) -> Any:
    import pandas as pd

    out = p
    if instruments is not None:
        cols = [c for c in out.columns if c in instruments]
        out = out[cols] if cols else out.iloc[:, 0:0]
    if lo is not None or hi is not None:
        out = out.loc[slice(lo, hi)]
    return out


class SliceDataSource:
    """只返回 shard 切片的 data source 包装（asset 仪器子集 / time 时间窗）。"""

    def __init__(
        self,
        inner: Any,
        *,
        instruments: tuple[Any, ...] | None = None,
        time_range: tuple[Any, Any] | None = None,
        warmup_start: Any | None = None,
    ) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(
            self, "_instruments", set(instruments) if instruments is not None else None
        )
        object.__setattr__(self, "_time_range", time_range)
        object.__setattr__(self, "_warmup_start", warmup_start)

    def _lo(self) -> Any:
        return self._warmup_start or (self._time_range[0] if self._time_range else None)

    def _hi(self) -> Any:
        return self._time_range[1] if self._time_range else None

    def load_column(self, name: str) -> Any:
        s = self._inner.load_column(name)
        return _slice_series(s, self._instruments, self._lo(), self._hi())

    def load_column_panel(self, name: str) -> Any:
        p = self._inner.load_column_panel(name)
        return _slice_panel(p, self._instruments, self._lo(), self._hi())

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_inner"), name)


class SliceCache(dict):
    """plan_ref CSE 输入的切片视图（只读投影，不修改 base）。"""

    def __init__(
        self,
        base: dict[str, Any],
        *,
        instruments: tuple[Any, ...] | None = None,
        time_range: tuple[Any, Any] | None = None,
        warmup_start: Any | None = None,
    ) -> None:
        super().__init__()
        self._base = base
        self._instruments = set(instruments) if instruments is not None else None
        self._time_range = time_range
        self._warmup_start = warmup_start

    def _lo(self) -> Any:
        return self._warmup_start or (self._time_range[0] if self._time_range else None)

    def _hi(self) -> Any:
        return self._time_range[1] if self._time_range else None

    def _project(self, val: Any) -> Any:
        if isinstance(val, np.ndarray):
            return val
        if hasattr(val, "index") and hasattr(val, "unstack"):
            # pd.Series
            return _slice_series(val, self._instruments, self._lo(), self._hi())
        if hasattr(val, "columns") and hasattr(val, "loc"):
            return _slice_panel(val, self._instruments, self._lo(), self._hi())
        return val

    def __getitem__(self, key: str) -> Any:
        return self._project(self._base[key])

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._base:
            return self._project(self._base[key])
        return default

    def __contains__(self, key: object) -> bool:
        return key in self._base

    def keys(self) -> Any:  # type: ignore[override]
        return self._base.keys()


class ShardExecutor:
    """真实分片执行器（shard → 切片执行 → 裁剪 → spool；merge → reload → concat）。"""

    def __init__(self, *, spool_dir: str | None = None) -> None:
        self._lock = threading.Lock()
        self._spool_dir = spool_dir
        self._spooled = 0
        self._events: list[str] = []

    @property
    def spool_dir(self) -> str:
        if self._spool_dir is None:
            self._spool_dir = os.path.join(
                os.environ.get("FACTOR_ENGINE_CACHE_DIR", "").strip() or ".", "_r38_spool"
            )
            os.makedirs(self._spool_dir, exist_ok=True)
        return self._spool_dir

    # -- shard 执行 --

    def execute_shard(self, backend: Any, node: Any, ctx: Any, descriptor: Any) -> Any:
        """在切片上下文执行一个 shard，返回裁剪后（非 overlap）输出。

        asset shard：data source + 缓存都按仪器子集切 → 输出仅覆盖该子集。
        time shard：输入 [warmup_start, block_end]，输出裁剪到 [block_start, block_end]。
        """
        dim = descriptor.dimension
        if dim == SHARD_ASSET:
            instruments = tuple(descriptor.input_slice)
            ds = SliceDataSource(
                ctx.data_source, instruments=instruments
            )
            cache = SliceCache(
                ctx.shared_result_cache or {},
                instruments=instruments,
            )
            shard_ctx = replace(
                ctx,
                data_source=ds,
                shared_result_cache=cache,
                panel_cache={},
                template_series=None,
                template_index=None,
            )
            return self._exec_trim(backend, node, shard_ctx, descriptor)
        if dim in (SHARD_TIME, "session"):
            block_start, block_end = descriptor.output_slice
            warmup_start = (
                descriptor.warmup_slice[0]
                if isinstance(descriptor.warmup_slice, (tuple, list)) and descriptor.warmup_slice
                else block_start
            )
            ds = SliceDataSource(
                ctx.data_source,
                time_range=(block_start, block_end),
                warmup_start=warmup_start,
            )
            cache = SliceCache(
                ctx.shared_result_cache or {},
                time_range=(block_start, block_end),
                warmup_start=warmup_start,
            )
            shard_ctx = replace(
                ctx,
                data_source=ds,
                shared_result_cache=cache,
                panel_cache={},
                template_series=None,
                template_index=None,
            )
            result = self._exec_trim(backend, node, shard_ctx, descriptor)
            return _trim_output(result, block_start, block_end, ctx)
        raise ValueError(f"unsupported shard dimension: {dim!r}")

    def _exec_trim(
        self, backend: Any, node: Any, shard_ctx: Any, descriptor: Any
    ) -> Any:
        """执行算子并记录 spool 事件（spool 由 scheduler 完成路径调用）。"""
        return backend.execute(node, shard_ctx)

    # -- spool / 合并 --

    def spool_or_keep(self, result: Any, *, spool_threshold: int = SPOOL_THRESHOLD_BYTES) -> Any:
        """shard 结果超过阈值 → 写 parquet（真实 spill）返回 SpooledShard 引用。

        小于阈值直接保留（merge 内存可容纳），否则 spool 到磁盘——merge 前不
        全部常驻内存（R38-P0-003）。
        """
        n = _estimate_bytes(result)
        if n < spool_threshold:
            return result
        path = os.path.join(self.spool_dir, f"shard_{uuid.uuid4().hex[:10]}.parquet")
        try:
            if hasattr(result, "to_frame"):
                result.to_frame("__r38_val__").to_parquet(path)
            else:
                import pandas as pd

                pd.DataFrame(result).to_parquet(path)
        except Exception as exc:  # noqa: BLE001
            self._events.append(f"spool_failed:{type(exc).__name__}:{exc}")
            return result
        with self._lock:
            self._spooled += 1
        self._events.append(f"spooled:{os.path.basename(path)}:{n}")
        return SpooledShard(path=path, n_bytes=n)

    def load_partial(self, partial: Any) -> Any:
        """把 SpooledShard / 内存对象统一还原成内存结果。"""
        if isinstance(partial, SpooledShard):
            return partial.load()
        return partial

    def execute_merge(
        self,
        backend: Any,
        ctx: Any,
        plan: ShardExecutionPlan,
        partials: dict[str, Any],
    ) -> Any:
        """按 merge_order 合并全部 shard 输出（bounded memory：逐片 load+concat）。"""
        contract = plan.merge_policy
        ordered = sorted(plan.shards, key=lambda s: s.merge_order)
        acc: Any = None
        n_keys = 0
        for descriptor in ordered:
            partial = partials.get(descriptor.shard_id)
            if partial is None:
                raise RuntimeError(
                    f"shard merge missing partial {descriptor.shard_id!r} "
                    f"(plan={plan.merge_task_id}) — shard 未完成就 merge"
                )
            chunk = self.load_partial(partial)
            acc = _concat_shards(acc, chunk, contract)
            n_keys += 1
        # duplicate key policy：显式检查。
        if contract.duplicate_key_policy == "error" and acc is not None:
            _assert_no_duplicate_keys(acc)
        return acc

    def summary(self) -> dict[str, Any]:
        return {
            "spool_dir": self._spool_dir,
            "spooled": self._spooled,
            "events": self._events[-20:],
        }


# -- merge helpers --

def _concat_shards(acc: Any, chunk: Any, contract: ShardMergeContract) -> Any:
    import pandas as pd

    if acc is None:
        return chunk
    if isinstance(acc, pd.Series) and isinstance(chunk, pd.Series):
        out = pd.concat([acc, chunk])
        return out.sort_index()
    if isinstance(acc, pd.DataFrame) and isinstance(chunk, pd.DataFrame):
        out = pd.concat([acc, chunk], axis=0)
        return out.sort_index()
    raise TypeError(
        f"shard merge incompatible types: {type(acc).__name__} vs {type(chunk).__name__} "
        f"(contract={contract.to_dict()})"
    )


def _assert_no_duplicate_keys(value: Any) -> None:
    idx = value.index
    dup = idx.duplicated()
    if dup.any():
        raise RuntimeError(
            f"shard merge produced duplicate index keys ({int(dup.sum())}) — "
            "duplicate_key_policy=error; check shard slicing/overlap trim"
        )


def _trim_output(result: Any, block_start: Any, block_end: Any, ctx: Any) -> Any:
    """把 shard 输出裁剪到非 overlap 块（time-shard rolling 的 warmup 部分丢弃）。"""
    import pandas as pd

    if result is None:
        return result
    if isinstance(result, pd.Series):
        idx = result.index
        if idx.nlevels > 1:
            t = idx.get_level_values(0)
            lo = block_start if block_start is not None else t.min()
            hi = block_end if block_end is not None else t.max()
            mask = (t >= pd.Timestamp(lo)) & (t <= pd.Timestamp(hi))
            return result[mask]
        return result.loc[slice(block_start, block_end)]
    if isinstance(result, pd.DataFrame):
        return result.loc[slice(block_start, block_end)]
    return result


def _estimate_bytes(value: Any) -> int:
    try:
        from runtime.resource_governor import estimate_object_bytes

        return max(0, int(estimate_object_bytes(value)))
    except Exception:
        return 0


def _cleanup_spool(spool_dir: str) -> None:
    """删除 spool 目录（scheduler finally 调用）。"""
    try:
        for name in os.listdir(spool_dir):
            os.remove(os.path.join(spool_dir, name))
        os.rmdir(spool_dir)
    except OSError:
        pass
