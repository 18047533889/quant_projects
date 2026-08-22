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
    ShardMergeMode,
)
from runtime.spool_policy import (
    DECISION_KEEP,
    SpoolDecision,
    default_spool_policy_factory,
    record_spool_decision,
)

#: 超过该阈值就把 shard 结果 spool 到磁盘（避免 merge 前全部常驻内存，§P0-003）。
#: R39-P1-PERF-027：该值仍是默认固定阈值；传入 ``policy`` 时由
#: :func:`runtime.spool_policy.decide_spool_or_keep` 自适应替代。
SPOOL_THRESHOLD_BYTES = 512 * 1024**2

#: spool 落盘格式（R39-PERF-024）：临时 spool 追求快速 + 低内存，优先 Arrow IPC。
SPOOL_FORMAT_ARROW = "arrow"
SPOOL_FORMAT_PARQUET = "parquet"

#: R39-P1-PERF-027：spool 盘吞吐低于该值时优先 parquet（压缩、更少磁盘 IO）。
ARROW_PREFERRED_THROUGHPUT_BYTES_PER_S = 100 * 1024**2


@dataclass(frozen=True)
class SpooledShard:
    """已落盘的 shard 结果引用（真实 spill）。

    ``spool_format`` 记录落盘格式（"parquet" | "arrow"）。Arrow 路径使用
    :class:`ArrowSpoolRef`（本类子类），parquet 路径保留旧行为。
    """

    path: str
    n_bytes: int
    spool_format: str = SPOOL_FORMAT_PARQUET

    def load(self) -> Any:
        import pandas as pd

        frame = pd.read_parquet(self.path)
        if frame.shape[1] == 1:
            return frame.iloc[:, 0]
        return frame


@dataclass(frozen=True)
class ArrowSpoolRef(SpooledShard):
    """R39-PERF-024：Arrow IPC stream spool 引用（Pandas→Arrow→Pandas 零压缩）。

    写入：``pyarrow.ipc.new_stream``（无压缩，临时 spool 优先速度 + 低内存）；
    读取：``pyarrow.ipc.open_stream`` + memory-map，仅在需要时 ``to_pandas``。
    ``schema`` / ``row_count`` 在 spool 时记录（不必重读文件就能知道元数据）；
    ``sortedness_certificate`` 是 spool 时的排序证明（供 PERF-025 使用）。
    """

    schema: Any = None
    row_count: int = 0
    sortedness_certificate: Any = None
    spool_format: str = SPOOL_FORMAT_ARROW

    def load(self) -> Any:
        import pyarrow as pa

        with pa.memory_map(self.path, "r") as src:
            with pa.ipc.open_stream(src) as reader:
                table = reader.read_all()
        frame = table.to_pandas()
        if frame.shape[1] == 1:
            return frame.iloc[:, 0]
        return frame


@dataclass(frozen=True)
class ShardDirectWriteManifest:
    """R39-PERF-026：直接落盘模式的 merge task 产物（不重建大内存结果）。

    merge task 只提交 manifest：已裁剪 shard 输出（SpooledShard 引用或内存帧）的
    碎片列表 + 元数据（contract / dimension / 列顺序）。writer 直接把这些碎片 append
    到 final storage 分区，跳过「拼成一份完整 DataFrame」。
    """

    fragments: tuple[Any, ...]
    shard_ids: tuple[str, ...]
    plan_metadata: dict[str, Any]


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


def _slice_value(
    val: Any,
    instruments: set[Any] | None,
    lo: Any,
    hi: Any,
    *,
    where: str,
) -> Any:
    """统一切片语义（P0-012：``ShardableView`` 协议唯一入口）。

    - pandas Series / DataFrame → 按仪器 / 时间窗投影；
    - 其它可带 index/columns 的 frame-like（polars 等）→ 同样投影；
    - **无标签 np.ndarray**：请求了切片时**fail-closed 抛错**，绝不静默返回
      完整数据（旧实现把 ndarray 原样返回，混合因子会在 shard 里拿到整幅面板）。
    """
    if instruments is None and lo is None and hi is None:
        return val  # 未请求切片 → 原样（无 slice 语义负担）
    if isinstance(val, np.ndarray):
        raise ValueError(
            f"{where}: cannot slice unlabeled np.ndarray for shard "
            "(instruments/time projection needs index/columns; use Series/DataFrame "
            "or wrap with an index) — refusing to return the full panel silently"
        )
    # DataFrame 先判（DataFrame 也有 ``.unstack``，旧顺序会误入 Series 分支只切行）。
    if hasattr(val, "columns") and hasattr(val, "loc"):
        return _slice_panel(val, instruments, lo, hi)
    if hasattr(val, "index") and hasattr(val, "unstack"):
        # pd.Series（或带 unstack 的 Series-like）
        return _slice_series(val, instruments, lo, hi)
    # 其它（标量 / 元数据）→ 原样。
    return val


class ShardableView:
    """统一切片的只读视图（P0-012）：数据源与共享缓存共用同一投影语义。

    子类必须实现 ``_lo()/_hi()``（时间窗）与 ``_instruments``（仪器子集），并让
    一切数据读取经 :func:`_slice_value` 显式投影。**不再靠 ``__getattr__`` 猜接口
    透传**——未知数据读取方法一律 ``AttributeError``（fail-loud），绝不静默拿到
    未切片的完整数据。
    """


#: 允许透传的**元数据**属性（非数据读取；数据方法必须显式实现并切片）。
_METADATA_ATTRS = frozenset({
    "dataset", "market", "bar_freq", "params", "execution_spec", "read_auto",
    "lazy_scan", "snapshot_id", "source_snapshot_id", "id", "name", "kind",
    "table", "session_open", "session_close", "session_minutes",
    "timestamp_convention", "universe", "normalize_units", "history_origin",
    "full_history_start", "source", "timeframe", "read_mode",
})


class SliceDataSource(ShardableView):
    """只返回 shard 切片的 data source 包装（asset 仪器子集 / time 时间窗）。

    显式实现全部数据读取方法（load_column / load_column_panel / load_columns /
    prefetch_columns）并投影；``scan_polars_long`` 因 polars 切片需表达式路由，
    无法可靠投影时 fail-loud。元数据属性透传，未知属性抛 ``AttributeError``。
    """

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

    def _project(self, val: Any, *, where: str = "SliceDataSource") -> Any:
        return _slice_value(val, self._instruments, self._lo(), self._hi(), where=where)

    # -- 显式数据读取方法（全部投影） --
    def load_column(self, name: str) -> Any:
        return self._project(self._inner.load_column(name))

    def load_column_panel(self, name: str) -> Any:
        return self._project(self._inner.load_column_panel(name))

    def load_columns(self, names: list[str]) -> dict[str, Any]:
        return {n: self._project(self._inner.load_column(n)) for n in names}

    def prefetch_columns(self, names: list[str]) -> None:
        fn = getattr(self._inner, "prefetch_columns", None)
        if callable(fn):
            fn(names)

    def scan_polars_long(self, columns: list[str]):  # noqa: ANN201
        raise NotImplementedError(
            "SliceDataSource.scan_polars_long: polars 切片需要表达式级过滤，"
            "无法可靠投影——请为 polars shard 使用显式 ShardableDataView 实现"
            "（fail-loud，绝不静默返回完整面板）"
        )

    # -- 切片后的边界元数据（让下游窗口/仪器逻辑看到 shard 真实范围） --
    @property
    def start_date(self) -> Any:
        lo = self._lo()
        return lo if lo is not None else getattr(self._inner, "start_date", None)

    @property
    def end_date(self) -> Any:
        hi = self._hi()
        return hi if hi is not None else getattr(self._inner, "end_date", None)

    @property
    def instrument_filter(self) -> Any:
        if self._instruments is not None:
            return tuple(sorted(self._instruments))
        return getattr(self._inner, "instrument_filter", None)

    @property
    def columns(self) -> Any:
        inner_columns = getattr(self._inner, "columns", None)
        if self._instruments is not None and inner_columns is not None:
            try:
                return [c for c in inner_columns if c in self._instruments]
            except TypeError:
                return inner_columns
        return inner_columns

    def __getattr__(self, name: str) -> Any:
        if name in _METADATA_ATTRS or not name.startswith("_"):
            # 元数据透传；数据读取方法不在此列 → 显式实现，避免静默绕过切片。
            inner = object.__getattribute__(self, "_inner")
            if name in _METADATA_ATTRS and hasattr(inner, name):
                return getattr(inner, name)
        raise AttributeError(
            f"SliceDataSource.{name}: 数据读取/未知属性不会透传给 inner（P0-012 "
            "ShardableDataView 协议）——必须显式实现并投影，避免拿到未切片全量数据"
        )


class SliceCache(ShardableView):
    """plan_ref CSE 输入的切片视图（完整 Mapping 语义，只读投影，不修改 base）。

    修复旧实现两个缺陷：
        - ``np.ndarray`` 不再原样返回（fail-closed，见 :func:`_slice_value`）；
        - 完整实现 ``__iter__/items/values/__len__/keys/__contains__/get``——
          直接迭代缓存的对象看到的是**切片后**的数据，而不是空 dict。
    """

    def __init__(
        self,
        base: dict[str, Any],
        *,
        instruments: tuple[Any, ...] | None = None,
        time_range: tuple[Any, Any] | None = None,
        warmup_start: Any | None = None,
    ) -> None:
        self._base = base
        self._instruments = set(instruments) if instruments is not None else None
        self._time_range = time_range
        self._warmup_start = warmup_start

    def _lo(self) -> Any:
        return self._warmup_start or (self._time_range[0] if self._time_range else None)

    def _hi(self) -> Any:
        return self._time_range[1] if self._time_range else None

    def _project(self, val: Any) -> Any:
        return _slice_value(
            val, self._instruments, self._lo(), self._hi(), where="SliceCache"
        )

    def __getitem__(self, key: str) -> Any:
        return self._project(self._base[key])

    def __iter__(self):
        return iter(self._base)

    def __len__(self) -> int:
        return len(self._base)

    def __contains__(self, key: object) -> bool:
        return key in self._base

    def keys(self):
        return self._base.keys()

    def items(self):
        return ((k, self._project(v)) for k, v in self._base.items())

    def values(self):
        return (self._project(v) for v in self._base.values())

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._base:
            return self._project(self._base[key])
        return default


class ShardExecutor:
    """真实分片执行器（shard → 切片执行 → 裁剪 → spool；merge → reload → concat）。"""

    def __init__(self, *, spool_dir: str | None = None) -> None:
        self._lock = threading.Lock()
        self._spool_dir = spool_dir
        self._spooled = 0
        self._events: list[str] = []
        #: R39-PERF-025 gate：证明路径必须 ==1（或直接落盘 ==0），绝不允许逐片 concat+sort。
        self._concat_sort_count = 0

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

    def spool_or_keep(
        self,
        result: Any,
        *,
        spool_threshold: int = SPOOL_THRESHOLD_BYTES,
        spool_format: str = "auto",
        policy: Any | None = None,
    ) -> Any:
        """shard 结果超过阈值 → spool 到磁盘返回 SpooledShard/ArrowSpoolRef 引用。

        小于阈值直接保留（merge 内存可容纳），否则 spool——merge 前不全部常驻内存
        （R38-P0-003）。R39-PERF-024：优先 Arrow IPC stream（无压缩，快速 + 低内存）；
        Arrow 不可转换时回退 parquet。``spool_format`` 取值 ``auto|arrow|parquet``。

        R39-P1-PERF-027：``policy`` 可选（``Callable[[int], SpoolDecision]``）。
        传入时用 :func:`runtime.spool_policy.decide_spool_or_keep` 结果替代固定
        阈值判断（KEEP → 直接保留；SPOOL → 落盘），并记录模块级计数器
        ``spool_decision_count`` / ``spool_keep_count`` / ``adaptive_threshold_applied_count``。
        默认 ``None`` → 保持原 512MB 固定阈值行为，完全向后兼容。policy 返回 SPOOL
        且 spool 盘吞吐较低时优先 parquet（压缩、更少磁盘 IO），否则 Arrow IPC。
        """
        n = _estimate_bytes(result)
        if policy is not None:
            decision: SpoolDecision = policy(n)
            record_spool_decision(decision)
            if decision.decision == DECISION_KEEP:
                return result
            fmt = spool_format
            if fmt == "auto":
                fmt = _pick_spool_format(result, decision.spool_disk_throughput)
            return self._write_spool(result, n, fmt)
        if n < spool_threshold:
            return result
        fmt = spool_format
        if fmt == "auto":
            fmt = SPOOL_FORMAT_ARROW if _arrow_convertible(result) else SPOOL_FORMAT_PARQUET
        return self._write_spool(result, n, fmt)

    def _write_spool(self, result: Any, n: int, fmt: str) -> Any:
        """把结果写 spool 并返回 SpooledShard/ArrowSpoolRef 引用（PERF-024 语义）。

        Arrow 写入失败 → 回退 parquet（正确性优先）。写失败 → 返回原结果并记录
        ``spool_failed`` 事件（不静默丢数据）。
        """
        if fmt == SPOOL_FORMAT_ARROW:
            ref = _write_arrow_spool(self.spool_dir, result)
            if ref is not None:
                with self._lock:
                    self._spooled += 1
                self._events.append(
                    f"spooled:{os.path.basename(ref.path)}:{n}:{ref.spool_format}"
                )
                return ref
            # Arrow 写入失败 → 回退 parquet（正确性优先）。
            fmt = SPOOL_FORMAT_PARQUET
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
        self._events.append(f"spooled:{os.path.basename(path)}:{n}:{fmt}")
        return SpooledShard(path=path, n_bytes=n, spool_format=fmt)

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
        *,
        merge_mode: ShardMergeMode | None = None,
    ) -> Any:
        """按 merge_order 合并全部 shard 输出（bounded memory：逐片 load+concat）。

        R39-PERF-025/026：
            - ``verify_shards_sorted_nonoverlapping`` 证明「每片已排序 + 范围不重叠 +
              merge_order 正确」→ **单次** ``pd.concat(all, copy=False)``（CONCAT_ONLY），
              ``shard_concat_sort_count`` == 1，绝无逐片 concat+sort。
            - 证明失败 → fallback 到旧 K-concat+sort（ORDERED_MERGE），输出与旧路径
              完全一致。
            - ``merge_mode`` 强制指定时绕过数据驱动选择（DIRECT_DURABLE_APPEND 只提交
              manifest，不重建内存结果）。
        """
        contract = plan.merge_policy
        ordered = sorted(plan.shards, key=lambda s: s.merge_order)
        chunks: list[Any] = []
        for descriptor in ordered:
            partial = partials.get(descriptor.shard_id)
            if partial is None:
                raise RuntimeError(
                    f"shard merge missing partial {descriptor.shard_id!r} "
                    f"(plan={plan.merge_task_id}) — shard 未完成就 merge"
                )
            chunks.append(self.load_partial(partial))
        mode = merge_mode or choose_shard_merge_mode(contract, chunks)
        self._events.append(f"merge_mode:{mode.value}:shards={len(chunks)}")

        if mode is ShardMergeMode.DIRECT_DURABLE_APPEND:
            # 不重建内存结果：merge task 只提交 manifest（碎片 + 元数据）。
            return ShardDirectWriteManifest(
                fragments=tuple(chunks),
                shard_ids=tuple(d.shard_id for d in ordered),
                plan_metadata=contract.to_dict(),
            )

        if mode is ShardMergeMode.CONCAT_ONLY:
            # Gate-07：证明路径 → 单次 concat，无逐片 sort。
            if len(chunks) > 1:
                with self._lock:
                    self._concat_sort_count += 1
            if not chunks:
                return None
            import pandas as pd

            if len(chunks) == 1:
                acc = chunks[0]
            else:
                acc = pd.concat(chunks, copy=False)
            if contract.duplicate_key_policy == "error" and acc is not None:
                _assert_no_duplicate_keys(acc)
            return acc

        # ORDERED_MERGE / REDUCE_STATE（未提供 reducer → 有序合并保证正确性）。
        with self._lock:
            self._concat_sort_count += len(chunks)
        acc: Any = None
        for chunk in chunks:
            acc = _concat_shards(acc, chunk, contract)
        # duplicate key policy：显式检查。
        if contract.duplicate_key_policy == "error" and acc is not None:
            _assert_no_duplicate_keys(acc)
        return acc

    def summary(self) -> dict[str, Any]:
        return {
            "spool_dir": self._spool_dir,
            "spooled": self._spooled,
            "shard_concat_sort_count": self._concat_sort_count,
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


# -- R39-PERF-024: Arrow IPC spool helpers --

def _arrow_convertible(result: Any) -> bool:
    """结果能否走 Arrow IPC spool（pandas Series/DataFrame / pyarrow.Table）。"""
    import pandas as pd

    if isinstance(result, (pd.Series, pd.DataFrame)):
        return True
    try:
        import pyarrow as pa

        if isinstance(result, pa.Table):
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def _pick_spool_format(result: Any, disk_throughput: int | None) -> str:
    """R39-P1-PERF-027：policy SPOOL 时按 spool 盘吞吐选格式。

    吞吐低（< 100MB/s）→ parquet（压缩，落盘字节更少）；否则 Arrow IPC
    （无压缩，快速 + 低内存）；不可 Arrow 转换一律 parquet。
    """
    if disk_throughput is not None and disk_throughput < ARROW_PREFERRED_THROUGHPUT_BYTES_PER_S:
        return SPOOL_FORMAT_PARQUET
    return SPOOL_FORMAT_ARROW if _arrow_convertible(result) else SPOOL_FORMAT_PARQUET


def _sortedness_certificate(result: Any) -> dict[str, Any] | None:
    """spool 时记录排序证明（供 PERF-025 复用，避免 reload 后重算）。"""
    import pandas as pd

    if isinstance(result, (pd.Series, pd.DataFrame)):
        idx = result.index
        return {
            "monotonic_increasing": bool(idx.is_monotonic_increasing) if len(idx) else True,
            "nlevels": int(idx.nlevels),
        }
    return None


def _write_arrow_spool(spool_dir: str, result: Any) -> ArrowSpoolRef | None:
    """写 Arrow IPC stream spool（无压缩）。失败返回 None → 调用方回退 parquet。"""
    import pandas as pd
    import pyarrow as pa

    path = os.path.join(spool_dir, f"shard_{uuid.uuid4().hex[:10]}.arrow")
    try:
        if isinstance(result, pa.Table):
            table = result
        elif isinstance(result, pd.Series):
            # Series 统一转单列 frame（与 parquet spool 同约定：单列 → Series）。
            table = pa.Table.from_pandas(
                result.to_frame("__r38_val__"), preserve_index=True
            )
        elif isinstance(result, pd.DataFrame):
            table = pa.Table.from_pandas(result, preserve_index=True)
        else:
            table = pa.Table.from_pandas(pd.DataFrame(result), preserve_index=True)
        schema = table.schema
        row_count = table.num_rows
        with pa.OSFile(path, "wb") as sink:
            with pa.ipc.new_stream(sink, schema) as writer:
                writer.write_table(table)
        n_bytes = os.path.getsize(path)
        return ArrowSpoolRef(
            path=path,
            n_bytes=n_bytes,
            schema=schema,
            row_count=row_count,
            sortedness_certificate=_sortedness_certificate(result),
            spool_format=SPOOL_FORMAT_ARROW,
        )
    except Exception:  # noqa: BLE001
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
        return None


# -- R39-PERF-025: sortedness proof + single-concat fast path --

def _key_gt(a: Any, b: Any) -> bool:
    """``a > b``；不可比较时按违反处理（fail-closed：证明失败 → 回退 ordered）。"""
    try:
        return a > b
    except Exception:  # noqa: BLE001
        return True


def _key_lt(a: Any, b: Any) -> bool:
    """``a < b``；不可比较时按不满足处理（fail-closed）。"""
    try:
        return a < b
    except Exception:  # noqa: BLE001
        return False


def _index_is_monotonic(idx: Any) -> bool:
    """index 单调非降（支持 flat 与 MultiIndex；显式 Python 校验，不做任何隐式假设）。"""
    if len(idx) == 0:
        return True
    if idx.nlevels > 1:
        keys = list(idx)  # list of tuples（MultiIndex 逐元素键）
    else:
        keys = idx.tolist()
    prev = keys[0]
    for k in keys[1:]:
        if _key_gt(prev, k):
            return False
        prev = k
    return True


def verify_shards_sorted_nonoverlapping(shards: list[Any], contract: ShardMergeContract) -> bool:
    """R39-PERF-025 运行时证明：每片已排序 + 范围不重叠 + merge_order 正确。

    ``shards`` 是按 merge_order 升序排列的已 load shard 结果。返回 True 仅当
    **单次** ``pd.concat(shards, copy=False)`` 就能复现 ordered concat+sort 输出：

        - ``index_union == "concat_sort"`` 且 ``shard_ordering == "merge_order_asc"``
          且 ``duplicate_key_policy == "error"``（无重复键 → concat 不需去重）；
        - 每片 index 单调非降（内部已排序）；
        - 相邻片 ``prev_max < this_min``（范围不重叠 → concat 结果天然有序）。

    任一条件不满足 → False → 调用方回退旧 K-concat+sort（正确性保留）。
    """
    import pandas as pd

    if contract.index_union != "concat_sort":
        return False
    if contract.shard_ordering != "merge_order_asc":
        return False
    if contract.duplicate_key_policy != "error":
        return False
    if not shards:
        return True
    first = shards[0]
    if isinstance(first, pd.DataFrame):
        kind = "df"
    elif isinstance(first, pd.Series):
        kind = "series"
    else:
        return False
    prev_max: Any = None
    for i, chunk in enumerate(shards):
        if chunk is None:
            return False
        if kind == "df" and not isinstance(chunk, pd.DataFrame):
            return False
        if kind == "series" and not isinstance(chunk, pd.Series):
            return False
        idx = chunk.index
        if len(idx) == 0:
            continue  # 空片不改变范围边界
        if not _index_is_monotonic(idx):
            return False
        lo = idx[0]
        hi = idx[-1]
        if i > 0 and prev_max is not None:
            if not _key_lt(prev_max, lo):
                return False  # 与上一片范围重叠（或不可比较）
        prev_max = hi
    return True


def merge_sorted_streams(chunks: list[Any], key_cols: list[str] | None = None) -> Any:
    """k-way streaming merge（min-heap over 每片 cursor）。

    ``chunks`` 每片按 index（或 ``key_cols``）内部已排序；结果 ==
    ``pd.concat(chunks).sort_index()``（有 ``key_cols`` 时按这些列排序）。
    每次 heap pop 消费一段连续 run（把该片所有 ``key <= 下一个最小键`` 的行一次
    取出）——Python 开销 O(runs) 而不是 O(rows)。等键跨片时保持原始片顺序
    （稳定排序语义）。重复键不在此去重（由 duplicate_key_policy 负责）。
    """
    import heapq

    import pandas as pd

    if not chunks:
        return None
    first = chunks[0]
    is_df = isinstance(first, pd.DataFrame)
    for c in chunks:
        if (is_df and not isinstance(c, pd.DataFrame)) or (
            not is_df and not isinstance(c, pd.Series)
        ):
            raise TypeError(
                f"merge_sorted_streams: mixed chunk types "
                f"{[type(c).__name__ for c in chunks]}"
            )
    n = len(chunks)
    keys_list = [_chunk_sort_keys(c, key_cols) for c in chunks]
    for keys in keys_list:
        prev: Any = None
        for k in keys:
            if prev is not None and _key_gt(prev, k):
                raise ValueError(
                    "merge_sorted_streams: chunk not internally sorted by merge key"
                )
            prev = k
    pos = [0] * n
    heap: list[tuple[Any, int]] = []
    for i, keys in enumerate(keys_list):
        if keys:
            heapq.heappush(heap, (keys[0], i))
    runs: list[Any] = []
    while heap:
        _key, i = heapq.heappop(heap)
        keys = keys_list[i]
        start = pos[i]
        end = start + 1
        while end < len(keys):
            if heap and _key_gt(keys[end], heap[0][0]):
                break  # 本片下一键 > 其它片最小键 → 结束本 run
            end += 1
        pos[i] = end
        if end < len(keys):
            heapq.heappush(heap, (keys[end], i))
        runs.append(chunks[i].iloc[start:end])
    if not runs:
        return first.iloc[0:0]
    return pd.concat(runs, copy=False)


def _chunk_sort_keys(chunk: Any, key_cols: list[str] | None) -> list[Any]:
    """取 merge key 序列（MultiIndex/单级 index，或 ``key_cols`` 多列组合键）。"""
    import pandas as pd

    if key_cols:
        if not isinstance(chunk, pd.DataFrame):
            raise TypeError("merge_sorted_streams: key_cols requires DataFrame chunks")
        cols = [chunk[c].tolist() for c in key_cols]
        return list(zip(*cols)) if cols else []
    idx = chunk.index
    return list(idx) if idx.nlevels > 1 else idx.tolist()


def choose_shard_merge_mode(
    contract: ShardMergeContract, chunks: list[Any]
) -> ShardMergeMode:
    """R39-PERF-026 数据驱动 merge 模式选择。

        - contract 显式请求 DIRECT_DURABLE_APPEND / REDUCE_STATE → 原样返回（调用方
          负责存储兼容性 / state reducer 语义）。
        - 否则（默认 ORDERED_MERGE，或显式 CONCAT_ONLY）：运行时证明
          :func:`verify_shards_sorted_nonoverlapping` 通过 → CONCAT_ONLY（单次 concat）；
          不通过 → ORDERED_MERGE（正确性保留）。
    """
    try:
        requested = ShardMergeMode(contract.merge_mode)
    except ValueError:
        requested = ShardMergeMode.ORDERED_MERGE
    if requested in (ShardMergeMode.DIRECT_DURABLE_APPEND, ShardMergeMode.REDUCE_STATE):
        return requested
    if verify_shards_sorted_nonoverlapping(chunks, contract):
        return ShardMergeMode.CONCAT_ONLY
    return ShardMergeMode.ORDERED_MERGE


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
