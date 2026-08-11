# -*- coding: utf-8 -*-
"""R39-P0-PERF-017：output warmup trim 下推 —— 零拷贝位置切片。

:class:`OutputSlice` 描述终端结果的 ``[start_offset, end_offset)`` 位置窗口。
native backend（DuckDB / Polars / SQL）在 warmup 结果上直接返回位置视图，取代
``_trim_batch_result`` 里每次因子结果的 Pandas boolean-mask copy/slice。

原则：
- warmup 仍参与计算（语义不变，PIT / NaN / Inf / dtype 不变）；
- 只有输出裁剪从「boolean-mask 重建索引」降级为「位置 ``.iloc`` 视图」（底层
  连续 block 时零拷贝，``np.shares_memory`` 可验证）；
- 结果若已携带 ``_output_slice``（backend 已切好视图），下游 writer 直接消费
  ``(BufferRef, slice)``，不再二次切片。

本模块只提供契约与纯函数；主链接线在 ``runtime.batch_service._trim_batch_result``。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OutputSlice:
    """终端结果的位置切片窗口（半开区间 ``[start_offset, end_offset)``）。

    ``end_offset=None`` 表示切到末尾。``is_full`` 表示未裁剪（可跳过切片）。
    """

    start_offset: int
    end_offset: int | None

    def __post_init__(self) -> None:
        if self.start_offset < 0:
            raise ValueError(f"start_offset must be >= 0, got {self.start_offset}")
        if self.end_offset is not None and self.end_offset < self.start_offset:
            raise ValueError(
                f"end_offset ({self.end_offset}) must be >= start_offset "
                f"({self.start_offset})"
            )

    @property
    def is_full(self) -> bool:
        return self.start_offset == 0 and self.end_offset is None

    @property
    def stop(self) -> int:
        return self.end_offset if self.end_offset is not None else -1

    def to_dict(self) -> dict[str, Any]:
        return {"start_offset": self.start_offset, "end_offset": self.end_offset}


def compute_output_slice(
    result: Any,
    run_window: Any,
    *,
    bars_per_day: int = 1,
) -> OutputSlice | None:
    """计算输出位置切片；无法定位（非 Series / 索引未排序 / 无窗口）返回 ``None``。

    与 :func:`storage.time_window.slice_series_time_window` 的边界语义完全一致
    （start 含 / end 含）：先用同一套 bound 归一化，再在已排序的 timestamp 层上
    ``searchsorted`` 得到位置。索引非单调递增时回退 ``None``（调用方走旧 mask 路径）。
    """
    if run_window is None:
        return None
    if not (getattr(run_window, "trim_output", False) and getattr(run_window, "requested_start", None)):
        return None
    if result is None or not hasattr(result, "index"):
        return None
    try:
        import pandas as pd

        from storage.time_window import _normalize_bound_for_index
    except Exception:  # noqa: BLE001
        return None
    index = result.index
    ts = (
        index.get_level_values(0)
        if isinstance(index, pd.MultiIndex)
        else index
    )
    if not bool(getattr(ts, "is_monotonic_increasing", False)):
        return None
    tz = getattr(ts, "tz", None)
    try:
        start_cmp = _normalize_bound_for_index(
            pd.Timestamp(run_window.requested_start), tz
        )
        end_cmp = (
            _normalize_bound_for_index(pd.Timestamp(run_window.requested_end), tz)
            if run_window.requested_end
            else None
        )
    except Exception:  # noqa: BLE001
        return None
    start_offset = 0
    end_offset: int | None = len(ts)
    if start_cmp is not None:
        start_offset = int(ts.searchsorted(start_cmp, side="left"))
    if end_cmp is not None:
        end_offset = int(ts.searchsorted(end_cmp, side="right"))
    if start_offset == 0 and (end_offset is None or end_offset >= len(ts)):
        return OutputSlice(0, None)
    return OutputSlice(start_offset, end_offset)


def apply_output_slice(result: Any, output_slice: OutputSlice) -> Any:
    """把 :class:`OutputSlice` 应用到结果：位置视图（连续 block 时零拷贝）。

    返回的 Series/DataFrame 与 mask 切片数值、索引、dtype 完全一致；并在结果上
    挂 ``_output_slice`` 属性，供 writer 以 ``(BufferRef, slice)`` 消费。
    """
    if result is None or output_slice is None or output_slice.is_full:
        return result
    stop = output_slice.end_offset if output_slice.end_offset is not None else None
    sliced = result.iloc[output_slice.start_offset:stop]
    _attach_output_slice(sliced, output_slice)
    return sliced


def _attach_output_slice(result: Any, output_slice: OutputSlice) -> None:
    """把 slice 元数据挂到结果（尽力而为，失败不阻塞）。"""
    try:
        object.__setattr__(result, "_output_slice", output_slice)
    except Exception:  # noqa: BLE001
        pass


def carried_output_slice(result: Any) -> OutputSlice | None:
    """读取结果已携带的 ``_output_slice``（native backend 已切好视图时）。"""
    if result is None:
        return None
    value = getattr(result, "_output_slice", None)
    if isinstance(value, OutputSlice):
        return value
    return None
