# -*- coding: utf-8 -*-
"""R39-P0-PERF-030：FactorBlockRef —— 多个因子共享一份 axis。

规格::

    @dataclass(frozen=True)
    class FactorBlockRef:
        axis_ref: AxisBufferRef
        factor_ids: tuple[str, ...]
        values_ref: BufferRef      # rows × factors
        dtype: str
        row_count: int
        factor_count: int
        validity_ref: BufferRef | None

收益（规格原文）：不重复 MultiIndex；不重复 ticker string；writer queue 少 Python
object；多因子一次 Arrow/Parquet write；matrix 天然接收。

实现要点：
  * ``AxisBufferRef`` 在 ``runtime/buffer_ref.py`` 中**不存在**（该模块只有
    ``BufferRef`` / ``SourceBufferRef``），因此在此定义极简形态：``axis_key`` 是
    axis 的稳定 hash，``index`` 是共享的 pandas Index/MultiIndex 对象引用（不复制）。
    ``values_ref`` / ``validity_ref`` 复用 ``runtime.buffer_ref.BufferRef``。
  * ``build_factor_block`` 是 producer：单次 index 引用（不复制）、values 单 block
    numpy 数组（rows×factors，列顺序与 factor_ids 对齐）、可空 validity（NaN/Inf
    掩码，全有效为 None）。
  * ``share_axis_across`` / ``group_by_shared_axis`` 判定/分组共享 axis（证明 axis
    不重复；同 index 只记一次）。
  * ``to_arrow_table`` / ``to_parquet`` 整 block 一次 Arrow/Parquet 写（供 batch
    writer / matrix 消费）。

接线（真路径）：
  * ``storage/materialize/factor_matrix_materializer.py`` 的 block 写路径在
    column-stack 完成后构造 ``FactorBlockRef`` 并 ``record_block_ref_used``；
  * ``runtime/materialize_batch.py`` 的 batch writer 对共享同一 MultiIndex 的多个
    item 构造 ``FactorBlockRef`` 并记录 ``factor_block_shared_axis_count``。
    两者都是额外的观测/载体，**不改变逐因子写入文件**。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from runtime.buffer_ref import BufferRef

# ---- R39-PERF-030 计数器（模块级） ------------------------------------------
factor_block_ref_built = 0            # build_factor_block 构造次数
factor_block_ref_used = 0             # matrix block 写路径真实使用的 FactorBlockRef 数
factor_block_shared_axis_count = 0    # batch writer 中不同的共享 axis 数（同 index 不重复）


def reset_counters() -> None:
    """Reset the R39-PERF-030 counters (tests / benchmarks)."""
    global factor_block_ref_built, factor_block_ref_used, factor_block_shared_axis_count
    factor_block_ref_built = 0
    factor_block_ref_used = 0
    factor_block_shared_axis_count = 0


def get_counters() -> dict[str, int]:
    """Snapshot of the module-level counters."""
    return {
        "factor_block_ref_built": factor_block_ref_built,
        "factor_block_ref_used": factor_block_ref_used,
        "factor_block_shared_axis_count": factor_block_shared_axis_count,
    }


@dataclass(frozen=True)
class AxisBufferRef:
    """共享 axis 引用：多个因子共用一份 axis 的稳定句柄。

    参数:
        axis_key: axis 的稳定 hash（``axis_key_of``），label 相等的轴 key 相等，
            与对象身份无关；同 axis 判定只需比 key。
        row_count: 轴行数。
        index: 共享的 pandas Index / MultiIndex 对象引用（不复制）。
    """

    axis_key: str
    row_count: int
    index: Any


@dataclass(frozen=True)
class FactorBlockRef:
    """多个因子共享一份 axis 的整块引用（R39-P0-PERF-030 规格）。

    参数:
        axis_ref: 共享 axis 引用（一份，不重复 MultiIndex / ticker string）。
        factor_ids: block 内因子 id 元组（列顺序）。
        values_ref: rows×factors 单 block numpy 引用（``BufferRef.location`` 为
            ``np.ndarray``；列顺序与 ``factor_ids`` 对齐）。
        dtype: 值 numpy dtype 名称（如 ``"float32"``）。
        row_count: 行数。
        factor_count: 因子数。
        validity_ref: NaN/Inf 掩码（True=非有限）的 ``BufferRef``；全有效为 None。
    """

    axis_ref: AxisBufferRef
    factor_ids: tuple[str, ...]
    values_ref: BufferRef
    dtype: str
    row_count: int
    factor_count: int
    validity_ref: BufferRef | None

    def to_arrow_table(self):
        """整 block 一次 Arrow Table 写（共享 axis 作为 index 列）。

        每个 factor 一列，行是共享的 MultiIndex（``preserve_index=True``），
        一次 ``pa.Table.from_pandas`` —— 避免逐因子重复 MultiIndex/ticker object。
        """
        import pyarrow as pa

        idx = self.axis_ref.index
        values = np.asarray(self.values_ref.location, dtype=self.dtype)
        df = pd.DataFrame(values, index=idx, columns=list(self.factor_ids))
        return pa.Table.from_pandas(df, preserve_index=True)

    def to_parquet(self, path: str | Path) -> Path:
        """整 block 一次 Parquet 写（供 batch writer / matrix 消费）。"""
        import pyarrow.parquet as pq

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(self.to_arrow_table(), str(path))
        return path


def _sorted_bytes(values: Any) -> bytes:
    """轴 level 值的顺序无关字节摘要（label 集合相等 -> 摘要相等）。"""
    vals = np.asarray(values)
    if vals.dtype.kind in "mM":  # datetime / timedelta
        return np.sort(vals.astype("datetime64[ns]").view("int64")).tobytes()
    if vals.dtype.kind in "USO":  # object / str / bytes
        codes, _ = pd.factorize(np.asarray(values, dtype=object), sort=True)
        return np.sort(np.asarray(codes, dtype=np.int64)).tobytes()
    try:
        return np.sort(vals).tobytes()
    except TypeError:  # pragma: no cover - 混合类型兜底
        codes, _ = pd.factorize(np.asarray(values, dtype=object), sort=True)
        return np.sort(np.asarray(codes, dtype=np.int64)).tobytes()


def axis_key_of(index: pd.Index) -> str:
    """轴稳定 hash：len / nlevels / names / 每 level dtype / 排序后 label 摘要。

    label 相等的轴（即使对象身份不同）产生相同 key，供 ``share_axis_across`` /
    ``group_by_shared_axis`` 证明 axis 不重复。
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(str(len(index)).encode("ascii"))
    h.update(str(getattr(index, "nlevels", 1)).encode("ascii"))
    names = tuple(str(n) for n in getattr(index, "names", [index.name]))
    h.update(repr(names).encode("utf-8"))
    if isinstance(index, pd.MultiIndex):
        for lvl in range(index.nlevels):
            values = index.get_level_values(lvl)
            h.update(str(values.dtype).encode("ascii"))
            h.update(_sorted_bytes(values))
    else:
        h.update(str(index.dtype).encode("ascii"))
        h.update(_sorted_bytes(index))
    return h.hexdigest()


def build_factor_block(
    factor_ids: Iterable[str],
    values_2d_or_dict: Any,
    index: pd.Index,
    dtype: str = "float32",
) -> FactorBlockRef:
    """producer：把多个同 axis 因子构造成单 block 的 ``FactorBlockRef``。

    参数:
        factor_ids: 因子 id 序列（决定列顺序）。
        values_2d_or_dict: ``np.ndarray``（rows×factors，列顺序与 ``factor_ids``
            对齐）或 ``{fid: 1D array/Series}``（内部按 ``factor_ids`` 顺序
            column-stack）。
        index: 共享的 pandas Index / MultiIndex 对象引用（**不复制**）。
        dtype: 值 numpy dtype 名称（缺省 ``"float32"``）。

    返回:
        FactorBlockRef：``values_ref.location`` 是 ``rows × factor_count`` 的连续
        numpy block；``validity_ref`` 为 NaN/Inf 掩码（True=非有限），全有效为 None。
    """
    fids = tuple(str(f) for f in factor_ids)
    if not fids:
        raise ValueError("build_factor_block: factor_ids 不能为空")
    dtype = str(dtype or "float64")
    np_dtype = np.dtype(dtype)
    if isinstance(values_2d_or_dict, dict):
        arrays = [
            np.asarray(values_2d_or_dict[f], dtype=np_dtype) for f in fids
        ]
        values = np.column_stack(arrays)
    else:
        values = np.asarray(values_2d_or_dict, dtype=np_dtype)
    if values.ndim != 2:
        raise ValueError(
            f"build_factor_block: values 必须为 rows×factors 2D block，got ndim={values.ndim}"
        )
    if values.shape[1] != len(fids):
        raise ValueError(
            f"build_factor_block: values 列数 {values.shape[1]} != factor_ids 数 {len(fids)}"
        )
    if len(index) != values.shape[0]:
        raise ValueError(
            f"build_factor_block: index 行数 {len(index)} != values 行数 {values.shape[0]}"
        )
    values = np.ascontiguousarray(values, dtype=np_dtype)
    row_count = int(values.shape[0])
    finite = np.isfinite(values)
    if finite.all():
        validity_ref: BufferRef | None = None
    else:
        invalid = ~finite
        validity_ref = BufferRef(
            representation="numpy_block",
            schema=("invalid_mask",),
            rows=row_count,
            bytes=int(invalid.nbytes),
            location=invalid,
            ownership="shared",
        )
    values_ref = BufferRef(
        representation="numpy_block",
        schema=tuple(fids),
        rows=row_count,
        bytes=int(values.nbytes),
        location=values,
        ownership="shared",
    )
    axis_ref = AxisBufferRef(
        axis_key=axis_key_of(index), row_count=row_count, index=index
    )
    global factor_block_ref_built
    factor_block_ref_built += 1
    return FactorBlockRef(
        axis_ref=axis_ref,
        factor_ids=fids,
        values_ref=values_ref,
        dtype=dtype,
        row_count=row_count,
        factor_count=len(fids),
        validity_ref=validity_ref,
    )


def share_axis_across(refs: Sequence[FactorBlockRef]) -> bool:
    """多个 ``FactorBlockRef`` 是否共享同一 axis_ref（轴相等）。

    用 8 字节 ``axis_key`` + ``row_count`` 判定，不重走完整 MultiIndex 对象；
    返回 True 即证明这些 block 的 axis 只有一份（不重复）。
    """
    refs = list(refs)
    if not refs:
        return True
    first = refs[0]
    return all(
        r.axis_ref.axis_key == first.axis_ref.axis_key
        and r.axis_ref.row_count == first.axis_ref.row_count
        for r in refs[1:]
    )


def group_by_shared_axis(
    factor_series: dict[str, pd.Series],
) -> list[tuple[AxisBufferRef, list[str]]]:
    """按相等 row axis 对因子分组；返回共享 axis（>=2 因子）的分组。

    每个**不同**的共享 axis 只返回一次（同 index 不重复记录）。对象身份
    （``index is``）先短路，``Index.equals`` 作正确性兜底。

    Axis keys provide a bounded candidate set for equality checks. The prior
    pairwise scan made this batch observability path O(n²) in factor count.
    """
    groups_by_key: dict[str, list[tuple[pd.Index, list[str]]]] = {}
    identity_to_group: dict[int, tuple[pd.Index, list[str]]] = {}
    for factor_id, series in factor_series.items():
        index = series.index
        group = identity_to_group.get(id(index))
        if group is None:
            key = axis_key_of(index)
            candidates = groups_by_key.setdefault(key, [])
            group = next(
                (candidate for candidate in candidates if candidate[0].equals(index)),
                None,
            )
            if group is None:
                group = (index, [factor_id])
                candidates.append(group)
            else:
                group[1].append(factor_id)
            identity_to_group[id(index)] = group
        else:
            group[1].append(factor_id)

    return [
        (
            AxisBufferRef(axis_key=axis_key_of(index), row_count=len(index), index=index),
            members,
        )
        for candidates in groups_by_key.values()
        for index, members in candidates
        if len(members) >= 2
    ]


def record_block_ref_used(ref: FactorBlockRef) -> None:
    """matrix block 写路径：记录一个真实构造并使用的 ``FactorBlockRef``。"""
    global factor_block_ref_used
    factor_block_ref_used += 1


def block_ref_meta(
    ref: FactorBlockRef, block_id: str | None = None
) -> dict[str, Any]:
    """可序列化的 block 元数据（供 manifest/summary 记录，不含大 numpy 对象）。"""
    meta: dict[str, Any] = {
        "factor_ids": list(ref.factor_ids),
        "axis_key": ref.axis_ref.axis_key,
        "row_count": ref.row_count,
        "factor_count": ref.factor_count,
        "dtype": ref.dtype,
    }
    if block_id is not None:
        meta["block_id"] = str(block_id)
    return meta
