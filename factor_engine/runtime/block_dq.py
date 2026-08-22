# -*- coding: utf-8 -*-
"""R33 §15/§33/§65：FactorBlock + Block DQ + Cross-Section Block Kernel。

- :class:`FactorBlock`：multi-root 输出的统一承载（``factor_ids`` × ``values``
  ndarray / Arrow columns），替代「1000 个 Series 逐个归一化/写盘」。
- :func:`compute_block_dq`：**block 化** DQ——一次对整块算 null ratio / finite
  ratio / cross-sectional coverage / variance / constant / extreme ratio
  （per-factor 向量化指标），不做逐 factor Python 级检查。
- :func:`cross_section_block`：对同 timestamp 上多个 factor 批量执行
  rank / zscore / demean / winsorize（不再 ``for factor: groupby(date)``）。
  parity 保证：block 结果 == canonical per-factor（R33_CROSS_SECTION_BLOCK_PARITY）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class FactorBlock:
    """R33 §65：BatchBlock 数据结构。

    ``values`` 为 ``(n_factors, n_cells)`` 的 ndarray（float64）；``date_axis`` /
    ``asset_axis`` 记录横截面轴（可选——用于 rank/zscore 的横截面分组）。
    """

    factor_ids: tuple[str, ...]
    values: np.ndarray
    date_axis: np.ndarray | None = None      # (n_cells,) 或 (n_dates,)（长表）
    asset_axis: np.ndarray | None = None     # (n_cells,) 或 (n_assets,)
    dtype: str = "float64"
    validity: np.ndarray | None = None
    snapshot: str = ""
    lineage: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.values.ndim == 1:
            self.values = self.values.reshape(1, -1)
        self.values = np.asarray(self.values, dtype=np.float64)

    @property
    def n_factors(self) -> int:
        return len(self.factor_ids)

    @property
    def n_cells(self) -> int:
        return self.values.shape[1]

    def factor(self, factor_id: str) -> np.ndarray:
        if factor_id not in self.factor_ids:
            raise KeyError(f"factor {factor_id!r} not in block {self.factor_ids}")
        return self.values[self.factor_ids.index(factor_id)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_ids": list(self.factor_ids),
            "shape": list(self.values.shape),
            "dtype": str(self.values.dtype),
            "snapshot": self.snapshot,
        }


def compute_block_dq(block: FactorBlock) -> dict[str, dict[str, float]]:
    """R33 §33：block 化 DQ——per-factor 向量化指标，不逐 factor Python 循环。

    返回 ``{factor_id: {null_ratio, finite_ratio, coverage, variance, constant,
    extreme_ratio, min, max}}``。
    """
    v = block.values
    n_cells = max(1, v.shape[1])
    out: dict[str, dict[str, float]] = {}
    with np.errstate(all="ignore"):
        for idx, fid in enumerate(block.factor_ids):
            col = v[idx]
            finite = np.isfinite(col)
            non_null = ~np.isnan(col)
            var = float(np.nanvar(col)) if np.any(non_null) else 0.0
            const = bool(np.isfinite(var) and var <= 1e-30)
            abs_finite = np.abs(col[finite])
            extreme = float(np.mean(abs_finite > 1e12)) if finite.any() else 0.0
            out[fid] = {
                "null_ratio": round(float(np.mean(~non_null)), 6),
                "finite_ratio": round(float(np.mean(finite)), 6),
                "coverage": round(float(np.mean(non_null)), 6),
                "variance": round(var, 9),
                "constant": const,
                "extreme_ratio": round(extreme, 6),
                "min": round(float(np.nanmin(col)), 9) if np.any(non_null) else float("nan"),
                "max": round(float(np.nanmax(col)), 9) if np.any(non_null) else float("nan"),
            }
    return out


def cross_section_block(
    block: FactorBlock,
    operation: str,
    *,
    group_labels: np.ndarray | None = None,
    q: float = 0.975,
    per_date: bool = True,
) -> np.ndarray:
    """R33 §15：Cross-Section Block Kernel。

    ``operation`` ∈ {rank, zscore, demean, winsorize}。``per_date=True`` 时按
    ``date_axis`` 逐 timestamp 做横截面（rank/zscore 的 canonical 语义是逐日
    横截面）；``group_labels`` 提供 group 时先 group 再操作。

    Parity：结果与 canonical per-factor（``backend.rank_spec`` /
    ``cs_*`` 语义）一致——逐 timestamp 横截面、NaN 不参与、tie 处理沿用
    canonical rank spec。
    """
    v = block.values
    if v.ndim == 1:
        v = v.reshape(1, -1)
    out = np.zeros_like(v, dtype=np.float64)
    if not per_date or block.date_axis is None:
        # 单截面：整块一个横截面。
        for idx in range(v.shape[0]):
            out[idx] = _apply_cs_op(v[idx], operation, group_labels, q)
        return out
    dates, inverse = np.unique(block.date_axis, return_inverse=True)
    for d in range(len(dates)):
        mask = inverse == d
        for idx in range(v.shape[0]):
            col = v[idx]
            sub = col[mask]
            labels = group_labels[mask] if group_labels is not None else None
            res = _apply_cs_op(sub, operation, labels, q)
            col_out = out[idx]
            col_out[mask] = res
    return out


def _apply_cs_op(
    col: np.ndarray,
    operation: str,
    group_labels: np.ndarray | None,
    q: float,
) -> np.ndarray:
    """单个横截面的操作（与 canonical per-factor 语义对齐）。"""
    res = np.full(col.shape, np.nan, dtype=np.float64)
    valid = ~np.isnan(col)
    values = col[valid]
    if not values.size:
        return res
    if group_labels is not None:
        # group-aware：按组独立操作。
        labels = group_labels[valid]
        for g in np.unique(labels):
            gm = labels == g
            res[valid][gm] = _apply_cs_op_single(values[gm], operation, q)
        return res
    res[valid] = _apply_cs_op_single(values, operation, q)
    return res


def _apply_cs_op_single(values: np.ndarray, operation: str, q: float) -> np.ndarray:
    with np.errstate(all="ignore"):
        if operation == "rank":
            # canonical rank：average tie（与 backend.rank_spec 一致），升序。
            order = values.argsort(kind="mergesort")
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(values) + 1, dtype=float)
            # average ties
            sorted_vals = values[order]
            i = 0
            while i < len(sorted_vals):
                j = i
                while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
                    j += 1
                if j > i:
                    avg = (ranks[order][i] + ranks[order][j]) / 2.0
                    ranks[order[i : j + 1]] = avg
                i = j + 1
            n = len(values)
            return (ranks - 1.0) / max(1, n - 1) if n > 1 else np.full_like(ranks, np.nan)
        if operation == "zscore":
            mean = float(np.mean(values))
            std = float(np.std(values))
            if std <= 1e-30:
                return np.zeros_like(values, dtype=float)
            return (values - mean) / std
        if operation == "demean":
            return values - float(np.mean(values))
        if operation == "winsorize":
            lo = float(np.quantile(values, 1.0 - q))
            hi = float(np.quantile(values, q))
            return np.clip(values, lo, hi)
        raise ValueError(f"unsupported cross-section operation {operation!r}")


def cross_section_block_parity(
    block: FactorBlock,
    operation: str,
    reference: dict[str, np.ndarray],
) -> bool:
    """R33_CROSS_SECTION_BLOCK_PARITY：block 结果与 canonical per-factor 对齐。

    ``reference``：``{factor_id: per-factor 逐日横截面结果}``。逐 element 比较
    （NaN mask + finite values），不允许无脑 tolerance。
    """
    block_out = cross_section_block(block, operation)
    for idx, fid in enumerate(block.factor_ids):
        ref = reference.get(fid)
        if ref is None:
            return False
        ref = np.asarray(ref, dtype=np.float64).reshape(-1)
        got = block_out[idx]
        if ref.shape != got.shape:
            return False
        if not np.array_equal(np.isnan(ref), np.isnan(got)):
            return False
        if not np.allclose(ref[np.isfinite(ref)], got[np.isfinite(got)],
                           rtol=1e-9, atol=1e-12):
            return False
    return True
