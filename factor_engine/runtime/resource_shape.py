# -*- coding: utf-8 -*-
"""R36 §10/§309：ResourceShapeKey —— 按算子族/后端/形状分桶的资源模型键。

R36 P0-004/005：不能按「operator name 一个值永久估计」；资源预测必须按
``(canonical_family, backend, execution_variant, rows_bucket, instruments_bucket,
window_bucket, input_count, dtype, representation, group_count_bucket)`` 学习
（§10）。本模块定义该键 + 离散桶（§309）。

设计要点：
    - 用**语义形状**而不是精确数值做键：1e6 rows 与 1.05e6 rows 同一桶，避免
      calibration 数据碎片化（§308：EWMA / rolling quantile / bucket table 优先）。
    - ``ResourceShapeKey`` 是 frozen dataclass（可 hash / 可作 parquet key）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: 离散桶边界（§309）
_ROW_BUCKETS = (0, 10_000, 100_000, 1_000_000, 10_000_000, float("inf"))
_ASSET_BUCKETS = (0, 100, 500, 1_000, 5_000, float("inf"))
_WINDOW_BUCKETS = (0, 20, 60, 120, 252, float("inf"))


def _bucket(value: float, edges: tuple[float, ...]) -> int:
    for i, edge in enumerate(edges):
        if value < edge:
            return i
    return len(edges) - 1


def rows_bucket(rows: int) -> int:
    return _bucket(max(0, int(rows or 0)), _ROW_BUCKETS)


def assets_bucket(assets: int) -> int:
    return _bucket(max(0, int(assets or 0)), _ASSET_BUCKETS)


def window_bucket(window: int | float | None) -> int:
    if window is None:
        return len(_WINDOW_BUCKETS) - 1  # full
    return _bucket(float(window), _WINDOW_BUCKETS)


@dataclass(frozen=True)
class ResourceShapeKey:
    """§10：一个资源形状的唯一键（冷启动/学习共用）。"""

    canonical_family: str
    backend: str
    execution_variant: str = "default"
    rows_bucket: int = 0
    instruments_bucket: int = 0
    window_bucket: int = 0
    input_count: int = 1
    dtype: str = "float64"
    representation: str = "panel"
    group_count_bucket: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_family": self.canonical_family,
            "backend": self.backend,
            "execution_variant": self.execution_variant,
            "rows_bucket": self.rows_bucket,
            "instruments_bucket": self.instruments_bucket,
            "window_bucket": self.window_bucket,
            "input_count": self.input_count,
            "dtype": self.dtype,
            "representation": self.representation,
            "group_count_bucket": self.group_count_bucket,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ResourceShapeKey":
        return cls(
            canonical_family=str(d.get("canonical_family", "")),
            backend=str(d.get("backend", "")),
            execution_variant=str(d.get("execution_variant", "default")),
            rows_bucket=int(d.get("rows_bucket", 0)),
            instruments_bucket=int(d.get("instruments_bucket", 0)),
            window_bucket=int(d.get("window_bucket", 0)),
            input_count=int(d.get("input_count", 1)),
            dtype=str(d.get("dtype", "float64")),
            representation=str(d.get("representation", "panel")),
            group_count_bucket=int(d.get("group_count_bucket", 0)),
        )

    @classmethod
    def from_task(cls, task: Any) -> "ResourceShapeKey":
        """从 PhysicalFactorTask 提取形状键（rows/instruments/window 尽力而为）。"""
        op = str(getattr(task, "op", "") or "unknown")
        backend = str(getattr(task, "preferred_backend", "") or "unknown")
        contract = getattr(task, "resource_contract", None)
        spec = getattr(task, "source_scan_spec", None)
        rows = 0
        if spec is not None:
            rows = int(getattr(spec, "expected_rows", 0) or 0)
        if rows <= 0 and contract is not None:
            rows = int(getattr(contract, "input_bytes", 0) or 0) // 8
        assets = 0
        scope = getattr(task, "instrument_scope", None)
        if scope is not None:
            try:
                assets = int(getattr(scope, "count", 0) or len(scope) or 0)
            except Exception:
                assets = 0
        window = None
        tr = getattr(task, "time_range", None)
        if tr is not None:
            try:
                start = getattr(tr, "start", None)
                end = getattr(tr, "end", None)
                if start is not None and end is not None:
                    window = max(1, int((end - start).days))
            except Exception:
                window = None
        return cls(
            canonical_family=op,
            backend=backend,
            execution_variant="default",
            rows_bucket=rows_bucket(rows),
            instruments_bucket=assets_bucket(assets),
            window_bucket=window_bucket(window),
            input_count=max(1, int(getattr(task, "inputs", None) is not None and len(task.inputs) or 1)),
            dtype="float64",
            representation="panel",
        )


def hardware_fingerprint() -> dict[str, Any]:
    """§33：HardwareFingerprint —— calibration 持久化的 key 维度。"""
    import os

    try:
        import numpy as np

        numpy_version = np.__version__
    except Exception:
        numpy_version = "?"
    try:
        import duckdb

        duckdb_version = duckdb.__version__
    except Exception:
        duckdb_version = "?"
    try:
        import polars

        polars_version = polars.__version__
    except Exception:
        polars_version = "?"
    try:
        import resource
        import socket

        hostname_hash = abs(hash(socket.gethostname())) % (10**10)
        from runtime.resource_governor import effective_cpu_slots, effective_memory_limit_bytes

        return {
            "hostname_hash": hostname_hash,
            "cpu_model": "?",
            "cpu_quota": effective_cpu_slots(),
            "affinity_count": effective_cpu_slots(),
            "ram_limit": effective_memory_limit_bytes(),
            "duckdb_version": duckdb_version,
            "polars_version": polars_version,
            "numpy_version": numpy_version,
            "blas_backend": "?",
        }
    except Exception:
        return {
            "hostname_hash": 0,
            "cpu_model": "?",
            "cpu_quota": 0,
            "affinity_count": 0,
            "ram_limit": 0,
            "duckdb_version": duckdb_version,
            "polars_version": polars_version,
            "numpy_version": numpy_version,
            "blas_backend": "?",
        }
