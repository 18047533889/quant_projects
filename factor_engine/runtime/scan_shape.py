# -*- coding: utf-8 -*-
"""R39-PERF-075：ScanShapeKey —— shape-aware 的 ScanCost 校准键 + P50/P95。

当前 ``runtime/runtime_calibration.py`` 的 ScanCost 校准只用
``(operator, backend, shape_bucket, window_bucket, market, frequency)`` 一个键
（dataset 维度只体现在 operator），没有 dataset / local_or_remote / file_count /
selected_bytes / rowgroup / 投影比例 / 仪器数 / manifest_hit / cache_heat /
storage_class。本模块新增：

    - :class:`ScanShapeKey`：frozen dataclass，覆盖以上 10 个维度，``from_scan_cost``
      从 ``data_access.read.scan_cost.ScanCost``（duck-typed）构造，``to_key`` 输出
      canonical 字符串。
    - :class:`ScanShapeCalibrator`：每 shape 记录有界样本窗口
      （open_ms/scan_ms/decode_ms/rows/bytes/throughput），输出 P50/P95。
    - 原 EMA 路径（``runtime_calibration.record_task_actual``）保持不变，作为
      shape 信息缺失时的 fallback。
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable

#: 每 shape 保留的最大样本数（有界窗口）
DEFAULT_MAX_SAMPLES_PER_SHAPE = 256


# ---------------------------------------------------------------------------
# 离散桶
# ---------------------------------------------------------------------------


def file_count_bucket(n: int) -> str:
    n = max(0, int(n or 0))
    if n <= 0:
        return "none"
    if n == 1:
        return "one"
    if n <= 10:
        return "few"
    if n <= 100:
        return "dozens"
    return "hundreds"


def selected_bytes_bucket(b: int | None) -> str:
    b = max(0, int(b or 0))
    if b <= 0:
        return "unknown"
    if b < 1024 * 1024:
        return "lt1mb"
    if b < 64 * 1024 * 1024:
        return "1to64mb"
    if b < 512 * 1024 * 1024:
        return "64to512mb"
    if b < 2 * 1024 * 1024 * 1024:
        return "512mbto2gb"
    return "gt2gb"


def rowgroup_count_bucket(n: int | None) -> str:
    n = max(0, int(n or 0))
    if n <= 0:
        return "none"
    if n <= 4:
        return "tiny"
    if n <= 20:
        return "small"
    if n <= 100:
        return "medium"
    return "large"


def projected_column_ratio_bucket(ratio: float | None) -> str:
    if ratio is None or math.isnan(float(ratio)):
        return "unknown"
    r = max(0.0, min(1.0, float(ratio)))
    if r < 0.1:
        return "lt0.1"
    if r < 0.5:
        return "0.1to0.5"
    return "0.5to1.0"


def instrument_count_bucket(n: int) -> str:
    n = max(0, int(n or 0))
    if n <= 0:
        return "unknown"
    if n <= 100:
        return "tiny"
    if n <= 1000:
        return "small"
    if n <= 5000:
        return "medium"
    return "large"


def cache_heat_bucket(cache_heat: str | None) -> str:
    h = str(cache_heat or "cold").strip().lower()
    if h in {"hot", "warm", "cold"}:
        return h
    if h in {"0", "1", "2", "3"}:
        return {"0": "cold", "1": "cold", "2": "warm", "3": "hot"}.get(h, "cold")
    return "cold"


def storage_class_bucket(storage_class: str | None) -> str:
    s = str(storage_class or "unknown").strip().lower()
    if s in {"nvme", "ssd", "network", "unknown", "memory"}:
        return s
    return "unknown"


# ---------------------------------------------------------------------------
# ScanShapeKey
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanShapeKey:
    """R39-PERF-075：shape-aware ScanCost 校准键（10 维度）。"""

    dataset: str
    local_or_remote: str = "local"
    file_count_bucket: str = "unknown"
    selected_bytes_bucket: str = "unknown"
    rowgroup_count_bucket: str = "unknown"
    projected_column_ratio_bucket: str = "unknown"
    instrument_count_bucket: str = "unknown"
    manifest_hit: bool | None = None
    cache_heat: str = "cold"
    storage_class: str = "unknown"

    @classmethod
    def from_scan_cost(cls, cost: Any) -> "ScanShapeKey":
        """从 ``ScanCost``（duck-typed）构造；缺字段时用 safe 默认。"""
        dataset = str(getattr(cost, "dataset", "") or "unknown")
        local_or_remote = "remote" if bool(getattr(cost, "remote", False)) else "local"
        fc = file_count_bucket(int(getattr(cost, "selected_files", 0) or getattr(cost, "file_count", 0) or 0))
        sb = selected_bytes_bucket(getattr(cost, "selected_bytes", None) or getattr(cost, "total_bytes", None))
        rg = rowgroup_count_bucket(getattr(cost, "selected_rowgroups", None))
        projected = int(getattr(cost, "projected_columns", 0) or 0)
        total = int(getattr(cost, "total_columns", 0) or 0)
        ratio = (projected / total) if (projected and total) else None
        pr = projected_column_ratio_bucket(ratio)
        ic = instrument_count_bucket(int(getattr(cost, "instrument_count", 0) or 0))
        manifest_hit = None
        try:
            mh = getattr(cost, "manifest_hit", None)
            if mh is not None:
                manifest_hit = bool(mh)
        except Exception:
            manifest_hit = None
        cache_heat = cache_heat_bucket(getattr(cost, "cache_heat", None))
        storage = storage_class_bucket(getattr(cost, "storage_class", None))
        return cls(
            dataset=dataset,
            local_or_remote=local_or_remote,
            file_count_bucket=fc,
            selected_bytes_bucket=sb,
            rowgroup_count_bucket=rg,
            projected_column_ratio_bucket=pr,
            instrument_count_bucket=ic,
            manifest_hit=manifest_hit,
            cache_heat=cache_heat,
            storage_class=storage,
        )

    def to_key(self) -> str:
        return "|".join(
            [
                str(self.dataset),
                str(self.local_or_remote),
                str(self.file_count_bucket),
                str(self.selected_bytes_bucket),
                str(self.rowgroup_count_bucket),
                str(self.projected_column_ratio_bucket),
                str(self.instrument_count_bucket),
                "hit" if self.manifest_hit else ("miss" if self.manifest_hit is False else "unknown"),
                str(self.cache_heat),
                str(self.storage_class),
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "local_or_remote": self.local_or_remote,
            "file_count_bucket": self.file_count_bucket,
            "selected_bytes_bucket": self.selected_bytes_bucket,
            "rowgroup_count_bucket": self.rowgroup_count_bucket,
            "projected_column_ratio_bucket": self.projected_column_ratio_bucket,
            "instrument_count_bucket": self.instrument_count_bucket,
            "manifest_hit": self.manifest_hit,
            "cache_heat": self.cache_heat,
            "storage_class": self.storage_class,
        }


def percentile(sorted_values: list[float], p: float) -> float:
    """nearest-rank 百分位（0<p<=1）。空列表返回 0.0。"""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if p <= 0.0:
        return sorted_values[0]
    if p >= 1.0:
        return sorted_values[-1]
    idx = max(0, min(n - 1, math.ceil(p * n) - 1))
    return float(sorted_values[idx])


# ---------------------------------------------------------------------------
# ScanShapeCalibrator
# ---------------------------------------------------------------------------


@dataclass
class _ShapeSamples:
    samples: list[dict[str, float]] = field(default_factory=list)


class ScanShapeCalibrator:
    """每 shape 记录有界样本窗口，输出 P50/P95。

    - ``record(shape, open_ms, scan_ms, decode_ms, rows, bytes)`` 累加一个观测；
      超过 ``max_samples_per_shape`` 时丢最旧（有界窗口）。
    - ``summary(shape)`` 返回该 shape 的 P50/P95 统计（无样本返回空 dict）。
    - ``all_summaries()`` 返回全部 shape 的统计（用于持久化/报告）。
    """

    def __init__(self, max_samples_per_shape: int = DEFAULT_MAX_SAMPLES_PER_SHAPE) -> None:
        self._max = max(4, int(max_samples_per_shape))
        self._samples: dict[str, _ShapeSamples] = {}
        self._lock = threading.RLock()

    # -- 内部 --

    def _ensure(self, key: str) -> _ShapeSamples:
        entry = self._samples.get(key)
        if entry is None:
            entry = _ShapeSamples()
            self._samples[key] = entry
        return entry

    # -- 对外 API --

    def record(
        self,
        shape: ScanShapeKey,
        *,
        open_ms: float = 0.0,
        scan_ms: float = 0.0,
        decode_ms: float = 0.0,
        rows: int = 0,
        bytes_: int = 0,
    ) -> None:
        key = shape.to_key()
        with self._lock:
            entry = self._ensure(key)
            entry.samples.append(
                {
                    "open_ms": float(open_ms or 0.0),
                    "scan_ms": float(scan_ms or 0.0),
                    "decode_ms": float(decode_ms or 0.0),
                    "rows": float(rows or 0.0),
                    "bytes": float(bytes_ or 0.0),
                }
            )
            if len(entry.samples) > self._max:
                del entry.samples[: len(entry.samples) - self._max]

    @staticmethod
    def _percentiles(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        s = sorted(values)
        return {
            "p50": round(percentile(s, 0.50), 3),
            "p95": round(percentile(s, 0.95), 3),
            "mean": round(sum(s) / len(s), 3),
            "count": len(s),
        }

    def summary(self, shape: ScanShapeKey) -> dict[str, Any]:
        key = shape.to_key()
        with self._lock:
            entry = self._samples.get(key)
            if entry is None or not entry.samples:
                return {}
            samples = list(entry.samples)
        open_ms = [s["open_ms"] for s in samples]
        scan_ms = [s["scan_ms"] for s in samples]
        decode_ms = [s["decode_ms"] for s in samples]
        rows = [s["rows"] for s in samples]
        bytes_ = [s["bytes"] for s in samples]
        total_scan_ms = [a + b + c for a, b, c in zip(open_ms, scan_ms, decode_ms)]
        # throughput: bytes / (open+scan+decode) ms  → MB/s
        throughput = []
        for b, t in zip(bytes_, total_scan_ms):
            if t > 0:
                throughput.append((b / 1024.0 / 1024.0) / (t / 1000.0))  # MB/s
        return {
            "shape": shape.to_dict(),
            "key": key,
            "samples": len(samples),
            "open_ms": self._percentiles(open_ms),
            "scan_ms": self._percentiles(scan_ms),
            "decode_ms": self._percentiles(decode_ms),
            "rows": self._percentiles(rows),
            "bytes": self._percentiles(bytes_),
            "throughput_mb_per_s": self._percentiles(throughput) if throughput else {},
        }

    def all_summaries(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            keys = list(self._samples.keys())
        out: dict[str, dict[str, Any]] = {}
        for key in keys:
            shape = reconstruct_scan_shape_key(key)
            if shape is None:
                continue
            out[key] = self.summary(shape)
        return out

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()


def reconstruct_scan_shape_key(key: str) -> ScanShapeKey | None:
    """从 ``to_key()`` 字符串重建 ScanShapeKey（all_summaries / 持久化用）。"""
    parts = key.split("|")
    if len(parts) != 10:
        return None
    mh: bool | None = None
    if parts[7] == "hit":
        mh = True
    elif parts[7] == "miss":
        mh = False
    return ScanShapeKey(
        dataset=parts[0],
        local_or_remote=parts[1],
        file_count_bucket=parts[2],
        selected_bytes_bucket=parts[3],
        rowgroup_count_bucket=parts[4],
        projected_column_ratio_bucket=parts[5],
        instrument_count_bucket=parts[6],
        manifest_hit=mh,
        cache_heat=parts[8],
        storage_class=parts[9],
    )
