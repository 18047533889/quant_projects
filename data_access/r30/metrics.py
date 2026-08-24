"""data_access.r30.metrics —— R30-P0-010 进程级指标注册表 + Prometheus/OTel 导出。

纯 additive 层：不依赖 opentelemetry 库，输出 OTel 风格 dict 与 Prometheus text
exposition format，供接入方自行桥接。

设计要点
    - ``MetricsRegistry`` 线程安全：histogram 计数（count / sum / min / max /
      last），按 ``name + labels`` 聚合。
    - 进程级默认 registry 用 ``get_metrics()`` 取，``reset_metrics()`` 清空。
    - ``record_trace(trace)``：把 ``r30.query_trace.QueryTrace`` 落进 registry
      （query_duration_ms / rows / cache_hit / resolution_cache_hit /
      errors_by_type 等），是 query_trace → metrics 的桥。
"""
from __future__ import annotations

import threading
import time
from typing import Any

#: 预定义指标（canonical 集合）。record() 允许任意名字，这里声明 R30-P0-010 约定的。
METRICS: tuple[str, ...] = (
    "query_duration_ms",
    "scan_bytes",
    "rows",
    "cache_hit",
    "resolution_cache_hit",
    "session_source_reuse",
    "remote_head_latency_ms",
    "remote_list_latency_ms",
    "duckdb_wait_ms",
    "resource_wait_ms",
    "pit_join_latency_ms",
    "errors_by_type",
)


class _Series:
    __slots__ = ("count", "sum", "min", "max", "last")

    def __init__(self) -> None:
        self.count: int = 0
        self.sum: float = 0.0
        self.min: float | None = None
        self.max: float | None = None
        self.last: float | None = None

    def record(self, value: float) -> None:
        self.count += 1
        self.sum += float(value)
        if self.min is None or value < self.min:
            self.min = value
        if self.max is None or value > self.max:
            self.max = value
        self.last = value

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "sum": self.sum,
            "min": self.min,
            "max": self.max,
            "last": self.last,
        }


def _labels_key(**labels: Any) -> tuple[tuple[str, str], ...]:
    """labels → 稳定排序的 (str,str) 键；非标量值 str() 化保证可哈希。"""
    return tuple(
        sorted((str(k), str(v)) for k, v in labels.items())
    )


class MetricsRegistry:
    """线程安全的 histogram 指标注册表。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        #: name → (labels_key → _Series)
        self._series: dict[str, dict[tuple[tuple[str, str], ...], _Series]] = {}

    def record(self, name: str, value: Any, **labels: Any) -> None:
        """记录一个观测值（值非数值时 best-effort 转 float，失败忽略）。"""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return
        key = _labels_key(**labels)
        with self._lock:
            by_labels = self._series.setdefault(name, {})
            series = by_labels.get(key)
            if series is None:
                series = _Series()
                by_labels[key] = series
            series.record(numeric)

    def count(self, name: str, **labels: Any) -> int:
        """取某序列的样本数（无则 0）。"""
        key = _labels_key(**labels)
        with self._lock:
            by_labels = self._series.get(name)
            if not by_labels:
                return 0
            series = by_labels.get(key)
            return series.count if series else 0

    def snapshot(self) -> dict[str, dict[tuple[tuple[str, str], ...], dict[str, Any]]]:
        """全量快照：name → {labels_key: {count,sum,min,max,last}}。

        labels_key 是稳定排序的 ``(k, v)`` 元组（可哈希）；要 dict 用
        ``dict(labels_key)``。
        """
        with self._lock:
            return {
                name: {
                    key: series.to_dict()
                    for key, series in by_labels.items()
                }
                for name, by_labels in self._series.items()
            }

    def reset(self) -> None:
        """清空全部序列（测试 / 重新计数）。"""
        with self._lock:
            self._series.clear()


#: 进程级默认 registry。
_default_registry = MetricsRegistry()


def get_metrics() -> MetricsRegistry:
    """进程级默认指标 registry。"""
    return _default_registry


def reset_metrics() -> None:
    """清空进程级默认 registry（主要给测试用）。"""
    _default_registry.reset()


def _escape_label_value(value: Any) -> str:
    """Prometheus label value 转义（\\ " \n）。"""
    s = str(value)
    s = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return s


def _format_labels(key: tuple[tuple[str, str], ...]) -> str:
    if not key:
        return ""
    inner = ",".join(f'{k}="{_escape_label_value(v)}"' for k, v in key)
    return "{" + inner + "}"


def prometheus_text(registry: MetricsRegistry | None = None) -> str:
    """渲染 Prometheus text exposition format（每序列 4 行）。

    对每个 name+labels 序列输出：
      - ``name_total{labels}``    = count
      - ``name_sum{labels}``      = sum
      - ``name_count{labels}``    = count
      - ``name_bucket{le="+Inf",labels}`` = count
    （简单 histogram；不做分桶插值。）
    """
    registry = registry or get_metrics()
    snap = registry.snapshot()
    lines: list[str] = []
    for name in sorted(snap):
        lines.append(f"# HELP {name} data_access metric")
        lines.append(f"# TYPE {name} counter")
        for key, stats in snap[name].items():
            label_str = _format_labels(key)
            count = int(stats["count"])
            lines.append(f"{name}_total{label_str} {count}")
            lines.append(f"{name}_sum{label_str} {stats['sum']:g}")
            lines.append(f"{name}_count{label_str} {count}")
            bucket_labels = (
                f'{{le="+Inf",' + label_str.lstrip("{") if label_str else '{le="+Inf"}'
            )
            lines.append(f"{name}_bucket{bucket_labels} {count}")
    return "\n".join(lines) + ("\n" if lines else "")


def otel_dict(registry: MetricsRegistry | None = None) -> list[dict[str, Any]]:
    """OTel 风格数据点列表（不硬依赖 opentelemetry 库）。

    每个 name+labels 序列一个点：{name, attributes, value(=`last`), count, sum,
    min, max, timestamp_ns}。
    """
    registry = registry or get_metrics()
    snap = registry.snapshot()
    ts_ns = time.time_ns()
    points: list[dict[str, Any]] = []
    for name in sorted(snap):
        for key, stats in snap[name].items():
            points.append(
                {
                    "name": name,
                    "attributes": dict(key),
                    "value": stats["last"],
                    "count": stats["count"],
                    "sum": stats["sum"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "timestamp_ns": ts_ns,
                }
            )
    return points


def record_trace(trace: Any, registry: MetricsRegistry | None = None) -> None:
    """把 ``r30.query_trace.QueryTrace`` 落进 registry（best-effort）。

    用 getattr 防御性读取——不 import query_trace，避免与 query_trace 的
    _record_metrics 形成强依赖；任何字段缺失都不抛。
    """
    registry = registry or get_metrics()
    elapsed_ms = getattr(trace, "elapsed_ms", None)
    if elapsed_ms is not None:
        registry.record("query_duration_ms", elapsed_ms, dataset=str(getattr(trace, "dataset", "") or ""))
    rows = getattr(trace, "rows", 0) or 0
    if rows:
        registry.record("rows", rows)
    result_bytes = getattr(trace, "result_bytes", 0) or 0
    if result_bytes:
        registry.record("scan_bytes", result_bytes)
    cache_status = getattr(trace, "cache_status", None)
    if cache_status is not None:
        registry.record("cache_hit", 1 if str(cache_status).lower() in ("hit", "true", "1") else 0)
    registry.record(
        "resolution_cache_hit",
        1 if getattr(trace, "resolution_cache_hit", False) else 0,
    )
    reuse = getattr(trace, "session_source_reuse", 0) or 0
    if reuse:
        registry.record("session_source_reuse", reuse)
    wait_ms = getattr(trace, "resource_wait_ms", 0.0) or 0.0
    if wait_ms:
        registry.record("resource_wait_ms", wait_ms)
    error = getattr(trace, "error", None)
    if error:
        err_type = str(error).split(":", 1)[0].strip() or "Unknown"
        registry.record("errors_by_type", 1, type=err_type)


__all__ = [
    "METRICS",
    "MetricsRegistry",
    "get_metrics",
    "reset_metrics",
    "prometheus_text",
    "otel_dict",
    "record_trace",
]
