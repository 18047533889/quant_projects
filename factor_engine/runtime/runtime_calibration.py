# -*- coding: utf-8 -*-
"""R27-016..019/135/136: 在线成本校准 + 服务器指纹。

要点
    - R27-016/017：task 运行后记录 actual_elapsed_ms / rss_before/after /
      read_bytes / output_bytes / backend / threads / rows / instruments / window，
      按 ``(operator_canonical, backend, shape_bucket, window_bucket, market,
      frequency)`` 建 key。
    - R27-018：EMA 维护 elapsed_factor / memory_factor（类似 DataAccess scan
      calibration）。
    - R27-019：``predicted_peak = static_peak * calibrated_memory_factor *
      uncertainty_margin``。
    - R27-135/136：per-server fingerprint → ``~/.cache/factor_engine/perf/
      <fingerprint>.json`` 持久化 scan/operator/write calibration。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runtime.task_resource_contract import (
    UNCERTAINTY_CALIBRATED,
    UNCERTAINTY_COLD,
    UNCERTAINTY_WARM,
)

_EMA_ALPHA = 0.2
_LOCK = threading.RLock()
#: key -> {"elapsed_factor": float, "memory_factor": float, "samples": int}
_CALIBRATION: dict[str, dict[str, float]] = {}


# ---------------------------------------------------------------------------
# FE-P0-019: Typed CalibrationFactors result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationFactors:
    """FE-P0-019: Typed calibration result replacing ambiguous tuple.

    Returned by calibrated_factors() to make elapsed_factor vs memory_factor
    explicit at call sites. Prevents bugs like FE-P0-020 where the wrong
    factor was used.
    """
    elapsed_factor: float
    memory_factor: float
    samples: int

    def to_tuple(self) -> tuple[float, float, int]:
        """Compatibility accessor for legacy callers expecting tuple unpacking."""
        return (self.elapsed_factor, self.memory_factor, self.samples)


def _server_fingerprint() -> str:
    """R27-135：CPU model / slots / RAM / NUMA / disk class / duckdb / polars 版本。"""
    try:
        import platform

        slots = os.cpu_count() or 0
        try:
            import psutil  # type: ignore[import-untyped]

            ram = int(psutil.virtual_memory().total)
        except Exception:
            ram = 0
        duckdb = ""
        try:
            import duckdb  # type: ignore[import-untyped]

            duckdb = duckdb.__version__ if hasattr(duckdb, "__version__") else ""
        except Exception:
            pass
        polars = ""
        try:
            import polars  # type: ignore[import-untyped]

            polars = polars.__version__ if hasattr(polars, "__version__") else ""
        except Exception:
            pass
        payload = json.dumps(
            {
                "machine": platform.machine(),
                "cpu_slots": slots,
                "ram": ram,
                "node": platform.node(),
                "duckdb": duckdb,
                "polars": polars,
            },
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        # SHA-1 used only for server fingerprint cache key, not cryptographic security
        return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    except Exception:
        return "unknown"


def _calib_path() -> Path:
    return Path(os.path.expanduser("~/.cache/factor_engine/perf")).joinpath(
        f"{_server_fingerprint()}.json"
    )


def load_server_calibration() -> dict[str, Any]:
    """R27-136：读持久化 calibration（不存在返回空 dict）。"""
    try:
        path = _calib_path()
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def save_server_calibration(data: dict[str, Any]) -> None:
    """R27-136：持久化 calibration。"""
    try:
        path = _calib_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass


def calibration_key(
    *,
    operator: str,
    backend: str,
    shape_bucket: str = "",
    window_bucket: str = "",
    market: str = "",
    frequency: str = "",
) -> str:
    """R27-017：calibration key。"""
    return "|".join(
        [str(operator), str(backend), str(shape_bucket), str(window_bucket),
         str(market), str(frequency)]
    )


def _shape_bucket(rows: int, instruments: int) -> str:
    if rows <= 1_000_000 and instruments <= 1000:
        return "small"
    if rows <= 10_000_000:
        return "medium"
    return "large"


def _window_bucket(window: int | None) -> str:
    if window is None:
        return ""
    if window <= 10:
        return "short"
    if window <= 60:
        return "medium"
    return "long"


def record_task_actual(
    *,
    operator: str,
    backend: str,
    actual_elapsed_ms: float,
    rss_delta_bytes: int,
    rows: int = 0,
    instruments: int = 0,
    window: int | None = None,
    market: str = "",
    frequency: str = "",
    predicted_ms: float = 0.0,
    predicted_peak_bytes: int = 0,
) -> None:
    """R27-016/018：记录一次 task 实际成本并 EMA 更新校准因子。"""
    key = calibration_key(
        operator=operator,
        backend=backend,
        shape_bucket=_shape_bucket(rows, instruments),
        window_bucket=_window_bucket(window),
        market=market,
        frequency=frequency,
    )
    with _LOCK:
        entry = _CALIBRATION.setdefault(
            key, {"elapsed_factor": 1.0, "memory_factor": 1.0, "samples": 0}
        )
        entry["samples"] = int(entry.get("samples", 0)) + 1
        # elapsed factor：predicted_ms>0 时 ratio = actual/predicted，EMA 平滑。
        if predicted_ms > 0 and actual_elapsed_ms > 0:
            ratio = min(10.0, max(0.1, actual_elapsed_ms / predicted_ms))
            cur = float(entry.get("elapsed_factor", 1.0))
            entry["elapsed_factor"] = cur + _EMA_ALPHA * (ratio - cur)
        # memory factor：predicted_peak>0 时 ratio = actual/predicted。
        if predicted_peak_bytes > 0 and rss_delta_bytes > 0:
            ratio = min(5.0, max(0.1, rss_delta_bytes / predicted_peak_bytes))
            cur = float(entry.get("memory_factor", 1.0))
            entry["memory_factor"] = cur + _EMA_ALPHA * (ratio - cur)


def calibrated_factors(key: str) -> CalibrationFactors:
    """返回 CalibrationFactors (elapsed_factor, memory_factor, samples)；无样本返回 (1.0, 1.0, 0)。

    FE-P0-019: Now returns typed CalibrationFactors instead of ambiguous tuple.
    Legacy callers can use .to_tuple() for compatibility, but new code should
    access fields directly (.elapsed_factor, .memory_factor, .samples).
    """
    with _LOCK:
        entry = _CALIBRATION.get(key)
        if entry is None:
            return CalibrationFactors(1.0, 1.0, 0)
        return CalibrationFactors(
            elapsed_factor=float(entry.get("elapsed_factor", 1.0)),
            memory_factor=float(entry.get("memory_factor", 1.0)),
            samples=int(entry.get("samples", 0)),
        )


def calibrated_peak_bytes(
    static_peak_bytes: int,
    key: str,
    *,
    samples_for_calibrated_uncertainty: int = 5,
) -> tuple[int, float]:
    """R27-019：``predicted_peak = static_peak * calibrated_memory_factor *
    uncertainty_margin``。

    返回 ``(predicted_peak, uncertainty)``。样本少 → 高 uncertainty（1.50）；
    样本足够且校准收敛 → 降到 1.15（R27-040）。

    FE-P0-020: Fixed bug where elapsed_factor was incorrectly used instead of
    memory_factor. Now correctly extracts .memory_factor from CalibrationFactors.

    FE-P0-027: Zero/unknown static_peak_bytes now returns conservative bound
    (8MB minimum) instead of zero, preventing production admission failures.
    """
    factors = calibrated_factors(key)

    # FE-P0-027: Prevent zero estimate in production paths
    if static_peak_bytes <= 0:
        # Conservative fallback: 8MB minimum (prevents zero admission)
        static_peak_bytes = 8 * 1024 * 1024

    # FE-P0-020: Use memory_factor, not elapsed_factor
    if factors.samples < 1:
        uncertainty = UNCERTAINTY_WARM
    elif factors.samples < samples_for_calibrated_uncertainty:
        uncertainty = UNCERTAINTY_COLD
    else:
        uncertainty = UNCERTAINTY_CALIBRATED

    predicted = max(0, int(static_peak_bytes * factors.memory_factor * uncertainty))

    # FE-P0-027: Final safeguard - never return zero for production paths
    if predicted == 0:
        predicted = 8 * 1024 * 1024

    return predicted, uncertainty


def reset_calibration() -> None:
    with _LOCK:
        _CALIBRATION.clear()


def calibration_summary() -> dict[str, Any]:
    with _LOCK:
        return {
            "keys": len(_CALIBRATION),
            "server_fingerprint": _server_fingerprint(),
            "persisted_path": str(_calib_path()),
        }


# ---------------------------------------------------------------------------
# R39-PERF-075：shape-aware ScanCost 校准（ScanShapeKey + P50/P95）
#
# 原有 EMA 路径（record_task_actual / calibrated_factors）保留，作为 shape 信息
# 缺失时的 fallback。shape 信息可用时走 ScanShapeCalibrator（有界样本窗口 +
# P50/P95）。
# ---------------------------------------------------------------------------

from runtime.scan_shape import (  # noqa: E402
    ScanShapeCalibrator,
    ScanShapeKey,
    reconstruct_scan_shape_key,
)

_scan_shape_calibrator: ScanShapeCalibrator | None = None


def scan_shape_calibrator() -> ScanShapeCalibrator:
    """进程级 shape-aware ScanCost 校准器（懒构造单例）。"""
    global _scan_shape_calibrator
    if _scan_shape_calibrator is None:
        _scan_shape_calibrator = ScanShapeCalibrator()
    return _scan_shape_calibrator


def reset_scan_shape_calibration() -> None:
    """清空 shape-aware 校准样本（测试用）。"""
    global _scan_shape_calibrator
    if _scan_shape_calibrator is not None:
        _scan_shape_calibrator.reset()
        _scan_shape_calibrator = None


def record_scan_shape_actual(
    *,
    shape: ScanShapeKey,
    open_ms: float = 0.0,
    scan_ms: float = 0.0,
    decode_ms: float = 0.0,
    rows: int = 0,
    bytes_: int = 0,
) -> None:
    """记录一次 shape-aware ScanCost 实际观测（open/scan/decode/rows/bytes）。"""
    scan_shape_calibrator().record(
        shape, open_ms=open_ms, scan_ms=scan_ms, decode_ms=decode_ms, rows=rows, bytes_=bytes_
    )


def scan_shape_summary(shape: ScanShapeKey) -> dict[str, Any]:
    """某 shape 的 P50/P95 校准汇总（无样本返回 {}）。"""
    return scan_shape_calibrator().summary(shape)
