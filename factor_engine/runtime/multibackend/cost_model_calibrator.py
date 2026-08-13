# -*- coding: utf-8 -*-
"""MB-P1-017: Cost model calibration with real observations.

真实执行观测 → 成本模型校准（闭环优化）：
    - 预测成本 vs 实际成本的误差统计
    - 按 (backend, operator, shape) 分层校准
    - P50/P90/P99 分位数模型（处理长尾）
    - 自适应调整：持续学习，模型随负载特征演进

校准数据持久化（跨会话复用）→ 冷启动后快速收敛。
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


@dataclass
class CostObservation:
    """单次成本观测（预测 vs 实际）。"""

    backend: str
    operator: str
    shape_key: str  # 资源 shape 签名（rows/cols/window）
    predicted_ms: float
    actual_ms: float
    predicted_bytes: int
    actual_bytes: int
    timestamp_ms: float

    @property
    def error_ratio(self) -> float:
        """误差比例（actual / predicted）。"""
        if self.predicted_ms <= 0:
            return 1.0
        return self.actual_ms / self.predicted_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "operator": self.operator,
            "shape_key": self.shape_key,
            "predicted_ms": round(self.predicted_ms, 2),
            "actual_ms": round(self.actual_ms, 2),
            "predicted_bytes": self.predicted_bytes,
            "actual_bytes": self.actual_bytes,
            "timestamp_ms": round(self.timestamp_ms, 2),
            "error_ratio": round(self.error_ratio, 3),
        }


@dataclass
class CalibratedCostModel:
    """校准后的成本模型（分位数 + 调整因子）。"""

    backend: str
    operator: str
    shape_key: str
    p50_adjustment: float = 1.0  # 中位数调整因子
    p90_adjustment: float = 1.0  # P90 调整因子
    p99_adjustment: float = 1.0  # P99 调整因子
    sample_count: int = 0
    last_updated_ms: float = 0.0

    def adjusted_cost(self, predicted_ms: float, percentile: str = "p50") -> float:
        """返回校准后的成本估算。

        Args:
            predicted_ms: 原始预测成本
            percentile: 使用哪个分位数模型（"p50" | "p90" | "p99"）

        Returns:
            校准后的成本（毫秒）
        """
        factor = {
            "p50": self.p50_adjustment,
            "p90": self.p90_adjustment,
            "p99": self.p99_adjustment,
        }.get(percentile, 1.0)
        return predicted_ms * factor

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "operator": self.operator,
            "shape_key": self.shape_key,
            "p50_adjustment": round(self.p50_adjustment, 4),
            "p90_adjustment": round(self.p90_adjustment, 4),
            "p99_adjustment": round(self.p99_adjustment, 4),
            "sample_count": self.sample_count,
            "last_updated_ms": round(self.last_updated_ms, 2),
        }


class CostModelCalibrator:
    """成本模型校准器（真实观测 → 模型更新）。

    集成点：
        - AdaptiveBatchScheduler._record_task_calibration() 记录观测
        - PhysicalLowerer.estimate_cost() 查询校准后的模型
        - ResourceCalibrationStore 持久化校准数据
    """

    def __init__(
        self,
        *,
        calibration_file: str | None = None,
        min_samples: int = 10,
        update_interval_s: float = 300.0,
    ) -> None:
        """
        Args:
            calibration_file: 校准数据持久化路径（None 则仅内存）
            min_samples: 最少样本数（达到后才生效校准）
            update_interval_s: 模型更新间隔（秒）
        """
        self.calibration_file = calibration_file
        self.min_samples = min_samples
        self.update_interval_s = update_interval_s

        # (backend, operator, shape_key) → list[CostObservation]
        self._observations: dict[tuple[str, str, str], list[CostObservation]] = {}
        # (backend, operator, shape_key) → CalibratedCostModel
        self._models: dict[tuple[str, str, str], CalibratedCostModel] = {}
        self._lock = threading.RLock()
        self._last_update_ms = 0.0

        # 加载持久化数据
        if calibration_file:
            self._load_from_disk()

    def record_observation(
        self,
        backend: str,
        operator: str,
        shape_key: str,
        predicted_ms: float,
        actual_ms: float,
        predicted_bytes: int = 0,
        actual_bytes: int = 0,
    ) -> None:
        """记录一次成本观测（预测 vs 实际）。

        Args:
            backend: 执行 backend
            operator: 算子名称
            shape_key: 资源 shape 签名
            predicted_ms: 预测耗时（毫秒）
            actual_ms: 实际耗时（毫秒）
            predicted_bytes: 预测内存峰值
            actual_bytes: 实际内存峰值
        """
        import time

        key = (backend.lower(), operator, shape_key)
        obs = CostObservation(
            backend=backend,
            operator=operator,
            shape_key=shape_key,
            predicted_ms=predicted_ms,
            actual_ms=actual_ms,
            predicted_bytes=predicted_bytes,
            actual_bytes=actual_bytes,
            timestamp_ms=time.monotonic() * 1000.0,
        )

        with self._lock:
            if key not in self._observations:
                self._observations[key] = []
            self._observations[key].append(obs)

            # 保留最近 1000 次（避免无限增长）
            if len(self._observations[key]) > 1000:
                self._observations[key] = self._observations[key][-1000:]

        # 定期更新模型
        now_ms = time.monotonic() * 1000.0
        if now_ms - self._last_update_ms >= self.update_interval_s * 1000.0:
            self._update_models()
            self._last_update_ms = now_ms

    def _update_models(self) -> None:
        """更新校准模型（从观测计算调整因子）。"""
        import time

        now_ms = time.monotonic() * 1000.0

        with self._lock:
            for key, observations in self._observations.items():
                if len(observations) < self.min_samples:
                    continue

                backend, operator, shape_key = key
                error_ratios = [obs.error_ratio for obs in observations]
                error_ratios.sort()

                # 计算分位数调整因子
                n = len(error_ratios)
                p50 = error_ratios[int(n * 0.50)]
                p90 = error_ratios[int(n * 0.90)]
                p99 = error_ratios[int(n * 0.99)] if n >= 100 else p90

                # 更新或创建模型
                if key in self._models:
                    model = self._models[key]
                    # 指数移动平均（新观测 20% 权重）
                    alpha = 0.2
                    model.p50_adjustment = model.p50_adjustment * (1 - alpha) + p50 * alpha
                    model.p90_adjustment = model.p90_adjustment * (1 - alpha) + p90 * alpha
                    model.p99_adjustment = model.p99_adjustment * (1 - alpha) + p99 * alpha
                    model.sample_count = len(observations)
                    model.last_updated_ms = now_ms
                else:
                    self._models[key] = CalibratedCostModel(
                        backend=backend,
                        operator=operator,
                        shape_key=shape_key,
                        p50_adjustment=p50,
                        p90_adjustment=p90,
                        p99_adjustment=p99,
                        sample_count=len(observations),
                        last_updated_ms=now_ms,
                    )

        # 持久化到磁盘
        if self.calibration_file:
            self._save_to_disk()

    def get_calibrated_cost(
        self,
        backend: str,
        operator: str,
        shape_key: str,
        predicted_ms: float,
        *,
        percentile: str = "p90",
    ) -> float:
        """返回校准后的成本估算（查询模型）。

        Args:
            backend: 执行 backend
            operator: 算子名称
            shape_key: 资源 shape 签名
            predicted_ms: 原始预测成本
            percentile: 使用哪个分位数模型（"p50" | "p90" | "p99"）

        Returns:
            校准后的成本（毫秒）
        """
        key = (backend.lower(), operator, shape_key)

        with self._lock:
            model = self._models.get(key)
            if model is None or model.sample_count < self.min_samples:
                # 无校准数据：返回原始预测
                return predicted_ms

            return model.adjusted_cost(predicted_ms, percentile)

    def get_model(
        self, backend: str, operator: str, shape_key: str
    ) -> CalibratedCostModel | None:
        """返回指定 key 的校准模型（诊断用）。"""
        key = (backend.lower(), operator, shape_key)
        with self._lock:
            return self._models.get(key)

    def _save_to_disk(self) -> None:
        """持久化校准数据到磁盘（JSON）。"""
        if not self.calibration_file:
            return

        try:
            path = Path(self.calibration_file)
            path.parent.mkdir(parents=True, exist_ok=True)

            with self._lock:
                data = {
                    "models": {
                        f"{k[0]}:{k[1]}:{k[2]}": m.to_dict()
                        for k, m in self._models.items()
                    },
                }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            _logger.debug("calibration data saved to %s", path)
        except Exception as exc:
            _logger.warning("failed to save calibration data: %s", exc)

    def _load_from_disk(self) -> None:
        """从磁盘加载校准数据。"""
        if not self.calibration_file or not os.path.exists(self.calibration_file):
            return

        try:
            with open(self.calibration_file, encoding="utf-8") as f:
                data = json.load(f)

            with self._lock:
                for key_str, model_dict in data.get("models", {}).items():
                    backend, operator, shape_key = key_str.split(":", 2)
                    key = (backend, operator, shape_key)
                    self._models[key] = CalibratedCostModel(
                        backend=model_dict["backend"],
                        operator=model_dict["operator"],
                        shape_key=model_dict["shape_key"],
                        p50_adjustment=model_dict["p50_adjustment"],
                        p90_adjustment=model_dict["p90_adjustment"],
                        p99_adjustment=model_dict["p99_adjustment"],
                        sample_count=model_dict["sample_count"],
                        last_updated_ms=model_dict["last_updated_ms"],
                    )

            _logger.info(
                "loaded %d calibrated models from %s",
                len(self._models), self.calibration_file,
            )
        except Exception as exc:
            _logger.warning("failed to load calibration data: %s", exc)

    def summary(self) -> dict[str, Any]:
        """返回校准器状态摘要。"""
        with self._lock:
            return {
                "total_observations": sum(len(v) for v in self._observations.values()),
                "calibrated_models": len(self._models),
                "min_samples": self.min_samples,
                "calibration_file": self.calibration_file,
            }


# 全局单例
_global_calibrator: CostModelCalibrator | None = None
_global_lock = threading.Lock()


def get_global_cost_calibrator() -> CostModelCalibrator:
    """返回全局 CostModelCalibrator（进程级单例）。"""
    global _global_calibrator
    if _global_calibrator is None:
        with _global_lock:
            if _global_calibrator is None:
                # 默认持久化到 ~/.cache/factor_engine/cost_calibration.json
                cache_dir = os.path.expanduser("~/.cache/factor_engine")
                calibration_file = os.path.join(cache_dir, "cost_calibration.json")
                _global_calibrator = CostModelCalibrator(
                    calibration_file=calibration_file
                )
    return _global_calibrator
