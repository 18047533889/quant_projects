# -*- coding: utf-8 -*-
"""R36 §9/§11/§34/§209/§210/§211：ResourceCalibrationStore —— 真实资源结果反哺。

R36 P0-004/005/006：
    - 资源预测从「固定 2GB/worker」「每 worker 3GB」升级为 **per-shape 分布**
      （§9：P50/P90/P95/P99/max/sample_count）。
    - 每个 task 完成后记录 actual（§11）：elapsed / peak / output / spill →
      更新 ``ResourceCalibrationStore``（持久化到 ``resource_calibration.parquet``，
      §209）。
    - 预测反馈闭环（P0-006）：``prediction_error = actual / predicted`` 更新
      P99 correction factor（§12）；OOM 立即提高 tail safety（§211），不能被均值
      稀释；长期过度保守只能慢慢下调（§283/R31）。
    - Calibration aging（§210）：样本按指数衰减权重，旧 workload 不永久影响。

数据模型：一个 shape 一个 :class:`ShapeCalibration`。样本 capped + 按 recency
加权（EWMA 均值 + weighted quantile 尾部）。
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from runtime.resource_shape import ResourceShapeKey, hardware_fingerprint


def _utc_now_ms() -> float:
    """R21-COST-CALIBRATION-KEY：UTC wall-clock 毫秒（持久化校准时间戳）。"""
    return datetime.now(timezone.utc).timestamp() * 1000.0


class _CalibrationAtomicWrite:
    """R21-COST-CALIBRATION-KEY：原子写 + 进程锁。

    - 进程锁：O_EXCL 独占 lock file（跨进程互斥，防止并发 save 互相覆盖）。
    - 原子写：先写同目录 ``.tmp``，fsync 后 ``os.replace`` 覆盖目标文件。
    """

    def __init__(self, target: str, lock_timeout_s: float = 5.0) -> None:
        self._target = Path(target)
        self._lock_path = self._target.with_suffix(self._target.suffix + ".lock")
        self._lock_timeout_s = lock_timeout_s
        self._tmp_path = self._target.with_suffix(self._target.suffix + ".tmp")
        self._held = False

    def __enter__(self) -> Path:
        self._target.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self._lock_timeout_s
        while True:
            try:
                fd = os.open(str(self._lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode("utf-8"))
                os.close(fd)
                self._held = True
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"calibration lock busy after {self._lock_timeout_s}s: "
                        f"{self._lock_path}"
                    )
                time.sleep(0.05)
        return self._tmp_path

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            if exc_type is None and self._tmp_path.exists():
                try:
                    with open(self._tmp_path, "rb") as fh:
                        os.fsync(fh.fileno())
                except OSError:
                    pass
                os.replace(self._tmp_path, self._target)
        finally:
            for leftover in (self._tmp_path, self._lock_path):
                try:
                    os.remove(leftover)
                except OSError:
                    pass
            self._held = False


def _calibration_atomic_write_lock(target: str) -> _CalibrationAtomicWrite:
    """返回进程锁 + 原子写的上下文管理器（写校准持久化文件用）。"""
    return _CalibrationAtomicWrite(target)

#: 每 shape 保留的最大原始样本数（超出做 recency 抽样）。
MAX_SAMPLES_PER_SHAPE = 256
#: R21-COST-CALIBRATION-KEY：持久化校准时间戳是 UTC wall-clock（不是
#: monotonic）。monotonic 是 boot 相对时间，写入持久化文件在跨进程/跨重启后
#: 无意义 —— 校准新鲜度判断必须以墙上时钟为准。
#: 指数衰减半衰期（样本数）：约 N 个样本后权重减半。
AGING_HALFLIFE_SAMPLES = 128
#: correction factor bounds（§311 safety floor：learned model 不能把安全系数
#: 降到配置下限以下）。
CORRECTION_MIN = 1.0
CORRECTION_MAX = 2.5
_UNDERPREDICT_STREAK_TRIGGER = 3
_OVERPREDICT_STREAK_TRIGGER = 5
#: R38 P0-010（§6）：calibration schema 版本。R36 时代用「预测值当真实值」污染了
#: memory_obs —— 升版后旧 schema 数据不进入新 P99（只作历史诊断）。
CALIBRATION_SCHEMA_VERSION = 2
#: 本实现写入的 ``calibration_source_version``（记录在每行）。
CALIBRATION_SOURCE_VERSION = f"fe-r38-schema-v{CALIBRATION_SCHEMA_VERSION}"


@dataclass
class ShapeCalibration:
    """一个 resource shape 的学习结果（§9）。"""

    shape_key: ResourceShapeKey
    memory_obs: list[float] = field(default_factory=list)
    elapsed_obs: list[float] = field(default_factory=list)
    output_obs: list[float] = field(default_factory=list)
    spill_obs: list[float] = field(default_factory=list)
    max_obs: int = 0
    correction_factor: float = 1.0
    underpredict_streak: int = 0
    overpredict_streak: int = 0
    oom_events: int = 0
    near_oom_events: int = 0
    updated_at_ms: float = 0.0

    def record(
        self,
        *,
        elapsed_ms: float,
        peak_mem: int,
        output_bytes: int = 0,
        spill_bytes: int = 0,
        near_oom: bool = False,
        oom: bool = False,
        attribution_quality: str = "unattributed",
        peak_is_trusted: bool | None = None,
    ) -> None:
        """记录一次观测并做 P99 correction 反馈（P0-006）。

        R38 P0-008（§6）：``peak_mem`` 只有在 ``attribution_quality`` 可信时
        （isolated / low-concurrency / concurrent-marginal）才进入 ``memory_obs``
        与 P99 主模型；``unattributed``（拿预测值当真实值）只记录 elapsed/output，
        峰值内存被拒——避免「模型学自己的预测」的污染闭环。
        """
        # P0-006：prediction_error = actual / predicted——用**加入本次之前的** P99
        # 比较，否则本次样本会成为自己的尾部，ratio 恒≈1，低估永远不触发。
        predicted_before = self.predict_memory_p99()

        from runtime.task_run_observation import P99_TRUSTED_ATTRIBUTION

        trusted = (
            peak_is_trusted
            if peak_is_trusted is not None
            else attribution_quality in P99_TRUSTED_ATTRIBUTION
        )
        # R38 P0-008：可信观测才进内存分布（不被预测值污染）。
        if trusted:
            self.memory_obs.append(float(peak_mem))
            self.max_obs = max(self.max_obs, int(peak_mem))
        self.elapsed_obs.append(float(elapsed_ms))
        self.output_obs.append(float(output_bytes))
        self.spill_obs.append(float(spill_bytes))
        # R21-COST-CALIBRATION-KEY：持久化校准时间戳是 UTC wall-clock。
        self.updated_at_ms = _utc_now_ms()
        self._cap_samples()

        # OOM 不能等均值稀释（§211）：立即提高 tail safety（与 attribution 无关）。
        if oom:
            self.oom_events += 1
            self.correction_factor = min(CORRECTION_MAX, self.correction_factor * 1.30)
            self.underpredict_streak = 0
            self.overpredict_streak = 0
            return
        if near_oom:
            self.near_oom_events += 1
            self.correction_factor = min(CORRECTION_MAX, self.correction_factor * 1.15)

        # prediction_error 反馈：低估 streak → 提高；长期高估 → 慢慢下调（§12/283）。
        # 只有可信观测参与低估判定（预测值对比预测值没有意义）。
        if trusted and predicted_before is not None and predicted_before > 0:
            ratio = peak_mem / predicted_before
            if ratio > 1.0:
                self.underpredict_streak += 1
                self.overpredict_streak = 0
                if self.underpredict_streak >= _UNDERPREDICT_STREAK_TRIGGER:
                    self.correction_factor = min(
                        CORRECTION_MAX, self.correction_factor * 1.15
                    )
                    self.underpredict_streak = 0
            else:
                self.overpredict_streak += 1
                self.underpredict_streak = 0
                if self.overpredict_streak >= _OVERPREDICT_STREAK_TRIGGER:
                    self.correction_factor = max(
                        CORRECTION_MIN, self.correction_factor * 0.98
                    )
                    self.overpredict_streak = 0

    def _cap_samples(self) -> None:
        """样本超出上限时做 recency 抽样（保留近半，抽稀旧半）。"""
        if len(self.memory_obs) <= MAX_SAMPLES_PER_SHAPE:
            return
        keep = MAX_SAMPLES_PER_SHAPE
        self.memory_obs = self.memory_obs[-keep:]
        self.elapsed_obs = self.elapsed_obs[-keep:]
        self.output_obs = self.output_obs[-keep:]
        self.spill_obs = self.spill_obs[-keep:]

    def _recency_weights(self, n: int) -> list[float]:
        """§210：指数衰减权重——越近的样本权重越大。"""
        return [math.exp(-math.log(2) * (n - 1 - i) / AGING_HALFLIFE_SAMPLES)
                for i in range(n)]

    def _weighted_quantile(self, values: list[float], q: float) -> float:
        if not values:
            return 0.0
        n = len(values)
        if n == 1:
            return values[0]
        weights = self._recency_weights(n)
        order = sorted(range(n), key=lambda i: values[i])
        total = sum(weights)
        cum = 0.0
        for idx in order:
            cum += weights[idx] / total
            if cum >= q:
                return values[idx]
        return values[order[-1]]

    def predict_memory_p99(self) -> float | None:
        """P99 峰值内存（× correction factor，P0-006）。无样本返回 None。"""
        if not self.memory_obs:
            return None
        p99 = self._weighted_quantile(self.memory_obs, 0.99)
        return p99 * self.correction_factor

    def predict(self) -> dict[str, float | int | None]:
        """完整预测：P50/P90/P95/P99/max/sample_count（§9）。"""
        if not self.memory_obs:
            return {
                "memory_p50": None, "memory_p90": None, "memory_p95": None,
                "memory_p99": None, "max_observed": self.max_obs,
                "sample_count": 0,
                "elapsed_sample_count": len(self.elapsed_obs),
                "output_sample_count": len(self.output_obs),
                "elapsed_p50": self._weighted_quantile(self.elapsed_obs, 0.50) if self.elapsed_obs else None,
                "correction_factor": self.correction_factor,
                "oom_events": self.oom_events,
                "near_oom_events": self.near_oom_events,
            }
        return {
            "memory_p50": self._weighted_quantile(self.memory_obs, 0.50),
            "memory_p90": self._weighted_quantile(self.memory_obs, 0.90),
            "memory_p95": self._weighted_quantile(self.memory_obs, 0.95),
            "memory_p99": self.predict_memory_p99(),
            "max_observed": self.max_obs,
            "sample_count": len(self.memory_obs),
            # R38 P0-008：elapsed/output 与 memory 分离统计——unattributed 观测
            # 不进入 memory P99，但仍贡献真实的 duration/output 分布。
            "elapsed_sample_count": len(self.elapsed_obs),
            "output_sample_count": len(self.output_obs),
            "elapsed_p50": self._weighted_quantile(self.elapsed_obs, 0.50) if self.elapsed_obs else None,
            "correction_factor": round(self.correction_factor, 3),
            "oom_events": self.oom_events,
            "near_oom_events": self.near_oom_events,
        }


class ResourceCalibrationStore:
    """进程级 calibration store（§209 持久化到 ``resource_calibration.parquet``）。

    key = hardware fingerprint（§34：同型号服务器共享校准）+ ResourceShapeKey。
    """

    def __init__(self, *, path: str | os.PathLike | None = None) -> None:
        self._lock = threading.RLock()
        self._shapes: dict[tuple[tuple, tuple], ShapeCalibration] = {}
        self._hw = hardware_fingerprint()
        self._path: str | None = str(path) if path is not None else None
        if self._path is not None and os.path.exists(self._path):
            self._load(self._path)

    # -- record / predict --

    def _shape_key_tuple(self, key: ResourceShapeKey) -> tuple[tuple, tuple]:
        return (
            tuple(sorted(self._hw.items())),
            tuple(sorted(key.to_dict().items())),
        )

    def get(self, key: ResourceShapeKey) -> ShapeCalibration | None:
        with self._lock:
            return self._shapes.get(self._shape_key_tuple(key))

    def record(
        self,
        key: ResourceShapeKey,
        *,
        elapsed_ms: float = 0.0,
        peak_mem: int = 0,
        output_bytes: int = 0,
        spill_bytes: int = 0,
        near_oom: bool = False,
        oom: bool = False,
        attribution_quality: str = "unattributed",
        peak_is_trusted: bool | None = None,
    ) -> None:
        """记录一次真实任务观测（§11）。R38 P0-008：峰值内存只在 attribution
        可信时进入 P99 主模型。"""
        with self._lock:
            sk = self._shape_key_tuple(key)
            cal = self._shapes.get(sk)
            if cal is None:
                cal = ShapeCalibration(shape_key=key)
                self._shapes[sk] = cal
            cal.record(
                elapsed_ms=elapsed_ms,
                peak_mem=peak_mem,
                output_bytes=output_bytes,
                spill_bytes=spill_bytes,
                near_oom=near_oom,
                oom=oom,
                attribution_quality=attribution_quality,
                peak_is_trusted=peak_is_trusted,
            )

    def predict(self, key: ResourceShapeKey) -> dict[str, float | int | None] | None:
        """shape 的预测（P50..P99）；cold shape 返回 None（走 static × uncertainty）。"""
        with self._lock:
            cal = self._shapes.get(self._shape_key_tuple(key))
            if cal is None:
                return None
            return cal.predict()

    def adapt_uncertainty_feedback(
        self, *, underpredict_streak: int = 0, overpredict_streak: int = 0
    ) -> None:
        """全局 uncertainty 反馈（R27-201..203 对接）。"""
        from runtime.task_resource_contract import DEFAULT_UNCERTAINTY

        # 由 broker 的 adapt_uncertainty 保留；此处仅记录（shape 级反馈已在
        # ShapeCalibration.record 内闭环）。
        return

    # -- persistence（§209） --

    def save(self, path: str | os.PathLike | None = None) -> str:
        """持久化全部 shape calibration 到 parquet。返回实际写入路径。"""
        target = str(path or self._path)
        if target is None:
            target = os.path.join(
                os.environ.get("FACTOR_ENGINE_CACHE_DIR", "").strip() or ".",
                "resource_calibration.parquet",
            )
        rows: list[dict[str, Any]] = []
        with self._lock:
            for cal in self._shapes.values():
                rows.append({
                    "schema_version": CALIBRATION_SCHEMA_VERSION,
                    "calibration_source_version": CALIBRATION_SOURCE_VERSION,
                    "hardware_fingerprint": json.dumps(self._hw, sort_keys=True),
                    "shape_key": json.dumps(cal.shape_key.to_dict(), sort_keys=True),
                    "max_observed": cal.max_obs,
                    "correction_factor": cal.correction_factor,
                    "oom_events": cal.oom_events,
                    "near_oom_events": cal.near_oom_events,
                    "sample_count": len(cal.memory_obs),
                    "memory_p99": cal.predict_memory_p99(),
                    "memory_obs": cal.memory_obs,
                    "elapsed_obs": cal.elapsed_obs,
                    "output_obs": cal.output_obs,
                    "spill_obs": cal.spill_obs,
                    "updated_at_ms": cal.updated_at_ms,
                })
        try:
            import pandas as pd

            with _calibration_atomic_write_lock(target) as tmp_path:
                pd.DataFrame(rows).to_parquet(tmp_path, index=False)
        except Exception:
            # 无 pandas/parquet 环境降级 JSON（calibration 是性能数据，非正确性数据）。
            with _calibration_atomic_write_lock(target) as tmp_path:
                with open(tmp_path, "w", encoding="utf-8") as fh:
                    json.dump(rows, fh)
        self._path = target
        return target

    def _load(self, path: str) -> None:
        rows: list[dict[str, Any]] | None = None
        try:
            import pandas as pd

            df = pd.read_parquet(path)
            rows = [dict(r) for _, r in df.iterrows()]
        except Exception:
            pass
        if rows is None:
            try:
                with open(path, encoding="utf-8") as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, list):
                    rows = loaded
            except Exception:
                pass
        if not rows:
            return  # 无历史 calibration 时冷启动（性能数据，不 fail）
        for row in rows:
            try:
                hw = json.loads(row["hardware_fingerprint"])
                sk = ResourceShapeKey.from_dict(json.loads(row["shape_key"]))
            except Exception:
                continue
            if hw != self._hw:
                continue  # 不同硬件不共享 calibration（§34）
            # R38 P0-010：旧 schema（R36 预测值污染时代）的数据不进入新 P99，
            # 只保留作历史诊断（跳过，不加载进主 store）。
            row_schema = int(row.get("schema_version", 0) or 0)
            if row_schema < CALIBRATION_SCHEMA_VERSION:
                continue
            skt = (tuple(sorted(hw.items())), tuple(sorted(sk.to_dict().items())))
            cal = self._shapes.get(skt)
            if cal is None:
                cal = ShapeCalibration(shape_key=sk)
                self._shapes[skt] = cal
            cal.correction_factor = max(
                CORRECTION_MIN, min(CORRECTION_MAX, float(row.get("correction_factor", 1.0)))
            )
            cal.oom_events = int(row.get("oom_events", 0))
            cal.near_oom_events = int(row.get("near_oom_events", 0))
            cal.max_obs = int(row.get("max_observed", 0))
            # 恢复原始观测（§9 P50..P99 需要分布，不只是聚合）。
            # parquet 列可能是 numpy 数组，``or []`` 对数组会歧义 → 显式 None 判空。
            def _obs(raw: Any) -> list[float]:
                if raw is None:
                    return []
                try:
                    return [float(v) for v in raw][-MAX_SAMPLES_PER_SHAPE:]
                except (TypeError, ValueError):
                    return []

            cal.memory_obs = _obs(row.get("memory_obs"))
            cal.elapsed_obs = _obs(row.get("elapsed_obs"))
            cal.output_obs = _obs(row.get("output_obs"))
            cal.spill_obs = _obs(row.get("spill_obs"))

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "shape_count": len(self._shapes),
                "hardware_fingerprint": self._hw,
                "path": self._path,
                "samples": sum(len(c.memory_obs) for c in self._shapes.values()),
            }


#: 进程级单例（跨 batch 共享校准）。
_CALIBRATION_STORE: ResourceCalibrationStore | None = None
_CALIBRATION_LOCK = threading.Lock()


def global_calibration_store() -> ResourceCalibrationStore:
    global _CALIBRATION_STORE
    if _CALIBRATION_STORE is None:
        with _CALIBRATION_LOCK:
            if _CALIBRATION_STORE is None:
                path = os.environ.get("FACTOR_ENGINE_CALIBRATION_PATH", "")
                _CALIBRATION_STORE = ResourceCalibrationStore(path=path or None)
    return _CALIBRATION_STORE


def reset_calibration_store() -> None:
    global _CALIBRATION_STORE
    with _CALIBRATION_LOCK:
        _CALIBRATION_STORE = None
