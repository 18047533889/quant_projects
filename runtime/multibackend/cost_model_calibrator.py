# -*- coding: utf-8 -*-
"""MB-P1-017: Cost model calibration with real observations.

真实执行观测 → 成本模型校准（闭环优化）：
    - 预测成本 vs 实际成本的误差统计
    - 按完整身份 key 分层校准（R21-COST-CALIBRATION-KEY）
    - P50/P90/P99 分位数模型（处理长尾）
    - 自适应调整：持续学习，模型随负载特征演进

R21-COST-CALIBRATION-KEY（本模块本次修复）：
    - 校准 key 从 ``(backend, operator, shape)`` 升级为**完整身份** key：
      PI-ID / build / CPU model / cores / threads / backend version /
      representation / dtype / market + (backend, operator, shape)。
    - 持久化校准时间戳为 **UTC wall-clock**（不再是 monotonic —— monotonic
      是 boot 相对时间，写入持久化文件在跨进程/跨重启后无意义）。
    - 持久化采用 **原子写**（tmp + os.replace）+ **进程锁**（O_EXCL lock file）。
    - **research / production 校准命名空间隔离**（按 run mode 分目录），
      research 校准数据绝不污染 production 成本模型。

校准数据持久化（跨会话复用）→ 冷启动后快速收敛。
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import platform
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

#: R21-COST-CALIBRATION-KEY: 校准 schema 版本。key 从 (backend, operator, shape)
#: 升级为完整身份后升版 —— 旧 schema（monotonic 时间戳 / 三元组 key）文件不再加载。
CALIBRATION_SCHEMA_VERSION = 2

#: 每 shape 保留的最大原始观测数（避免无限增长）。
_MAX_OBSERVATIONS_PER_KEY = 1000


@dataclass(frozen=True)
class CalibrationKey:
    """完整身份的校准 key（R21-COST-CALIBRATION-KEY）。

    在 ``(backend, operator, shape_key)`` 之外绑定 9 个身份维度：
    PI-ID / build / CPU model / cores / threads / backend version /
    representation / dtype / market —— 校准数据绝不跨这些维度复用。

    设计动机：不同 PI-ID（实现换 kernel）、不同 build（代码升级）、不同机器
    （CPU model / cores / threads）、不同 backend 版本、不同 representation /
    dtype / market 下测得的校准因子不能互相套用。
    """

    pi_id: str
    build: str
    cpu_model: str
    cores: int
    threads: int
    backend_version: str
    representation: str
    dtype: str
    market: str
    backend: str
    operator: str
    shape_key: str

    def to_key(self) -> str:
        """稳定 canonical 字符串（可直接作持久化 key / dict key）。"""
        return "|".join([
            str(self.pi_id),
            str(self.build),
            str(self.cpu_model),
            str(int(self.cores)),
            str(int(self.threads)),
            str(self.backend_version),
            str(self.representation),
            str(self.dtype),
            str(self.market),
            str(self.backend),
            str(self.operator),
            str(self.shape_key),
        ])

    def digest(self) -> str:
        """完整 256-bit SHA-256（避免长 key 拼接歧义 / 碰撞）。"""
        return hashlib.sha256(self.to_key().encode("utf-8")).hexdigest()

    def identity(self) -> dict[str, Any]:
        """9 个身份维度（backend/operator/shape 之外的部分）。"""
        return {
            "pi_id": self.pi_id,
            "build": self.build,
            "cpu_model": self.cpu_model,
            "cores": self.cores,
            "threads": self.threads,
            "backend_version": self.backend_version,
            "representation": self.representation,
            "dtype": self.dtype,
            "market": self.market,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.identity(),
            "backend": self.backend,
            "operator": self.operator,
            "shape_key": self.shape_key,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CalibrationKey":
        return cls(
            pi_id=str(d.get("pi_id", "")),
            build=str(d.get("build", "")),
            cpu_model=str(d.get("cpu_model", "")),
            cores=int(d.get("cores", 0) or 0),
            threads=int(d.get("threads", 0) or 0),
            backend_version=str(d.get("backend_version", "")),
            representation=str(d.get("representation", "")),
            dtype=str(d.get("dtype", "")),
            market=str(d.get("market", "")),
            backend=str(d.get("backend", "")),
            operator=str(d.get("operator", "")),
            shape_key=str(d.get("shape_key", "")),
        )


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


# ---------------------------------------------------------------------------
# 身份维度解析（R21-COST-CALIBRATION-KEY）
# ---------------------------------------------------------------------------


def _utc_now_ms() -> float:
    """UTC wall-clock 毫秒（持久化安全；不是 monotonic）。"""
    return datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000.0


def _cpu_model() -> str:
    try:
        return platform.processor() or "unknown"
    except Exception:
        return "unknown"


def _effective_cores() -> int:
    try:
        from runtime.resource_governor import effective_cpu_slots

        return max(1, effective_cpu_slots())
    except Exception:
        return max(1, os.cpu_count() or 1)


def _effective_threads() -> int:
    """实际 backend 线程数：优先读线程 env（POLARS_MAX_THREADS 等）。

    约束要求线程 env vars 在测试中设置并被 key 反映 —— 这里显式消费
    ``POLARS_MAX_THREADS`` / ``DUCKDB_THREADS`` / ``OMP_NUM_THREADS``。
    """
    for env in ("POLARS_MAX_THREADS", "DUCKDB_THREADS", "OMP_NUM_THREADS"):
        raw = os.environ.get(env, "").strip()
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                pass
    return _effective_cores()


def _backend_version(backend: str) -> str:
    """backend 运行库版本（duckdb/polars/pandas/numba/clickhouse …）。"""
    b = str(backend or "").lower()
    if "duckdb" in b:
        mod = "duckdb"
    elif "polars" in b:
        mod = "polars"
    elif "pandas" in b or b in {"numpy", "pandas_numpy"}:
        mod = "pandas"
    elif "numba" in b:
        mod = "numba"
    elif "clickhouse" in b:
        mod = "clickhouse"
    elif b in {"q", "q_kdb", "kdb", "q_kdb+", "q_kdb_plus"}:
        return "unknown"  # q/pykx 版本不在此处解析
    else:
        mod = b or "unknown"
    try:
        m = __import__(mod)
        return str(getattr(m, "__version__", "") or "unknown")
    except Exception:
        return "unknown"


def _build_identity() -> str:
    """engine build identity：``factor_engine==<version>[@<git sha>]``。"""
    pkg = "unknown"
    try:
        from importlib.metadata import version as _pkg_version

        pkg = f"factor_engine=={_pkg_version('factor_engine')}"
    except Exception:
        pass
    try:
        from runtime.lineage import resolve_git_commit_hash

        git = resolve_git_commit_hash() or ""
    except Exception:
        git = ""
    if git:
        return f"{pkg}@{git[:12]}"
    return pkg


def _market_id() -> str:
    """市场标识：``FACTOR_ENGINE_MARKET`` env 优先，否则尽力推断（ashare/us）。"""
    raw = os.environ.get("FACTOR_ENGINE_MARKET", "").strip().lower()
    if raw:
        return raw
    try:
        from storage.trading_calendar import infer_market

        return infer_market() or "unknown"
    except Exception:
        return "unknown"


def _resolve_pi_id(operator: str, backend: str) -> str:
    """解析算子/backend 的 PhysicalImplementationID（best-effort，失败 "unknown"）。"""
    try:
        from cleaned_operators.registry import OperatorRegistry
        from backend.operator_capability import _declared_physical_spec

        impl = OperatorRegistry.get(str(operator), str(backend), mode="any")
        if impl is None:
            return "unknown"
        spec = _declared_physical_spec(impl, str(backend))
        if spec is None:
            return "unknown"
        pid = spec.physical_implementation_id
        return str(pid) if pid is not None else "unknown"
    except Exception:
        return "unknown"


@lru_cache(maxsize=1)
def _static_machine_identity() -> tuple[str, int, str]:
    """进程内静态的机器/build 身份（CPU model, cores, build）。"""
    return (_cpu_model(), _effective_cores(), _build_identity())


def _build_calibration_key(
    *,
    backend: str,
    operator: str,
    shape_key: str,
    pi_id: str = "",
    representation: str = "",
    dtype: str = "",
    market: str = "",
) -> CalibrationKey:
    """组装完整身份 CalibrationKey。

    显式传入的身份维度优先；缺省时自动解析。自动解析维度中 CPU model / cores /
    build 是进程静态值（lru_cache），threads / market / pi_id 每次新鲜取值，
    保证 env 变化（测试里设置线程 env）能被 key 反映。
    """
    cpu_model, cores, build = _static_machine_identity()
    return CalibrationKey(
        pi_id=pi_id or _resolve_pi_id(operator, backend),
        build=build,
        cpu_model=cpu_model,
        cores=cores,
        threads=_effective_threads(),
        backend_version=_backend_version(backend),
        representation=representation or "unknown",
        dtype=dtype or "unknown",
        market=market or _market_id(),
        backend=str(backend).lower(),
        operator=str(operator),
        shape_key=str(shape_key),
    )


# ---------------------------------------------------------------------------
# 进程锁（R21-COST-CALIBRATION-KEY：原子写 + 进程锁）
# ---------------------------------------------------------------------------


def _acquire_process_lock(lock_path: Path, timeout_s: float = 5.0) -> bool:
    """跨进程互斥锁（O_EXCL 独占创建）。

    返回是否成功获得锁；超时返回 False（calibration 是性能数据，不阻塞主流程）。
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            os.close(fd)
            return True
        except FileExistsError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.05)


def default_calibration_path() -> str:
    """Run-mode 命名空间的默认校准路径（research/production 隔离）。

    research 校准数据绝不与 production 共用同一文件 —— 不同 run mode 的
    calibration 样本不可互相复用。
    """
    run_mode = "research"
    try:
        from runtime.production_policy import resolve_run_mode

        run_mode = resolve_run_mode()
    except Exception:
        pass
    cache_dir = os.environ.get("FACTOR_ENGINE_CACHE_DIR", "").strip()
    base = Path(cache_dir) if cache_dir else Path(os.path.expanduser("~/.cache/factor_engine"))
    return str(base / "calibration" / run_mode / "cost_calibration.json")


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

        # CalibrationKey → list[CostObservation]
        self._observations: dict[CalibrationKey, list[CostObservation]] = {}
        # CalibrationKey → CalibratedCostModel
        self._models: dict[CalibrationKey, CalibratedCostModel] = {}
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
        *,
        pi_id: str = "",
        representation: str = "",
        dtype: str = "",
        market: str = "",
    ) -> None:
        """记录一次成本观测（预测 vs 实际）。

        R21-COST-CALIBRATION-KEY：观测按完整身份 key 归档 —— 不同身份维度
        （PI-ID / build / 机器 / backend 版本 / representation / dtype / market）
        的观测互不共享校准模型。

        Args:
            backend: 执行 backend
            operator: 算子名称
            shape_key: 资源 shape 签名
            predicted_ms: 预测耗时（毫秒）
            actual_ms: 实际耗时（毫秒）
            predicted_bytes: 预测内存峰值
            actual_bytes: 实际内存峰值
            pi_id: 物理实现 ID（缺省自动解析）
            representation: 数据表示（缺省 "unknown"）
            dtype: 数据类型（缺省 "unknown"）
            market: 市场标识（缺省自动解析 ashare/us）
        """
        key = _build_calibration_key(
            backend=backend,
            operator=operator,
            shape_key=shape_key,
            pi_id=pi_id,
            representation=representation,
            dtype=dtype,
            market=market,
        )
        obs = CostObservation(
            backend=backend,
            operator=operator,
            shape_key=shape_key,
            predicted_ms=predicted_ms,
            actual_ms=actual_ms,
            predicted_bytes=predicted_bytes,
            actual_bytes=actual_bytes,
            # R21: 观测时间戳用 UTC wall-clock（不是 monotonic）。
            timestamp_ms=_utc_now_ms(),
        )

        with self._lock:
            if key not in self._observations:
                self._observations[key] = []
            self._observations[key].append(obs)

            # 保留最近 _MAX_OBSERVATIONS_PER_KEY 次（避免无限增长）
            if len(self._observations[key]) > _MAX_OBSERVATIONS_PER_KEY:
                self._observations[key] = self._observations[key][-_MAX_OBSERVATIONS_PER_KEY:]

        # 定期更新模型 —— 节流用 monotonic（仅进程内，不持久化）。
        now_mono_ms = time.monotonic() * 1000.0
        if now_mono_ms - self._last_update_ms >= self.update_interval_s * 1000.0:
            self._update_models()
            self._last_update_ms = now_mono_ms

    def _update_models(self) -> None:
        """更新校准模型（从观测计算调整因子）。

        R21: 持久化的 ``last_updated_ms`` 使用 UTC wall-clock（不是 monotonic）。
        """
        now_wall_ms = _utc_now_ms()

        with self._lock:
            for key, observations in self._observations.items():
                if len(observations) < self.min_samples:
                    continue

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
                    model.last_updated_ms = now_wall_ms
                else:
                    self._models[key] = CalibratedCostModel(
                        backend=key.backend,
                        operator=key.operator,
                        shape_key=key.shape_key,
                        p50_adjustment=p50,
                        p90_adjustment=p90,
                        p99_adjustment=p99,
                        sample_count=len(observations),
                        last_updated_ms=now_wall_ms,
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
        pi_id: str = "",
        representation: str = "",
        dtype: str = "",
        market: str = "",
    ) -> float:
        """返回校准后的成本估算（查询模型）。

        Args:
            backend: 执行 backend
            operator: 算子名称
            shape_key: 资源 shape 签名
            predicted_ms: 原始预测成本
            percentile: 使用哪个分位数模型（"p50" | "p90" | "p99"）
            pi_id: 物理实现 ID（缺省自动解析）
            representation: 数据表示（缺省 "unknown"）
            dtype: 数据类型（缺省 "unknown"）
            market: 市场标识（缺省自动解析）

        Returns:
            校准后的成本（毫秒）
        """
        key = _build_calibration_key(
            backend=backend,
            operator=operator,
            shape_key=shape_key,
            pi_id=pi_id,
            representation=representation,
            dtype=dtype,
            market=market,
        )

        with self._lock:
            model = self._models.get(key)
            if model is None or model.sample_count < self.min_samples:
                # 无校准数据：返回原始预测
                return predicted_ms

            return model.adjusted_cost(predicted_ms, percentile)

    def get_model(
        self,
        backend: str,
        operator: str,
        shape_key: str,
        *,
        pi_id: str = "",
        representation: str = "",
        dtype: str = "",
        market: str = "",
    ) -> CalibratedCostModel | None:
        """返回指定完整身份 key 的校准模型（诊断用）。"""
        key = _build_calibration_key(
            backend=backend,
            operator=operator,
            shape_key=shape_key,
            pi_id=pi_id,
            representation=representation,
            dtype=dtype,
            market=market,
        )
        with self._lock:
            return self._models.get(key)

    def _save_to_disk(self) -> None:
        """持久化校准数据到磁盘（JSON，原子写 + 进程锁）。"""
        if not self.calibration_file:
            return

        try:
            path = Path(self.calibration_file)
            path.parent.mkdir(parents=True, exist_ok=True)

            with self._lock:
                data = {
                    "schema_version": CALIBRATION_SCHEMA_VERSION,
                    "models": {
                        k.to_key(): {**m.to_dict(), "calibration_key": k.to_dict()}
                        for k, m in self._models.items()
                    },
                }

            # 进程锁：避免多进程并发写互相覆盖/损坏。
            lock_path = path.with_suffix(path.suffix + ".lock")
            if not _acquire_process_lock(lock_path):
                _logger.debug("calibration lock busy; skipping save to %s", path)
                return
            try:
                # 原子写：先写同目录 .tmp，fsync 后 os.replace 覆盖。
                tmp_path = path.with_suffix(path.suffix + ".tmp")
                with open(tmp_path, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_path, path)
                _logger.debug("calibration data saved to %s", path)
            finally:
                try:
                    os.remove(lock_path)
                except OSError:
                    pass
        except Exception as exc:
            _logger.warning("failed to save calibration data: %s", exc)

    def _load_from_disk(self) -> None:
        """从磁盘加载校准数据（仅接受当前 schema 版本）。"""
        if not self.calibration_file or not os.path.exists(self.calibration_file):
            return

        try:
            with open(self.calibration_file, encoding="utf-8") as f:
                data = json.load(f)

            schema_version = int(data.get("schema_version", 0) or 0)
            if schema_version != CALIBRATION_SCHEMA_VERSION:
                _logger.info(
                    "calibration file %s schema v%s != current v%s; skipping",
                    self.calibration_file, schema_version, CALIBRATION_SCHEMA_VERSION,
                )
                return

            with self._lock:
                for key_str, model_dict in data.get("models", {}).items():
                    key_payload = model_dict.get("calibration_key")
                    if not isinstance(key_payload, dict):
                        continue
                    key = CalibrationKey.from_dict(key_payload)
                    if key.to_key() != key_str:
                        continue  # 完整性校验：文件 key 必须与 payload 一致
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
    """返回全局 CostModelCalibrator（进程级单例）。

    R21-COST-CALIBRATION-KEY：默认持久化路径按 run mode 命名空间隔离
    （research / production 不共享同一校准文件）。
    """
    global _global_calibrator
    if _global_calibrator is None:
        with _global_lock:
            if _global_calibrator is None:
                _global_calibrator = CostModelCalibrator(
                    calibration_file=default_calibration_path()
                )
    return _global_calibrator


def reset_global_cost_calibrator() -> None:
    """清空全局单例（测试用）。"""
    global _global_calibrator
    with _global_lock:
        _global_calibrator = None
