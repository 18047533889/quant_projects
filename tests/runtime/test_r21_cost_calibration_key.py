# -*- coding: utf-8 -*-
"""R21-COST-CALIBRATION-KEY regression tests.

行为探针：
    - CostModelCalibrator 校准 key 从 ``(backend, operator, shape)`` 升级为完整
      身份 key —— PI-ID / build / CPU model / cores / threads / backend version /
      representation / dtype / market 任何一个维度不同 → 观测不共享校准模型。
    - 持久化校准时间戳是 UTC wall-clock（不是 monotonic）—— 写入文件的
      ``last_updated_ms`` / ``updated_at_ms`` 是墙上时钟量级（epoch 毫秒，
      ~1.7e12），不是 boot 相对单调时钟（monotonic*1000 在重启后很小）。
    - 校准持久化是原子写 + 进程锁 —— 保存后无 .tmp/.lock 残留；并发写被
      O_EXCL lock 互斥（超时 fail-closed 抛 RuntimeError）。
    - research / production 校准命名空间隔离 —— 不同 run mode 的默认路径
      不同，research 校准数据不污染 production。
"""
from __future__ import annotations

import json
import os
import tempfile
import time

import pytest

from runtime.multibackend.cost_model_calibrator import (
    CALIBRATION_SCHEMA_VERSION,
    CalibrationKey,
    CostModelCalibrator,
    _build_calibration_key,
    default_calibration_path,
)
from runtime.resource_calibration_store import (
    ResourceCalibrationStore,
    ShapeCalibration,
)
from runtime.resource_shape import ResourceShapeKey


# ---------------------------------------------------------------------------
# 1. 完整身份 key（PI-ID / build / CPU / threads / backend version / rep / dtype / market）
# ---------------------------------------------------------------------------


def _key_calibrator(tmp_path) -> CostModelCalibrator:
    return CostModelCalibrator(
        calibration_file=str(os.path.join(str(tmp_path), "cal.json")),
        min_samples=2,
        update_interval_s=0.0,
    )


def test_key_binds_pi_id() -> None:
    """不同 PI-ID → 不同校准 key（观测不共享模型）。"""
    cal = _key_calibrator(tempfile.mkdtemp())
    cal.record_observation(
        "polars", "ts_mean", "1Mx20", 100.0, 200.0,
        pi_id="pi:v3:aaa", market="ashare", representation="polars_native", dtype="float64",
    )
    key_a = _build_calibration_key(
        backend="polars", operator="ts_mean", shape_key="1Mx20",
        pi_id="pi:v3:aaa", market="ashare", representation="polars_native", dtype="float64",
    )
    key_b = _build_calibration_key(
        backend="polars", operator="ts_mean", shape_key="1Mx20",
        pi_id="pi:v3:bbb", market="ashare", representation="polars_native", dtype="float64",
    )
    assert key_a != key_b
    assert key_a in cal._observations
    assert key_b not in cal._observations


def test_key_binds_threads_env() -> None:
    """线程 env（POLARS_MAX_THREADS）变化 → 校准 key 变化。

    用户约束要求线程 env vars 被 key 反映：相同机器不同线程数测得的校准因子
    不能互相套用。先保存全部候选线程 env 的旧值，结束后逐一恢复 —— 避免
    污染同会话内后续测试（串行 pytest）。
    """
    candidates = ("POLARS_MAX_THREADS", "DUCKDB_THREADS", "OMP_NUM_THREADS")
    old = {name: os.environ.get(name) for name in candidates}
    try:
        for name in candidates:
            os.environ.pop(name, None)
        a = _build_calibration_key(
            backend="polars", operator="ts_mean", shape_key="1Mx20",
            pi_id="pi:v3:x", market="ashare",
        )
        os.environ["POLARS_MAX_THREADS"] = "2"
        b = _build_calibration_key(
            backend="polars", operator="ts_mean", shape_key="1Mx20",
            pi_id="pi:v3:x", market="ashare",
        )
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    # env 全未设时 threads=cores(8)；设置后 threads=2 → key 不同。
    assert a.threads == 8
    assert b.threads == 2
    assert a != b
    assert a.to_key() != b.to_key()


def test_key_binds_representation_dtype_market() -> None:
    """representation / dtype / market 任一不同 → 校准 key 不同。"""
    base = dict(
        backend="polars", operator="ts_mean", shape_key="1Mx20",
        pi_id="pi:v3:base", market="ashare", representation="polars_native", dtype="float64",
    )
    key_base = _build_calibration_key(**base)
    for field, value in (
        ("representation", "pandas_wide"),
        ("dtype", "float32"),
        ("market", "us"),
    ):
        changed = dict(base)
        changed[field] = value
        assert _build_calibration_key(**changed) != key_base, field


def test_key_binds_backend_version_and_cpu() -> None:
    """backend version / CPU model / cores / build 都在 key 里。"""
    key = _build_calibration_key(
        backend="duckdb_sql", operator="ts_mean", shape_key="1Mx20",
        pi_id="pi:v3:x", market="ashare",
    )
    identity = key.identity()
    assert identity["backend_version"]
    assert identity["backend_version"] != "unknown"
    assert identity["cpu_model"]
    assert identity["cores"] >= 1
    assert identity["threads"] >= 1
    assert identity["build"]
    # digest 完整 256-bit
    assert len(key.digest()) == 64


def test_key_backend_lowercased_and_stable() -> None:
    """key 稳定：相同输入 → 相同 key/digest；backend 归一化为小写。"""
    k1 = _build_calibration_key(
        backend="Polars", operator="ts_mean", shape_key="1Mx20", pi_id="pi:v3:same",
    )
    k2 = _build_calibration_key(
        backend="polars", operator="ts_mean", shape_key="1Mx20", pi_id="pi:v3:same",
    )
    assert k1 == k2
    assert k1.backend == "polars"
    assert k1.digest() == k2.digest()


# ---------------------------------------------------------------------------
# 2. 持久化时间戳是 UTC wall-clock（不是 monotonic）
# ---------------------------------------------------------------------------


def test_cost_observation_timestamp_is_wall_clock() -> None:
    """观测 timestamp_ms 是 UTC epoch 毫秒量级（~1.7e12），不是 monotonic*1000。"""
    cal = _key_calibrator(tempfile.mkdtemp())
    cal.record_observation(
        "polars", "ts_mean", "1Mx20", 100.0, 150.0,
        pi_id="pi:v3:wall", market="ashare",
    )
    obs = list(cal._observations.values())[0][0]
    assert obs.timestamp_ms > 1_000_000_000_000  # epoch ms（2023+）


def test_persisted_model_timestamp_is_wall_clock() -> None:
    """写入文件的模型 last_updated_ms 是 UTC wall-clock 量级。"""
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "cal.json")
    cal = CostModelCalibrator(calibration_file=path, min_samples=2, update_interval_s=0.0)
    cal.record_observation(
        "polars", "ts_mean", "1Mx20", 100.0, 150.0,
        pi_id="pi:v3:wall", market="ashare",
    )
    cal.record_observation(
        "polars", "ts_mean", "1Mx20", 100.0, 130.0,
        pi_id="pi:v3:wall", market="ashare",
    )
    data = json.load(open(path, encoding="utf-8"))
    model = next(iter(data["models"].values()))
    assert model["last_updated_ms"] > 1_000_000_000_000


def test_resource_shape_calibration_timestamp_is_wall_clock() -> None:
    """ResourceCalibrationStore 持久化的 updated_at_ms 也是 UTC wall-clock。"""
    key = ResourceShapeKey("ts_mean", "pandas_numpy", rows_bucket=2, instruments_bucket=2, window_bucket=1)
    store = ResourceCalibrationStore()
    store.record(key, elapsed_ms=5, peak_mem=777, attribution_quality="isolated", peak_is_trusted=True)
    cal = store.get(key)
    assert cal is not None
    assert cal.updated_at_ms > 1_000_000_000_000  # wall clock, not monotonic

    path = os.path.join(tempfile.mkdtemp(), "cal.parquet")
    store.save(path)
    store2 = ResourceCalibrationStore(path=path)
    cal2 = store2.get(key)
    assert cal2 is not None
    assert cal2.updated_at_ms > 1_000_000_000_000
    assert cal2.updated_at_ms == pytest.approx(cal.updated_at_ms, rel=1e-3)


def test_shape_calibration_updated_at_is_wall_clock() -> None:
    """ShapeCalibration.record 直接写入的 updated_at_ms 是 UTC wall-clock。"""
    key = ResourceShapeKey("ts_mean", "pandas_numpy", rows_bucket=2, instruments_bucket=2, window_bucket=1)
    cal = ShapeCalibration(shape_key=key)
    t0 = time.monotonic() * 1000.0  # boot 相对
    cal.record(elapsed_ms=1, peak_mem=1, attribution_quality="isolated", peak_is_trusted=True)
    assert cal.updated_at_ms > 1_000_000_000_000
    # monotonic boot 时钟量级远小于 epoch ms；wall-clock 一定大于任何 boot 相对值
    assert cal.updated_at_ms > t0


# ---------------------------------------------------------------------------
# 3. 原子写 + 进程锁
# ---------------------------------------------------------------------------


def test_save_leaves_no_tmp_or_lock_leftover() -> None:
    """原子写成功后无 .tmp / .lock 残留。"""
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "cal.json")
    cal = CostModelCalibrator(calibration_file=path, min_samples=2, update_interval_s=0.0)
    cal.record_observation(
        "polars", "ts_mean", "1Mx20", 100.0, 150.0,
        pi_id="pi:v3:atomic", market="ashare",
    )
    cal.record_observation(
        "polars", "ts_mean", "1Mx20", 100.0, 130.0,
        pi_id="pi:v3:atomic", market="ashare",
    )
    leftovers = [f for f in os.listdir(tmp) if f.endswith(".tmp") or f.endswith(".lock")]
    assert leftovers == []


def test_save_round_trip_schema_v2() -> None:
    """保存的文件带 schema_version=2 且加载回完整模型。"""
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "cal.json")
    cal = CostModelCalibrator(calibration_file=path, min_samples=2, update_interval_s=0.0)
    cal.record_observation(
        "polars", "ts_mean", "1Mx20", 100.0, 150.0,
        pi_id="pi:v3:rt", market="ashare", representation="polars_native", dtype="float64",
    )
    cal.record_observation(
        "polars", "ts_mean", "1Mx20", 100.0, 130.0,
        pi_id="pi:v3:rt", market="ashare", representation="polars_native", dtype="float64",
    )
    data = json.load(open(path, encoding="utf-8"))
    assert data["schema_version"] == CALIBRATION_SCHEMA_VERSION
    # 持久化 key 包含完整身份维度
    key_str = next(iter(data["models"]))
    assert "pi:v3:rt" in key_str
    assert "ashare" in key_str
    assert "polars_native" in key_str
    assert "float64" in key_str

    # 重新加载
    cal2 = CostModelCalibrator(calibration_file=path, min_samples=2)
    assert len(cal2._models) == 1


def test_old_schema_file_rejected() -> None:
    """旧 schema（v1，monotonic 时间戳时代）文件不加载。"""
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "cal.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"schema_version": 1, "models": {}}, fh)
    cal = CostModelCalibrator(calibration_file=path, min_samples=2)
    assert len(cal._models) == 0


def test_process_lock_busy_fails_closed() -> None:
    """进程锁被占用时 fail-closed：抛 RuntimeError（不静默覆盖）。"""
    from runtime.resource_calibration_store import _CalibrationAtomicWrite

    tmp = tempfile.mkdtemp()
    target = os.path.join(tmp, "out.json")
    lock_path = os.path.join(tmp, "out.json.lock")
    with open(lock_path, "w", encoding="utf-8") as fh:
        fh.write("9999")
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="lock busy"):
        with _CalibrationAtomicWrite(target, lock_timeout_s=0.2) as _tmp:
            pass
    assert time.monotonic() - start < 2.0
    # 他人的 lock 不被移除
    assert os.path.exists(lock_path)


def test_atomic_write_success() -> None:
    """原子写成功：目标文件完整，无残留。"""
    from runtime.resource_calibration_store import _CalibrationAtomicWrite

    tmp = tempfile.mkdtemp()
    target = os.path.join(tmp, "out.json")
    with _CalibrationAtomicWrite(target) as tmp_path:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump({"ok": 1}, fh)
    assert json.load(open(target, encoding="utf-8")) == {"ok": 1}
    leftovers = [f for f in os.listdir(tmp) if f.endswith(".tmp") or f.endswith(".lock")]
    assert leftovers == []


# ---------------------------------------------------------------------------
# 4. research / production 校准命名空间隔离
# ---------------------------------------------------------------------------


def test_default_calibration_path_run_mode_namespaced() -> None:
    """默认校准路径按 run mode 隔离（research 与 production 不同文件）。"""
    old_mode = os.environ.get("FACTOR_ENGINE_RUN_MODE")
    old_cache = os.environ.get("FACTOR_ENGINE_CACHE_DIR")
    try:
        os.environ["FACTOR_ENGINE_RUN_MODE"] = "research"
        os.environ.pop("FACTOR_ENGINE_CACHE_DIR", None)
        research_path = default_calibration_path()
        os.environ["FACTOR_ENGINE_RUN_MODE"] = "production"
        production_path = default_calibration_path()
    finally:
        if old_mode is None:
            os.environ.pop("FACTOR_ENGINE_RUN_MODE", None)
        else:
            os.environ["FACTOR_ENGINE_RUN_MODE"] = old_mode
        if old_cache is None:
            os.environ.pop("FACTOR_ENGINE_CACHE_DIR", None)
        else:
            os.environ["FACTOR_ENGINE_CACHE_DIR"] = old_cache
    assert "research" in research_path
    assert "production" in production_path
    assert research_path != production_path


# ---------------------------------------------------------------------------
# 5. CalibrationKey dataclass 完整性
# ---------------------------------------------------------------------------


def test_calibration_key_round_trip_dict() -> None:
    """CalibrationKey to_dict/from_dict 往返一致。"""
    key = CalibrationKey(
        pi_id="pi:v3:x", build="b", cpu_model="cpu", cores=8, threads=4,
        backend_version="1.0", representation="wide_panel", dtype="float64",
        market="ashare", backend="polars", operator="ts_mean", shape_key="1Mx20",
    )
    assert CalibrationKey.from_dict(key.to_dict()) == key
    assert CalibrationKey.from_dict(key.to_dict()).to_key() == key.to_key()
