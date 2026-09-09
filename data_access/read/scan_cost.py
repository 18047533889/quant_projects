"""
data_access.read.scan_cost —— 读路径成本估算与执行路由

职责
    1. 估算一次读的扫描成本：file_count / total_bytes / estimated_rows /
       projected_columns / total_columns / remote / selectivity / 引擎启动成本
    2. 给出 (engine, result_mode) 路由建议——不只按行数，而是综合成本

设计要点
    1. estimated_rows 优先来自 manifest（不扫 footer）；否则 fallback
       ``dataset_read_stats`` 的 footer 估算。
    2. result_mode：
       - 小（行数少 + 本地）→ arrow（一次 materialize）
       - 大（行数多 / 结果字节大）→ stream
       - 宽表 + prefer_polars → lazy（Polars pushdown）
    3. engine：
       - 格式 arrow/feather → pyarrow
       - prefer_polars 且体量大 → polars
       - 默认 duckdb（共享连接，footer cache 复用）

非职责
    不做真正的文件 IO；不替换 QueryBudget（那是硬拦截，这里是软路由）。
"""

from __future__ import annotations

import enum
import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from data_access.core.storage import is_remote_storage
from data_access.read.formats import format_adapter_for_dataset
from data_access.read.query_budget import QueryBudget

if TYPE_CHECKING:
    from .store import DataAccessStore

logger = logging.getLogger("data_access.scan_cost")

_ARROW_MAX_ROWS = 5_000_000
_STREAM_MIN_ROWS = 1_000_000
_ENGINE_STARTUP_MS = {"duckdb": 5.0, "polars": 15.0, "pyarrow": 3.0}

# R40 #53 + R32-P0-065：typed UnknownCost 而非 magic integer。
COST_UNKNOWN_CONSERVATIVE = "cost_unknown_conservative"

#: 保守估计的 safe ceiling（estimated_rows=MAX，total_bytes=SAFE_CEILING）。
COST_MAX_ESTIMATED_ROWS = 10**12
COST_SAFE_CEILING_BYTES = (1 << 63) - 1


class CostDimension:
    """R32-P0-066: Cost calibration 有量纲模型。

    区分各种成本维度：
    - IO_BYTES: 远程/本地字节传输成本
    - CPU_ROWS: 行数处理成本（解码/反序列化）
    - MEMORY_PROJECTION: 投影列内存占用
    - ENGINE_STARTUP: 引擎启动固定成本
    - FILE_OVERHEAD: 小文件开销（打开/关闭/元数据）
    """
    IO_BYTES = "io_bytes"
    CPU_ROWS = "cpu_rows"
    MEMORY_PROJECTION = "memory_projection"
    ENGINE_STARTUP = "engine_startup"
    FILE_OVERHEAD = "file_overhead"


@dataclass(frozen=True)
class UnknownCost:
    """R32-P0-065: 类型化的未知成本（不用 sentinel 整数）。

    reason: 为何成本未知（"no_manifest" / "stats_failed" / "empty_dataset"）
    conservative_bound: 保守上界（None = 无法估算保守值，production reject）
    """
    reason: str
    conservative_bound: int | None = None

    def is_usable(self, production: bool = False) -> bool:
        """production 下必须有保守上界才可用。"""
        if production:
            return self.conservative_bound is not None
        return True


@dataclass(frozen=True)
class ScanCost:
    dataset: str
    file_count: int
    total_bytes: int | None
    estimated_rows: int
    projected_columns: int
    total_columns: int | None
    remote: bool
    selectivity: float | None = None      # time_range/instrument 估算选择率
    instrument_count: int = 0
    engine_startup_ms: float = 5.0
    score: float = 0.0
    # #30 数据集文件格式：arrow/feather 没有 DuckDB 原生 reader，auto 路由
    # 必须选 pyarrow，不能拿 scan_parquet 硬扫。
    file_format: str | None = None
    # ---- #25 真实 CBO 扩展：已选文件/字节/row-group、投影字节、IO 估算 ----
    selected_files: int = 0
    selected_bytes: int | None = None
    selected_rowgroups: int | None = None
    projection_bytes: int | None = None       # 选定列 × 行数的近似物化字节
    estimate_ms: float = 0.0                  # 成本估算本身的耗时（校准用）
    calibrated_factor: float = 1.0            # #27 estimate-vs-actual 在线校准乘子
    # R40 #53：cost basis —— "manifest" | "stats" | COST_UNKNOWN_CONSERVATIVE |
    # "unknown"。production 对 "unknown"（无法计算保守估计）reject。
    cost_basis: str = "stats"
    # R32-P0-066/067: 多维成本模型 {dimension: value}
    cost_by_dimension: dict[str, float] | None = None
    # R32-P0-067: 多维校准 scope（dataset × format × remote × time_range_selectivity）
    calibration_scope: dict[str, str] | None = None

    @property
    def calibrated_score(self) -> float:
        return self.score * self.calibrated_factor

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "estimated_rows": self.estimated_rows,
            "projected_columns": self.projected_columns,
            "total_columns": self.total_columns,
            "remote": self.remote,
            "selectivity": self.selectivity,
            "instrument_count": self.instrument_count,
            "engine_startup_ms": self.engine_startup_ms,
            "score": self.score,
            "selected_files": self.selected_files,
            "selected_bytes": self.selected_bytes,
            "selected_rowgroups": self.selected_rowgroups,
            "projection_bytes": self.projection_bytes,
            "estimate_ms": self.estimate_ms,
            "calibrated_factor": self.calibrated_factor,
            "calibrated_score": self.calibrated_score,
            "file_format": self.file_format,
            "cost_basis": self.cost_basis,
        }


def _avg_row_width(ds: Any, columns: Sequence[str] | None) -> float:
    """按 schema dtype 估算平均行宽（字节），用于投影字节估算。"""
    schema = getattr(ds, "schema", None) or {}
    cols = columns or list(schema.keys())
    widths = {
        "double": 8, "float": 4, "int": 8, "int64": 8, "int32": 4,
        "date": 4, "timestamp": 8, "bool": 1, "string": 16, "varchar": 16,
    }
    total = 0.0
    for c in cols:
        d = str(schema.get(c, "")).lower()
        total += widths.get(d, 8)
    return total


def estimate_scan_cost(
    store: "DataAccessStore",
    dataset: str,
    *,
    columns: Sequence[str] | None = None,
    time_range: tuple[Any, Any] | None = None,
    instrument_filter: Sequence[str] | None = None,
    prefer_polars: bool = False,
    **params: Any,
) -> ScanCost:
    """估算一次读的扫描成本（#25 真实 CBO）。

    优先用 manifest 的文件级 min/max + rows/bytes 做裁剪后的真实估算
    （selected_files / selected_bytes / estimated_rows 都是裁剪后的值，
    不再是全数据集估算）；无 manifest 才回退 ``dataset_read_stats`` 全量
    footer 估算。projection 用平均行宽 × 行数近似。
    """
    t0 = time.monotonic()
    ds = store._registry.get(dataset)
    estimated_rows = 0
    file_count = 0
    selected_files = 0
    selected_bytes: int | None = None
    selected_rowgroups: int | None = None
    # R40 #53：cost basis（manifest / stats / unknown_conservative / unknown）。
    basis = "stats"

    manifest = None
    try:
        from data_access.read.manifest import load_manifest_for_dataset

        manifest = load_manifest_for_dataset(store, dataset, **params)
    except Exception:
        manifest = None

    if manifest is not None and manifest.files:
        basis = "manifest"
        by_path = {f.path: f for f in manifest.files}
        pruned_paths = manifest.prune(
            time_range=time_range, instrument_filter=instrument_filter
        )
        selected = [by_path[p] for p in pruned_paths if p in by_path]
        selected_files = len(selected)
        selected_bytes = sum(f.bytes or 0 for f in selected)
        estimated_rows = sum(f.rows or 0 for f in selected)
        file_count = len(by_path)
        if manifest.row_groups:
            # #P1-17 row-group 统计按 (path, row_group) 去重：ManifestRowGroup 是
            # path × column × row_group 的**列统计**，不是物理 row-group 数。
            selected_files_set = {f.path for f in selected}
            selected_rowgroups = len(
                {
                    (rg.path, rg.row_group)
                    for rg in manifest.row_groups
                    if rg.path in selected_files_set
                }
            ) or None
    else:
        stats_failed = False
        try:
            stats = store.dataset_read_stats(
                dataset,
                columns=list(columns) if columns else None,
                time_range=time_range,
                prefer_polars=prefer_polars,
                **params,
            )
            estimated_rows = stats.estimated_rows
            file_count = stats.parquet_files
        except Exception:
            # 只有估算**真的失败**（异常）才视为成本未知；空数据集（stats 成功但
            # 0 行）是合法零成本，绝不当成 MAX。
            stats_failed = True
            estimated_rows = 0
            file_count = 0

    # R40 #53：既无 manifest 又**估算失败** → 成本未知（``basis="unknown"``）。
    # 不在此处篡改 estimated_rows/bytes 为 MAX——那会误伤写路径/新建数据集
    # （写 generation 时 manifest/stats 天然不可用）。保守 MAX 由
    # :func:`conservative_scan_cost` 显式构造，production 由
    # :func:`assert_scan_cost_usable` 对 ``"unknown"`` reject。
    if manifest is None and stats_failed:
        basis = "unknown"

    total_bytes = selected_bytes
    if total_bytes is None:
        try:
            if manifest is not None:
                total_bytes = manifest.total_bytes
        except Exception:
            pass

    schema = getattr(ds, "schema", None) or {}
    total_columns = len(schema) if schema else None
    projected = len(columns) if columns else 0
    projection_bytes = None
    if estimated_rows and total_columns:
        width = _avg_row_width(ds, columns)
        projection_bytes = int(estimated_rows * width)

    remote = is_remote_storage(ds)
    # #30 数据集格式（arrow/feather 无 DuckDB reader → pyarrow；csv/tsv/jsonl 可
    # 走 duckdb/polars；parquet 全引擎）。用 normalize_format_name 防别名漂移。
    file_format = None
    try:
        from data_access.read.formats import normalize_format_name

        file_format = normalize_format_name(str(getattr(ds, "format", "parquet") or "parquet"))
    except Exception:
        file_format = str(getattr(ds, "format", "parquet") or "parquet").lower()
    startup = _ENGINE_STARTUP_MS.get("polars" if prefer_polars else "duckdb", 5.0)

    # 选择率：有 manifest 时用「裁剪后字节 / 全量字节」；否则经验值
    selectivity = 1.0
    if manifest is not None and manifest.total_bytes:
        sel = (selected_bytes or 0) / manifest.total_bytes if manifest.total_bytes else 1.0
        selectivity = max(0.0, min(1.0, sel))
    elif time_range is not None:
        start, end = time_range
        if start is not None and end is not None:
            selectivity = 0.3
        elif start is not None or end is not None:
            selectivity = 0.5

    score = 0.0
    rows_factor = max(estimated_rows, 0)
    col_factor = (projected / total_columns) if (projected and total_columns) else 1.0
    remote_factor = 3.0 if remote else 1.0
    file_factor = 1.0 + 0.05 * max(0, selected_files - 1)  # 大量小文件惩罚
    # #P1-14：manifest 路径下 ``estimated_rows`` 已经来自裁剪后的 selected 文件
    # （selection effect 已算进去），score 再乘 selectivity 会算两次。只有无
    # manifest（estimated_rows 是全集估算）时才用 selectivity 折扣。
    sel_effective = 1.0 if manifest is not None else (selectivity or 1.0)
    score = rows_factor * col_factor * remote_factor * sel_effective * file_factor
    if remote:
        score += (selected_bytes or 0) / 1024.0  # 远程按字节加成本

    estimate_ms = (time.monotonic() - t0) * 1000.0
    calibrated = _calibrated_factor(dataset)

    # R32-P0-066: 多维成本模型
    cost_by_dimension = {
        CostDimension.IO_BYTES: float(selected_bytes or total_bytes or 0),
        CostDimension.CPU_ROWS: float(estimated_rows),
        CostDimension.MEMORY_PROJECTION: float(projection_bytes or 0),
        CostDimension.ENGINE_STARTUP: startup,
        CostDimension.FILE_OVERHEAD: file_factor * 100.0,  # 归一化
    }

    # R32-P0-067: 多维校准 scope
    calibration_scope = {
        "dataset": dataset,
        "format": file_format or "parquet",
        "remote": "remote" if remote else "local",
        "selectivity_bin": (
            "full" if selectivity >= 0.9
            else "high" if selectivity >= 0.5
            else "medium" if selectivity >= 0.1
            else "low"
        ),
    }

    return ScanCost(
        dataset=dataset,
        file_count=file_count,
        total_bytes=total_bytes,
        estimated_rows=estimated_rows,
        projected_columns=projected,
        total_columns=total_columns,
        remote=remote,
        selectivity=selectivity,
        instrument_count=len(instrument_filter) if instrument_filter else 0,
        engine_startup_ms=startup,
        score=score,
        selected_files=selected_files,
        selected_bytes=selected_bytes,
        selected_rowgroups=selected_rowgroups,
        projection_bytes=projection_bytes,
        estimate_ms=estimate_ms,
        calibrated_factor=calibrated,
        file_format=file_format,
        cost_basis=basis,
        cost_by_dimension=cost_by_dimension,
        calibration_scope=calibration_scope,
    )


def conservative_scan_cost(
    dataset: str,
    *,
    instrument_count: int = 0,
    remote: bool = False,
) -> ScanCost:
    """R40 #53：成本未知时的保守估计（estimated_rows=MAX, total_bytes=SAFE_CEILING）。

    ``cost_basis = COST_UNKNOWN_CONSERVATIVE``——与「真未知」（``unknown``）区分：
    有保守估计兜底 → 可继续；完全无法估计 → production reject。
    """
    return ScanCost(
        dataset=dataset,
        file_count=0,
        total_bytes=COST_SAFE_CEILING_BYTES,
        estimated_rows=COST_MAX_ESTIMATED_ROWS,
        projected_columns=0,
        total_columns=None,
        remote=remote,
        instrument_count=instrument_count,
        cost_basis=COST_UNKNOWN_CONSERVATIVE,
    )


def assert_scan_cost_usable(
    cost: ScanCost | None,
    *,
    production: bool | None = None,
    dataset: str = "",
) -> bool:
    """R40 #53：production 下 cost basis 为 ``unknown``（无法计算保守估计）→ reject。

    - ``cost is None`` → 无法估计 → production reject；
    - ``cost_basis == "unknown"`` → production reject（无保守估计兜底）；
    - ``COST_UNKNOWN_CONSERVATIVE`` / ``manifest`` / ``stats`` → 可继续。

    research 模式只记录 warning，不拒绝。返回是否可用。
    """
    from data_access.core.exceptions import ResourceAdmissionError
    from data_access.read.query_budget import is_strict_semantics

    if production is None:
        production = is_strict_semantics()
    if cost is None:
        if production:
            raise ResourceAdmissionError(
                f"scan cost 无法估计（R40 #53）：dataset={dataset or '<unknown>'} "
                "production 拒绝无成本的读路径。"
            )
        return False
    basis = getattr(cost, "cost_basis", "stats") or "unknown"
    if basis == "unknown":
        if production:
            raise ResourceAdmissionError(
                f"scan cost basis 未知且无法计算保守估计（R40 #53）：dataset="
                f"{dataset or cost.dataset or '<unknown>'} basis={basis!r}。"
                "production 拒绝以未知成本继续。"
            )
        return False
    return True


# ---- #27 estimate-vs-actual 在线校准（轻量 EMA） ----

_EMA_ALPHA = 0.2
_calibration_lock = None
if _calibration_lock is None:
    import threading

    _calibration_lock = threading.Lock()
_calibration: dict[str, float] = {}   # dataset -> 校准乘子


def _calibrated_factor(dataset: str) -> float:
    with _calibration_lock:
        return _calibration.get(dataset, 1.0)


def reset_calibration() -> None:
    with _calibration_lock:
        _calibration.clear()
    with _shape_calibration_lock:
        _shape_calibration_samples.clear()


# ---- R42 形状级校准（shape × cache-state 隔离的逐样本校准） ----

class ScanCacheState(str, enum.Enum):
    """scan 缓存状态维度：同一 dataset 的冷/暖/稳态读取成本不同。

    COLD:  未缓存/首次读（读物理文件）
    WARM:  部分缓存命中
    STEADY_STATE: 充分缓存/稳定态

    用于把校准样本按「shape × cache-state」分桶，避免把冷/暖读数混进
    同一个样本集而扭曲分位数校准。
    """
    COLD = "cold"
    WARM = "warm"
    STEADY_STATE = "steady_state"


@dataclass(frozen=True)
class CalibrationStats:
    """一次校准样本集的统计摘要（分位数 + MAD）。"""
    sample_count: int
    p50: float
    p95: float
    mad: float
    mean: float


def scan_shape_key(
    cost: ScanCost,
    *,
    cache_state: ScanCacheState = ScanCacheState.STEADY_STATE,
) -> tuple[str, ...]:
    """构造形状校准 key（tuple）：dataset × cache-state × projection。

    设计：把「同一 dataset 的冷/暖读数」与「不同 projection（宽/窄）」
    分开，使在线 EMA（record_scan_actual）不会用不同形状的读数互相污染。
    返回元组 key（兼容记录/分位数/落盘三方）。仅返回形状标识，不含校准值。
    """
    cache_state = ScanCacheState(cache_state)
    return (cost.dataset, cache_state.value, f"proj={cost.projected_columns}")


# R42：形状校准样本分桶（scan_shape_key -> list[elapsed_ms]）。
# 与上面的轻量 EMA（_calibration, dataset 级）正交：EMA 给出稳定乘子，
# 这里给出逐形状的分位数/MAD 诊断。生产读取只依赖保守估计（COST_*），
# 分位数仅用于路由/监控，因此样本缺失时 fail-closed 返回 None。
_shape_calibration_samples: dict[tuple[str, ...], list[float]] = {}
_shape_calibration_lock = threading.Lock()


def record_scan_actual(
    dataset: str,
    *,
    estimated_score: float,
    actual_elapsed_ms: float,
    shape_key: tuple[str, ...] | None = None,
) -> None:
    """#27 + R42：记录一次实际读数。

    兼容既有签名（不传 shape_key 时只更新轻量 EMA）。传入 shape_key 时
    同时把 elapsed 加入对应形状样本桶（R42 形状分位数校准）。
    """
    if estimated_score <= 0 or actual_elapsed_ms <= 0:
        return
    ratio = actual_elapsed_ms / max(1.0, estimated_score / 1e6)
    correction = max(0.1, min(10.0, ratio))
    with _calibration_lock:
        cur = _calibration.get(dataset, 1.0)
        _calibration[dataset] = cur + _EMA_ALPHA * (correction - cur)
    if shape_key is not None:
        key = tuple(shape_key)
        with _shape_calibration_lock:
            _shape_calibration_samples.setdefault(key, []).append(actual_elapsed_ms)


def calibration_quantiles(
    shape_key: tuple[str, ...] | None,
) -> CalibrationStats | None:
    """按形状 key 返回分位数摘要；无样本 → None（fail-closed）。"""
    if shape_key is None:
        return None
    key = tuple(shape_key)
    with _shape_calibration_lock:
        samples = list(_shape_calibration_samples.get(key, ()))
    if not samples:
        return None
    samples = sorted(samples)
    n = len(samples)
    p50 = samples[n // 2] if n else 0.0
    p95 = samples[min(n - 1, int(round(n * 0.95)))] if n else 0.0
    mean = sum(samples) / n
    mad = statistics.median(sorted(abs(x - mean) for x in samples))
    return CalibrationStats(
        sample_count=n, p50=p50, p95=p95, mad=mad, mean=mean,
    )


def save_calibration(
    path: Path,
    *,
    host_class: str,
    storage_class: str,
    build_id: str,
) -> None:
    """把形状校准样本落盘，绑定 host/storage/build（R42 生成绑定）。"""
    payload = {
        "schema": "r42-scan-cost-shape-calibration-v1",
        "host_class": host_class,
        "storage_class": storage_class,
        "build_id": build_id,
        "samples": {
            "::".join(str(part) for part in k): v for k, v in _shape_calibration_samples.items()
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_calibration(
    path: Path,
    *,
    host_class: str,
    storage_class: str,
    build_id: str,
) -> bool:
    """加载与当前 (host, storage, build) 匹配的形状校准；不匹配 → 丢弃（返回 False）。

    只有三把钥匙全对才载入样本（否则会用别的主机/存储/构建的读数污染）。
    返回是否加载成功。
    """
    if not Path(path).exists():
        return False
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if (
        payload.get("host_class") != host_class
        or payload.get("storage_class") != storage_class
        or payload.get("build_id") != build_id
    ):
        return False
    samples = payload.get("samples", {})
    with _shape_calibration_lock:
        for key, vals in samples.items():
            _shape_calibration_samples[tuple(key.split("::"))] = list(vals)
    return True

def suggest_read_strategy(
    cost: ScanCost,
    *,
    prefer_polars: bool = False,
    engine: str = "auto",
    result: str = "auto",
    downstream_backend: str | None = None,
) -> tuple[str, str]:
    """根据 ScanCost 建议 (engine, result_mode)。

    engine ∈ {duckdb, polars, pyarrow}；result_mode ∈ {arrow, pandas, polars, lazy, stream}。
    显式传入的 engine/result 直接返回（auto 才路由）。

    ``downstream_backend``：消费方是 polars-native（``polars_native``）时，
    即使行数不够大也建议 polars+lazy，避免中间 Arrow 物化（R42）。
    """
    if engine not in {"auto", "duckdb", "polars", "pyarrow"}:
        raise ValueError(f"engine 必须是 auto|duckdb|polars|pyarrow，收到 {engine!r}")
    if result not in {"auto", "arrow", "pandas", "polars", "lazy", "stream"}:
        raise ValueError(
            f"result 必须是 auto|arrow|pandas|polars|lazy|stream，收到 {result!r}"
        )

    if engine == "auto":
        fmt = str(cost.file_format or "parquet").lower()
        if fmt in {"arrow", "feather", "ipc", "feather-v2"}:
            engine = "pyarrow"
        elif downstream_backend == "polars_native":
            engine = "polars"
        elif cost.file_count == 0 and cost.estimated_rows == 0:
            engine = "duckdb"
        elif prefer_polars and cost.estimated_rows >= _STREAM_MIN_ROWS:
            engine = "polars"
        else:
            engine = "duckdb"

    if result == "auto":
        if engine == "polars":
            result = "lazy"
        elif cost.estimated_rows >= _STREAM_MIN_ROWS or (
            cost.total_bytes is not None and cost.total_bytes >= 512 * 1024 * 1024
        ):
            result = "stream"
        elif cost.estimated_rows <= _ARROW_MAX_ROWS:
            result = "arrow"
        else:
            result = "stream"
    return engine, result


def log_routing(cost: ScanCost, engine: str, result: str) -> None:
    logger.info(
        "read_auto dataset=%s rows=%d files=%d bytes=%s engine=%s result=%s score=%.0f",
        cost.dataset,
        cost.estimated_rows,
        cost.file_count,
        cost.total_bytes,
        engine,
        result,
        cost.score,
    )
