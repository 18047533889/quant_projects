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

import logging
import time
from dataclasses import dataclass
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

    manifest = None
    try:
        from data_access.read.manifest import load_manifest_for_dataset

        manifest = load_manifest_for_dataset(store, dataset, **params)
    except Exception:
        manifest = None

    if manifest is not None and manifest.files:
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
            estimated_rows = 0
            file_count = 0

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
    )


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


def record_scan_actual(
    dataset: str,
    *,
    estimated_score: float,
    actual_elapsed_ms: float,
) -> None:
    """#27 估算 vs 实际：EMA 调整每数据集的校准乘子。

    实际时间显著高于估算 → 乘子上调（后续路由更保守/更倾向 stream）；
    显著低于 → 下调。只在有意义的样本上更新，避免抖动。
    """
    if estimated_score <= 0 or actual_elapsed_ms <= 0:
        return
    ratio = actual_elapsed_ms / max(1.0, estimated_score / 1e6)
    # ratio 理论上接近常数；偏离 1 太多说明估算系统性偏差
    correction = max(0.1, min(10.0, ratio))
    with _calibration_lock:
        cur = _calibration.get(dataset, 1.0)
        _calibration[dataset] = cur + _EMA_ALPHA * (correction - cur)


def reset_calibration() -> None:
    with _calibration_lock:
        _calibration.clear()


def suggest_read_strategy(
    cost: ScanCost,
    *,
    prefer_polars: bool = False,
    engine: str = "auto",
    result: str = "auto",
) -> tuple[str, str]:
    """根据 ScanCost 建议 (engine, result_mode)。

    engine ∈ {duckdb, polars, pyarrow}；result_mode ∈ {arrow, pandas, polars, lazy, stream}。
    显式传入的 engine/result 直接返回（auto 才路由）。
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
            # #30 文档说 Arrow/Feather 自动选 PyArrow，但 router 从没实现——
            # arrow/feather 没有 DuckDB/scan_parquet 原生 reader，硬走会报错。
            engine = "pyarrow"
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
