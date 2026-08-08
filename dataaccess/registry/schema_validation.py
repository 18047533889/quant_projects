"""
data_access.schema_validation —— 首访 schema 自检（PR8）

目的
    每个 dataset 第一次被真正读时，对 parquet header 做一次 schema 对齐校验：
      1. 声明在 registry 的列必须实际存在
      2. 实际类型必须和声明类型兼容（宽松匹配，见 _normalize_dtype）
    不一致时按模式（strict / warn / off）处理：strict 抛 ValidationError，
    warn 打 WARNING 继续放行，off 跳过。

    行为开关（env）：
        QUANT_SCHEMA_CHECK = off | warn | strict   默认 warn
        warn 模式适合灰度：先把差异暴露到日志/告警上，不阻塞业务；
        生产稳定后切 strict。

设计要点
    1. 通过 `DESCRIBE SELECT * FROM read_parquet(...) LIMIT 0` 拿 schema，
       只扫 footer 不扫数据；列类型来自 DuckDB 的 information_schema。
    2. 校验结果缓存在 store 里按 dataset name 存一次，避免重复 DESCRIBE；
       schema 在运行过程中是不变量——一旦校验通过，后续所有读都省略。
    3. 类型宽松匹配：业务声明 "timestamp" 能匹配 DuckDB 的任一
       TIMESTAMP / TIMESTAMP_NS / TIMESTAMP WITH TIME ZONE；声明 "int" 能
       匹配 TINYINT/SMALLINT/INTEGER/BIGINT/HUGEINT；想精确匹配就声明
       DuckDB 原生类型名，normalize 过后仍能匹配。
    4. Hive 分区列（year= 等）在 parquet header 里不存在，但业务用户把它当列
       访问；校验时对分区列做豁免——registry 没法当前感知具体分区列名，
       我们用启发式：如果声明列在 hive_partitioning dataset 的实际 schema 里
       没有，但名字等于 ds 的 `hive_partition_cols`（现在没显式字段；保守跳过），
       就记为 missing 但不当致命。保守起见 PR8 只对非分区列强校验。

非职责
    不做列顺序校验（parquet 允许任意列顺序）
    不做 nullability 校验（DuckDB 默认 NULL-friendly）
    不做分区列类型校验（hive 分区 year 可能是 INT 也可能是 VARCHAR，业务不关心）

维护人：quant 基础平台组    最后更新：2026-04-20
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from data_access.core.exceptions import ValidationError

if TYPE_CHECKING:
    from data_access.core.engine import DuckDBEngine
    from .loader import Dataset


logger = logging.getLogger("data_access.schema_validation")


# ---- 类型归一化 -------------------------------------------------------------
#
# 业务声明的宽松类型 → 一组匹配的 DuckDB 类型前缀。
# DuckDB 的 DESCRIBE 会返回形如 "TIMESTAMP WITH TIME ZONE" / "TIMESTAMP_NS" /
# "DECIMAL(18,4)" 等带括号和修饰的类型，我们只按大写前缀匹配。
_TYPE_ALIASES: dict[str, tuple[str, ...]] = {
    # 时间戳：覆盖 TIMESTAMP / TIMESTAMP_NS / TIMESTAMP_MS / TIMESTAMPTZ /
    # TIMESTAMP WITH TIME ZONE
    "timestamp": ("TIMESTAMP",),
    "datetime": ("TIMESTAMP",),
    "date": ("DATE",),
    "time": ("TIME",),
    # 整数：INTEGER / BIGINT / SMALLINT / TINYINT / HUGEINT
    "int": ("INTEGER", "BIGINT", "SMALLINT", "TINYINT", "HUGEINT", "UBIGINT", "UINTEGER"),
    "int32": ("INTEGER",),
    "int64": ("BIGINT",),
    # 浮点
    "float": ("DOUBLE", "REAL", "DECIMAL", "FLOAT"),
    "double": ("DOUBLE", "REAL", "DECIMAL", "FLOAT"),
    "decimal": ("DECIMAL",),
    # 字符串
    "string": ("VARCHAR",),
    "str": ("VARCHAR",),
    "varchar": ("VARCHAR",),
    # 布尔
    "bool": ("BOOLEAN",),
    "boolean": ("BOOLEAN",),
    # 二进制 / blob
    "bytes": ("BLOB",),
    "blob": ("BLOB",),
}


def _is_compatible(declared: str, actual: str) -> bool:
    """declared 是业务 yaml 里声明的类型；actual 是 DuckDB DESCRIBE 返回的类型字符串。

    匹配规则：
      1. 大小写不敏感的精确子串匹配（declared 是 DuckDB 原生类型名时）
      2. declared 在 _TYPE_ALIASES 里时，actual 以任一别名前缀开头即匹配

    例子：
      _is_compatible("timestamp", "TIMESTAMP WITH TIME ZONE") → True
      _is_compatible("int", "BIGINT") → True
      _is_compatible("VARCHAR", "VARCHAR") → True
      _is_compatible("double", "BIGINT") → False
    """
    if not declared or not actual:
        return False
    declared_lc = declared.strip().lower()
    actual_upper = actual.strip().upper()

    # 别名命中
    if declared_lc in _TYPE_ALIASES:
        return any(actual_upper.startswith(pref) for pref in _TYPE_ALIASES[declared_lc])

    # 原生类型：精确等值 or actual 以 declared 开头（兼容 DECIMAL(18,4) 这种参数化类型）
    declared_upper = declared.strip().upper()
    if actual_upper == declared_upper:
        return True
    return actual_upper.startswith(declared_upper)


# ---- 校验结果 ---------------------------------------------------------------

@dataclass(frozen=True)
class SchemaCheckResult:
    """单次校验的结果。ok == True 时其它字段应为空。"""

    ok: bool
    dataset: str
    missing: tuple[str, ...] = field(default_factory=tuple)          # 声明了但 parquet 里没有
    type_mismatch: tuple[tuple[str, str, str], ...] = field(default_factory=tuple)  # (col, declared, actual)
    actual_schema: tuple[tuple[str, str], ...] = field(default_factory=tuple)       # (col, duckdb_type)

    def format_message(self) -> str:
        lines = [f"dataset '{self.dataset}' schema 校验失败:"]
        if self.missing:
            lines.append(f"  缺失列（声明了 registry 但 parquet 里没有）: {list(self.missing)}")
        if self.type_mismatch:
            lines.append("  类型不匹配:")
            for col, declared, actual in self.type_mismatch:
                lines.append(f"    - {col}: 声明={declared} 实际={actual}")
        if self.actual_schema:
            lines.append("  实际 parquet schema:")
            for col, typ in self.actual_schema:
                lines.append(f"    - {col}: {typ}")
        lines.append(
            "  修复方式："
            "a) 若声明错了，改 data_access/config/datasets.yaml 的 schema 块；"
            "b) 若数据确实漂移了，先联系 #quant-platform，确认上游生产路径是否变更。"
        )
        return "\n".join(lines)


# ---- 模式开关 ---------------------------------------------------------------

_VALID_MODES = ("off", "warn", "strict")


def _resolve_mode() -> str:
    """读 QUANT_SCHEMA_CHECK；production 默认 strict。"""
    raw = os.environ.get("QUANT_SCHEMA_CHECK", "").strip().lower()
    if not raw:
        prod = os.environ.get("QUANT_PRODUCTION_MODE", "").lower() in {"1", "true", "yes"}
        strict_read = os.environ.get("DATA_ACCESS_STRICT_READ", "").lower() in {"1", "true", "yes"}
        raw = "strict" if (prod or strict_read) else "warn"
    if raw not in _VALID_MODES:
        logger.warning(
            "QUANT_SCHEMA_CHECK='%s' 不合法，使用默认 'warn'（合法值：%s）",
            raw, _VALID_MODES,
        )
        return "warn"
    return raw


# ---- 核心 API ---------------------------------------------------------------

def _partition_columns_of(ds) -> set[str]:
    """数据集的分区列集合（partitioning / partition_columns / hive 布局 / bucket）。

    #35：hive 分区列存在目录名而非 parquet header，schema 自检时应豁免。
    """
    cols: set[str] = set()
    partitioning = getattr(ds, "partitioning", None)
    if isinstance(partitioning, dict):
        keys = partitioning.get("columns") or partitioning.get("partition_by")
        if isinstance(keys, (list, tuple)):
            cols.update(str(k) for k in keys)
    pcols = getattr(ds, "partition_columns", ()) or ()
    cols.update(str(c) for c in pcols)
    layout_policy = getattr(ds, "layout_policy", None)
    bucket = getattr(layout_policy, "bucket", None)
    if bucket is not None and getattr(bucket, "column", None):
        cols.add(str(bucket.column))
    return cols


def _fetch_actual_schema(
    engine: "DuckDBEngine",
    ds: "Dataset",
    paths: list[str],
) -> list[tuple[str, str]]:
    """用 DESCRIBE 拿 parquet header 的列和类型。只扫 footer，不读数据。

    返回 [(col_name, duckdb_type_string), ...]
    """
    if not paths:
        raise ValidationError(
            f"dataset '{ds.name}' 的 resolve_paths 返回空，无法做 schema 校验"
        )
    from data_access.read.formats import format_adapter_for_dataset

    adapter = format_adapter_for_dataset(ds)
    if not adapter.uses_duckdb:
        raise ValidationError(
            f"dataset '{ds.name}' 的格式 '{ds.format}' 不能走 DuckDB DESCRIBE；"
            "schema 校验请切到 pyarrow 引擎路径。"
        )
    path_param = paths if len(paths) > 1 else paths[0]
    # DESCRIBE 返回 column_name, column_type, null, key, default, extra 六列
    sql = adapter.describe_sql(
        path_param,
        hive_partitioning=ds.hive_partitioning,
        union_by_name=ds.union_by_name,
    )
    sql = f"DESCRIBE {sql}"
    tbl = engine.execute_arrow(sql, [path_param])
    if "column_name" not in tbl.column_names or "column_type" not in tbl.column_names:
        # 不同 DuckDB 版本列名略有差别；兜底取前两列
        col_names = [c for c in tbl.column_names]
        name_col = col_names[0]
        type_col = col_names[1]
    else:
        name_col = "column_name"
        type_col = "column_type"
    names = tbl.column(name_col).to_pylist()
    types = tbl.column(type_col).to_pylist()
    return list(zip(names, types))


def check_schema(
    engine: "DuckDBEngine",
    ds: "Dataset",
    paths: list[str],
) -> SchemaCheckResult:
    """对给定 dataset 做一次 schema 对齐校验；纯函数，不改任何状态。

    如果 ds.schema 为空，直接 ok；调用方自己决定要不要跳。
    """
    declared = dict(ds.schema) if getattr(ds, "schema", None) else {}
    if not declared:
        return SchemaCheckResult(ok=True, dataset=ds.name)
    # #21 manifest 空裁剪：没有文件可校验（也无可裁剪的列），直接 ok。
    # _fetch_actual_schema 对空路径会抛错，这里提前短路。
    if not paths:
        return SchemaCheckResult(ok=True, dataset=ds.name)

    actual_pairs = _fetch_actual_schema(engine, ds, paths)
    actual_map = {name: typ for name, typ in actual_pairs}

    # #35 分区列豁免：hive 分区列（partitioning/partition_columns 声明的）存在
    # 于目录名而非 parquet header，不能算「missing」。
    partition_cols = _partition_columns_of(ds)
    missing: list[str] = []
    type_mismatch: list[tuple[str, str, str]] = []
    for col, declared_type in declared.items():
        if col not in actual_map:
            if col in partition_cols:
                continue
            missing.append(col)
            continue
        if not _is_compatible(declared_type, actual_map[col]):
            type_mismatch.append((col, declared_type, actual_map[col]))

    ok = not missing and not type_mismatch
    return SchemaCheckResult(
        ok=ok,
        dataset=ds.name,
        missing=tuple(missing),
        type_mismatch=tuple(type_mismatch),
        actual_schema=tuple(actual_pairs),
    )


def enforce_schema_or_raise(result: SchemaCheckResult, *, mode: str | None = None) -> None:
    """按 QUANT_SCHEMA_CHECK 模式处理校验结果。

    ok 时无事发生；strict 下抛 ValidationError；warn 下记 WARNING；off 下跳过。
    """
    if result.ok:
        return
    effective = mode if mode is not None else _resolve_mode()
    if effective == "off":
        return
    message = result.format_message()
    if effective == "strict":
        raise ValidationError(message)
    # warn
    logger.warning(
        "schema 校验未通过（当前模式=warn，不阻塞；QUANT_SCHEMA_CHECK=strict 切为强制）:\n%s",
        message,
    )


# ---- 进程级缓存 -------------------------------------------------------------
#
# 缓存 key = dataset + params fingerprint + file manifest hash（见 schema_cache_key）。
_validated: set[str] = set()
_validated_lock = threading.Lock()


def schema_cache_key(
    dataset_name: str,
    *,
    params_fingerprint: str = "",
    manifest_hash: str = "",
) -> str:
    """首访 schema 校验缓存键（含 params 与 manifest，避免 factor_id 间漂移漏检）。"""
    return f"{dataset_name}:{params_fingerprint}:{manifest_hash}"


def mark_validated(cache_key: str) -> None:
    with _validated_lock:
        _validated.add(cache_key)


def is_validated(cache_key: str) -> bool:
    with _validated_lock:
        return cache_key in _validated


def reset_validated_cache() -> None:
    """测试用：清掉首访缓存。"""
    with _validated_lock:
        _validated.clear()
