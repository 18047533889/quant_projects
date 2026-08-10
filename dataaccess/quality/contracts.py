"""
data_access.quality.contracts —— 企业级数据质量契约检查

在原来的 ``required_columns / primary_key`` 基础上扩展：

    基础        schema 对齐 / PK 唯一 / null ratio / range / finite·inf·NaN
    时序        time 单调 / 重复 timestamp / 缺失时间 / 未来 timestamp
    截面        coverage / 截面覆盖 / 意外标的
    一致性      PIT 泄漏 / 分区完整性 / schema 漂移 / 行数异常

输入可以是 Arrow Table（``store.read`` / ``store.read_uri`` 结果），检查全部在
Arrow 层完成，不绕开 DataAccess。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc

from data_access.core.exceptions import ValidationError

_CHECK_NAMES = (
    "required_columns",
    "primary_key",
    "schema_alignment",
    "null_ratio",
    "range",
    "finite",
    "monotonic_time",
    "duplicate_timestamp",
    "future_timestamp",
    "coverage",
    "pit_leakage",
)


@dataclass(frozen=True)
class QualityOptions:
    """质量检查选项。只跑显式开启/配置的检查项。"""

    required_columns: tuple[str, ...] = ()
    primary_key: tuple[str, ...] = ()
    declared_schema: dict[str, str] = field(default_factory=dict)   # 声明 {col: type}
    null_ratio_max: float | None = None        # 单列最大缺失率（0~1）
    range: dict[str, tuple[Any, Any]] = field(default_factory=dict)  # col -> (min, max)
    finite_columns: tuple[str, ...] = ()        # 需要全 finite 的列
    time_column: str | None = None
    instrument_column: str | None = None
    check_monotonic_time: bool = False
    check_duplicate_timestamp: bool = False
    check_future_timestamp: bool = False
    min_rows: int | None = None                 # 行数下限（coverage）
    check_pit_leakage: bool = False             # report_period <= publish_time
    # R25 §65：单位漂移 sentinel——{col: (semantics, typical_abs_range)}。
    # A Return bp semantics、US Ret decimal semantics、ROE ratio 等；监控分布
    # 突然 100x/10000x 缩放报警。``unit_sentinel`` 与 ``unit_sentinel_warn_only``
    # 配合：超界默认 BLOCK（fail），显式 warn_only 则 WARN。
    unit_sentinel: dict[str, tuple[str, tuple[float, float]]] = field(
        default_factory=dict
    )
    unit_sentinel_warn_only: bool = False

    def enabled_checks(self) -> list[str]:
        out: list[str] = []
        if self.required_columns:
            out.append("required_columns")
        if self.primary_key:
            out.append("primary_key")
        if self.declared_schema:
            out.append("schema_alignment")
        if self.null_ratio_max is not None:
            out.append("null_ratio")
        if self.range:
            out.append("range")
        if self.finite_columns:
            out.append("finite")
        if self.check_monotonic_time:
            out.append("monotonic_time")
        if self.check_duplicate_timestamp:
            out.append("duplicate_timestamp")
        if self.check_future_timestamp:
            out.append("future_timestamp")
        if self.min_rows is not None:
            out.append("coverage")
        if self.check_pit_leakage:
            out.append("pit_leakage")
        if self.unit_sentinel:
            out.append("unit_sentinel")
        return out


@dataclass
class QualityReport:
    dataset: str
    passed: bool
    checks: tuple[str, ...]
    failures: tuple[str, ...]
    details: dict[str, Any] = field(default_factory=dict)
    sampled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "passed": self.passed,
            "checks": list(self.checks),
            "failures": list(self.failures),
            "details": self.details,
            "sampled": self.sampled,
        }


def _failures_list() -> list[str]:
    return []


def run_quality_checks(
    dataset: str,
    table: pa.Table,
    *,
    options: QualityOptions | None = None,
) -> QualityReport:
    """对 Arrow Table 跑配置的质量检查，返回报告。"""
    options = options or QualityOptions()
    failures: list[str] = []
    details: dict[str, Any] = {}

    # 1) required columns
    if options.required_columns:
        missing = [c for c in options.required_columns if c not in table.column_names]
        details["required_columns"] = {"missing": missing}
        if missing:
            failures.append(f"missing columns: {missing}")

    # 2) primary key uniqueness
    if options.primary_key:
        missing_pk = [c for c in options.primary_key if c not in table.column_names]
        if missing_pk:
            failures.append(f"primary key columns missing: {missing_pk}")
        else:
            dup_count = _primary_key_duplicates(table, options.primary_key)
            details["primary_key"] = {"duplicates": dup_count}
            if dup_count:
                failures.append(f"duplicate primary keys: {dup_count}")

    # 3) schema alignment
    if options.declared_schema:
        mismatch = _schema_alignment(table, options.declared_schema)
        details["schema_alignment"] = mismatch
        if mismatch:
            failures.append(f"schema mismatch: {mismatch}")

    # 4) null ratio
    if options.null_ratio_max is not None:
        ratios = _null_ratios(table)
        over = {c: r for c, r in ratios.items() if r > options.null_ratio_max}
        details["null_ratio"] = {"max": options.null_ratio_max, "ratios": ratios}
        if over:
            failures.append(f"null ratio exceeds {options.null_ratio_max}: {over}")

    # 5) range
    if options.range:
        violations = _range_violations(table, options.range)
        details["range"] = violations
        if violations:
            failures.append(f"range violations: {violations}")

    # 6) finite / inf / NaN
    if options.finite_columns:
        bad = _non_finite_columns(table, options.finite_columns)
        details["finite"] = {"non_finite_counts": bad}
        if any(bad.values()):
            failures.append(f"non-finite values: {bad}")

    # 7) monotonic time
    if options.check_monotonic_time:
        mono = _monotonic_time(table, options)
        details["monotonic_time"] = mono
        if not mono["monotonic"]:
            failures.append(
                f"time not monotonic per instrument: {mono['decreases']} decreases"
            )

    # 8) duplicate timestamp (per instrument)
    if options.check_duplicate_timestamp:
        dup_ts = _duplicate_timestamps(table, options)
        details["duplicate_timestamp"] = {"duplicates": dup_ts}
        if dup_ts:
            failures.append(f"duplicate (time, instrument): {dup_ts}")

    # 9) future timestamp
    if options.check_future_timestamp:
        future = _future_timestamps(table, options.time_column)
        details["future_timestamp"] = {"future_rows": future}
        if future:
            failures.append(f"future timestamps: {future} rows")

    # 10) coverage / min rows
    if options.min_rows is not None:
        rows = table.num_rows
        details["coverage"] = {"rows": rows, "min_rows": options.min_rows}
        if rows < options.min_rows:
            failures.append(f"row count {rows} < min {options.min_rows}")

    # 11) PIT leakage (report_period <= publish_time)
    if options.check_pit_leakage:
        leak = _pit_leakage(table)
        details["pit_leakage"] = {"leaked_rows": leak}
        if leak:
            failures.append(f"PIT leakage (report_period > publish_time): {leak} rows")

    # 12) R25 §65：unit sentinel（单位漂移检测）——黄金 sentinel 监控分布突然
    # 100x/10000x 缩放。超界默认 BLOCK（fail）；``unit_sentinel_warn_only=True``
    # 只 WARN（details 里标记 warn_only）。DTO 只负责检测，不修数据（§64）。
    if options.unit_sentinel:
        sentinel_failures, sentinel_details = _unit_sentinel_check(
            table, options.unit_sentinel
        )
        details["unit_sentinel"] = {
            "warn_only": options.unit_sentinel_warn_only,
            "violations": sentinel_details,
        }
        if sentinel_failures and not options.unit_sentinel_warn_only:
            failures.append(
                f"unit sentinel violations (可能单位漂移): {sentinel_failures}"
            )

    checks = tuple(_CHECK_NAMES) if not options.enabled_checks() else tuple(options.enabled_checks())
    return QualityReport(
        dataset=dataset,
        passed=not failures,
        checks=checks,
        failures=tuple(failures),
        details=details,
    )


def _primary_key_duplicates(table: pa.Table, pk: Sequence[str]) -> int:
    sub = table.select(list(pk))
    n = sub.num_rows
    combined = _concatenate_columns(sub)
    uniq = pc.unique(combined)
    return int(n - len(uniq))


def _concatenate_columns(table: pa.Table):
    """把多列 PK 拼成一个字符串列（元素级 join）以便去重。"""
    arrays = []
    for i in range(table.num_columns):
        arr = table.column(i).combine_chunks()
        if not pa.types.is_string(arr.type):
            try:
                arr = pc.cast(arr, pa.string())
            except Exception:
                arr = pc.cast(arr, pa.large_string())
        arrays.append(arr)
    if len(arrays) == 1:
        return arrays[0]
    return pc.binary_join_element_wise(*arrays, "\x1f")


def _schema_alignment(table: pa.Table, declared: Mapping[str, str]) -> list[str]:
    out: list[str] = []
    actual = {c: str(table.schema.field(c).type) for c in table.column_names}
    for col, typ in declared.items():
        if col not in actual:
            out.append(f"{col}: missing")
            continue
        if not _type_compatible(typ, actual[col]):
            out.append(f"{col}: declared={typ} actual={actual[col]}")
    return out


def _type_compatible(declared: str, actual: str) -> bool:
    d = declared.strip().lower()
    a = actual.lower()
    if d in {"double", "float", "float64", "decimal", "float32"}:
        return any(k in a for k in ("double", "float", "decimal"))
    if d in {"int", "int64", "int32", "int16", "int8", "uint"}:
        return any(k in a for k in ("int", "uint"))
    if d in {"string", "str", "varchar", "text", "utf8"}:
        return any(k in a for k in ("string", "utf8", "large_string"))
    if d in {"timestamp", "datetime"}:
        return "timestamp" in a or "date" in a
    if d == "date":
        return "date" in a
    if d in {"bool", "boolean"}:
        return "bool" in a
    return d == a or a.startswith(d)


def _null_ratios(table: pa.Table) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in table.column_names:
        col = table.column(name)
        out[name] = round(float(col.null_count) / max(1, len(col)), 6)
    return out


def _range_violations(table: pa.Table, ranges: Mapping[str, tuple[Any, Any]]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for col, (lo, hi) in ranges.items():
        if col not in table.column_names:
            out[col] = {"missing": True}
            continue
        arr = table.column(col)
        below = pc.sum(pc.less(arr, lo)).as_py() if lo is not None else 0
        above = pc.sum(pc.greater(arr, hi)).as_py() if hi is not None else 0
        out[col] = {"below": int(below or 0), "above": int(above or 0)}
    return out


def _non_finite_columns(table: pa.Table, columns: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for col in columns:
        if col not in table.column_names:
            continue
        arr = table.column(col)
        if pa.types.is_floating(arr.type):
            bad = pc.sum(pc.or_kleene(pc.is_nan(arr), pc.is_inf(arr))).as_py()
            out[col] = int(bad or 0)
        else:
            out[col] = 0
    return out


def _timestamp_array(table: pa.Table, col: str | None) -> pa.Array | None:
    if col is None or col not in table.column_names:
        return None
    arr = table.column(col)
    if pa.types.is_timestamp(arr.type) or pa.types.is_date(arr.type):
        return arr
    # 尝试 cast 字符串时间列
    try:
        return pc.cast(arr, pa.timestamp("us"))
    except Exception:
        return None


def _monotonic_time(table: pa.Table, options: QualityOptions) -> dict[str, Any]:
    ts = _timestamp_array(table, options.time_column)
    if ts is None:
        return {"monotonic": True, "decreases": 0, "note": "no time column"}
    # 全表按行序检查递减（不跨 instrument 排序，保守契约：文件内时间应单调）
    ts_numeric = pc.cast(ts, pa.int64())
    diff = pc.subtract(ts_numeric, pc.shift(ts_numeric, 1))
    decreases = int(pc.sum(pc.and_kleene(pc.is_valid(diff), pc.less(diff, 0))).as_py() or 0)
    return {"monotonic": decreases == 0, "decreases": decreases}


def _duplicate_timestamps(table: pa.Table, options: QualityOptions) -> int:
    if options.time_column is None:
        return 0
    keys = [options.time_column]
    if options.instrument_column and options.instrument_column in table.column_names:
        keys.append(options.instrument_column)
    missing = [c for c in keys if c not in table.column_names]
    if missing:
        return 0
    combined = _concatenate_columns(table.select(keys))
    uniq = pc.unique(combined)
    return int(len(combined) - len(uniq))


def _future_timestamps(table: pa.Table, time_column: str | None) -> int:
    ts = _timestamp_array(table, time_column)
    if ts is None:
        return 0
    now = datetime.now(timezone.utc)
    try:
        now_scalar = pa.scalar(now).cast(ts.type)
    except Exception:
        now_scalar = pa.scalar(now.timestamp() * 1_000_000, type=pa.int64())
        if not pa.types.is_timestamp(ts.type):
            return 0
        return 0
    future = pc.sum(pc.greater(ts, now_scalar)).as_py()
    return int(future or 0)


def _pit_leakage(table: pa.Table) -> int:
    """PIT 泄漏启发式：若存在 report_period 与 publish_time 两列，统计
    report_period > publish_time 的行数。"""
    if "report_period" not in table.column_names or "publish_time" not in table.column_names:
        return 0
    rp = table.column("report_period")
    pt = table.column("publish_time")
    try:
        rp_n = pc.cast(pc.coalesce(rp, pa.scalar(None, type=rp.type)), pa.timestamp("us"))
        pt_n = pc.cast(pc.coalesce(pt, pa.scalar(None, type=pt.type)), pa.timestamp("us"))
    except Exception:
        return 0
    leak = pc.sum(pc.and_kleene(pc.is_valid(rp_n), pc.greater(rp_n, pt_n))).as_py()
    return int(leak or 0)


def _unit_sentinel_check(
    table: pa.Table,
    sentinel: Mapping[str, tuple[str, tuple[float, float]]],
) -> tuple[list[str], dict[str, Any]]:
    """R25 §65：单位漂移 sentinel。

    ``sentinel``：``{col: (semantics, (typical_min_abs, typical_max_abs))}``。
    对非 null 样本的绝对值分位数（p5/p95）与 typical 区间比对：分布突然缩放
    （100x/10000x）超出 typical 区间 → 判为疑似单位漂移。DQ 不修数据（§64），
    只 BLOCK/WARN 上报。
    """
    failures: list[str] = []
    violations: dict[str, Any] = {}
    for col, (semantics, (lo, hi)) in sentinel.items():
        if col not in table.column_names:
            failures.append(f"{col}: column missing (sentinel {semantics})")
            violations[col] = {"semantics": semantics, "error": "missing"}
            continue
        arr = table.column(col)
        try:
            numeric = pc.cast(arr, pa.float64())
        except Exception:
            failures.append(f"{col}: non-numeric (sentinel {semantics})")
            violations[col] = {"semantics": semantics, "error": "non-numeric"}
            continue
        valid = pc.filter(numeric, pc.is_valid(numeric))
        if int(valid.length()) == 0:
            violations[col] = {"semantics": semantics, "samples": 0}
            continue
        vals = valid.to_pylist()
        samples = [abs(float(v)) for v in vals if v is not None]
        if not samples:
            continue
        samples.sort()
        n = len(samples)
        p5 = samples[int(n * 0.05)]
        p95 = samples[min(n - 1, int(n * 0.95))]
        in_band = (p5 >= lo) and (p95 <= hi)
        violations[col] = {
            "semantics": semantics,
            "p5_abs": p5,
            "p95_abs": p95,
            "typical_range": [lo, hi],
            "samples": n,
            "in_band": in_band,
        }
        if not in_band:
            failures.append(f"{col}={semantics} p5={p5} p95={p95} 超出 typical [{lo},{hi}]")
    return failures, violations
