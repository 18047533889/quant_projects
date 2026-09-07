# -*- coding: utf-8 -*-
"""输入侧数据质量检查：因子执行前校验依赖列可用性。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class InputDQThresholds:
    """输入侧 DQ 阈值。

    R21-106/107: ``expected_coverage`` 允许按列覆盖统一 ``min_non_null_ratio``
    （fundamental/event/news/snapshot 字段天然稀疏，阈值来自 FieldSpec /
    provider expected coverage，而不是一刀切 0.01）。
    """

    min_rows: int = 1
    min_non_null_ratio: float = 0.01
    min_instruments: int = 1
    max_inf_ratio: float = 0.0
    expected_coverage: dict[str, float] | None = None

    def coverage_for(self, column: str) -> float | None:
        if not self.expected_coverage:
            return None
        return self.expected_coverage.get(column) or self.expected_coverage.get(str(column).lower())


@dataclass
class InputColumnReport:
    """单列输入 DQ 检查结果。"""

    column: str
    passed: bool
    row_count: int
    non_null_ratio: float
    instrument_count: int
    message: str


@dataclass
class InputDQReport:
    """多列输入 DQ 报告。"""

    columns: list[InputColumnReport] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """是否全部列检查通过。"""
        return all(c.passed for c in self.columns)

    def to_dict(self) -> dict[str, Any]:
        """序列化为 JSON 友好字典。"""
        return {
            "passed": self.passed,
            "columns": [
                {
                    "column": c.column,
                    "passed": c.passed,
                    "row_count": c.row_count,
                    "non_null_ratio": c.non_null_ratio,
                    "instrument_count": c.instrument_count,
                    "message": c.message,
                }
                for c in self.columns
            ],
        }


class InputDQError(RuntimeError):
    """输入 DQ 门禁未通过。"""

    def __init__(self, report: InputDQReport):
        self.report = report
        failed = [c.column for c in report.columns if not c.passed]
        super().__init__(f"输入 DQ 未通过，列: {', '.join(failed)}")


def _series_stats(series: pd.Series) -> tuple[int, float, int]:
    n = len(series)
    if n == 0:
        return 0, 0.0, 0
    non_null = int(series.notna().sum())
    ratio = non_null / n
    inst = 0
    if isinstance(series.index, pd.MultiIndex) and series.index.nlevels >= 2:
        inst = int(series.index.get_level_values(-1).nunique())
    return n, ratio, inst


def evaluate_input_columns(
    data_source: Any,
    columns: set[str] | list[str],
    *,
    thresholds: InputDQThresholds | None = None,
    prefetched: dict[str, Any] | None = None,
) -> InputDQReport:
    """加载依赖列并检查行数/覆盖率/标的数。"""
    th = thresholds or InputDQThresholds()
    cols = sorted(set(columns))
    reports: list[InputColumnReport] = []

    load_columns = getattr(data_source, "load_columns", None)
    fetched: dict[str, Any] = dict(prefetched or {})
    missing_after_batch: list[str] = [n for n in cols if n not in fetched]
    if callable(load_columns) and len(missing_after_batch) > 1:
        try:
            fetched.update(load_columns(missing_after_batch))
            missing_after_batch = [n for n in cols if n not in fetched]
        except Exception:
            missing_after_batch = [n for n in cols if n not in fetched]

    for name in cols:
        try:
            if name in fetched:
                series = fetched[name]
            elif name in missing_after_batch:
                series = data_source.load_column(name)
            else:
                continue
        except Exception as exc:
            reports.append(
                InputColumnReport(
                    column=name,
                    passed=False,
                    row_count=0,
                    non_null_ratio=0.0,
                    instrument_count=0,
                    message=f"加载失败: {exc}",
                )
            )
            continue

        if not isinstance(series, pd.Series):
            reports.append(
                InputColumnReport(
                    column=name,
                    passed=False,
                    row_count=0,
                    non_null_ratio=0.0,
                    instrument_count=0,
                    message=f"期望 Series，得到 {type(series).__name__}",
                )
            )
            continue

        row_count, non_null_ratio, inst_count = _series_stats(series)
        inf_ratio = 0.0
        if row_count > 0:
            numeric = pd.to_numeric(series, errors="coerce")
            inf_ratio = float(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum()) / float(
                row_count
            )

        passed = True
        msgs: list[str] = []
        if row_count < th.min_rows:
            passed = False
            msgs.append(f"行数 {row_count} < {th.min_rows}")
        # R21-106/107: 列级 expected coverage（来自 FieldSpec/provider）优先；
        # 否则用统一 min_non_null_ratio。
        column_floor = th.coverage_for(name)
        effective_floor = column_floor if column_floor is not None else th.min_non_null_ratio
        if non_null_ratio < effective_floor:
            passed = False
            msgs.append(f"非空率 {non_null_ratio:.4f} < {effective_floor:.4f}")
        if inst_count < th.min_instruments:
            passed = False
            msgs.append(f"标的数 {inst_count} < {th.min_instruments}")
        if inf_ratio > th.max_inf_ratio:
            passed = False
            msgs.append(f"inf 比例 {inf_ratio:.4f} > {th.max_inf_ratio}")
        if isinstance(series.index, pd.MultiIndex) and series.index.nlevels >= 2:
            # R21-108: 输入列不得有重复 (timestamp, instrument) 主键。
            dup = int(series.index.duplicated().sum())
            if dup:
                passed = False
                msgs.append(f"重复主键 {dup}")

        reports.append(
            InputColumnReport(
                column=name,
                passed=passed,
                row_count=row_count,
                non_null_ratio=non_null_ratio,
                instrument_count=inst_count,
                message="OK" if passed else "; ".join(msgs),
            )
        )

    return InputDQReport(columns=reports)


def assert_input_dq(
    data_source: Any,
    columns: set[str] | list[str],
    *,
    thresholds: InputDQThresholds | None = None,
    raise_on_fail: bool = True,
    prefetched: dict[str, Any] | None = None,
) -> InputDQReport:
    """执行输入 DQ 检查；``raise_on_fail=True`` 时未通过则抛 :class:`InputDQError`。"""
    report = evaluate_input_columns(
        data_source, columns, thresholds=thresholds, prefetched=prefetched
    )
    if not report.passed and raise_on_fail:
        raise InputDQError(report)
    return report


def load_dataset_stats_for_source(data_source: Any) -> Any | None:
    """从 ``DataAccessSource`` 等加载 sidecar 统计（若存在）。"""
    dataset = getattr(data_source, "dataset", None)
    if not dataset:
        return None
    try:
        from data_access.read.stats import load_stats_sidecar
    except ImportError:
        return None
    root = getattr(data_source, "dataset_root", None)
    if root is None:
        try:
            from data_access import get_store

            ds = get_store().get_dataset(str(dataset))
            root = getattr(ds, "root", None)
        except Exception:
            return None
    if root is None:
        return None
    return load_stats_sidecar(Path(root) if not isinstance(root, Path) else root)


def adjust_input_dq_thresholds_from_stats(
    thresholds: InputDQThresholds | None,
    stats: Any | None,
    columns: set[str] | list[str],
    *,
    slack_ratio: float = 0.95,
) -> InputDQThresholds:
    """用 DataAccess sidecar stats 动态设定 ``min_non_null_ratio`` 下限。

    R21-111..113: ``DatasetStatsSnapshot.column_null_ratio`` 是 *null* 率
    （已核实 ``data_access/read/stats.py``，且有 [0,1] bounds 校验）。之前代码
    把 null 率当作非空率用 —— ``min(observed)*slack`` 会把一个 90% null 的列
    反而抬成 0.855 的“非空率”，与字面含义相反。这里先 ``1 - null_ratio``。
    """
    base = thresholds or InputDQThresholds()
    if stats is None:
        return base
    # R21-114: typed schema —— 分别解析 null/non-null/finite，不再混淆。
    null_map = getattr(stats, "column_null_ratio", None) or {}
    nonnull_map = getattr(stats, "column_non_null_ratio", None) or {}
    finite_map = getattr(stats, "column_finite_ratio", None) or {}
    cols = set(columns)
    observed: list[float] = []
    for c in cols:
        if c in nonnull_map:
            observed.append(float(nonnull_map[c]))
        elif c in finite_map:
            observed.append(float(finite_map[c]))
        elif c in null_map:
            observed.append(1.0 - float(null_map[c]))
    if not observed:
        return base
    floor = min(observed) * float(slack_ratio)
    return InputDQThresholds(
        min_rows=base.min_rows,
        min_non_null_ratio=max(base.min_non_null_ratio, min(floor, 1.0)),
        min_instruments=base.min_instruments,
        max_inf_ratio=base.max_inf_ratio,
        expected_coverage=dict(base.expected_coverage or {}),
    )


def expected_coverage_from_fields(
    columns: set[str] | list[str],
    *,
    field_registry: Any | None = None,
    provider_metadata: Any | None = None,
    default: float = 0.01,
) -> dict[str, float]:
    """R21-106/107: 从 FieldSpec / provider expected coverage 派生列级阈值。

    fundamental/event/news/snapshot 字段天然稀疏，不能统一用
    ``min_non_null_ratio=0.01``（那是针对量价日频列的）。返回
    ``{column: coverage_floor}``；未知列用 ``default``。
    """
    out: dict[str, float] = {}
    for col in columns:
        floor: float | None = None
        if field_registry is not None:
            try:
                spec = field_registry.get(str(col))
                if spec is not None:
                    cov = getattr(spec, "expected_coverage", None)
                    if cov is not None:
                        floor = float(cov)
            except Exception:
                floor = None
        if floor is None and provider_metadata is not None:
            try:
                cov = getattr(provider_metadata, str(col), None)
                if cov is not None:
                    floor = float(cov)
            except Exception:
                floor = None
        out[str(col)] = floor if floor is not None else default
    return out


def source_freshness_report(
    data_source: Any,
    *,
    latest_source_timestamp: Any | None = None,
    expected_trading_session: str | None = None,
    max_lag_days: float | None = None,
) -> InputDQReport:
    """R21-109: source freshness 检查（latest ts / expected session / lag / SLA）。

    返回一个只有一条 freshness 检查的 ``InputDQReport``；``max_lag_days`` 为
    None 时不做绝对 lag 门禁，仅报告。
    """
    import datetime as _dt

    report: list[InputColumnReport] = []
    lag_days: float | None = None
    try:
        latest_ts = latest_source_timestamp
        if latest_ts is None:
            latest_ts = getattr(data_source, "latest_timestamp", None)
        if hasattr(data_source, "max_timestamp"):
            latest_ts = latest_ts or data_source.max_timestamp()
        if latest_ts is not None and not isinstance(latest_ts, str):
            if hasattr(latest_ts, "to_pydatetime"):
                latest_ts = latest_ts.to_pydatetime()
            now = _dt.datetime.now(_dt.timezone.utc)
            if latest_ts.tzinfo is None:
                latest_ts = latest_ts.replace(tzinfo=_dt.timezone.utc)
            lag_days = (now - latest_ts).total_seconds() / 86400.0
    except Exception:
        lag_days = None
    passed = True
    msgs: list[str] = []
    if lag_days is not None and max_lag_days is not None and lag_days > max_lag_days:
        passed = False
        msgs.append(f"source lag {lag_days:.2f}d > SLA {max_lag_days}d")
    if expected_trading_session and latest_ts is None:
        passed = False
        msgs.append(f"expected trading session {expected_trading_session} but no timestamp found")
    report.append(
        InputColumnReport(
            column="__source_freshness__",
            passed=passed,
            row_count=0,
            non_null_ratio=1.0 if lag_days is not None else 0.0,
            instrument_count=0,
            message="OK" if passed else "; ".join(msgs) or f"lag_days={lag_days}",
        )
    )
    return InputDQReport(columns=report)
