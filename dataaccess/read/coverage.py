"""
data_access.read.coverage —— 数据集覆盖/完整性/陈旧度契约（#14）

为什么需要
    有些数据集覆盖明显落后于行情（如 A 股 StockCapitalDaily）、美股 Valuation/
    Indicator 只有约 49 日 X0、halt 辅助数据某些日期根本没文件。过去只能
    读出一堆 NaN 才发现。本模块在查询前回答：

        「这张表在你的研究窗口内是 complete / partial / unavailable」

声明侧（declared）
    来自 COS 契约：``coverage_start/end / expected_cadence / max_staleness /
    missing_partition_semantics``。

观测侧（observed）
    从数据集实际文件（本地 glob 展开 / manifest min-max）推导覆盖区间与文件数。

判定
    - 无文件 → unavailable（``missing_partition_semantics=empty_ok`` 除外）；
    - 声明区间 ⊆ 观测区间 → complete；
    - 否则 partial，并列出问题。

维护人：quant 基础平台组    最后更新：2026-08-08
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass
class CoverageReport:
    dataset: str
    declared_start: str | None = None
    declared_end: str | None = None
    expected_cadence: str | None = None
    max_staleness: str | None = None
    missing_semantics: str = "error"
    observed_start: str | None = None
    observed_end: str | None = None
    observed_files: int = 0
    status: str = "unknown"      # complete / partial / unavailable / unknown
    problems: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.problems is None:
            self.problems = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "declared_start": self.declared_start,
            "declared_end": self.declared_end,
            "expected_cadence": self.expected_cadence,
            "max_staleness": self.max_staleness,
            "missing_semantics": self.missing_semantics,
            "observed_start": self.observed_start,
            "observed_end": self.observed_end,
            "observed_files": self.observed_files,
            "status": self.status,
            "problems": list(self.problems or []),
        }


def _file_dates(paths: Sequence[str]) -> list[_dt.date]:
    """从文件路径里解析日期（``YYYY-MM-DD`` 片段）。"""
    out: list[_dt.date] = []
    for p in paths:
        for tok in str(p).split("/"):
            try:
                if len(tok) >= 10:
                    out.append(_dt.date.fromisoformat(tok[:10]))
            except ValueError:
                continue
    return out


def _observed_from_manifest(store: Any, ds: Any) -> tuple[str | None, str | None]:
    from data_access.read.manifest import (
        DatasetManifest,
        is_manifest_fresh,
        manifest_root_for_paths,
    )

    try:
        raw_paths = store._resolve_raw_paths(ds, time_range=None, params={})
    except Exception:
        return None, None
    root = manifest_root_for_paths(raw_paths)
    if root is None:
        return None, None
    try:
        manifest = DatasetManifest.load(root)
        if manifest is not None and is_manifest_fresh(manifest, raw_paths):
            return manifest.min_time_key, manifest.max_time_key
    except Exception:
        pass
    return None, None


def compute_coverage(
    store: Any,
    dataset: str,
    *,
    params: Mapping[str, Any] | None = None,
    max_glob: int = 50000,
) -> CoverageReport:
    """计算一个数据集的覆盖/完整性报告。"""
    from data_access.cos_contract import get_cos_contract

    contract = get_cos_contract(dataset)
    report = CoverageReport(dataset=dataset)
    if contract is not None:
        report.declared_start = contract.coverage_start
        report.declared_end = contract.coverage_end
        report.expected_cadence = contract.expected_cadence
        report.max_staleness = contract.max_staleness
        report.missing_semantics = contract.missing_partition_semantics

    ds = store._registry.get(dataset)
    try:
        paths = store._resolve_raw_paths(
            ds, time_range=None, params=dict(params or {})
        )
    except Exception:
        report.problems.append("路径解析失败")
        report.status = "unavailable"
        return report

    # 具体文件：manifest min/max 优先，否则 glob 展开
    obs_start, obs_end = _observed_from_manifest(store, ds)
    files: list[str] = []
    glob_failed = False
    glob_truncated = False
    for g in paths:
        if str(g).startswith(("s3://", "cos://")):
            continue
        try:
            tbl = store._engine.execute_arrow(
                "SELECT file FROM glob(?) LIMIT ?", [str(g), max_glob], deadline_ms=None
            )
        except Exception:
            # #P1-46 任一 glob root 失败 → 观测不完整，绝不能判 complete。
            glob_failed = True
            report.problems.append(f"glob 失败: {g}")
            continue
        rows = [str(r["file"]) for r in tbl.to_pylist()]
        files.extend(rows)
        # #P1-45 enumeration 触顶（安全阈值截断）→ 观测不完整。
        if len(rows) >= max_glob:
            glob_truncated = True
            report.problems.append(
                f"glob {g} 达到枚举上限 {max_glob}，可能被截断"
            )
    files = sorted(set(files))
    report.observed_files = len(files)

    if files:
        dates = _file_dates(files)
        if dates and obs_start is None:
            obs_start = min(dates).isoformat()
        if dates and obs_end is None:
            obs_end = max(dates).isoformat()
    report.observed_start = obs_start
    report.observed_end = obs_end

    if not files and not (obs_start or obs_end):
        report.status = "unavailable"
        report.problems.append(
            "无任何数据文件"
            + ("" if report.missing_semantics == "empty_ok" else "（missing_partition_semantics=error）")
        )
        return report

    # 判定
    if report.declared_start and report.declared_end:
        try:
            d_lo = _dt.date.fromisoformat(report.declared_start)
            d_hi = _dt.date.fromisoformat(report.declared_end)
        except ValueError:
            d_lo = d_hi = None
        o_lo = _dt.date.fromisoformat(report.observed_start) if report.observed_start else None
        o_hi = _dt.date.fromisoformat(report.observed_end) if report.observed_end else None
        if d_lo and d_hi and o_lo and o_hi:
            if d_lo >= o_lo and d_hi <= o_hi:
                # #P1-45/#P1-46 枚举触顶或某 root glob 失败 → 观测不完整，不能判 complete。
                if glob_failed or glob_truncated:
                    report.status = "partial"
                    report.problems.append(
                        "观测枚举不完整（glob 失败或触顶），无法证明声明区间全覆盖"
                    )
                else:
                    report.status = "complete"
                # #24 max_staleness 判定：末个 partition 落后超过阈值 → stale
                _maybe_mark_stale(report)
                return report
            if o_lo and o_hi:
                report.status = "partial"
                if o_lo and d_lo and o_lo > d_lo:
                    report.problems.append(
                        f"观测起始 {o_lo.isoformat()} 晚于声明起始 {d_lo.isoformat()}"
                    )
                if o_hi and d_hi and o_hi < d_hi:
                    report.problems.append(
                        f"观测终止 {o_hi.isoformat()} 早于声明终止 {d_hi.isoformat()}"
                    )
                _maybe_mark_stale(report)
                return report
    if glob_failed or glob_truncated:
        report.status = "partial"
        report.problems.append(
            "观测枚举不完整（glob 失败或触顶），不能判 complete"
        )
    else:
        report.status = "complete" if files else "unavailable"
    _maybe_mark_stale(report)
    return report


def _maybe_mark_stale(report: CoverageReport) -> None:
    """#24 max_staleness 真正执行：末个 partition 落后超过阈值 → status=stale。

    ``max_staleness`` 格式：``"5d"``（自然日）或 ``"5t"``（交易日）。无该声明或
    无 observed_end 时不判定。
    """
    ms = (report.max_staleness or "").strip().lower()
    if not ms or not report.observed_end:
        return
    try:
        unit = ms[-1]
        num = int(ms[:-1])
    except (ValueError, IndexError):
        return
    try:
        o_hi = _dt.date.fromisoformat(report.observed_end)
    except ValueError:
        return
    today = _dt.date.today()
    if unit == "d":
        lag = (today - o_hi).days
    elif unit == "t":
        # 交易日滞后：自然日 / 7 * 5 近似（无日历时）；有日历可精确算
        lag = max(0, int(round((today - o_hi).days * 5 / 7)))
    else:
        return
    if lag > num:
        report.status = "stale"
        report.problems.append(
            f"末个 partition {o_hi.isoformat()} 落后 {lag}（{'交易' if unit == 't' else '自然'}日），"
            f"超过 max_staleness={report.max_staleness}"
        )


def coverage_status(store: Any, dataset: str, **kwargs: Any) -> str:
    """便捷：直接返回状态字符串。"""
    return compute_coverage(store, dataset, **kwargs).status


__all__ = ["CoverageReport", "compute_coverage", "coverage_status"]
