"""
data_access.r30.coverage_service —— R30-P0-007 CoverageService（concept/market/
provider/timeframe/universe 全维度覆盖描述）

R30 全维度成熟度层是 **additive** 的。本模块在既有 ``read/coverage.compute_coverage``
（declared/observed 区间 + complete/partial/unavailable 状态）之上，扩展成
**多维度覆盖描述**：

    - 细粒度分布：``by_year`` / ``by_date`` / ``by_instrument``；
    - 覆盖率统计：``overall_coverage`` + ``quantiles``（by_date 的分位数）；
    - 每分区行数优先来自 ``DatasetManifest.files``（row_count/min_time/max_time）；
      取不到时用文件数近似并在 ``authority`` 标记 ``approximate``；
    - ``by_instrument`` 需要 instrument 列维度——manifest 没有该粒度时按
      「每文件存在即覆盖该文件内 instruments」近似，并在 ``problems`` 注明；
    - ``coverage_parity_fe``：迁移期与 FE ``HistoricalCoverageContract`` 逐字段对比。

**不修改** ``read/coverage.py`` / ``read/metadata_plane.py`` / ``read/manifest.py``。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Mapping, Sequence

from data_access.read.coverage import CoverageReport, compute_coverage


# ---------------------------------------------------------------------------
# 覆盖描述对象
# ---------------------------------------------------------------------------
@dataclass
class CoverageDescription:
    concept: str | None = None
    market: str | None = None
    dataset: str = ""
    provider: str | None = None
    timeframe: str | None = None
    universe: str | None = None
    source_snapshot: str | None = None
    first_valid_date: str | None = None
    last_valid_date: str | None = None
    overall_coverage: float = 0.0
    by_year: dict[int, float] = field(default_factory=dict)
    by_date: dict[str, float] = field(default_factory=dict)
    by_instrument: dict[str, float] = field(default_factory=dict)
    quantiles: dict[str, float] = field(default_factory=dict)
    source_version: str | None = None
    snapshot_id: str | None = None
    authority: str = "authoritative"
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept": self.concept,
            "market": self.market,
            "dataset": self.dataset,
            "provider": self.provider,
            "timeframe": self.timeframe,
            "universe": self.universe,
            "source_snapshot": self.source_snapshot,
            "first_valid_date": self.first_valid_date,
            "last_valid_date": self.last_valid_date,
            "overall_coverage": self.overall_coverage,
            "by_year": {str(k): float(v) for k, v in self.by_year.items()},
            "by_date": {str(k): float(v) for k, v in self.by_date.items()},
            "by_instrument": {str(k): float(v) for k, v in self.by_instrument.items()},
            "quantiles": {str(k): float(v) for k, v in self.quantiles.items()},
            "source_version": self.source_version,
            "snapshot_id": self.snapshot_id,
            "authority": self.authority,
            "problems": list(self.problems),
        }


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _infer_market(dataset: str | None) -> str:
    name = str(dataset or "").strip().lower()
    if name.startswith("ashare") or name.startswith("a_"):
        return "ashare"
    if name.startswith("us") or name.startswith("am"):
        return "us"
    return "any"


def _iter_days_between(lo: str | None, hi: str | None, *, cap: int = 5000) -> list[str]:
    """把 [lo, hi]（iso 字符串）展开成日期列表；跨度超 cap 只返回端点。"""
    if not lo or not hi:
        return []
    try:
        d_lo = date.fromisoformat(str(lo)[:10])
        d_hi = date.fromisoformat(str(hi)[:10])
    except ValueError:
        return []
    if d_hi < d_lo:
        return []
    if (d_hi - d_lo).days > cap:
        # 超长区间：只统计端点，避免爆炸（按年聚合仍然可用）
        return [d_lo.isoformat(), d_hi.isoformat()]
    out: list[str] = []
    cur = d_lo
    while cur <= d_hi:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def _file_rows(manifest: Any) -> list[dict[str, Any]]:
    """把 manifest（DatasetManifest 或 duck-typed）files 归一化成 dict 列表。"""
    files = getattr(manifest, "files", None) or ()
    out: list[dict[str, Any]] = []
    for f in files:
        out.append(
            {
                "path": getattr(f, "path", None) or str(f),
                "rows": getattr(f, "rows", None),
                "bytes": getattr(f, "bytes", None),
                "min_time": getattr(f, "min_time", None),
                "max_time": getattr(f, "max_time", None),
                "min_instrument": getattr(f, "min_instrument", None),
                "max_instrument": getattr(f, "max_instrument", None),
            }
        )
    return out


def _daily_distribution(
    files: Sequence[dict[str, Any]],
) -> tuple[dict[str, float], dict[int, float], bool]:
    """从每分区行数/时间算 by_date + by_year。

    返回 ``(by_date, by_year, has_rows)``：
        - 有行数 → by_date = 当日行数 / 最大单日行数；by_year = 按年行数占比；
        - 无行数（文件数近似）→ by_date = 每日期 1.0（有文件即覆盖）；
          by_year 按文件数占比。
    """
    daily: dict[str, float] = {}
    has_rows = any(f.get("rows") is not None for f in files)
    for f in files:
        rows = f.get("rows")
        if rows is None:
            rows = 1
        weight = float(rows)
        lo, hi = f.get("min_time"), f.get("max_time")
        if lo and hi:
            for d in _iter_days_between(str(lo), str(hi)):
                daily[d] = daily.get(d, 0.0) + weight
        else:
            # 无时间信息：从路径日期近似
            dt = _parse_path_date(f.get("path") or "")
            if dt:
                daily[dt] = daily.get(dt, 0.0) + weight
    if not daily:
        return {}, {}, has_rows
    if has_rows:
        max_day = max(daily.values())
        if max_day > 0:
            by_date = {d: v / max_day for d, v in daily.items()}
        else:
            by_date = {d: 1.0 for d in daily}
    else:
        by_date = {d: 1.0 for d in daily}
    # by_year：按年行数（或文件数）占比
    year_total: dict[int, float] = {}
    for d, v in daily.items():
        try:
            y = int(str(d)[:4])
        except (ValueError, TypeError):
            continue
        year_total[y] = year_total.get(y, 0.0) + v
    tot = sum(year_total.values())
    by_year = {y: v / tot for y, v in year_total.items()} if tot > 0 else {}
    return by_date, by_year, has_rows


def _parse_path_date(path: str) -> str | None:
    for tok in str(path).split("/"):
        if len(tok) >= 10:
            try:
                return date.fromisoformat(tok[:10]).isoformat()
            except ValueError:
                continue
    return None


def _percentile(values: Sequence[float], p: float) -> float:
    """nearest-rank 分位数（p ∈ (0, 100]）。空输入返回 0.0。"""
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    idx = max(0, int(math.ceil(p / 100.0 * len(vals))) - 1)
    return vals[min(idx, len(vals) - 1)]


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    return {
        "p25": _percentile(values, 25),
        "p50": _percentile(values, 50),
        "p75": _percentile(values, 75),
        "p90": _percentile(values, 90),
        "max": _percentile(values, 100),
    }


def _declared_days(report: CoverageReport) -> int | None:
    try:
        d_lo = date.fromisoformat(report.declared_start)
        d_hi = date.fromisoformat(report.declared_end)
    except (ValueError, TypeError):
        return None
    if d_hi < d_lo:
        return None
    return (d_hi - d_lo).days + 1


# ---------------------------------------------------------------------------
# CoverageService
# ---------------------------------------------------------------------------
class CoverageService:
    """R30-P0-007 覆盖描述服务。

    类方法/实例方法均可调用；``describe`` 是最小完整入口。
    """

    @staticmethod
    def _load_manifest(
        store: Any, dataset: str, params: Mapping[str, Any] | None = None
    ) -> Any | None:
        """加载含 per-file 细节的 manifest（复用 R30 索引层同一来源）。"""
        try:
            plane = store.metadata_plane(dataset, **dict(params or {}))
        except Exception:
            plane = None
        if plane is not None:
            try:
                return plane.manifest()
            except Exception:
                return None
        return None

    @classmethod
    def from_manifest(
        cls, store: Any, dataset: str, manifest: Any | None = None
    ) -> Any | None:
        """帮助函数：返回 manifest（未传则加载）。"""
        if manifest is not None:
            return manifest
        return cls._load_manifest(store, dataset)

    @classmethod
    def describe(
        cls,
        store: Any,
        dataset: str,
        *,
        concept: str | None = None,
        market: str | None = None,
        provider: str | None = None,
        timeframe: str | None = None,
        universe: str | None = None,
        source_snapshot: str | None = None,
    ) -> CoverageDescription:
        """多维度覆盖描述。

        - 既有 ``compute_coverage`` 提供 declared/observed 区间 + 状态 + authority；
        - per-partition 行数/时间优先从 ``DatasetManifest.files`` 取；
        - overall = 声明跨度内有数据的日期占比（无声明则 by_date 均值）；
        - quantiles = by_date 的 p25/p50/p75/p90/max；
        - by_instrument：manifest 无 instrument 粒度时近似为空 + problems 注明。
        """
        params = {
            k: v
            for k, v in (
                ("provider", provider),
                ("timeframe", timeframe),
                ("universe", universe),
                ("source_snapshot", source_snapshot),
            )
            if v is not None
        }
        try:
            report = compute_coverage(store, dataset, params=params or None)
        except Exception as exc:  # 防御：compute_coverage 失败不阻塞描述
            report = CoverageReport(
                dataset=dataset, status="unknown", problems=[f"compute_coverage 失败: {exc}"]
            )
        manifest = cls._load_manifest(store, dataset, params)
        files = _file_rows(manifest) if manifest is not None else []
        by_date, by_year, has_rows = _daily_distribution(files)

        problems = list(report.problems or [])
        authority = report.authority or "authoritative"
        if files and not has_rows:
            authority = "approximate"
            problems.append(
                "manifest 无每分区行数，by_year/by_date 用文件数近似（authority=approximate）"
            )
        if not files and report.status == "unknown":
            problems.append("无 manifest per-file 细节，分布为空")

        # by_instrument：manifest 无 instrument 粒度 → 近似空 + 注明
        by_instrument: dict[str, float] = {}
        if files:
            any_inst = any(
                f.get("min_instrument") or f.get("max_instrument") for f in files
            )
            if not any_inst:
                problems.append(
                    "manifest 无 instrument 粒度，by_instrument 无法精确计算；"
                    "近似为「每文件存在即覆盖该文件内 instruments」（返回空分布）"
                )
            else:
                problems.append(
                    "by_instrument 按「每文件存在即覆盖该文件内 instruments」近似，"
                    "未做 instrument 级精确统计"
                )

        # first/last valid date：observed 优先，其次 by_date 极值
        first_valid = report.observed_start
        last_valid = report.observed_end
        if not first_valid and by_date:
            first_valid = min(by_date)
        if not last_valid and by_date:
            last_valid = max(by_date)

        # overall coverage
        if _declared_days(report):
            observed_days = sum(1 for v in by_date.values() if v > 0)
            overall = min(1.0, observed_days / _declared_days(report))
        elif by_date:
            overall = sum(by_date.values()) / len(by_date) if by_date else 0.0
        else:
            overall = 1.0 if report.observed_files > 0 else 0.0

        source_version = None
        snapshot_id = None
        if manifest is not None:
            source_version = (
                getattr(manifest, "manifest_built_epoch", None)
                or getattr(manifest, "source_epoch", None)
                or getattr(manifest, "manifest_epoch", None)
            )
            snapshot_id = getattr(manifest, "dataset_version", None) or getattr(
                manifest, "partition_version", None
            )
        if source_version is None:
            source_version = report.observed_files and f"files={report.observed_files}" or None

        return CoverageDescription(
            concept=concept or dataset,
            market=market or _infer_market(dataset),
            dataset=dataset,
            provider=provider,
            timeframe=timeframe,
            universe=universe,
            source_snapshot=source_snapshot,
            first_valid_date=first_valid,
            last_valid_date=last_valid,
            overall_coverage=overall,
            by_year=by_year,
            by_date=by_date,
            by_instrument=by_instrument,
            quantiles=_quantiles(list(by_date.values())),
            source_version=source_version,
            snapshot_id=snapshot_id,
            authority=authority,
            problems=problems,
        )


# ---------------------------------------------------------------------------
# FE HistoricalCoverageContract parity（R30-P0-007 迁移期）
# ---------------------------------------------------------------------------
def _dict_close(a: Mapping[Any, float], b: Mapping[Any, float], tol: float = 1e-3) -> bool:
    ka, kb = set(a), set(b)
    if ka != kb:
        return False
    for k in ka:
        try:
            if abs(float(a[k]) - float(b[k])) > tol:
                return False
        except (TypeError, ValueError):
            if a[k] != b[k]:
                return False
    return True


def coverage_parity_fe(
    store: Any, dataset: str, historical_contract: Mapping[str, Any]
) -> dict[str, Any]:
    """对比 FE ``HistoricalCoverageContract`` 与 DA 观测，列出差异字段。

    ``historical_contract`` 可以是 dataclass 实例或 dict，字段：
    ``first_valid_date / coverage_by_year / coverage_by_stock / coverage_by_date /
    coverage_ratio``。返回 ``{"dataset", "fields", "differing_fields", "match_all"}``。
    """
    desc = CoverageService.describe(store, dataset)

    def _get(key: str) -> Any:
        try:
            return historical_contract[key]
        except (KeyError, TypeError):
            return getattr(historical_contract, key, None)

    fe_fvd = _get("first_valid_date")
    fields: dict[str, dict[str, Any]] = {
        "first_valid_date": {
            "fe": fe_fvd,
            "da": desc.first_valid_date,
            "match": fe_fvd == desc.first_valid_date,
        },
        "coverage_by_year": {
            "fe": _get("coverage_by_year") or {},
            "da": desc.by_year,
            "match": _dict_close(_get("coverage_by_year") or {}, desc.by_year),
        },
        "coverage_by_stock": {
            "fe": _get("coverage_by_stock") or {},
            "da": desc.by_instrument,
            "match": _dict_close(
                _get("coverage_by_stock") or {}, desc.by_instrument
            ),
        },
        "coverage_by_date": {
            "fe": _get("coverage_by_date") or {},
            "da": desc.by_date,
            "match": _dict_close(_get("coverage_by_date") or {}, desc.by_date),
        },
    }
    fe_ratio = _get("coverage_ratio")
    if fe_ratio is not None:
        try:
            ratio_match = abs(float(fe_ratio) - desc.overall_coverage) <= 1e-3
        except (TypeError, ValueError):
            ratio_match = False
        fields["coverage_ratio"] = {
            "fe": fe_ratio,
            "da": desc.overall_coverage,
            "match": ratio_match,
        }
    differing = [k for k, v in fields.items() if v["match"] is not True]
    return {
        "dataset": dataset,
        "fields": fields,
        "differing_fields": differing,
        "match_all": not differing,
    }


__all__ = [
    "CoverageDescription",
    "CoverageService",
    "coverage_parity_fe",
]
