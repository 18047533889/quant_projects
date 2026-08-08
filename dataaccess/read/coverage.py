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
    status: str = "unknown"      # complete / partial / unavailable / unknown / stale
    # #P0 收官（0.9.5）：判定权威性。``authoritative`` = 判定完全基于真实数据/
    # 交易日历；``approximate`` = 至少一部分（如 ``max_staleness=5t`` 的交易日陈旧度）
    # 用了自然日近似——strict 下近似会降级成 partial，绝不声称权威 complete/stale。
    authority: str = "authoritative"
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
            "authority": self.authority,
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


def _observed_from_manifest(
    store: Any, ds: Any, params: Mapping[str, Any] | None = None
) -> tuple[str | None, str | None]:
    """从 manifest min/max 读观测区间。

    #P0 收官（0.9.5）：``params`` 必须与主流程同源的 **validated** params 贯穿到
    这里——旧代码硬编码 ``params={}``，对 ``factor_id`` / ``universe`` / model 版本
    等 ParametricDataset，coverage 问的是某个实例，manifest lookup 却去查另一个
    （未参数化的）实例，min/max 完全对不上。
    """
    from data_access.read.manifest import (
        DatasetManifest,
        is_manifest_fresh,
        manifest_root_for_paths,
    )

    try:
        raw_paths = store._resolve_raw_paths(
            ds, time_range=None, params=dict(params or {})
        )
    except Exception:
        return None, None
    root = manifest_root_for_paths(raw_paths)
    if root is None:
        return None, None
    try:
        manifest = DatasetManifest.load(root)
        if manifest is not None and is_manifest_fresh(manifest, raw_paths):
            # #P0 收官（0.9.5）：DatasetManifest **没有** min_time_key/max_time_key
            # 属性——旧代码读不存在的属性被下面的 ``except Exception`` 静默吞掉，
            # manifest fast path 从未真正生效（每次都回退 glob）。从 files 的
            # min_time/max_time 推导观测区间。
            times = [f.min_time for f in manifest.files if f.min_time is not None]
            if times:
                return min(times), max(times)
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

    # 具体文件：manifest min/max 优先，否则 glob 展开（#P0 收官 0.9.5：manifest
    # lookup 用与主流程相同的 validated params）
    obs_start, obs_end = _observed_from_manifest(store, ds, params=dict(params or {}))
    files: list[str] = []
    glob_failed = False
    glob_truncated = False
    remote_only = True
    for g in paths:
        if str(g).startswith(("s3://", "cos://")):
            # #P1-final closure 20：remote-only 数据集本地无法 glob——覆盖区间只能
            # 由 manifest（或 COS 枚举）提供。这里不跳过就误判，显式记录 remote，
            # 避免「本地无文件 ⇒ unavailable」掩盖 remote 数据存在。
            continue
        remote_only = False
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
    if remote_only and paths:
        # 全部路径都是 s3/cos：本地枚举不到，覆盖由 manifest 决定；无 manifest 时
        # 下方会判 unavailable（fail-closed），但问题描述要说明是 remote-only。
        report.problems.append(
            "remote-only 数据集（s3:// 或 cos:// 路径），本地无法 glob 枚举；"
            "覆盖区间依赖 manifest 或 COS 枚举"
        )

    if files:
        dates = _file_dates(files)
        if dates and obs_start is None:
            obs_start = min(dates).isoformat()
        if dates and obs_end is None:
            obs_end = max(dates).isoformat()
    report.observed_start = obs_start
    report.observed_end = obs_end

    if not files and not (obs_start or obs_end):
        # #P1-final closure 20：``empty_ok`` 声明下「无分区」是**合法空数据集**——
        # 判 complete（没缺东西），不是 unavailable；否则远程/新数据集刚上线时
        # 会被误报「数据缺失」。
        if report.missing_semantics == "empty_ok":
            report.status = "complete"
            report.problems.append(
                "empty_ok：无分区是合法空数据集（声明允许空）"
            )
            return report
        if remote_only:
            # #35 收官轮：remote-only 数据集本地无法观测（s3:// / cos:// 路径，无
            # 本地文件/manifest），但 cos_contract 声明了权威区间——**不能永久误报
            # unavailable**（数据可能在 COS 上完好）。基于合同声明给出 partial +
            # authority=declared_remote，显式说明覆盖是「合同声明」而非本地观测；
            # 无合同声明时也是 partial（honest「本地不可观测」），不是 unavailable。
            report.status = "partial"
            if report.declared_start or report.declared_end:
                report.authority = "declared_remote"
                report.problems.append(
                    "remote-only 数据集：本地无法观测，覆盖基于 cos_contract 声明区间"
                    "（authority=declared_remote）。如需权威观测，请同步 mirror 或用 "
                    "COS 枚举（load_mirror_inventory）。"
                )
            else:
                report.problems.append(
                    "remote-only 且无合同声明区间：本地不可观测，无法判定覆盖"
                )
            return report
        report.status = "unavailable"
        report.problems.append("无任何数据文件（missing_partition_semantics=error）")
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
                _maybe_mark_stale(report, store=store, dataset=dataset)
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
                _maybe_mark_stale(report, store=store, dataset=dataset)
                return report
    if glob_failed or glob_truncated:
        report.status = "partial"
        report.problems.append(
            "观测枚举不完整（glob 失败或触顶），不能判 complete"
        )
    else:
        report.status = "complete" if files else "unavailable"
    _maybe_mark_stale(report, store=store, dataset=dataset)
    return report


def _infer_market(dataset: str) -> str:
    """从数据集名推断市场（coverage 交易日 staleness 用）。"""
    name = str(dataset or "").strip().lower()
    if name.startswith("ashare") or name.startswith("a_"):
        return "ashare"
    if name.startswith("us") or name.startswith("am"):
        return "us"
    return ""


def _trading_day_lag(
    o_hi: _dt.date,
    today: _dt.date,
    *,
    store: Any,
    dataset: str,
    strict: bool,
) -> tuple[int | None, bool]:
    """observed_end 到今天之间的**交易日**数。

    返回 ``(lag, authoritative)``：
        - 真实 MarketCalendar 可用 → (真实交易日数, True)；
        - 市场未知 / 日历不可用 / 加载失败：
            - ``strict`` → ``(None, False)``——**绝不静默退化**成自然日近似
              （production 的「5 个交易日」必须真·5 个交易日，否则 complete/stale
              判定撒谎）。调用方据此降级 status=partial + authority=approximate；
            - research → ``(round(days*5/7), False)`` 近似，并标记 approximate。
    """
    market = _infer_market(dataset)
    if market and store is not None:
        try:
            from data_access.read.session_calendar import get_market_calendar

            cal = get_market_calendar(market, store=store)
            if cal is not None and cal.has_data:
                lag = sum(1 for d in cal.trading_days if o_hi < d <= today)
                return lag, True
        except Exception:
            pass
    # 日历不可用 / 市场未知
    if strict:
        return None, False
    return max(0, int(round((today - o_hi).days * 5 / 7))), False


def _maybe_mark_stale(
    report: CoverageReport, *, store: Any = None, dataset: str = ""
) -> None:
    """#24 max_staleness 真正执行：末个 partition 落后超过阈值 → status=stale。

    ``max_staleness`` 格式：``"5d"``（自然日）或 ``"5t"``（交易日）。无该声明或
    无 observed_end 时不判定。

    #P0 收官（0.9.5）：``5t`` 需要真实交易日历。strict/production 下日历不可用
    ⇒ 无法权威判定——status 降级 ``partial`` + ``authority=approximate``（fail-closed，
    绝不把近似当成权威 complete/stale）；research 才允许自然日近似并显式标记。
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
        authoritative = True
    elif unit == "t":
        from data_access.read.query_budget import is_strict_semantics

        strict = is_strict_semantics()
        lag, authoritative = _trading_day_lag(
            o_hi, today, store=store, dataset=dataset, strict=strict
        )
        if lag is None:
            # strict + 日历不可用 → 无法权威判定交易日陈旧度（fail-closed）
            report.authority = "approximate"
            report.status = "partial"
            report.problems.append(
                f"max_staleness={report.max_staleness}（交易日）需要真实交易日历，"
                "但日历不可用（strict fail-closed）：无法权威判定 complete/stale。"
            )
            return
    else:
        return
    if not authoritative:
        report.authority = "approximate"
        report.problems.append(
            "交易日陈旧度使用自然日近似（真实交易日历不可用）"
        )
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
