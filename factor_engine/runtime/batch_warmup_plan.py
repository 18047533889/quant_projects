# -*- coding: utf-8 -*-
"""R39-P0-PERF-014 / PERF-015：全 batch warmup 规划只算一次 + 扫描成本感知分组。

老路径（``runtime.batch_service``）在不同位置多次调用
``prepare_run_warmup``（成本聚类、batch union warmup、recursive wave），每个因子
被重复算多次。本模块新增：

- :class:`BatchWarmupPlan`：一次性计算 ``analysis → factor RunWindow →
  扫描成本感知分组``，后续所有路径只消费这一个结果。
- :func:`compute_batch_warmup_plan`：按 PERF-015 的成本模型（合并两组的额外
  superset scan bytes vs 分开扫描的重复 scan/decode 成本）分组，替代固定
  short/medium/long 比例桶；``full_history`` 保持独立组。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class WarmupGroup:
    """扫描成本感知的 warmup 分组：共享一个 union 加载窗口。"""

    factor_names: tuple[str, ...]
    run_window: Any | None
    load_start: str | None
    load_end: str | None
    scan_bytes: int
    source_scope: str = ""


@dataclass(frozen=True)
class BatchWarmupPlan:
    """全 batch 只算一次的 warmup 规划结果。"""

    per_factor: dict[str, Any]          # factor.name -> RunWindow
    groups: list[WarmupGroup]

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_count": len(self.per_factor),
            "group_count": len(self.groups),
            "groups": [
                {
                    "factor_names": list(g.factor_names),
                    "load_start": g.load_start,
                    "load_end": g.load_end,
                    "scan_bytes": g.scan_bytes,
                    "source_scope": g.source_scope,
                }
                for g in self.groups
            ],
        }


class _FactorProxy:
    """dag.roots 上最小 Factor 兼容视图（以 ``factor_name`` 为 name）。"""

    __slots__ = ("_fp",)

    def __init__(self, fp: Any) -> None:
        self._fp = fp

    @property
    def name(self) -> str | None:
        return getattr(self._fp, "factor_name", None) or getattr(
            self._fp, "name", None
        )

    def __getattr__(self, item: str) -> Any:
        return getattr(self._fp, item)


def _instrument_count(data_source: Any) -> int:
    try:
        count = getattr(data_source, "instrument_count", None)
        if count:
            return int(count)
        scope = getattr(data_source, "instrument_filter", None)
        if scope:
            return max(1, len(scope))
    except Exception:  # noqa: BLE001
        pass
    return 500


def _days_between(start: str | None, end: str | None) -> int:
    if not start or not end:
        return 252
    try:
        import pandas as pd

        return max(1, len(pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))))
    except Exception:  # noqa: BLE001
        return 252


def _source_bounds(engine: Any) -> tuple[str | None, str | None]:
    try:
        from factor_engine.runtime.run_window import extract_source_date_bounds

        start, end = extract_source_date_bounds(engine.data_source)
        return start, end
    except Exception:  # noqa: BLE001
        return None, None


def _load_start(rw: Any, engine: Any) -> str | None:
    src_start, _ = _source_bounds(engine)
    return (
        getattr(rw, "actual_load_start", None)
        or getattr(rw, "requested_start", None)
        or src_start
    )


def _load_end(rw: Any, engine: Any) -> str | None:
    _, src_end = _source_bounds(engine)
    return (
        getattr(rw, "actual_load_end", None)
        or getattr(rw, "requested_end", None)
        or src_end
    )


def _estimate_factor_scan_bytes(engine: Any, analysis: Any, rw: Any) -> int:
    """按加载窗口业务日 × instrument 数 × 引用列数 × 8B 估算单因子 scan bytes。"""
    days = _days_between(_load_start(rw, engine), _load_end(rw, engine))
    instruments = _instrument_count(engine.data_source)
    cols = max(1, len(getattr(analysis, "referenced_columns", ()) or ()))
    return int(round(days * instruments * cols * 8.0))


def _union_scan_bytes(group: list[dict[str, Any]], instruments: int) -> int:
    start = min(e["load_start"] or "" for e in group) or None
    end = max(e["load_end"] or "" for e in group) or None
    union_days = _days_between(start, end)
    max_cols = max(e["cols"] for e in group)
    return int(round(union_days * instruments * max_cols * 8.0))


def _merge_beneficial(
    group: list[dict[str, Any]], nxt: dict[str, Any], instruments: int
) -> bool:
    """合并额外 superset scan bytes < 分开扫描的重复 scan/decode 成本时才合并。"""
    candidate = group + [nxt]
    separate_scan = sum(e["scan_bytes"] for e in candidate)
    union_scan = _union_scan_bytes(candidate, instruments)
    return union_scan < separate_scan


def _group_by_scan_cost(
    entries: list[dict[str, Any]], instruments: int
) -> list[list[dict[str, Any]]]:
    """扫描成本感知分组：非 full-history 按 load_start 排序后贪心合并；full-history 独立组置末。"""
    non_full = [e for e in entries if not e["full_history"]]
    full = [e for e in entries if e["full_history"]]
    ordered = sorted(
        non_full, key=lambda e: (e["load_start"] or "", e["load_end"] or "")
    )
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for entry in ordered:
        if current and _merge_beneficial(current, entry, instruments):
            current.append(entry)
        else:
            if current:
                groups.append(current)
            current = [entry]
    if current:
        groups.append(current)
    if full:
        groups.append(full)
    return groups


def _union_run_window(group: list[dict[str, Any]]) -> Any | None:
    from factor_engine.runtime.run_window import RunWindow

    rws = [e["run_window"] for e in group if e["run_window"] is not None]
    if not rws:
        return None
    load_start = min(e["load_start"] or "" for e in group) or None
    load_end = max(e["load_end"] or "" for e in group) or None
    return RunWindow(
        requested_start=rws[0].requested_start,
        requested_end=rws[0].requested_end,
        actual_load_start=load_start,
        actual_load_end=load_end,
        warmup_bars=0,
        trim_output=bool(rws[0].trim_output),
        full_history_required=any(
            bool(getattr(rw, "full_history_required", False)) for rw in rws
        ),
    )


def compute_batch_warmup_plan(
    analyses: dict[str, Any],
    dag: Any,
    engine: Any,
    *,
    factors: Sequence[Any] | None = None,
    auto_warmup: bool = False,
    trim_warmup: bool = True,
    market: str | None = None,
    strict: bool = True,
) -> BatchWarmupPlan:
    """一次计算全 batch 的 warmup 规划（PERF-014）与扫描成本感知分组（PERF-015）。

    ``strict=True``（默认）：任一因子 warmup 解析失败直接抛出（与
    ``runtime.warmup_service`` 的 production fail-closed / 老
    ``_maybe_prepare_batch_warmup`` 行为一致）；``strict=False`` 时跳过失败因子
    （供成本聚类等研究级容错路径）。
    """
    if not auto_warmup:
        return BatchWarmupPlan(per_factor={}, groups=[])
    from factor_engine.runtime.warmup_service import prepare_run_warmup

    if factors is None:
        derived: list[Any] = []
        for fp in getattr(dag, "roots", None) or []:
            derived.append(_FactorProxy(fp))
        factors = derived
    name_to_factor: dict[str, Any] = {}
    for factor in factors:
        name = getattr(factor, "name", None)
        if name:
            name_to_factor[name] = factor

    per_factor: dict[str, Any] = {}
    entries: list[dict[str, Any]] = []
    for factor in factors:
        name = getattr(factor, "name", None)
        if not name:
            continue
        analysis = (analyses or {}).get(name)
        if analysis is None:
            continue
        try:
            wctx = prepare_run_warmup(
                engine,
                factor,
                analysis,
                auto_warmup=auto_warmup,
                trim_warmup=trim_warmup,
                market=market,
            )
        except Exception:
            if strict:
                raise
            continue
        rw = wctx.run_window
        if rw is None:
            continue
        per_factor[name] = rw
        entries.append(
            {
                "factor_name": name,
                "run_window": rw,
                "load_start": _load_start(rw, engine),
                "load_end": _load_end(rw, engine),
                "scan_bytes": _estimate_factor_scan_bytes(engine, analysis, rw),
                "cols": max(1, len(getattr(analysis, "referenced_columns", ()) or ())),
                "full_history": bool(getattr(rw, "full_history_required", False)),
            }
        )
    instruments = _instrument_count(engine.data_source)
    groups: list[WarmupGroup] = []
    for grouped in _group_by_scan_cost(entries, instruments):
        if not grouped:
            continue
        names = tuple(e["factor_name"] for e in grouped)
        start = min(e["load_start"] or "" for e in grouped) or None
        end = max(e["load_end"] or "" for e in grouped) or None
        groups.append(
            WarmupGroup(
                factor_names=names,
                run_window=_union_run_window(grouped),
                load_start=start,
                load_end=end,
                scan_bytes=_union_scan_bytes(grouped, instruments),
                source_scope=str(getattr(engine.data_source, "dataset", None) or ""),
            )
        )
    return BatchWarmupPlan(per_factor=per_factor, groups=groups)
