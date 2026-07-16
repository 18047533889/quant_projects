# -*- coding: utf-8 -*-
"""数据更新事件 → 受影响因子 → 最小增量重算窗口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cleaned_operators.operator_policy import effective_lookback
from runtime.incremental import build_incremental_plan
from storage.catalog import FactorCatalog


@dataclass(frozen=True)
class DataEvent:
    """源数据列更新事件。"""

    dataset: str
    column: str
    updated_date: str


@dataclass(frozen=True)
class FactorUpdatePlan:
    """单因子增量重算计划。"""

    factor_id: str
    column: str
    lookback_bars: int
    since: str | None
    end_date: str | None
    load_start: str | None
    load_end: str | None
    source_dataset: str | None
    is_full_run: bool

    def to_dict(self) -> dict[str, Any]:
        """序列化为可 JSON 化的字典。"""
        return {
            "factor_id": self.factor_id,
            "column": self.column,
            "lookback_bars": self.lookback_bars,
            "since": self.since,
            "end_date": self.end_date,
            "load_start": self.load_start,
            "load_end": self.load_end,
            "source_dataset": self.source_dataset,
            "is_full_run": self.is_full_run,
        }


def _resolve_source_dataset(data_source: Any) -> str | None:
    return getattr(data_source, "dataset", None)


def plan_updates_from_data_event(
    catalog: FactorCatalog,
    event: DataEvent,
    *,
    end_date: str | None = None,
    lookback_extra: int = 5,
    market: str | None = None,
) -> list[FactorUpdatePlan]:
    """``column`` 更新后，查询 catalog 中受影响因子并计算 lookback 窗口。"""
    deps = catalog.list_factors_for_column(event.column)
    plans: list[FactorUpdatePlan] = []
    for dep in deps:
        source_dataset = dep.get("source_dataset")
        if source_dataset and str(source_dataset) != str(event.dataset):
            continue
        factor_id = str(dep["factor_id"])
        lookback = int(dep.get("lookback") or 0)
        inc = build_incremental_plan(
            factor_id=factor_id,
            analysis_lookback=lookback,
            watermark={"end_date": event.updated_date},
            since=event.updated_date,
            end_date=end_date,
            lookback_extra=lookback_extra,
            market=market,
            factor_freq=dep.get("frequency"),
        )
        plans.append(
            FactorUpdatePlan(
                factor_id=factor_id,
                column=event.column,
                lookback_bars=int(inc.lookback_bars),
                since=event.updated_date,
                end_date=end_date,
                load_start=None if inc.load_start is None else inc.load_start.isoformat(),
                load_end=None if inc.load_end is None else inc.load_end.isoformat(),
                source_dataset=source_dataset,
                is_full_run=bool(inc.is_full_run),
            )
        )
    return plans


def record_factor_dependency_from_analysis(
    catalog: FactorCatalog,
    *,
    factor_id: str,
    analysis: Any,
    data_source: Any,
    frequency: str | None = None,
) -> None:
    """从 ``AnalysisResult`` 写入依赖 catalog。"""
    catalog.record_factor_dependency(
        factor_id,
        referenced_columns=getattr(analysis, "referenced_columns", set()),
        lookback=int(getattr(analysis, "lookback", 0)),
        frequency=frequency,
        source_dataset=_resolve_source_dataset(data_source),
    )


def factor_from_catalog_info(info: dict) -> Any:
    """catalog 行 → ``Factor``（需含 ``expression``）。"""
    from api.dsl_parser import parse_factor

    expression = info.get("expression")
    if not expression or not str(expression).strip():
        raise ValueError(f"因子 {info.get('factor_id')!r} 缺少 expression，无法重建")
    factor_id = str(info.get("factor_id") or info.get("name") or "unknown")
    return parse_factor(
        str(expression),
        name=factor_id,
        freq=str(info.get("frequency") or "1d"),
        description=info.get("description"),
    )


def execute_incremental_updates_from_event(
    engine: Any,
    event: DataEvent | dict[str, Any],
    *,
    lake_root: str | None = None,
    end_date: str | None = None,
    lookback_extra: int = 5,
    market: str | None = None,
    dry_run: bool = False,
    materialize_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """DataEvent → 计划 → （可选）逐因子 ``materialize_incremental``。"""
    from storage.materializer import ParquetMaterializer

    if not isinstance(event, DataEvent):
        event = DataEvent(
            dataset=str(event["dataset"]),
            column=str(event["column"]),
            updated_date=str(event["updated_date"]),
        )
    catalog = ParquetMaterializer(lake_root=lake_root).catalog
    plans = plan_updates_from_data_event(
        catalog,
        event,
        end_date=end_date,
        lookback_extra=lookback_extra,
        market=market,
    )
    out: dict[str, Any] = {
        "event": {
            "dataset": event.dataset,
            "column": event.column,
            "updated_date": event.updated_date,
        },
        "plans": [p.to_dict() for p in plans],
        "factor_count": len(plans),
        "dry_run": dry_run,
        "materializations": {},
    }
    if dry_run or not plans:
        return out

    mat_kwargs = dict(materialize_kwargs or {})
    mat_kwargs.setdefault("lake_root", lake_root)
    mat_kwargs.setdefault("lookback_extra", lookback_extra)
    mat_kwargs.setdefault("market", market)

    failures: list[dict[str, str]] = []
    for plan in plans:
        info = catalog.get_factor_info(plan.factor_id)
        if info is None:
            failures.append(
                {
                    "factor_id": plan.factor_id,
                    "error": "catalog 中无 factor_registry 记录",
                }
            )
            continue
        try:
            factor = factor_from_catalog_info({**info, "factor_id": plan.factor_id})
        except ValueError as exc:
            failures.append({"factor_id": plan.factor_id, "error": str(exc)})
            continue
        try:
            result = engine.materialize_incremental(
                factor,
                factor_id=plan.factor_id,
                since=plan.since,
                end_date=plan.end_date or end_date,
                expression=info.get("expression"),
                frequency=info.get("frequency"),
                description=info.get("description"),
                **mat_kwargs,
            )
            out["materializations"][plan.factor_id] = result.get(
                "materialization", result
            )
        except Exception as exc:
            failures.append({"factor_id": plan.factor_id, "error": str(exc)})

    if failures:
        out["failures"] = failures
    out["succeeded"] = len(out["materializations"])
    out["failed"] = len(failures)
    return out
