"""数据列更新事件 → 计划 / 增量物化编排（从 engine 抽离）。

R10 #53: ``DataEvent`` now carries revision fields (``field_id``,
``affected_start/end``, ``snapshot_before/after``, ``revision_kind``,
``deleted_keys``); the dict-normalization path forwards them so callers feeding
raw dicts keep the same convenience as the dataclass constructor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from runtime.incremental_scheduler import (
    DataEvent,
    execute_incremental_updates_from_event,
    normalize_data_event,
    plan_updates_from_data_event,
)
from storage.materializer import ParquetMaterializer


def normalize_data_event(event: DataEvent | dict[str, Any]) -> DataEvent:
    """Normalize a dict / ``DataEvent`` to :class:`DataEvent` (R10 #53)."""
    return normalize_data_event(event)


def _event_summary(event: DataEvent) -> dict[str, Any]:
    return event.to_dict()


def plan_incremental_from_event(
    event: DataEvent | dict[str, Any],
    *,
    lake_root: str | Path | None = None,
    end_date: str | None = None,
    lookback_extra: int = 5,
    market: str | None = None,
) -> dict[str, Any]:
    """数据列更新事件 → 受影响因子增量重算计划。"""
    normalized = normalize_data_event(event)
    catalog = ParquetMaterializer(lake_root=lake_root).catalog
    from runtime.dependency_catalog import DependencyCatalog

    dep_catalog = DependencyCatalog(catalog)
    plans = plan_updates_from_data_event(
        dep_catalog,
        normalized,
        end_date=end_date,
        lookback_extra=lookback_extra,
        market=market,
    )
    return {
        "event": _event_summary(normalized),
        "plans": [p.to_dict() for p in plans],
        "factor_count": len(plans),
    }


def materialize_incremental_from_event(
    engine: Any,
    event: DataEvent | dict[str, Any],
    *,
    lake_root: str | Path | None = None,
    end_date: str | None = None,
    lookback_extra: int = 5,
    market: str | None = None,
    dry_run: bool = False,
    engine_factory: Any | None = None,
    **materialize_kwargs: Any,
) -> dict[str, Any]:
    """数据列更新事件 → 受影响因子自动增量物化（R10 #51 per-factor engines）。"""
    return execute_incremental_updates_from_event(
        engine,
        event,
        lake_root=str(lake_root) if lake_root is not None else None,
        end_date=end_date,
        lookback_extra=lookback_extra,
        market=market,
        dry_run=dry_run,
        materialize_kwargs=materialize_kwargs,
        engine_factory=engine_factory,
    )


def materialize_incremental_many_from_config(
    engine_cls: Any,
    config_paths: Sequence[str | Path],
    *,
    profile: str | None = None,
    pipeline_overrides: Any | None = None,
) -> dict[str, Any]:
    """多 YAML 批量增量物化；逐配置 resolve + materialize_incremental。"""
    from runtime.config_runtime import resolve_materialize_kwargs_for_pipeline

    loaded: list[tuple[Any, Any, Any, str | Path]] = []
    for path in config_paths:
        engine, factor, config = engine_cls.from_config(path, profile=profile)
        loaded.append((engine, factor, config, path))
    outputs: dict[str, Any] = {}
    for engine, factor, config, path in loaded:
        opts = resolve_materialize_kwargs_for_pipeline(config, pipeline_overrides)
        out = engine.materialize_incremental(
            factor,
            **opts.to_incremental_materialize_kwargs(),
        )
        out["config"] = config
        out["config_path"] = str(path)
        outputs[factor.name] = out
    return {"materializations": outputs}
