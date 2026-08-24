"""数据列更新事件 → 计划 / 增量物化编排（从 engine 抽离）。

R10 #53: ``DataEvent`` now carries revision fields (``field_id``,
``affected_start/end``, ``snapshot_before/after``, ``revision_kind``,
``deleted_keys``); the dict-normalization path forwards them so callers feeding
raw dicts keep the same convenience as the dataclass constructor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from factor_engine.runtime.incremental_scheduler import (
    DataEvent,
    execute_incremental_updates_from_event,
    normalize_data_event as _normalize_data_event,
    plan_updates_from_data_event,
)
from factor_engine.storage.materializer import ParquetMaterializer


def normalize_data_event(event: DataEvent | dict[str, Any]) -> DataEvent:
    """Normalize a dict / ``DataEvent`` to :class:`DataEvent` (R10 #53)."""
    return _normalize_data_event(event)


def _production_for_event(event: DataEvent) -> bool:
    """P0-19/P0-20: effective production mode for a planning-only call."""
    from factor_engine.runtime.incremental_scheduler import _is_production_run

    return _is_production_run(event)


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
    from factor_engine.runtime.dependency_catalog import DependencyCatalog

    dep_catalog = DependencyCatalog(catalog)
    production = _production_for_event(normalized)
    plans = plan_updates_from_data_event(
        dep_catalog,
        normalized,
        end_date=end_date,
        lookback_extra=lookback_extra,
        market=market,
        production=production,
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


def _factor_from_config(config: Any) -> Any:
    """``config.factor`` → :class:`Factor`（与 ``from_loaded_config`` 完全同源）。"""
    from factor_engine.api.dsl_parser import parse_factor

    return parse_factor(
        config.factor.expr,
        name=config.factor.name,
        freq=config.factor.freq,
        universe=config.factor.universe,
        description=config.factor.description,
        surface=getattr(config.factor, "surface", "daily") or "daily",
    )


def _build_campaign_engine(engine_cls: Any, base_config: Any) -> Any:
    """Build ONE engine for a whole identity group (reuse across the group)."""
    engine, _base_factor = engine_cls.from_loaded_config(base_config)
    return engine


def materialize_incremental_many_from_config(
    engine_cls: Any,
    config_paths: Sequence[str | Path],
    *,
    profile: str | None = None,
    pipeline_overrides: Any | None = None,
) -> dict[str, Any]:
    """多 YAML 批量增量物化（R39 PERF-069/070）。

    流程：
      1. 先加载**全部** config（单遍，构造引擎前）；
      2. 按严格 :class:`ExecutionIdentity` 分组 —— source snapshot / dataset
         config / market / universe / frequency / calendar / PIT policy /
         writer target 任一不同即独立引擎；
      3. 同组共享一个 engine + source session（``FactorCampaignSession``），
         ``materialize_incremental`` 仍逐因子调用，但复用共享引擎；CSE 批编译
         通过 ``session.compile()``（engine 的 ``compile_many``）暴露。

    Correctness：不同 source config 绝不共享 engine；factor-level semantic
    identity 仍由引擎逐因子校验。

    输出结构不变：``materializations`` 按 ``factor.name`` 键、每个结果带
    ``config`` 与 ``config_path`` —— 与逐配置串行完全一致。
    """
    from factor_engine.runtime.config import load_config
    from factor_engine.runtime.config_runtime import resolve_materialize_kwargs_for_pipeline
    from factor_engine.runtime.execution_identity import execution_identity_from_config
    from factor_engine.runtime.factor_campaign_session import FactorCampaignSession

    # 1. load all configs first; resolve materialize opts (identity depends on
    #    the effective write target / market / PIT, including pipeline overrides).
    loaded: list[tuple[Any, Any, Any, str | Path]] = []
    for path in config_paths:
        config = load_config(path, profile=profile)
        factor = _factor_from_config(config)
        opts = resolve_materialize_kwargs_for_pipeline(config, pipeline_overrides)
        loaded.append((config, factor, opts, path))

    # 2. group by strict ExecutionIdentity (order-preserving first-appearance).
    groups: dict[Any, list[tuple[Any, Any, Any, str | Path]]] = {}
    order: list[Any] = []
    for config, factor, opts, path in loaded:
        identity = execution_identity_from_config(config, resolved=opts)
        if identity not in groups:
            groups[identity] = []
            order.append(identity)
        groups[identity].append((config, factor, opts, path))

    outputs: dict[str, Any] = {}
    for identity in order:
        group = groups[identity]
        base_config = group[0][0]
        # 3. one engine per identity — the whole campaign reuses it.
        engine = _build_campaign_engine(engine_cls, base_config)
        session = FactorCampaignSession(
            identity=identity,
            engine=engine,
            source_session=getattr(engine, "data_source", None),
        )
        for config, factor, opts, path in group:
            session.add_factor(config, factor, path, opts)
        # 4. per-factor materialize on the SHARED engine/source session.
        for rec in session.compiled_factors:
            out = engine.materialize_incremental(
                rec.factor,
                **rec.to_incremental_materialize_kwargs(),
            )
            out["config"] = rec.config
            out["config_path"] = rec.path
            outputs[rec.name] = out
    return {"materializations": outputs}
