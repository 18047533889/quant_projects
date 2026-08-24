# -*- coding: utf-8 -*-
"""Pipeline：数据更新事件 → 批量增量物化。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from factor_engine.runtime.config_runtime import build_data_source_config, resolve_materialize_kwargs
from factor_engine.runtime.engine import FactorEngine
from factor_engine.runtime.incremental_scheduler import DataEvent
from factor_engine.util.workspace_paths import default_factor_lake_root


def run_data_event(
    *,
    dataset: str,
    column: str,
    updated_date: str,
    lake_root: str | Path | None = None,
    end_date: str | None = None,
    lookback_extra: int = 5,
    market: str | None = None,
    dry_run: bool = False,
    profile: str | None = None,
    config_path: str | Path | None = None,
    output_root: str | Path | None = None,
    write_target: str | None = None,
    # R10 #53: extended revision fields — all optional, passed straight through
    # to the DataEvent so the scheduler sees the full revision context.
    field_id: str | None = None,
    affected_start: str | None = None,
    affected_end: str | None = None,
    snapshot_before: Any = None,
    snapshot_after: Any = None,
    revision_kind: str | None = None,
    deleted_keys: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """``DataEvent`` 驱动增量物化；可选 YAML 提供 data_source / materialize 默认。"""
    lake = Path(lake_root or default_factor_lake_root())
    root = Path(output_root) if output_root else lake.parent / "event_runs" / datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    root.mkdir(parents=True, exist_ok=True)

    mat_kwargs: dict[str, Any] = {}
    if write_target is not None:
        mat_kwargs["write_target"] = write_target

    engine: FactorEngine | None = None
    if config_path is not None:
        engine, _, config = FactorEngine.from_config(config_path, profile=profile)
        opts = resolve_materialize_kwargs(config, write_target_override=write_target)
        mat_kwargs.update(opts.to_engine_materialize_kwargs())
        mat_kwargs.update(
            {
                k: v
                for k, v in opts.to_run_kwargs().items()
                if k
                in {
                    "input_dq_check",
                    "input_dq_strict",
                    "input_dq_thresholds",
                    "auto_warmup",
                    "trim_warmup",
                    "pit_enforce",
                    "pit_forbid_forward_fill",
                }
            }
        )
        if engine.data_source is not None:
            mat_kwargs.setdefault(
                "data_source_config",
                build_data_source_config(config),
            )

    if engine is None:
        from factor_engine.backend.factory import build_backend

        engine = FactorEngine(
            backend=build_backend("pandas"),
            data_source=_MinimalEventSource(dataset=dataset),
        )

    event = DataEvent(
        dataset=dataset,
        column=column,
        updated_date=updated_date,
        field_id=field_id,
        affected_start=affected_start,
        affected_end=affected_end,
        snapshot_before=snapshot_before,
        snapshot_after=snapshot_after,
        revision_kind=revision_kind,
        deleted_keys=tuple(deleted_keys) if deleted_keys else None,
    )
    out = engine.materialize_incremental_from_event(
        event,
        lake_root=str(lake),
        end_date=end_date,
        lookback_extra=lookback_extra,
        market=market,
        dry_run=dry_run,
        **mat_kwargs,
    )

    summary = {
        "mode": "data_event",
        "dry_run": dry_run,
        "event": out.get("event"),
        "factor_count": out.get("factor_count", 0),
        "succeeded": out.get("succeeded", 0),
        "failed": out.get("failed", 0),
        "plans": out.get("plans", []),
        "failures": out.get("failures", []),
        "lake_root": str(lake),
        "output_root": str(root),
    }
    summary_path = root / "event_summary.json"
    summary_path.write_text(
        json.dumps({**summary, "materializations": out.get("materializations", {})},
                   ensure_ascii=False,
                   indent=2,
                   default=str),
        encoding="utf-8",
    )
    return {"summary": summary, "event_output": out, "summary_path": str(summary_path)}


class _MinimalEventSource:
    """事件模式占位数据源（增量物化从 catalog expression 重建，不读本源）。"""

    def __init__(self, *, dataset: str) -> None:
        self.dataset = dataset
