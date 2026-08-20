# -*- coding: utf-8
"""R39 PERF-069/070: strict immutable execution identity for incremental batching.

A group of configs sharing an identical :class:`ExecutionIdentity` may reuse one
engine + source session (``FactorCampaignSession``) so the expensive per-config
engine construction / source-session setup is hoisted out of the serial loop.

Correctness boundary: **different source configs never share an engine.**  A
difference in any of the identity fields — source snapshot, dataset config,
market, universe, frequency, calendar, PIT policy or writer target — forces its
own engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def freeze_value(value: Any) -> Any:
    """Recursively freeze a config value into a hashable, comparable form."""
    if isinstance(value, dict):
        return tuple(sorted((str(k), freeze_value(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(v) for v in value)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value)
    if value is None:
        return ("none", None)
    if isinstance(value, Path):
        return ("path", str(value))
    return ("str", str(value))


def freeze_options(options: dict[str, Any] | None) -> tuple[tuple[str, Any], ...]:
    """Canonical hashable form of a data-source ``options`` dict."""
    return tuple(sorted((str(k), freeze_value(v)) for k, v in (options or {}).items()))


@dataclass(frozen=True)
class ExecutionIdentity:
    """Strict immutable identity under which factors may share one engine/session.

    Two configs group together ONLY when every field matches.  ``source_snapshot``
    carries the resolved data-source config plus the build context (run mode /
    backend) — a research engine and a production engine over the same physical
    source are NOT the same execution identity.
    """

    source_snapshot: tuple[tuple[str, Any], ...]
    dataset_config: tuple[tuple[str, Any], ...]
    market: str | None
    universe: str | None
    frequency: str | None
    calendar: str | None
    pit_policy: tuple[bool, bool]
    writer_target: str


def execution_identity_from_config(
    config: Any,
    *,
    pipeline_overrides: Any | None = None,
    resolved: Any | None = None,
) -> ExecutionIdentity:
    """Build the strict :class:`ExecutionIdentity` for a loaded config.

    ``resolved`` (a ``ResolvedMaterializeKwargs``) may be passed to avoid
    re-resolving when the caller already has it — the resolved write target /
    market / PIT flags must enter the identity so CLI overrides affect grouping.
    """
    from runtime.config_runtime import (
        build_data_source_config,
        resolve_materialize_kwargs_for_pipeline,
    )

    opts = resolved or resolve_materialize_kwargs_for_pipeline(
        config, pipeline_overrides
    )
    ds = build_data_source_config(config)
    # source snapshot = resolved data source (type + options) + build context.
    source_payload: dict[str, Any] = {
        "type": ds.get("type"),
        "run_mode": getattr(config.run, "mode", None),
        "backend": getattr(config.backend, "type", None),
    }
    for key, value in ds.items():
        if key != "type":
            source_payload[str(key)] = freeze_value(value)
    source_snapshot = tuple(sorted(source_payload.items()))
    dataset_config = (
        ("type", getattr(config.data_source, "type", None)),
        ("options", freeze_options(getattr(config.data_source, "options", None))),
    )
    return ExecutionIdentity(
        source_snapshot=source_snapshot,
        dataset_config=dataset_config,
        market=str(opts.market) if opts.market is not None else None,
        universe=getattr(config.factor, "universe", None),
        frequency=getattr(config.factor, "freq", None),
        calendar=getattr(config.run, "calendar", None),
        pit_policy=(bool(opts.pit_enforce), bool(opts.pit_forbid_forward_fill)),
        writer_target=str(opts.write_target or "local"),
    )
