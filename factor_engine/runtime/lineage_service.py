# -*- coding: utf-8 -*-
"""Materialization lineage with primary and transitive logical-source identity."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from api.factor import Factor
from cleaned_operators.operator_policy import compute_operator_catalog_hash, effective_lookback
from runtime.lineage import build_run_lineage, hash_data_source_config, resolve_git_commit_hash
from storage.catalog import compute_ir_hash
from storage.composite_source import CompositeDataSource


def resolve_lineage_expression(factor: Factor, expression: str | None) -> str | None:
    if expression:
        return expression
    return getattr(factor, "source_expr", None) or None


def composite_lineage_from_source(data_source: Any) -> dict[str, Any]:
    if not isinstance(data_source, CompositeDataSource):
        return {}
    reports = data_source.collect_join_reports(clear=False)
    return {"composite_join_reports": reports} if reports else {}


def _logical_wrapper(data_source: Any) -> Any | None:
    if data_source is None:
        return None
    if callable(getattr(data_source, "collect_source_dependencies", None)):
        return data_source
    wrapper = getattr(data_source, "_factor_engine_lqtp_wrapper", None)
    if callable(getattr(wrapper, "collect_source_dependencies", None)):
        return wrapper
    return None


def logical_source_lineage(
    data_source: Any,
    *,
    output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect secondary snapshots/definition hashes used by SourceRef execution."""
    if output and output.get("source_dependencies") is not None:
        deps = [dict(x) for x in output.get("source_dependencies") or []]
        dep_hash = output.get("source_dependency_hash")
    else:
        wrapper = _logical_wrapper(data_source)
        deps = wrapper.collect_source_dependencies() if wrapper is not None else []
        dep_hash = wrapper.source_dependency_hash() if wrapper is not None and deps else None
    if not deps:
        return {}
    if not dep_hash:
        payload = json.dumps(deps, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        dep_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return {
        "source_dependencies": deps,
        "source_dependency_hash": str(dep_hash),
    }


def _combined_snapshot_id(
    primary: str | None,
    source_dependency_hash: str | None,
) -> str | None:
    if not source_dependency_hash:
        return primary
    payload = json.dumps(
        {"primary": primary or "", "secondary": source_dependency_hash},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_lineage_extra(
    *,
    data_source_config: dict | None,
    snapshot_id: str | None,
    input_dq: dict | None = None,
    data_source: Any = None,
    output: dict[str, Any] | None = None,
    **more: Any,
) -> dict[str, Any]:
    logical = logical_source_lineage(data_source, output=output) if data_source is not None else {}
    combined_snapshot = _combined_snapshot_id(
        snapshot_id,
        logical.get("source_dependency_hash"),
    )
    extra: dict[str, Any] = {
        "data_snapshot_id": combined_snapshot,
        "primary_data_snapshot_id": snapshot_id,
        "data_source_config": data_source_config,
        "git_commit": resolve_git_commit_hash(),
        **logical,
    }
    if input_dq is not None:
        extra["input_dq"] = input_dq
    if data_source is not None:
        extra.update(composite_lineage_from_source(data_source))
    extra.update(more)
    return extra


def resolve_data_snapshot_id(
    data_source: Any | None,
    data_source_config: dict | None,
) -> str | None:
    if data_source is not None:
        read_snap = getattr(data_source, "data_snapshot_id", None)
        if read_snap:
            return str(read_snap)
    return data_snapshot_id_from_config(data_source_config)


def data_snapshot_id_from_config(data_source_config: dict | None) -> str | None:
    if not data_source_config:
        return None
    return hash_data_source_config(data_source_config)


def build_materialize_lineage(
    *,
    factor: Factor,
    analysis: Any,
    output: dict[str, Any],
    factor_id: str | None,
    expression: str | None,
    data_source_config: dict | None,
    data_source: Any,
    mode: str = "full",
    incremental: dict | None = None,
):
    from backend.cleaned_bridge import ensure_cleaned_loaded

    ensure_cleaned_loaded()
    primary_snapshot_id = resolve_data_snapshot_id(data_source, data_source_config)
    op_hash = compute_operator_catalog_hash()
    lookback = effective_lookback(getattr(analysis, "lookback", 0))
    if mode == "incremental" and output.get("incremental"):
        lookback = output["incremental"]["lookback_bars"]

    extra_kwargs: dict[str, Any] = {}
    if mode == "incremental":
        extra_kwargs["mode"] = "incremental"
        extra_kwargs["incremental"] = output.get("incremental")
    if output.get("run_window") is not None:
        extra_kwargs["run_window"] = output.get("run_window")

    return build_run_lineage(
        factor_id=factor_id or factor.name,
        factor_name=factor.name,
        ast_hash=compute_ir_hash(analysis.ir),
        operator_catalog_hash=op_hash,
        expression=resolve_lineage_expression(factor, expression),
        lookback=lookback,
        referenced_columns=analysis.referenced_columns,
        result=output["result"],
        extra=build_lineage_extra(
            data_source_config=data_source_config,
            snapshot_id=primary_snapshot_id,
            input_dq=output.get("input_dq"),
            data_source=data_source,
            output=output,
            **extra_kwargs,
        ),
    )
