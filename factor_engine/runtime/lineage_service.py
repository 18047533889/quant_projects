# -*- coding: utf-8
"""物化 lineage 构建（从 FactorEngine 抽离）。"""

from __future__ import annotations

from typing import Any

from api.factor import Factor
from cleaned_operators.operator_policy import compute_operator_catalog_hash, effective_lookback
from runtime.lineage import build_run_lineage, hash_data_source_config, resolve_git_commit_hash
from storage.catalog import compute_ir_hash
from storage.composite_source import CompositeDataSource


def resolve_lineage_expression(factor: Factor, expression: str | None) -> str | None:
    if expression:
        return expression
    source = getattr(factor, "source_expr", None)
    return source or None


def composite_lineage_from_source(data_source: Any) -> dict[str, Any]:
    if not isinstance(data_source, CompositeDataSource):
        return {}
    reports = data_source.collect_join_reports(clear=False)
    if not reports:
        return {}
    return {"composite_join_reports": reports}


def build_lineage_extra(
    *,
    data_source_config: dict | None,
    snapshot_id: str | None,
    input_dq: dict | None = None,
    data_source: Any = None,
    **more: Any,
) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "data_snapshot_id": snapshot_id,
        "data_source_config": data_source_config,
        "git_commit": resolve_git_commit_hash(),
    }
    if input_dq is not None:
        extra["input_dq"] = input_dq
    if data_source is not None:
        extra.update(composite_lineage_from_source(data_source))
    extra.update(more)
    return extra


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
    snapshot_id = data_snapshot_id_from_config(data_source_config)
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
            snapshot_id=snapshot_id,
            input_dq=output.get("input_dq"),
            data_source=data_source,
            **extra_kwargs,
        ),
    )
