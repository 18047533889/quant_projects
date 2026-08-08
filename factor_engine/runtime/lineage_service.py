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
from planner.lowerer import Lowerer
from planner.plan_hash import structural_key


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
    # P0-034 / P1-024: universe-mask coverage recorded on the run lineage.
    # ``universe_mask`` is either a dict of per-market mask components (e.g.
    # ``{"tradability_state": <panel>, "close": <panel>}``) passed to
    # ``market.universe.apply_universe_mask_to_panel``, or a precomputed 2-D
    # boolean mask.  ``coverage_ratio``/``drop_reason`` may be passed directly
    # by the CS materialization entry when it already applied the mask.
    market: str | None = None,
    universe_mask: dict[str, Any] | Any | None = None,
    coverage_ratio: float | None = None,
    drop_reason: str | None = None,
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

    plan = Lowerer().to_logical_plan(analysis.ir)
    field_ids = sorted(
        str(spec.field_id) for spec in getattr(analysis, "referenced_fields", {}).values()
    )
    physical_sources = sorted({
        f"{spec.table}.{spec.source_name}"
        for spec in getattr(analysis, "referenced_fields", {}).values()
    })
    unit_normalizations = sorted(
        {
            "field_id": str(spec.field_id),
            "source_unit": str(spec.source_unit),
            "canonical_unit": str(spec.canonical_unit),
            "scale": float(spec.scale_to_canonical),
        }.items()
        for spec in getattr(analysis, "referenced_fields", {}).values()
    )
    extra_kwargs.update({
        "canonical_expression_source": "lowered_ir",
        "expression_structural_hash": compute_ir_hash(analysis.ir),
        "lowered_plan_hash": structural_key(plan),
        "field_ids": field_ids,
        "physical_sources": physical_sources,
        "unit_normalizations": [dict(items) for items in unit_normalizations],
        "source_contract_hash": hash_data_source_config(data_source_config or {}),
        "original_source_expr": getattr(factor, "source_expr", None),
    })

    # P0-034 / P1-024: record the applied universe mask + coverage on the lineage.
    # The CS materialization entry passes ``universe_mask`` (component dict or
    # precomputed 2-D mask) + optional ``market``; coverage is computed here from
    # the factor panel so it stays reproducible.
    coverage_mask_summary: dict[str, Any] | None = None
    if universe_mask is not None or coverage_ratio is not None or drop_reason is not None:
        try:
            import numpy as _np

            from market.universe import (
                apply_universe_mask,
                apply_universe_mask_to_panel,
                universe_mask_contract,
            )

            mkt = market or getattr(analysis, "market", None) or getattr(factor, "market", None)
            result = output.get("result")
            is_panel = result is not None and getattr(result, "ndim", 1) == 2
            if isinstance(universe_mask, dict) and universe_mask:
                if mkt is not None and is_panel:
                    _masked, _mask, _cov = apply_universe_mask_to_panel(
                        result, mkt, universe_mask
                    )
                    if coverage_ratio is None:
                        coverage_ratio = _cov
                    try:
                        _contract = universe_mask_contract(mkt)
                        _components = list(_contract.required_fields)
                    except Exception:  # pragma: no cover - defensive
                        _components = sorted(universe_mask.keys())
                    coverage_mask_summary = {
                        "market": mkt,
                        "component_fields": _components,
                        "coverage_ratio": float(_cov),
                        "shape": list(getattr(result, "shape", None) or list(_mask.shape)),
                    }
                else:
                    # Non-2-D result or unknown market: record the requested
                    # components without computing a panel coverage ratio.
                    coverage_mask_summary = {
                        "market": mkt,
                        "component_fields": sorted(universe_mask.keys()),
                        "coverage_ratio": None if coverage_ratio is None else float(coverage_ratio),
                        "note": "result not a 2-D panel or market unknown; coverage not computed",
                    }
            elif universe_mask is not None and is_panel:
                # Precomputed 2-D boolean mask.
                _masked = apply_universe_mask(result, universe_mask)
                _pfinite = _np.isfinite(_np.asarray(result, dtype=float))
                _ufinite = _np.isfinite(_np.asarray(_masked, dtype=float))
                _total = int(_pfinite.sum())
                _cov = float(_ufinite.sum() / _total) if _total else 0.0
                if coverage_ratio is None:
                    coverage_ratio = _cov
                coverage_mask_summary = {
                    "coverage_ratio": float(_cov),
                    "shape": list(_np.asarray(result).shape),
                }
        except Exception as exc:  # pragma: no cover - defensive
            if drop_reason is None:
                drop_reason = f"universe_mask application failed: {exc}"

    return build_run_lineage(
        factor_id=factor_id or factor.name,
        factor_name=factor.name,
        ast_hash=compute_ir_hash(analysis.ir),
        operator_catalog_hash=op_hash,
        expression=resolve_lineage_expression(factor, expression),
        lookback=lookback,
        referenced_columns=analysis.referenced_columns,
        result=output["result"],
        coverage_mask=coverage_mask_summary,
        coverage_ratio=coverage_ratio,
        drop_reason=drop_reason,
        extra=build_lineage_extra(
            data_source_config=data_source_config,
            snapshot_id=primary_snapshot_id,
            input_dq=output.get("input_dq"),
            data_source=data_source,
            output=output,
            **extra_kwargs,
        ),
    )
