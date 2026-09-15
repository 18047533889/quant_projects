# -*- coding: utf-8 -*-
"""Polars interfaces for cs_batch1 cross-sectional operators.

The four estimator classes use explicit full-panel Pandas-reference delegates,
including sklearn for isolation forests. They are not native Polars or GPU
implementations. The separately scoped graph-profile interface is retained below.
"""
from __future__ import annotations

from typing import Any
import hashlib
from pathlib import Path

import numpy as np
import polars as pl

from factor_engine.backend.contracts import ExecutionKind, PhysicalImplementationSpec

from factor_engine.cleaned_operators.base_polars import (
    OperatorMetadata,
    SeriesOperator,
    register_operator,
    PANEL_SKIP_COLUMNS,
)

__all__ = [
    "CsIsolationForestScorePolars",
    "CsFactorBucketReturnPolars",
    "CsEmpiricalBayesShrinkagePolars",
    "CsShrinkToGroupMeanPolars",
    "PanelPeerGraphAggregatePolars",
]

_MIN_BREADTH = 10
_EPS = 1e-12


def _metadata(
    name: str,
    description: str,
    params: list[str],
    *,
    domain: str,
    unit: str,
    cost: int = 2,
) -> OperatorMetadata:
    """Build operator metadata following R47 conventions."""
    output_unit = (
        unit
        if (unit.startswith("same_as:") or unit.startswith("unit(") or unit == "dimensionless")
        else None
    )
    return OperatorMetadata(
        name=name,
        category="cross_sectional",
        description=description,
        param_names=params,
        return_type="series",
        tags=[
            "cross_sectional",
            "daily",
            "pit_safe",
            "causal",
            "typed_v2",
            "polars",
            "native",
            f"signature:{','.join(params)}->series",
            f"domain:{domain}",
            f"unit:{unit}",
            f"cost:{cost}",
        ],
        output_unit=output_unit,
    )


def _value_cols(df: pl.DataFrame) -> list[str]:
    """Extract value column names (excluding metadata columns)."""
    return [c for c in df.columns if c not in PANEL_SKIP_COLUMNS]


# ---------------------------------------------------------------------------
# 1. cs_isolation_forest_score (UDF delegation)
# ---------------------------------------------------------------------------

@register_operator(
    name="cs_isolation_forest_score",
    category="cross_sectional",
    business_category="cross_sectional_anomaly",
    canonical="cs_isolation_forest_score",
    source="cs_batch1_polars",
    backend="polars",
    status="experimental",
)
class CsIsolationForestScorePolars(SeriesOperator):
    """Canonical estimator; honest full-panel reference delegation."""
    from factor_engine.cleaned_operators.common import cs_estimator_delegate as _delegate
    metadata = _delegate.metadata("cs_isolation_forest_score")
    _physical_spec = _delegate.physical_spec("cs_isolation_forest_score")

    def _calculate_series(self, *args, **kwargs):
        return self._delegate.calculate("cs_isolation_forest_score", *args, **kwargs)


# ---------------------------------------------------------------------------
# 2. cs_factor_bucket_return (native Polars)
# ---------------------------------------------------------------------------

@register_operator(
    name="cs_factor_bucket_return",
    category="cross_sectional",
    business_category="cross_sectional_bucket",
    canonical="cs_factor_bucket_return",
    source="cs_batch1_polars",
    backend="polars",
    status="experimental",
)
class CsFactorBucketReturnPolars(SeriesOperator):
    """Canonical estimator; honest full-panel reference delegation."""
    from factor_engine.cleaned_operators.common import cs_estimator_delegate as _delegate
    metadata = _delegate.metadata("cs_factor_bucket_return")
    _physical_spec = _delegate.physical_spec("cs_factor_bucket_return")

    def _calculate_series(self, *args, **kwargs):
        return self._delegate.calculate("cs_factor_bucket_return", *args, **kwargs)


# ---------------------------------------------------------------------------
# 3. cs_empirical_bayes_shrinkage (native Polars)
# ---------------------------------------------------------------------------

@register_operator(
    name="cs_empirical_bayes_shrinkage",
    category="cross_sectional",
    business_category="cross_sectional_shrinkage",
    canonical="cs_empirical_bayes_shrinkage",
    source="cs_batch1_polars",
    backend="polars",
    status="experimental",
)
class CsEmpiricalBayesShrinkagePolars(SeriesOperator):
    """Canonical estimator; honest full-panel reference delegation."""
    from factor_engine.cleaned_operators.common import cs_estimator_delegate as _delegate
    metadata = _delegate.metadata("cs_empirical_bayes_shrinkage")
    _physical_spec = _delegate.physical_spec("cs_empirical_bayes_shrinkage")

    def _calculate_series(self, *args, **kwargs):
        return self._delegate.calculate("cs_empirical_bayes_shrinkage", *args, **kwargs)


# ---------------------------------------------------------------------------
# 4. cs_shrink_to_group_mean (pandas bridge)
# ---------------------------------------------------------------------------

@register_operator(
    name="cs_shrink_to_group_mean",
    category="cross_sectional",
    business_category="cross_sectional_shrinkage",
    canonical="cs_shrink_to_group_mean",
    source="cs_batch1_polars",
    backend="polars",
    status="experimental",
)
class CsShrinkToGroupMeanPolars(SeriesOperator):
    """Canonical estimator; honest full-panel reference delegation."""
    from factor_engine.cleaned_operators.common import cs_estimator_delegate as _delegate
    metadata = _delegate.metadata("cs_shrink_to_group_mean")
    _physical_spec = _delegate.physical_spec("cs_shrink_to_group_mean")

    def _calculate_series(self, *args, **kwargs):
        return self._delegate.calculate("cs_shrink_to_group_mean", *args, **kwargs)


# ---------------------------------------------------------------------------
# 5. panel_peer_graph_aggregate (pandas bridge)
# ---------------------------------------------------------------------------

@register_operator(
    name="panel_peer_graph_aggregate",
    category="cross_sectional",
    business_category="cross_sectional_graph",
    canonical="panel_peer_graph_aggregate",
    source="cs_batch1_polars",
    backend="polars",
    status="experimental",
)
class PanelPeerGraphAggregatePolars(SeriesOperator):
    """Peer graph aggregation: Polars implementation via pandas bridge.

    Graph aggregation logic is complex, use pandas bridge for correctness.
    """

    from factor_engine.cleaned_operators.cs_batch1 import PanelPeerGraphAggregate as _reference
    _meta = _reference.metadata
    metadata = OperatorMetadata(
        name=_meta.name, category=_meta.category, description=_meta.description,
        param_names=list(_meta.param_names), panel_params=_meta.panel_params,
        scalar_params=_meta.scalar_params, param_specs=dict(_meta.param_specs),
        tags=[*_meta.tags, "polars", "pandas_delegate"], output_unit=_meta.output_unit,
    )
    _physical_spec = PhysicalImplementationSpec(
        canonical="panel_peer_graph_aggregate", backend="polars",
        execution_kind=ExecutionKind.POLARS_PANDAS_DELEGATE,
        materializes_full_panel=True, supports_nulls=True, supports_nan=True,
        supports_inf=True,
        implementation_source_hash=hashlib.sha256(
            Path(__file__).read_bytes()
            + Path(__import__("factor_engine.cleaned_operators.cs_batch1", fromlist=["x"]).__file__).read_bytes()
        ).hexdigest(),
        kernel_identity="cs_batch1.PanelPeerGraphAggregate",
        notes="Explicit full-panel pandas delegate over immutable static adjacency; CPU only.",
    )

    def _calculate_series(
        self,
        x: pl.DataFrame,
        similarity: Any,
        method: str = "mean",
        threshold: float = 0.0,
        **_: Any
    ) -> pl.DataFrame:
        """Delegate to pandas bridge for graph aggregation."""
        from factor_engine.cleaned_operators.base_polars import panel_pandas_bridge
        from factor_engine.cleaned_operators.cs_batch1 import PanelPeerGraphAggregate

        ref_impl = PanelPeerGraphAggregate()
        return panel_pandas_bridge(
            x,
            lambda x_pd: ref_impl._calculate_series(
                x_pd,
                similarity,
                method=method,
                threshold=threshold,
            ),
        )
