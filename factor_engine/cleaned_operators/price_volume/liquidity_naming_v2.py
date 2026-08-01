# -*- coding: utf-8 -*-
"""Resolve canonical field/operator collisions in liquidity extensions."""
from __future__ import annotations

from cleaned_operators.base import OperatorMetadata, SeriesOperator, register_operator
from cleaned_operators.price_volume.liquidity_v2 import average_volume
from cleaned_operators.registry import OperatorRegistry
from cleaned_operators import operator_surface


class TsAverageVolume(SeriesOperator):
    metadata = OperatorMetadata(
        name="ts_average_volume",
        category="price_volume_extension",
        description="Rolling average volume over an explicit window.",
        param_names=["volume", "window"],
        return_type="series",
        tags=[
            "pit_safe",
            "causal",
            "bounded_history",
            "production_extension",
            "field_collision_resolved",
        ],
    )

    def _calculate_series(self, volume, window):
        return average_volume(volume, window)


register_operator(
    name="ts_average_volume",
    category="price_volume_extension",
    business_category="price_volume",
    canonical="ts_average_volume",
    source="liquidity_naming_v2",
    backend="pandas_numpy",
    status="production",
)(TsAverageVolume)

# ``average_volume`` is an authoritative data field. It cannot remain a DSL
# function or alias without making formulas ambiguous. The Python helper remains
# private to implementation modules, while the Registry canonical is removed.
OperatorRegistry.unregister("average_volume")
for alias, target in list(OperatorRegistry._aliases.items()):
    if alias == "average_volume" or target == "average_volume":
        OperatorRegistry._aliases.pop(alias, None)

operator_surface._LIQUIDITY_V2_CANONICALS = frozenset(
    (set(operator_surface._LIQUIDITY_V2_CANONICALS) - {"average_volume"})
    | {"ts_average_volume"}
)
operator_surface.EXTENDED_ONLY_CANONICALS = frozenset(
    (set(operator_surface.EXTENDED_ONLY_CANONICALS) - {"average_volume"})
    | {"ts_average_volume"}
)
