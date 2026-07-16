# -*- coding: utf-8
"""聚合算子：全 NULL 窗口、NaN/Inf 参与规则。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

NanComparePolicy = Literal["treat_as_null", "ieee_compare"]
AllNullSumPolicy = Literal["null", "zero"]


@dataclass(frozen=True)
class AggregationSpec:
    all_null_sum: AllNullSumPolicy = "null"
    nan_compare_policy: NanComparePolicy = "treat_as_null"
    inf_in_min_max: bool = False


AGG_SPECS: dict[str, AggregationSpec] = {
    "ts_sum": AggregationSpec(all_null_sum="null"),
    "c_sum": AggregationSpec(all_null_sum="null"),
    "c_mean": AggregationSpec(all_null_sum="null"),
    "c_std": AggregationSpec(all_null_sum="null"),
    "cum_sum": AggregationSpec(all_null_sum="null"),
    "expanding_sum": AggregationSpec(all_null_sum="null"),
    "ts_max": AggregationSpec(nan_compare_policy="treat_as_null"),
    "ts_min": AggregationSpec(nan_compare_policy="treat_as_null"),
    "maximum": AggregationSpec(nan_compare_policy="treat_as_null"),
    "minimum": AggregationSpec(nan_compare_policy="treat_as_null"),
}


def aggregation_spec_for(canon: str) -> AggregationSpec:
    from cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return AGG_SPECS.get(name, AggregationSpec())


def sum_all_null_is_null() -> bool:
    return aggregation_spec_for("ts_sum").all_null_sum == "null"


def cs_agg_all_null_is_null(canon: str) -> bool:
    return aggregation_spec_for(canon).all_null_sum == "null"


def min_max_nan_propagates_null() -> bool:
    return aggregation_spec_for("maximum").nan_compare_policy == "treat_as_null"
