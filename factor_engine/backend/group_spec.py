# -*- coding: utf-8
"""组内算子最小样本与 singleton 策略。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SingletonPolicy = Literal["null", "zero", "one", "half", "identity"]


@dataclass(frozen=True)
class GroupSpec:
    min_valid: int = 2
    null_group_policy: Literal["null"] = "null"
    singleton_std: SingletonPolicy = "null"
    singleton_zscore: SingletonPolicy = "null"
    singleton_rank: SingletonPolicy = "one"
    constant_group_zscore: SingletonPolicy = "zero"
    constant_group_normalize: SingletonPolicy = "half"


GROUP_SPECS: dict[str, GroupSpec] = {
    "group_std": GroupSpec(min_valid=2, singleton_std="null"),
    "group_zscore": GroupSpec(min_valid=2, singleton_zscore="null", constant_group_zscore="zero"),
    "group_neutralize": GroupSpec(min_valid=2, singleton_zscore="null"),
    "group_rank": GroupSpec(min_valid=1, singleton_rank="one"),
    "group_winsorize": GroupSpec(min_valid=1),
    "group_normalize": GroupSpec(min_valid=2, constant_group_normalize="half"),
    "group_mean": GroupSpec(min_valid=1),
}


def group_spec_for(canon: str) -> GroupSpec:
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return GROUP_SPECS.get(name, GroupSpec())


def group_std_singleton_is_null() -> bool:
    return group_spec_for("group_std").singleton_std == "null"
