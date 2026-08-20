# -*- coding: utf-8
"""Group key 类型契约：字符串/分类/NULL/空串。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

NullGroupPolicy = Literal["null_output", "separate_bucket"]
EmptyStringPolicy = Literal["valid_bucket", "treat_as_null"]


@dataclass(frozen=True)
class GroupKeySpec:
    null_group_policy: NullGroupPolicy = "null_output"
    empty_string_policy: EmptyStringPolicy = "valid_bucket"
    categorical_order_independent: bool = True
    utf8_and_categorical_equivalent: bool = True
    int_category_code_allowed: bool = True


GROUP_KEY_SPEC = GroupKeySpec()


def null_group_outputs_null() -> bool:
    return GROUP_KEY_SPEC.null_group_policy == "null_output"


def categorical_encoding_not_business_semantic() -> bool:
    return GROUP_KEY_SPEC.categorical_order_independent
