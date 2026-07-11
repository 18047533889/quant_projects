# -*- coding: utf-8
"""空 universe 与过滤后截面 shape 契约。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EmptyUniversePolicy = Literal["preserve_keys_null", "zero_rows"]
FilteredAllInvalidPolicy = Literal["preserve_keys_null"]


@dataclass(frozen=True)
class UniverseSpec:
    empty_universe: EmptyUniversePolicy = "preserve_keys_null"
    filtered_all_invalid: FilteredAllInvalidPolicy = "preserve_keys_null"
    shape_preserving_cross_section: bool = True


UNIVERSE_SPEC = UniverseSpec()


def empty_universe_preserves_keys() -> bool:
    return UNIVERSE_SPEC.empty_universe == "preserve_keys_null"


def cross_section_shape_preserving() -> bool:
    return UNIVERSE_SPEC.shape_preserving_cross_section
