# -*- coding: utf-8
"""Optimizer fusion/rewrite 不得改变语义。"""
from __future__ import annotations

from typing import Literal

OptimizerPass = Literal[
    "constant_fold",
    "binary_fusion",
    "rolling_fusion",
    "composite_lowering",
    "cse",
]


OPTIMIZER_PARITY_PASSES: tuple[OptimizerPass, ...] = (
    "constant_fold",
    "binary_fusion",
    "rolling_fusion",
    "composite_lowering",
    "cse",
)


def optimizer_on_must_equal_off() -> bool:
    return True
