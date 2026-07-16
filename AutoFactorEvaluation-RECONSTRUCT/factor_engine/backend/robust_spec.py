# -*- coding: utf-8
"""Robust 统计契约：median / MAD / MAD-zscore。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MedianEvenPolicy = Literal["average", "lower", "higher", "nearest"]
MadDefinition = Literal["median_abs_dev_from_median"]
MadZscoreScaling = Literal["none", "gaussian_0_6745"]


@dataclass(frozen=True)
class TsMedianSpec:
    even_window: MedianEvenPolicy = "average"


@dataclass(frozen=True)
class CsMadSpec:
    definition: MadDefinition = "median_abs_dev_from_median"


@dataclass(frozen=True)
class CsMadZscoreSpec:
    """``cs_mad_zscore`` = (x - median) / MAD，无 0.6745 缩放。"""

    scaling: MadZscoreScaling = "none"
    zero_mad_is_null: bool = True


TS_MEDIAN_SPEC = TsMedianSpec()
CS_MAD_SPEC = CsMadSpec()
CS_MAD_ZSCORE_SPEC = CsMadZscoreSpec()


def cs_mad_is_median_abs_dev() -> bool:
    return CS_MAD_SPEC.definition == "median_abs_dev_from_median"


def cs_mad_zscore_no_gaussian_scaling() -> bool:
    return CS_MAD_ZSCORE_SPEC.scaling == "none"
