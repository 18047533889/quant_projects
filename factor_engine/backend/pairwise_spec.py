# -*- coding: utf-8
"""ts_corr / ts_cov / ts_beta 共享有效样本集合契约。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PairwiseRollingSpec:
    """cov/var/beta 须基于 x,y 同时有效的 pairwise 样本。"""

    shared_valid_mask: bool = True
    min_pairwise_samples: int = 2
    ddof: int = 1


PAIRWISE_SPECS: dict[str, PairwiseRollingSpec] = {
    "ts_corr": PairwiseRollingSpec(),
    "ts_cov": PairwiseRollingSpec(),
    "ts_beta": PairwiseRollingSpec(),
    "rolling_beta": PairwiseRollingSpec(),
    "ts_correlation": PairwiseRollingSpec(),
}


def pairwise_spec_for(canon: str) -> PairwiseRollingSpec:
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    name = OperatorRegistry._aliases.get(canon, canon)
    return PAIRWISE_SPECS.get(name, PairwiseRollingSpec())


def beta_uses_pairwise_var() -> bool:
    return pairwise_spec_for("ts_beta").shared_valid_mask
