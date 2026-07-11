# -*- coding: utf-8
"""Arrow→MultiIndex 键语义策略。"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class KeyPolicy:
    """控制无效键 / 重复键行为。"""

    invalid_key: str = "drop"  # drop | error
    duplicate_key: str = "keep_last"  # keep_last | error
    duplicate_resolution: str | None = None  # revision column name


def resolve_key_policy(policy: KeyPolicy | None = None) -> KeyPolicy:
    if policy is not None:
        return policy
    if os.environ.get("QUANT_PRODUCTION_MODE", "").lower() in {"1", "true", "yes"}:
        return KeyPolicy(invalid_key="error", duplicate_key="error")
    if os.environ.get("DATA_ACCESS_STRICT_READ", "").lower() in {"1", "true", "yes"}:
        return KeyPolicy(invalid_key="error", duplicate_key="error")
    return KeyPolicy()
