# -*- coding: utf-8
"""Arrow→MultiIndex 键语义策略。"""
from __future__ import annotations

import os
from dataclasses import dataclass

from data_access.core.exceptions import ValidationError

_VALID_INVALID_KEY = {"drop", "error"}
_VALID_DUPLICATE_KEY = {"keep_last", "error"}


@dataclass(frozen=True)
class KeyPolicy:
    """控制无效键 / 重复键行为。

    ``duplicate_resolution`` 指定一个确定性的版本列（例如 ``revision_id`` 或
    ``updated_at``）。一旦指定，适配器会按该列降序选择每个主键的最新记录；
    版本列本身出现 NULL 或同一主键同一版本重复时必须报错，避免依赖文件扫描顺序。
    """

    invalid_key: str = "drop"  # drop | error
    duplicate_key: str = "keep_last"  # keep_last | error
    duplicate_resolution: str | None = None  # revision column name

    def __post_init__(self) -> None:
        if self.invalid_key not in _VALID_INVALID_KEY:
            raise ValidationError(
                f"KeyPolicy.invalid_key 必须是 {sorted(_VALID_INVALID_KEY)}，"
                f"收到 {self.invalid_key!r}"
            )
        if self.duplicate_key not in _VALID_DUPLICATE_KEY:
            raise ValidationError(
                f"KeyPolicy.duplicate_key 必须是 {sorted(_VALID_DUPLICATE_KEY)}，"
                f"收到 {self.duplicate_key!r}"
            )
        if self.duplicate_resolution is not None:
            name = str(self.duplicate_resolution).strip()
            if not name:
                raise ValidationError("KeyPolicy.duplicate_resolution 不能为空字符串")
            if name in {"timestamp", "instrument"}:
                raise ValidationError(
                    "KeyPolicy.duplicate_resolution 必须是独立版本列，"
                    "不能复用 timestamp/instrument"
                )
            object.__setattr__(self, "duplicate_resolution", name)


def resolve_key_policy(policy: KeyPolicy | None = None) -> KeyPolicy:
    if policy is not None:
        return policy
    prod = os.environ.get("QUANT_PRODUCTION_MODE", "").lower() in {"1", "true", "yes"}
    strict = os.environ.get("DATA_ACCESS_STRICT_READ", "").lower() in {"1", "true", "yes"}
    revision_col = os.environ.get("DATA_ACCESS_REVISION_COLUMN", "").strip() or None
    if prod or strict:
        if revision_col:
            return KeyPolicy(
                invalid_key="error",
                duplicate_key="error",
                duplicate_resolution=revision_col,
            )
        return KeyPolicy(invalid_key="error", duplicate_key="error")
    if revision_col:
        return KeyPolicy(duplicate_resolution=revision_col)
    return KeyPolicy()
