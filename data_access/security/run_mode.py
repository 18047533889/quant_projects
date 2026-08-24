"""R25 P0-020 —— Runtime Mode：interactive_research / automated_research / production。

AlphaProbe / LLM mining 是机器，不看 warning。automated_research 在 PIT、calendar、
semantic ambiguity、required filters、units、source snapshot、authorization、
unknown field 上**必须 strict**（INV-07），但：
    - 不允许 publish production；
    - 预算可比 production 宽。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from data_access.core.exceptions import ValidationError

_RUN_MODES = frozenset(
    {"interactive_research", "automated_research", "production"}
)


@dataclass(frozen=True)
class RunMode:
    value: str

    def __post_init__(self) -> None:
        if self.value not in _RUN_MODES:
            raise ValidationError(
                f"run_mode={self.value!r} 非法；允许: {sorted(_RUN_MODES)}"
            )

    @property
    def is_production(self) -> bool:
        return self.value == "production"

    @property
    def is_automated_research(self) -> bool:
        return self.value == "automated_research"

    @property
    def is_interactive_research(self) -> bool:
        return self.value == "interactive_research"

    @property
    def strict_semantics(self) -> bool:
        """automated_research 和 production 都 strict（INV-07）。"""
        return self.value in {"automated_research", "production"}

    @property
    def allows_publish(self) -> bool:
        """只有 production 允许 publish。"""
        return self.value == "production"


def resolve_run_mode(
    explicit: str | None = None,
    *,
    env: str = "DATA_ACCESS_RUN_MODE",
) -> RunMode:
    """解析 run_mode。

    优先级：显式传参 > env（DATA_ACCESS_RUN_MODE）> 兼容旧开关
    （QUANT_PRODUCTION_MODE=1 → production）。缺省 interactive_research。
    """
    raw = explicit
    if raw is None:
        raw = os.environ.get(env, "")
    if not str(raw).strip():
        # 兼容旧 production 开关。
        if os.environ.get("QUANT_PRODUCTION_MODE", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            return RunMode("production")
        return RunMode("interactive_research")
    return RunMode(str(raw).strip().lower())


def require_publish_permission(run_mode: RunMode) -> None:
    """R25 P0-020：automated_research / interactive_research 不允许 publish。"""
    if not run_mode.allows_publish:
        raise ValidationError(
            f"run_mode={run_mode.value!r} 不允许 publish production（R25 P0-020："
            "只有 production 允许发布）。AlphaProbe 默认 automated_research。"
        )
