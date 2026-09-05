"""LLM model-class 路由协议（plan.md Task 22 / §30 / Part J1 留白）。

三档 model class（class 路由协议 + 配置）：

- ``cheap``：deterministic/local refine、参数变化、syntax repair；
- ``strong``：新 schema/logic、ToT、困难 crossover、trajectory critique；
- ``critic``：critique/裁判类调用（另设一档，供后续 trajectory/self-critic 用）。

Part J1 留白：**provider / model 名不实现**。本模块只把 action → model class
的路由判定和配置做完整；具体 provider/model 名由下游 LLM client 层按
``RouteResult.provider/model_name``（当前均为 None）补全。留白是显式契约，
不是 placeholder。

纪律（#30）：路由表 / 覆盖全部可配置；未知 action → 默认 cheap；未知 model
class → :class:`UnknownModelClassError`（fail-closed，不静默映射到最强模型）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

__all__ = [
    "MODEL_CLASSES",
    "DEFAULT_CHEAP_ACTIONS",
    "DEFAULT_STRONG_ACTIONS",
    "DEFAULT_CRITIC_ACTIONS",
    "ModelRoutingConfig",
    "RouteResult",
    "UnknownModelClassError",
    "classify_llm_route",
    "resolve_model_class",
]

MODEL_CLASSES: frozenset[str] = frozenset({"cheap", "strong", "critic"})

#: 默认 cheap：deterministic/local refine、参数变化、语法修复。
DEFAULT_CHEAP_ACTIONS: frozenset[str] = frozenset(
    {
        "REFINE",
        "GENERATION_REPAIR",
        "FIELD_SUBSTITUTION",
        "OPERATOR_SUBSTITUTION",
        "WINDOW_SCALE",
    }
)

#: 默认 strong：新 schema/logic、ToT、困难 crossover、状态条件。
DEFAULT_STRONG_ACTIONS: frozenset[str] = frozenset(
    {
        "CROSSOVER",
        "SCHEMA_EXPLORE",
        "STATE_CONDITION",
        "ROBUSTIFY",
    }
)

#: 默认 critic：批判/裁判类 action（trajectory critique / self-critic）。
DEFAULT_CRITIC_ACTIONS: frozenset[str] = frozenset(
    {"TRAJECTORY_CRITIQUE", "SELF_CRITIC", "HYPOTHESIS_CRITIQUE"}
)


class UnknownModelClassError(ValueError):
    """Model class 未知（fail-closed：不静默映射到最强模型）。"""


@dataclass
class ModelRoutingConfig:
    """三档 model class 配置（#30 全可配，无硬编码分支）。

    ``overrides``: action → model_class 覆盖表（比默认表优先）。
    """

    cheap: str = "cheap"
    strong: str = "strong"
    critic: str = "critic"
    cheap_actions: frozenset[str] = field(default_factory=lambda: DEFAULT_CHEAP_ACTIONS)
    strong_actions: frozenset[str] = field(default_factory=lambda: DEFAULT_STRONG_ACTIONS)
    critic_actions: frozenset[str] = field(default_factory=lambda: DEFAULT_CRITIC_ACTIONS)
    overrides: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteResult:
    """一条 model-class 路由判定。

    Part J1 留白：``provider`` / ``model_name`` 恒为 None（具体 provider/model
    名由下游 LLM client 层实现，本模块只做 class 路由协议 + 配置）。
    """

    model_class: str
    provider: str | None = None
    model_name: str | None = None

    @property
    def is_resolved(self) -> bool:
        """class 路由已判定；provider/model 名是否补全是下游的事。"""
        return self.model_class in MODEL_CLASSES


def classify_llm_route(action_type: str, *, cfg: ModelRoutingConfig | None = None) -> str:
    """action → model class（确定性默认表 + overrides 覆盖）。未知 → cheap。"""
    cfg = cfg or ModelRoutingConfig()
    action = str(action_type or "").upper()
    if action in cfg.overrides:
        return str(cfg.overrides[action])
    if action in cfg.cheap_actions:
        return cfg.cheap
    if action in cfg.critic_actions:
        return cfg.critic
    if action in cfg.strong_actions:
        return cfg.strong
    return cfg.cheap


def resolve_model_class(
    model_class: str,
    *,
    cfg: ModelRoutingConfig | None = None,
) -> RouteResult:
    """校验 model_class ∈ {cheap, strong, critic} → RouteResult。

    Raises
    ------
    UnknownModelClassError
        model_class 不在三档内（fail-closed：绝不静默降级到 cheap 或强模型）。
    """
    cfg = cfg or ModelRoutingConfig()
    cls = str(model_class or "")
    if cls not in MODEL_CLASSES:
        raise UnknownModelClassError(
            f"unknown model class {model_class!r}; "
            f"expected one of {sorted(MODEL_CLASSES)}"
        )
    if cls == "cheap":
        resolved = cfg.cheap
    elif cls == "strong":
        resolved = cfg.strong
    else:
        resolved = cfg.critic
    return RouteResult(model_class=str(resolved))
