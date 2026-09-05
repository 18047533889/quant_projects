"""LLM client 协议与 ExecutionMode（任务书 §30 / §56 / P0-A）。

职责：
- :class:`ExecutionMode`：挖掘链运行模式。只有 ``OFFLINE_TEST`` 允许
  stub LLM / 本地 seen / 合成 evaluator 静默降级；``PRODUCTION`` /
  ``RESEARCH_DEGRADED`` 下 stub 或缺失 LLM 一律 fail-closed（抛异常，
  绝不静默切 stub——硬规矩）。
- :class:`LLMClientProtocol`：真实 LLM client 的统一协议（结构化输出
  经 ``response_schema``；返回带 ``.text`` 的对象或 str，协议层宽松）。
- :class:`UsageRecord`（plan Task 22 cost accounting）：单次 LLM 调用的
  token / latency / cost 记账（prompt/completion tokens、latency_ms、
  cost、model_class）。真实 client 可在返回对象上挂 ``.usage``（可选，
  向后兼容——无 usage 的 client 照常工作）；账本层用
  ``usage_from_result`` 读取，读不到 → 中性零记账。
- :class:`DeterministicStubLLMClient`：把 pipeline.py 现有
  ``make_stub_llm_fn`` 包装成协议实现（**内部复用，不复制逻辑**）。
  显式标注：仅 ``OFFLINE_TEST``。
- :func:`resolve_llm_fn`：把 ``llm_client`` / ``mode`` 粘合成 pipeline
  消费的 ``llm_fn`` 的**唯一**入口。PRODUCTION 下 llm_client 缺失或为
  stub → 抛明确异常；OFFLINE_TEST 下缺失 → DeterministicStubLLMClient。

模块级不 import torch / openai（与 pipeline 同纪律）。

Cost accounting 扩展（plan Task 22）不改既有 fail-closed 语义：UsageRecord
只是数据容器，``resolve_llm_fn`` / ExecutionMode / stub 纪律原样保留。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, runtime_checkable


class ExecutionMode(str, Enum):
    """§56 运行模式：决定 stub / 降级允许面。

    - PRODUCTION：真实数据 + 真实 LLM + 真实 evaluator；缺 llm 即 fail-closed。
    - RESEARCH_DEGRADED：真实数据，可容忍部分降级（评估降级可记录 degraded，
      但 LLM 仍必须是真实 client——禁止 stub 冒充生产）。
    - OFFLINE_TEST：唯一允许 stub LLM / 本地 seen / 合成 evaluator 的模式。
    """

    PRODUCTION = "production"
    RESEARCH_DEGRADED = "research_degraded"
    OFFLINE_TEST = "offline_test"


class _LLMResultLike(Protocol):
    """宽松的结果协议：只要带 ``.text``（或本身是 str）即可。"""

    @property
    def text(self) -> str: ...  # pragma: no cover - protocol


@runtime_checkable
class LLMClientProtocol(Protocol):
    """真实 LLM client 的统一协议。

    ``generate`` 返回带 ``.text`` 的结果对象（或直接 str）；``response_schema``
    非 None 时请求结构化 JSON 输出。协议层宽松，避免把具体 SDK 类型绑死。
    """

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_class: str,
        response_schema: type | None = None,
    ) -> Any: ...  # pragma: no cover - protocol


class DeterministicStubLLMClient:
    """仅 OFFLINE_TEST：确定性 stub LLM client（复用 pipeline.make_stub_llm_fn）。

    不发起任何网络 / 模型调用；输出由 ``make_stub_llm_fn`` 规则化文本变异
    生成（含 structured 分支）。**生产禁用**——见 :func:`resolve_llm_fn` 的
    fail-closed 守卫。
    """

    def __init__(
        self,
        *,
        rng: Any | None = None,
        forced_candidates: list[dict[str, Any]] | None = None,
        structured: bool = True,
    ) -> None:
        # 内部复用 pipeline.make_stub_llm_fn（不复制生成逻辑）。
        from alphaprobe.pipeline import make_stub_llm_fn

        self._structured = bool(structured)
        self._llm_fn = make_stub_llm_fn(
            rng=rng,
            forced_candidates=forced_candidates,
            structured=self._structured,
        )

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_class: str,
        response_schema: type | None = None,
    ) -> Any:
        """协议实现：调内部 ``make_stub_llm_fn`` 并包一层 ``.text`` 结果。"""
        del response_schema  # stub 不走 schema 约束
        text = self._llm_fn(str(system_prompt or ""), str(user_prompt or ""), str(model_class or ""))
        return _StubResult(text)

    def __call__(self, system_prompt: str, user_prompt: str, model_class: str) -> str:
        """兼容 pipeline 的 llm_fn 直调签名（测试便捷，返回原文本）。"""
        return self._llm_fn(str(system_prompt or ""), str(user_prompt or ""), str(model_class or ""))


class _StubResult:
    """最小 ``.text`` 结果对象（DeterministicStubLLMClient 返回）。"""

    __slots__ = ("_text",)

    def __init__(self, text: str) -> None:
        self._text = str(text or "")

    @property
    def text(self) -> str:
        return self._text

    def __str__(self) -> str:  # pragma: no cover - 便利
        return self._text


# ---------------------------------------------------------------------------
# plan Task 22 cost accounting：UsageRecord + usage 读取
# ---------------------------------------------------------------------------


@dataclass
class UsageRecord:
    """单次 LLM 调用的成本记账（plan Task 22）。

    纯数据容器：不发起调用、不解析 SDK 类型。真实 client 可在返回对象上
    挂 ``.usage``（本类实例）；``usage_from_result`` 宽松读取——读不到 →
    中性零记账（向后兼容：无 usage 的 client 照常工作）。

    字段与 observability.CostLedger / ledger.attempts 的成本列对齐：
    prompt_tokens / completion_tokens / latency_ms / cost / model_class。
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    cost: float = 0.0
    model_class: str = ""

    @property
    def total_tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.completion_tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": int(self.prompt_tokens),
            "completion_tokens": int(self.completion_tokens),
            "latency_ms": int(self.latency_ms),
            "cost": float(self.cost),
            "model_class": str(self.model_class or ""),
        }

    @classmethod
    def from_result(cls, result: Any) -> "UsageRecord | None":
        """从 LLM 返回对象/文本宽松读取 usage（None = 无记账信息）。"""
        usage = getattr(result, "usage", None)
        if usage is None:
            return None
        if isinstance(usage, UsageRecord):
            return usage
        # duck-typed usage 对象（第三方 SDK 风格字段名）
        if not isinstance(usage, dict):
            usage = getattr(usage, "__dict__", None) or {}
        if isinstance(usage, dict):
            try:
                return cls(
                    prompt_tokens=int(usage.get("prompt_tokens", usage.get("promptTokens", 0)) or 0),
                    completion_tokens=int(
                        usage.get("completion_tokens", usage.get("completionTokens", 0)) or 0
                    ),
                    latency_ms=int(usage.get("latency_ms", usage.get("latencyMs", 0)) or 0),
                    cost=float(usage.get("cost", 0.0) or 0.0),
                    model_class=str(usage.get("model_class", usage.get("model", "")) or ""),
                )
            except (TypeError, ValueError):
                return None
        return None


def usage_from_result(result: Any) -> UsageRecord:
    """把一次 LLM 调用结果 → UsageRecord（读不到 → 中性零记账）。

    中性零记账 = 全 0 字段，绝不抛、绝不阻塞调用链（cost accounting 是
    记账层；缺失 usage 只意味着这笔调用没有成本信息，不是错误）。
    """
    rec = UsageRecord.from_result(result)
    return rec if rec is not None else UsageRecord()


class LLMConfigurationError(RuntimeError):
    """LLM client 配置错误（fail-closed 语义：缺 client / stub 冒充生产）。"""


def _is_stub(client: Any) -> bool:
    return isinstance(client, DeterministicStubLLMClient)


def resolve_llm_fn(
    llm_client: Any,
    mode: ExecutionMode | str,
    *,
    structured: bool = False,
) -> Callable[[str, str, str], Any] | None:
    """把 llm_client / mode 粘合成 pipeline 消费的 llm_fn。

    Parameters
    ----------
    llm_client
        ``LLMClientProtocol`` 实现（或 None）。stub 显式标注仅 OFFLINE_TEST。
    mode
        :class:`ExecutionMode`（或其 value str）。
    structured
        是否产出结构化 action JSON 输出（DeterministicStub 分支用）。

    Returns
    -------
    llm_fn : Callable | None
        非 None 即 ``(system_prompt, user_prompt, model_class) -> text/obj``。

    Raises
    ------
    LLMConfigurationError
        PRODUCTION / RESEARCH_DEGRADED 下 llm_client 为 None 或为 stub
        （fail-closed，绝不静默切 stub）；或 mode 非法。
    """
    mode_enum = ExecutionMode(mode) if not isinstance(mode, ExecutionMode) else mode
    if mode_enum in (ExecutionMode.PRODUCTION, ExecutionMode.RESEARCH_DEGRADED):
        if llm_client is None:
            raise LLMConfigurationError(
                f"{mode_enum.value} mode requires a real llm_client (got None). "
                "fail-closed: 绝不静默切 stub LLM"
            )
        if _is_stub(llm_client):
            raise LLMConfigurationError(
                f"{mode_enum.value} mode requires a real llm_client; "
                "DeterministicStubLLMClient 仅限 OFFLINE_TEST"
            )
        if not hasattr(llm_client, "generate"):
            raise LLMConfigurationError(
                f"{mode_enum.value} mode llm_client must implement LLMClientProtocol.generate; "
                f"got {type(llm_client).__name__}"
            )
        return _client_to_llm_fn(llm_client)
    if mode_enum == ExecutionMode.OFFLINE_TEST:
        if llm_client is None:
            client = DeterministicStubLLMClient(structured=structured)
            return client.__call__  # type: ignore[return-value]
        if _is_stub(llm_client):
            return llm_client.__call__  # type: ignore[union-attr]
        if hasattr(llm_client, "generate"):
            return _client_to_llm_fn(llm_client)
        if callable(llm_client):
            # 测试注入的裸 llm_fn 直接放行（离线可跑）。
            return llm_client
        raise LLMConfigurationError(f"invalid llm_client for OFFLINE_TEST: {type(llm_client).__name__}")
    raise LLMConfigurationError(f"unknown ExecutionMode: {mode!r}")


def _client_to_llm_fn(client: Any) -> Callable[[str, str, str], Any]:
    """把协议 client 的 ``generate`` 包成 pipeline 的 llm_fn 签名。"""

    def _llm_fn(system_prompt: str, user_prompt: str, model_class: str) -> Any:
        out = client.generate(
            system_prompt=str(system_prompt or ""),
            user_prompt=str(user_prompt or ""),
            model_class=str(model_class or ""),
        )
        # 协议层宽松：str 直接用；带 .text 的结果对象取 .text。
        if isinstance(out, str):
            return out
        text = getattr(out, "text", None)
        if text is not None:
            return str(text)
        return out

    return _llm_fn


__all__ = [
    "DeterministicStubLLMClient",
    "ExecutionMode",
    "LLMClientProtocol",
    "LLMConfigurationError",
    "UsageRecord",
    "resolve_llm_fn",
    "usage_from_result",
]
