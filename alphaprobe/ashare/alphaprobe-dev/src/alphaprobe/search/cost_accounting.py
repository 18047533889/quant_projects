"""Search cost accounting（plan.md Task 22 / Part F5）。

:class:`SearchCostLedger` 聚合搜索链路的可计量成本，并暴露可接入 EV 的
cost_source：

- LLM usage（token/latency/cost，来源 :class:`UsageRecord`，见 llm_client）；
- FE 计算秒；
- QE 计算秒；
- cache hit / duplicate 节省（duplicate 在 GlobalFactorGate 前被拒 → 从未进
  QE → 记账的 QE 成本 ≈ 0，且显式记一笔「节省」供审计）。

纪律（Non-negotiable #26 / #30）：
- **cost 只记账，不改变 reward 语义**：本模块不写 ledger.reward /
  settle_reward 的任何分支；reward 结算语义由
  :class:`CandidateAttemptLedger.settle_reward` 独立仲裁（DB 级唯一）。
- **#7 不手算指标**：本模块只聚合已发生的秒/元计数，不重算任何
  IC/Sharpe/收益。
- **#30 全开关**：cost_source 归一化（cap / 缩放 / 开关）全部走显式参数，
  无硬编码行为分支。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from alphaprobe.llm_client import UsageRecord

__all__ = [
    "SearchCostLedger",
    "CostSourceConfig",
    "action_cost_adjusted_ev",
    "DEFAULT_QE_SECONDS_REF",
]

#: 默认 QE 成本归一参考秒数（每 factor 的 QE 秒 → [0,1] 成本）。
DEFAULT_QE_SECONDS_REF = 3600.0


@dataclass
class CostSourceConfig:
    """cost_source 投影配置（#30：cap / 线性缩放 / 关成本全可配）。

    Attributes
    ----------
    qe_seconds_ref : float
        QE 秒数线性归一到 [0,1] 的参考值：cost = min(1, qe_seconds / ref)。
        0 或 None → 关闭 QE 成本投影（恒 0）。
    llm_cost_scale : float
        LLM 美元成本乘以该系数后与 QE 秒成本合并（0 → 不看 LLM 成本）。
    include_llm_cost : bool
        False → cost_source 不含 LLM 成本（只算 FE/QE compute）。
    """

    qe_seconds_ref: float = DEFAULT_QE_SECONDS_REF
    llm_cost_scale: float = 0.0
    include_llm_cost: bool = True


@dataclass
class SearchCostLedger:
    """搜索成本账本：只记账 + 聚合，不碰 reward。

    ``record_attempt`` 追加一条成本事件（kind ∈ ``qe / fe / llm / duplicate /
    cache_hit / other``）；``totals()`` 聚合全量；``cost_source()`` 返回
    ``factor_id -> 归一成本`` 的确定性函数，可直接注入 contextual EV /
    EVI promotion 的 cost_source。
    """

    config: CostSourceConfig = field(default_factory=CostSourceConfig)
    _entries: list[dict[str, Any]] = field(default_factory=list)
    _per_factor: dict[str, float] = field(default_factory=dict)

    # -- 写 ------------------------------------------------------------------

    def record_attempt(
        self,
        *,
        attempt_id: str,
        kind: str,
        factor_id: str | None = None,
        duplicate_of: str | None = None,
        fe_seconds: float | None = None,
        qe_seconds: float | None = None,
        llm_usage: UsageRecord | None = None,
    ) -> None:
        """记录一条成本事件。

        kind ∈ ``qe / fe / llm / duplicate / cache_hit``。duplicate / cache_hit
        事件只记节省标记，不增加 QE/FE 秒（前拒后从未执行）。
        """
        entry: dict[str, Any] = {
            "attempt_id": str(attempt_id),
            "kind": str(kind),
            "factor_id": str(factor_id) if factor_id else None,
            "duplicate_of": str(duplicate_of) if duplicate_of else None,
            "fe_seconds": max(float(fe_seconds), 0.0) if fe_seconds is not None else 0.0,
            "qe_seconds": max(float(qe_seconds), 0.0) if qe_seconds is not None else 0.0,
        }
        if llm_usage is not None:
            entry["prompt_tokens"] = int(llm_usage.prompt_tokens)
            entry["completion_tokens"] = int(llm_usage.completion_tokens)
            entry["latency_ms"] = int(llm_usage.latency_ms)
            entry["llm_cost"] = max(float(llm_usage.cost), 0.0)
            entry["model_class"] = llm_usage.model_class or ""
        else:
            entry.update(
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=0,
                llm_cost=0.0,
                model_class="",
            )
        self._entries.append(entry)
        if factor_id:
            cur = self._per_factor.setdefault(str(factor_id), 0.0)
            # duplicate/cache_hit 的 QE/FE 秒为 0 → 不污染 per-factor compute
            self._per_factor[str(factor_id)] = cur + entry["fe_seconds"] + entry["qe_seconds"]

    def record_duplicate_saving(self, *, factor_id: str, saved_qe_seconds: float) -> None:
        """显式记一笔 duplicate 节省（审计口径：净 QE = 实际 - 节省）。"""
        self._entries.append(
            {
                "attempt_id": f"save-{factor_id}",
                "kind": "duplicate_saving",
                "factor_id": str(factor_id),
                "duplicate_of": None,
                "fe_seconds": 0.0,
                "qe_seconds": 0.0,
                "saved_qe_seconds": max(float(saved_qe_seconds), 0.0),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms": 0,
                "llm_cost": 0.0,
                "model_class": "",
            }
        )

    def record_llm_usage(
        self, usage: UsageRecord, *, factor_id: str | None = None
    ) -> None:
        """把一次 LLM 调用的 UsageRecord 记入账本。"""
        self.record_attempt(
            attempt_id=f"llm-{len(self._entries)}",
            kind="llm",
            factor_id=factor_id,
            llm_usage=usage,
        )

    # -- 读 ------------------------------------------------------------------

    @property
    def duplicate_count(self) -> int:
        return sum(1 for e in self._entries if e["kind"] in ("duplicate", "duplicate_saving"))

    @property
    def total_qe_seconds(self) -> float:
        return sum(e["qe_seconds"] for e in self._entries)

    @property
    def total_fe_seconds(self) -> float:
        return sum(e["fe_seconds"] for e in self._entries)

    @property
    def total_qe_saved_seconds(self) -> float:
        return sum(float(e.get("saved_qe_seconds", 0.0)) for e in self._entries)

    @property
    def net_qe_seconds(self) -> float:
        return max(0.0, self.total_qe_seconds - self.total_qe_saved_seconds)

    @property
    def total_llm_tokens(self) -> int:
        return sum(int(e.get("prompt_tokens", 0)) + int(e.get("completion_tokens", 0)) for e in self._entries)

    @property
    def total_llm_cost(self) -> float:
        return sum(float(e.get("llm_cost", 0.0)) for e in self._entries)

    @property
    def total_llm_calls(self) -> int:
        return sum(1 for e in self._entries if e["kind"] == "llm")

    def qe_seconds_of(self, factor_id: str) -> float:
        """单个 factor 的实际 QE 秒（duplicate → 0，因从未进 QE）。"""
        return sum(
            e["qe_seconds"] for e in self._entries if e.get("factor_id") == factor_id
        )

    def fe_seconds_of(self, factor_id: str) -> float:
        return sum(
            e["fe_seconds"] for e in self._entries if e.get("factor_id") == factor_id
        )

    def totals(self) -> dict[str, float]:
        return {
            "attempts": float(len(self._entries)),
            "duplicates": float(self.duplicate_count),
            "qe_seconds": self.total_qe_seconds,
            "fe_seconds": self.total_fe_seconds,
            "qe_saved_seconds": self.total_qe_saved_seconds,
            "net_qe_seconds": self.net_qe_seconds,
            "llm_calls": float(self.total_llm_calls),
            "llm_tokens": float(self.total_llm_tokens),
            "llm_cost": self.total_llm_cost,
        }

    def cost_source(self) -> Callable[[str], float]:
        """factor_id -> 归一 [0,1] 成本（可注入 EV/EVI 的 cost_source）。

        cost = qe_seconds/ref（cap 1.0）+ llm_cost×scale（如启用）。
        重复/缓存命中的 factor 从未执行 → 0（近零 QE 成本语义）。
        """

        cfg = self.config

        def _cost(factor_id: str) -> float:
            qe = self.qe_seconds_of(factor_id)
            fe = self.fe_seconds_of(factor_id)
            total_sec = qe + fe
            compute_cost = 0.0
            if cfg.qe_seconds_ref and cfg.qe_seconds_ref > 0:
                compute_cost = min(1.0, total_sec / cfg.qe_seconds_ref)
            llm_cost = 0.0
            if cfg.include_llm_cost:
                llm_cost = min(1.0, sum(
                    float(e.get("llm_cost", 0.0)) * cfg.llm_cost_scale
                    for e in self._entries
                    if e.get("factor_id") == factor_id
                ))
            return compute_cost + llm_cost

        return _cost

    def state(self) -> dict[str, Any]:
        return self.totals()


def action_cost_adjusted_ev(*, gain: float, cost: float, p: float = 1.0) -> float:
    """cost-adjusted EV：``gain × p / cost``（cost 进 EV，非仅日志）。

    Task 22 测试 2 / Part F5：等增益下贵 action 的 cost-adjusted EV 更低。
    cost=0 → 无成本惩罚（EV = gain×p）。cost 必须 >= 0。
    """
    g = float(gain)
    if not (g >= 0.0):
        raise ValueError(f"gain must be >= 0, got {gain!r}")
    c = float(cost)
    if c < 0.0:
        raise ValueError(f"cost must be >= 0, got {cost!r}")
    prob = float(p)
    if c == 0.0:
        return g * prob
    return g * prob / c


def _default_usage_record() -> UsageRecord:  # pragma: no cover - 便利
    return UsageRecord()
