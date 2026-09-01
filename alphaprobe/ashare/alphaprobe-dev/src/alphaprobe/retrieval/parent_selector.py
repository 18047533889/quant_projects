"""ParentSelector —— 父节点选择（任务书 §21-§23）。

给 orchestrator/round_manager 用的最小选择器：

- ``enabled=False``（默认）：零行为变化，原样返回前 k 个（向后兼容，全量测试不挂）；
- ``enabled=True``：按 BayesianRetriever.select_parents（RetrieverScore 降序）选 k 个；
- 自动跳过与 main parent 相同 factor_id / 相同 formula 的候选（保留 main 本身）。

与 P1-pool 的 ``extra_parents``/``select_complement`` 兼容：本选择器只管选 top-k
补充 parent 池，是否做结构远度 complement 由 orchestrator 现有逻辑决定。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from alphaprobe.retrieval.bayesian_retriever import BayesianRetriever


@dataclass
class ParentSelector:
    """候选 parent 的 RetrieverScore 排序选择器。

    retriever : BayesianRetriever | None
        注入的检索器。None 时内部惰性构造（默认 config，enabled=False）。
    enabled : bool | None
        覆盖 retriever.config.enabled。None = 跟随 retriever。
    """

    retriever: BayesianRetriever | None = None
    enabled: bool | None = None

    def __post_init__(self) -> None:
        if self.retriever is None:
            self.retriever = BayesianRetriever()

    @property
    def active(self) -> bool:
        if self.enabled is not None:
            return bool(self.enabled)
        return bool(self.retriever.config.enabled)

    def select_parents(
        self,
        candidates: Sequence[Any],
        k: int = 5,
        *,
        main: dict[str, Any] | None = None,
        action_to_retrieve: str | None = None,
        visible_survival: dict[str, Any] | None = None,
    ) -> list[Any]:
        """返回排序后的前 k 个候选。

        disabled → 原样前 k 个（零行为变化）。main 给定且出现在 candidates 中时
        保留 main 并去重（同 factor_id / 同 formula）。
        """
        cands = list(candidates or [])
        if not cands:
            return []
        if main is not None:
            main_id = str(main.get("factor_id") or main.get("id") or "")
            main_f = str(
                main.get("formula") or main.get("canonical_formula") or ""
            ).strip()
            seen: set[str] = set()
            uniq: list[Any] = []
            for c in cands:
                cid = ""
                if isinstance(c, dict):
                    cid = str(c.get("factor_id") or c.get("id") or "")
                    cf = str(c.get("formula") or c.get("canonical_formula") or "").strip()
                else:
                    cid = str(getattr(c, "factor_id", "") or "")
                    cf = str(getattr(c, "formula", "") or getattr(c, "canonical_formula", "") or "").strip()
                key = cid or cf
                if key and key in seen:
                    continue
                if cid and main_id and cid == main_id:
                    continue
                if cf and main_f and cf == main_f:
                    continue
                if key:
                    seen.add(key)
                uniq.append(c)
            cands = uniq
        if not self.active:
            return cands[:k]
        return self.retriever.select_parents(
            cands, k, action_to_retrieve=action_to_retrieve, visible_survival=visible_survival
        )
