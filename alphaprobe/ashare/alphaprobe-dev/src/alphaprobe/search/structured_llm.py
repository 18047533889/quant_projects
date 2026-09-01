"""LLM 结构化输出（任务书 §31）。

优先 JSON constrained output；regex fallback 只做兼容并打 warning（不依赖从大段
自然语言硬抓 JSON）。Phase 9 不调用真实 LLM：parse 只消费文本，测试注入 stub。
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

from alphaprobe.contracts import SearchActionType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# §31 GeneratedCandidate
# ---------------------------------------------------------------------------

@dataclass
class GeneratedCandidate:
    """§31：LLM 单条生成的结构化结果。"""

    formula: str
    explanation: str = ""
    hypothesis: str = ""
    action_type: str = "REFINE"
    parent_ids: list[str] = field(default_factory=list)
    schema_tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GeneratedCandidate":
        return cls(
            formula=str(d.get("formula", "")).strip(),
            explanation=str(d.get("explanation", "") or ""),
            hypothesis=str(d.get("hypothesis", "") or ""),
            action_type=_normalize_action_type(d.get("action_type")),
            parent_ids=[str(p) for p in (d.get("parent_ids") or []) if p],
            schema_tags={str(k): str(v) for k, v in (d.get("schema_tags") or {}).items()},
        )


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

def _normalize_action_type(value: Any) -> str:
    if value is None:
        return "REFINE"
    s = str(value)
    # 兼容 SearchActionType.REFINE / "REFINE" 两种写法
    if s.startswith("SearchActionType."):
        s = s.split(".", 1)[1]
    return s.upper()


# 兼容现有 trainer.py 的 regex 逻辑：```json {...} ``` 或裸 {...}
_JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*(\{[\s\S]*?\})\s*```|\{([\s\S]*?)\}", re.MULTILINE
)


def _balanced_json_candidates(text: str) -> list[str]:
    # 括号平衡扫描：从每个 '{' 出发找第一个深度归零的 '}'。
    # 只做兼容 fallback——比 trainer 原 regex（非贪婪 \\{([\s\S]*?)\\} 会在第一个
    # 内层 '}' 截断）更能处理含嵌套对象的 JSON。不处理字符串内的大括号（DSL
    # 公式不含 '{'/'}'，可接受）。
    out: list[str] = []
    for m in re.finditer(r"\{", text):
        depth = 0
        i = m.start()
        while i < len(text):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out.append(text[m.start() : i + 1])
                    break
            i += 1
    return out


def _find_first_json_obj(text: str) -> dict[str, Any] | None:
    """返回第一个能 json.loads 的 JSON 对象（优先整体解析，再逐块尝试）。"""
    stripped = str(text).strip()
    if not stripped:
        return None
    # 整体直接可解析 → 最优先
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass
    # markdown 包裹 / 大段文字中的 JSON 块（兼容 trainer 的 regex 逻辑）
    for match in _JSON_BLOCK_RE.finditer(stripped):
        candidate = match.group(1) or f"{{{match.group(2)}}}"
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    # 括号平衡 fallback（处理含嵌套对象的裸 JSON 块）
    for candidate in _balanced_json_candidates(stripped):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _extract_candidates(obj: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("candidates", "expressions_fixed", "expressions"):
        val = obj.get(key)
        if isinstance(val, list):
            return [item for item in val if isinstance(item, dict)]
        if isinstance(val, dict):
            return [val]
    # 顶层本身就是单条 candidate
    if any(k in obj for k in ("formula", "expression")):
        return [obj]
    return []


def parse_generated(
    text: str,
    expected_num: int,
    *,
    default_action_type: str = "REFINE",
    allow_skip: bool = True,
) -> list[GeneratedCandidate]:
    """解析 LLM 输出为 GeneratedCandidate 列表。

    优先 ``json.loads``；失败时 regex fallback 并打 warning 计数。
    非法条目跳过不抛（allow_skip=True）。首版跳过空 formula 与缺 candidates 结构。
    """
    obj = _find_first_json_obj(text)
    candidates: list[GeneratedCandidate] = []
    if obj is None:
        warnings.warn(
            "parse_generated: JSON parse failed; regex fallback also empty "
            f"(expected_num={expected_num})",
            RuntimeWarning,
            stacklevel=2,
        )
        logger.warning(
            "parse_generated fallback empty; input text head=%r", str(text)[:120]
        )
        return candidates
    raw_items = _extract_candidates(obj)
    for item in raw_items:
        try:
            cand = GeneratedCandidate.from_dict(item)
        except Exception as exc:  # noqa: BLE001 - 非法条目跳过不抛
            logger.debug("parse_generated skip invalid item: %s", exc)
            continue
        formula = cand.formula.strip()
        if not formula:
            logger.debug("parse_generated skip empty formula")
            continue
        if default_action_type and cand.action_type == "REFINE":
            cand.action_type = default_action_type
        candidates.append(cand)
    if len(candidates) != expected_num:
        warnings.warn(
            f"parse_generated: expected {expected_num} candidates, got {len(candidates)}",
            RuntimeWarning,
            stacklevel=2,
        )
        logger.warning(
            "parse_generated count mismatch expected=%s got=%s", expected_num, len(candidates)
        )
    return candidates


def candidates_to_json(cands: Sequence[GeneratedCandidate]) -> str:
    """把候选列表序列化为 §31 的 JSON（测试 stub / 离线 replay 用）。"""
    return json.dumps(
        {"candidates": [c.to_dict() for c in cands]},
        ensure_ascii=False,
    )


__all__ = [
    "GeneratedCandidate",
    "parse_generated",
    "candidates_to_json",
    "SearchActionType",
]
