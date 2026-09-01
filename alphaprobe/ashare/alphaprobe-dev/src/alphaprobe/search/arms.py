"""Search Arms（任务书 §25 / §27 / §30 / §31）。

5 个 arm 统一 ``generate(parents, action, llm_fn)`` 接口：

- ``llm_fn: (system_prompt, user_prompt, model_class) -> str`` —— 测试注入 stub，
  不直接依赖 openai/shared.utils.llm，Phase 9 不调用真实 LLM。
- ``parents`` 为 dict 列表（formula/explanation/topic/fitness/schema_tags...）。
- ``action`` 为 ``SearchAction``（search/__init__.py 的 make_action 构造）。

每个 arm 返回 ``list[GeneratedCandidate]``（structured_llm.GeneratedCandidate），
与任务书 §31 的生成结果一致。GenerationRepairArm 复用 search/__init__.py 的
同名词（§25.5：只把无效生成改成合法公式，不改变统计形状）。
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from alphaprobe.search import GenerationRepairArm  # §25.5 复用（勿重写）
from alphaprobe.search.structured_llm import GeneratedCandidate, parse_generated

LLMFn = Callable[[str, str, str], str]

# §27 Role-aware Formula Representation：公式角色白名单
SIGNAL_ROLES = {
    "signal_core",
    "context_condition",
    "quality_filter",
    "normalization",
    "direction",
    "output_transform",
}
_DEFAULT_ROLE = "signal_core"

# 简单角色启发（§27 首版，不做字符串拼接 "(a)+(b)"）
_QUALITY_PATTERNS = (
    re.compile(r"\b(rank|zscore|cs_pct_rank|cs_mad_zscore)\s*\(", re.IGNORECASE),
    re.compile(r"\b(topk_|ts_rank|ts_zscore)\s*\(", re.IGNORECASE),
)
_CONTEXT_PATTERNS = (
    re.compile(r"\bif_else\s*\(", re.IGNORECASE),
    re.compile(r"\bwhere\s*\(", re.IGNORECASE),
)
# §27 crossover 组合条件子串：if_else(cond, val, otherwise) 的第一实参（括号平衡）。
# §27 crossover 组合条件子串：if_else(cond, val, otherwise) 的第一实参（括号平衡）。
# 条件本身可含函数调用，如 gt(ts_mean(volume, 5), 0.5)。
_IF_ELSE_ARG_RE = re.compile(
    r"\bif_else\s*\(\s*((?:[^(),]|\([^()]*\)|\([^()]*\([^()]*\)[^()]*\))*)",
    re.IGNORECASE,
)

# §25.4 SchemaExploreArm：9 个 schema 维度（值只做表面归一，语义标注交给 LLM/记忆）
SCHEMA_DIMENSIONS: tuple[str, ...] = (
    "Event",
    "Context",
    "Qualities",
    "Direction",
    "Output",
    "DataDomain",
    "Horizon",
    "Normalization",
    "Tradability",
)


# ---------------------------------------------------------------------------
# prompt 拼装（正文走 configs/prompts/*.txt，已由 shared.utils.prompt 加载）
# ---------------------------------------------------------------------------

def _build_prompt(
    parents: Sequence[dict[str, Any]],
    action_type: str,
    num: int,
    instruction: str,
    *,
    schema_tags: dict[str, str] | None = None,
) -> tuple[str, str]:
    """构造 (system_prompt, user_prompt)。

    PROMPT_HEAD 惰性加载；configs/prompts 目录缺失/未配置时降级为内置最小头
    （不阻塞 import，测试可离线跑）。
    """
    try:
        from shared.utils.prompt import PROMPT_HEAD
        system_prompt = PROMPT_HEAD
    except Exception:
        system_prompt = (
            "You are an expert quantitative researcher specializing in alpha factor "
            "mining for A-share equities. Output exactly one JSON object."
        )
    lines = [f"# Task: {instruction}"]
    lines.append(f"Search action type: {action_type}")
    lines.append(f"Generate exactly {num} candidates.")
    if schema_tags:
        lines.append(
            "Target schema: "
            + ", ".join(f"{k}={v}" for k, v in schema_tags.items())
        )
    lines.append("## Parent factor(s)")
    for p in parents:
        f = p.get("formula") or p.get("canonical_formula") or "?"
        e = p.get("explanation") or p.get("description") or ""
        lines.append(f"- {f} :: {e}".rstrip())
    lines.append(
        '## Output JSON (one object only): {"candidates": [{"formula": "...", '
        '"explanation": "...", "hypothesis": "...", "action_type": "REFINE", '
        '"parent_ids": [...], "schema_tags": {...}}]}'
    )
    user_prompt = "\n".join(lines)
    return system_prompt, user_prompt


def _base_action_type(action: Any, arm_fallback: str) -> str:
    """从 SearchAction 取 action_type，缺省回落 arm 缺省类型。"""
    if action is not None:
        at = getattr(action, "action_type", None)
        if at is not None:
            return str(getattr(at, "value", at))
    return arm_fallback


def _parent_ids(parents: Sequence[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for p in parents:
        pid = p.get("factor_id") or p.get("id")
        if pid:
            ids.append(str(pid))
    return ids


def _schema_tags_of(parent: dict[str, Any] | None) -> dict[str, str]:
    if not parent:
        return {}
    st = parent.get("schema_tags") or parent.get("schema")
    return dict(st) if isinstance(st, dict) else {}


def _clone_with_role(
    formula: str, role: str, parent: dict[str, Any] | None
) -> dict[str, Any]:
    """为 role-aware crossover 构造角色标注片段（§27，非字符串拼接）。"""
    base = _schema_tags_of(parent)
    return {
        "formula": str(formula),
        "role": role,
        "schema_tags": base,
    }


# ---------------------------------------------------------------------------
# §27 简单角色启发
# ---------------------------------------------------------------------------

def infer_formula_roles(formula: str) -> dict[str, str]:
    """把一条公式拆成 role → 子串 映射。

    首版启发：
    - if_else/where 条件 → context_condition（整体包裹）；
    - rank/zscore/topk 等截面变换 → quality_filter；
    - 其余 → signal_core。

    对拍口径：只做表面启发，不做真 AST 标注（AST 标注留给 FactorEngine/记忆层）。
    """
    s = str(formula).strip()
    roles: dict[str, str] = {}
    if any(p.search(s) for p in _CONTEXT_PATTERNS):
        # 首版启发：if_else/where 外层 = 整条 context_condition。抽取条件子串
        # 时对 if_else 进一步取第一实参（条件部分），而非整条（否则组合出的
        # if_else(if_else(...), core, 0.0) 会让真实条件嵌套失真）。
        roles["context_condition"] = s
        m = _IF_ELSE_ARG_RE.match(s)
        if m:
            roles["context_condition"] = m.group(1).strip()
    elif any(p.search(s) for p in _QUALITY_PATTERNS):
        roles["quality_filter"] = s
    else:
        roles["signal_core"] = s
    return roles


def _extract_role(formula: str, role: str) -> str:
    return infer_formula_roles(formula).get(role, formula)


def _default_role(formula: str) -> str:
    for role in ("context_condition", "quality_filter", "signal_core"):
        if role in infer_formula_roles(formula):
            return role
    return _DEFAULT_ROLE


# ---------------------------------------------------------------------------
# §25.1 RefineArm
# ---------------------------------------------------------------------------

@dataclass
class RefineArm:
    """小步修改：operator/field/window/normalization/context/quality filter。

    无 LLM 时（llm_fn=None）退化为确定性本地变换，保证全离线可用。
    """

    _transformations: tuple[tuple[str, str], ...] = (
        ("ts_mean(close, 20)", "ts_std(close, 20)"),
        ("ts_mean(close, 20)", "ts_mean(close, 60)"),
        ("ts_mean(close, 20)", "ts_median(close, 20)"),
        ("rank(close)", "zscore(close)"),
        ("ts_corr(close, volume, 10)", "ts_corr(close, amount, 10)"),
    )

    def generate(
        self,
        parents: Sequence[dict[str, Any]],
        action: Any | None = None,
        llm_fn: LLMFn | None = None,
    ) -> list[GeneratedCandidate]:
        action_type = _base_action_type(action, "REFINE")
        if llm_fn is not None:
            sys_p, usr_p = _build_prompt(
                parents, action_type, 3,
                "Produce small refinements of the parent factor "
                "(operator/field/window/normalization/context/quality filter).",
            )
            text = llm_fn(sys_p, usr_p, _route(action_type))
            return parse_generated(text, expected_num=3, default_action_type=action_type)
        out: list[GeneratedCandidate] = []
        for p in parents:
            base = str(p.get("formula") or p.get("canonical_formula") or "")
            if not base:
                continue
            for i, (old, new) in enumerate(self._transformations[:3]):
                if old in base:
                    out.append(
                        GeneratedCandidate(
                            formula=base.replace(old, new, 1),
                            explanation=f"refine #{i + 1}: {old} -> {new}",
                            hypothesis="小步局部替换，保持整体结构",
                            action_type=action_type,
                            parent_ids=_parent_ids([p]),
                            schema_tags=_schema_tags_of(p),
                        )
                    )
                    break
        return out


# ---------------------------------------------------------------------------
# §25.2 BranchArm
# ---------------------------------------------------------------------------

@dataclass
class BranchArm:
    """Tree-of-Thought 风格：一个 parent → 多个机制假设 → 多个 branch。"""

    _mechanisms: tuple[str, ...] = (
        "momentum",
        "reversal",
        "volatility",
        "liquidity",
        "value",
        "flow",
    )

    def generate(
        self,
        parents: Sequence[dict[str, Any]],
        action: Any | None = None,
        llm_fn: LLMFn | None = None,
    ) -> list[GeneratedCandidate]:
        action_type = _base_action_type(action, "STATE_CONDITION")
        if llm_fn is not None:
            sys_p, usr_p = _build_prompt(
                parents, action_type, 3,
                "Propose distinct mechanism hypotheses for the parent factor and "
                "branch each into a candidate.",
            )
            text = llm_fn(sys_p, usr_p, _route(action_type))
            return parse_generated(text, expected_num=3, default_action_type=action_type)
        out: list[GeneratedCandidate] = []
        for p in parents:
            base = str(p.get("formula") or p.get("canonical_formula") or "")
            if not base:
                continue
            for m in self._mechanisms[:3]:
                out.append(
                    GeneratedCandidate(
                        formula=base,
                        explanation=f"branch under mechanism: {m}",
                        hypothesis=f"假设该信号由{m}机制驱动，先低成本 scout 验证",
                        action_type=action_type,
                        parent_ids=_parent_ids([p]),
                        schema_tags=_schema_tags_of(p),
                    )
                )
        return out


# ---------------------------------------------------------------------------
# §25.3 EvolutionArm：mutation + role-aware crossover
# ---------------------------------------------------------------------------

@dataclass
class EvolutionArm:
    """mutation 做局部变换；crossover 做 §27 角色感知组合。

    不做 ``f"({a}) + ({b})"`` 字符串拼接：crossover 输出 = A.signal_core 包裹
    B.context_condition 的结构化组合，并在 hypothesis 里写明角色出处。
    """

    _mutations: tuple[tuple[str, str], ...] = (
        ("ts_mean(close, 20)", "ts_mean(close, 60)"),
        ("rank(close)", "rank(ts_std(close, 20))"),
        ("ts_corr(close, volume, 10)", "ts_corr(close, volume, 20)"),
        ("ts_mean(close, 20)", "ts_pct(close, 5)"),
    )

    def _mutate(self, formula: str, rng: Any, parent: dict[str, Any]) -> GeneratedCandidate:
        for old, new in self._mutations:
            if old in formula:
                return GeneratedCandidate(
                    formula=formula.replace(old, new, 1),
                    explanation=f"mutation: {old} -> {new}",
                    hypothesis="随机局部变异，保持信号角色不变",
                    action_type="REFINE",
                    parent_ids=_parent_ids([parent]),
                    schema_tags=_schema_tags_of(parent),
                )
        return GeneratedCandidate(
            formula=formula,
            explanation="mutation (no-op on unknown structure)",
            hypothesis="未能定位可变异结构",
            action_type="REFINE",
            parent_ids=_parent_ids([parent]),
            schema_tags=_schema_tags_of(parent),
        )

    def _crossover(
        self,
        a: dict[str, Any],
        b: dict[str, Any],
        role: str,
        *,
        cond_override: str | None = None,
    ) -> GeneratedCandidate:
        """role-aware crossover：A.signal_core 包裹 B.context_condition。

        cond_override 由调用方传入 B 的已抽取条件子串（B 可能没有 context_condition
        角色而用了 quality_filter/signal_core 段作条件）；None 时退回自动抽取。
        """
        fa = str(a.get("formula") or a.get("canonical_formula") or "")
        fb = str(b.get("formula") or b.get("canonical_formula") or "")
        core_a = _extract_role(fa, "signal_core")
        if cond_override is not None:
            cond_b = cond_override
        else:
            cond_b = _extract_role(fb, "context_condition")
            cond_b = cond_b if cond_b != fb else _extract_role(fb, _default_role(fb))
        combined = f"if_else({cond_b}, {core_a}, 0.0)"
        tags = dict(_schema_tags_of(a))
        tags.update(_schema_tags_of(b))
        return GeneratedCandidate(
            formula=combined,
            explanation=(
                f"role-aware crossover: A[{_default_role(fa)}] wrapped with "
                f"B[{_default_role(fb)}] as condition"
            ),
            hypothesis="A 的核心信号只在 B 的条件成立时暴露，非简单线性相加",
            action_type="CROSSOVER",
            parent_ids=_parent_ids([a]) + [pid for pid in _parent_ids([b]) if pid not in _parent_ids([a])],
            schema_tags=tags,
        )

    def generate(
        self,
        parents: Sequence[dict[str, Any]],
        action: Any | None = None,
        llm_fn: LLMFn | None = None,
        rng: Any | None = None,
    ) -> list[GeneratedCandidate]:
        import random as _random
        rng = rng or _random.Random(0)
        action_type = _base_action_type(action, "CROSSOVER")
        if llm_fn is not None:
            sys_p, usr_p = _build_prompt(
                parents, action_type, 3,
                "Produce evolution candidates: mutation and role-aware crossover "
                "(A.signal_core wrapped by B.context_condition).",
            )
            text = llm_fn(sys_p, usr_p, _route(action_type))
            return parse_generated(text, expected_num=3, default_action_type=action_type)
        ps = list(parents)
        if not ps:
            return []
        out: list[GeneratedCandidate] = []
        p = ps[0]
        out.append(self._mutate(str(p.get("formula") or ""), rng, p))
        # §27 真 multi-parent role-aware crossover：≥2 parents 时按角色抽取
        # 产出 if_else(B.context_condition, A.signal_core, 0.0)；角色抽取失败
        # 或 parents 不足 → 回落单 parent 路径（不抛）。parents 间无序
        # （补充 parent 不一定排在 ps[1]，角色解析前先做确定性去重）。
        uniq: list[dict[str, Any]] = []
        _seen_f: set[str] = set()
        for _p in ps:
            _pf = str(_p.get("formula") or _p.get("canonical_formula") or "").strip()
            if _pf and _pf not in _seen_f:
                _seen_f.add(_pf)
                uniq.append(_p)
        if not uniq:
            uniq = ps
        core = _extract_role(str(uniq[0].get("formula") or uniq[0].get("canonical_formula") or ""), "signal_core")
        for b in uniq[1:]:
            fb = str(b.get("formula") or b.get("canonical_formula") or "")
            cond = _extract_role(fb, "context_condition")
            if cond != fb:
                out.append(self._crossover(uniq[0], b, "signal_core", cond_override=cond))
                continue
            # B 无 context 角色：quality_filter / signal_core 任一段都可能是条件
            cond = _extract_role(fb, "quality_filter")
            if cond != fb:
                out.append(self._crossover(uniq[0], b, "signal_core", cond_override=cond))
                continue
            cond = _extract_role(fb, "signal_core")
            if cond != fb:
                out.append(self._crossover(uniq[0], b, "signal_core", cond_override=cond))
        # 没有任何 pair 产出 crossover（全部无 context）→ 单 parent 回落：
        # 补一条基于 ps[0] 的 no-op mutation（不抛、不静默变空）。
        if not any(c.action_type == "CROSSOVER" for c in out):
            # 全部 pair 都无 context 角色可抽 → 单 parent 回落：直接用核心 parent
            # 的 signal_core 再试一次 B 的 context_condition（B 可能整条就是条件）。
            if len(uniq) >= 2:
                b = uniq[1]
                fb2 = str(b.get("formula") or b.get("canonical_formula") or "")
                cond2 = _extract_role(fb2, "context_condition")
                if cond2 != fb2 and core:
                    out.append(self._crossover(uniq[0], b, "signal_core", cond_override=cond2))
            if not any(c.action_type == "CROSSOVER" for c in out):
                # 二次兜底：uniq[0] 本身带 context（本身就是 if_else 条件包装）→
                # 用 uniq[0] 作条件、uniq[1] 作核心再试（保证 ≥2 parents 必有 pair）。
                if len(uniq) >= 2 and core == str(uniq[0].get("formula") or uniq[0].get("canonical_formula") or ""):
                    cond0 = _extract_role(str(uniq[0].get("formula") or ""), "context_condition")
                    if cond0 != str(uniq[0].get("formula") or ""):
                        out.append(self._crossover(uniq[1], uniq[0], "signal_core", cond_override=cond0))
            if not any(c.action_type == "CROSSOVER" for c in out):
                out.append(
                    GeneratedCandidate(
                        formula=str(p.get("formula") or p.get("canonical_formula") or ""),
                        explanation="crossover fallback: no context_condition parent; single-parent path",
                        hypothesis="缺少可作条件的 context parent，回落单 parent 生成",
                        action_type=action_type,
                        parent_ids=_parent_ids([p]),
                        schema_tags=_schema_tags_of(p),
                    )
                )
        return out


# ---------------------------------------------------------------------------
# §25.4 SchemaExploreArm：9 维 schema 维护 + schema_tags 候选
# ---------------------------------------------------------------------------

@dataclass
class SchemaExploreArm:
    """按 §25.4 的 9 个 schema 维度主动找"还没被挖烂"的机制。

    维护内部维度覆盖率 dict（key=维度，value=已见值集合），每次探索优先挑
    覆盖率最低的维度；候选带 schema_tags（父因子 tags 合并目标维度新值）。
    """

    dimensions: tuple[str, ...] = SCHEMA_DIMENSIONS
    _coverage: dict[str, set[str]] | None = None
    _values: dict[str, tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        if self._coverage is None:
            self._coverage = {d: set() for d in self.dimensions}
        if self._values is None:
            self._values = {
                "Event": ("earnings_surprise", "limit_up", "index_rebalance"),
                "Context": ("bull_market", "high_vol", "low_liquidity"),
                "Qualities": ("momentum", "reversal", "quality"),
                "Direction": ("long", "short"),
                "Output": ("rank", "zscore", "raw"),
                "DataDomain": ("price", "volume", "fundamental"),
                "Horizon": ("short", "medium", "long"),
                "Normalization": ("cross_section", "time_series", "none"),
                "Tradability": ("liquid", "illiquid", "cap_small", "cap_large"),
            }

    def coverage(self) -> dict[str, int]:
        return {d: len(self._coverage.get(d, set())) for d in self.dimensions}

    def _target_dimension(self) -> str:
        return min(
            self.dimensions,
            key=lambda d: len(self._coverage.get(d, set())),
        )

    def _candidate_value(self, dim: str) -> str:
        seen = self._coverage.get(dim, set())
        for v in self._values.get(dim, ()):
            if v not in seen:
                return v
        return self._values.get(dim, ())[0] if self._values.get(dim) else "unknown"

    def generate(
        self,
        parents: Sequence[dict[str, Any]],
        action: Any | None = None,
        llm_fn: LLMFn | None = None,
    ) -> list[GeneratedCandidate]:
        action_type = _base_action_type(action, "SCHEMA_EXPLORE")
        dim = self._target_dimension()
        value = self._candidate_value(dim)
        if llm_fn is not None:
            sys_p, usr_p = _build_prompt(
                parents, action_type, 3,
                f"Explore schema dimension '{dim}' with target value '{value}'.",
                schema_tags={dim: value},
            )
            text = llm_fn(sys_p, usr_p, _route(action_type))
            cands = parse_generated(
                text, expected_num=3, default_action_type=action_type
            )
            for c in cands:
                c.schema_tags.setdefault(dim, value)
                self._mark(dim, value)
            return cands
        out: list[GeneratedCandidate] = []
        for p in parents:
            base = str(p.get("formula") or p.get("canonical_formula") or "")
            if not base:
                continue
            tags = dict(_schema_tags_of(p))
            tags[dim] = value
            out.append(
                GeneratedCandidate(
                    formula=base,
                    explanation=f"schema explore: {dim}={value}",
                    hypothesis=f"探索尚未覆盖的 schema 维度 {dim}={value}",
                    action_type=action_type,
                    parent_ids=_parent_ids([p]),
                    schema_tags=tags,
                )
            )
        self._mark(dim, value)
        return out

    def _mark(self, dim: str, value: str) -> None:
        self._coverage.setdefault(dim, set()).add(value)


# ---------------------------------------------------------------------------
# §30 LLM 模型路由（与 search/__init__.py 的 model_route 保持同一语义）
# ---------------------------------------------------------------------------

def _route(action_type: str) -> str:
    from alphaprobe.search import model_route
    return model_route(action_type)
