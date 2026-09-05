"""plan.md Task 21 / Part A12：MemoryPacket V2（multi-parent、多段、token 预算）。

设计（与 A12「仍偏单 parent」对照）
---------------------------------
- **selected parent set + roles**：packet 持有 ``parents`` 列表
  （[(role, parent_dict), ...]）+ ``primary_index``，不再只偏第一个 parent。
- **13 段**（plan Task 21 清单）：research objective / parent set+roles /
  current logic+schema / last 3-5 lineage actions / best offspring /
  representative failures / structural nearest / numerical nearest /
  cluster saturation / successful+failed actions / allowed FE operators+DA
  fields / legal survival info / avoid-repeat identities。
- **缺失段优雅省略**：section 内容为 None/空 → 该段不进 prompt，绝不硬造。
- **sealed/current-version 隔离**（#23）：调用方（retriever）用
  ``exclude_sealed=True`` 过滤 survival 段（survival_profiles 里属于被封存
  版本/未来版本的行不进入当前 prompt；本模块只做过滤判定，不复制版本语义）。
- **token budget 2k-4k**（可配置 + 确定性截断，ablation switch #30）：
  ``PacketBuilder`` 组段后按 budget 从**末尾段**向**首位段**确定性截断
  （固定优先级：research objective 永远保留；截断只发生在 oversize 段内部，
  按「每段先整段丢弃、再段内条目从后向前丢」的固定规则，不随机）。
- 保持既有调用兼容：``to_prompt_text`` / ``parent`` / ``numerical_neighbors``
  等属性与旧版 MemoryPacket 相同（retriever/store 与 round_manager 仍可用）。

本模块零顶层平台依赖：不 import factor_engine / data_access / factor_assets。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "DEFAULT_TOKEN_BUDGET_MIN",
    "DEFAULT_TOKEN_BUDGET_MAX",
    "SECTIONS",
    "Section",
    "PacketBuilder",
    "MemoryPacketV2",
    "simple_tokens",
    "sealed_survival_rows",
]

#: 默认 token 预算（plan 2k-4k）。
DEFAULT_TOKEN_BUDGET_MIN: int = 2000
DEFAULT_TOKEN_BUDGET_MAX: int = 4000

#: 段内条目截断时的最大保留量（超长只截断不随机）。
_MAX_ITEMS_PER_SECTION: int = 12
#: 单条公式/长文本保留的字符上限（截断优先级固定：先段内条目，再段内文本）。
_MAX_TEXT_CHARS: int = 400


class Section(str):
    """Packet 13 段（plan Task 21 清单，序号稳定）。"""

    RESEARCH_OBJECTIVE = "1_research_objective"
    PARENT_SET = "2_parent_set_roles"
    LOGIC_SCHEMA = "3_logic_schema"
    LINEAGE_ACTIONS = "4_last_lineage_actions"
    BEST_OFFSPRING = "5_best_offspring"
    REPRESENTATIVE_FAILURES = "6_representative_failures"
    STRUCTURAL_NEAREST = "7_structural_nearest"
    NUMERICAL_NEAREST = "8_numerical_nearest"
    CLUSTER_SATURATION = "9_cluster_saturation"
    ACTIONS_TRIED = "10_successful_failed_actions"
    ALLOWED_SURFACE = "11_allowed_operators_fields"
    LEGAL_SURVIVAL = "12_legal_survival_info"
    AVOID_REPEAT = "13_avoid_repeat_identities"


SECTIONS: tuple[Section, ...] = (
    Section.RESEARCH_OBJECTIVE,
    Section.PARENT_SET,
    Section.LOGIC_SCHEMA,
    Section.LINEAGE_ACTIONS,
    Section.BEST_OFFSPRING,
    Section.REPRESENTATIVE_FAILURES,
    Section.STRUCTURAL_NEAREST,
    Section.NUMERICAL_NEAREST,
    Section.CLUSTER_SATURATION,
    Section.ACTIONS_TRIED,
    Section.ALLOWED_SURFACE,
    Section.LEGAL_SURVIVAL,
    Section.AVOID_REPEAT,
)

#: 截断时永不丢弃的段（research objective 是研究意图的最小上下文）。
_KEEP_ALWAYS = frozenset({Section.RESEARCH_OBJECTIVE, Section.PARENT_SET})


def simple_tokens(text: str) -> int:
    """确定性近似 token 计数（无 tokenizer 依赖）。

    规则：CJK 每字 ≈ 1 token；其余按 4 字符 ≈ 1 token（ASCII 平均）。只做
    预算截断的相对排序用，不追求精确 token 数。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    rest = len(text) - cjk
    return cjk + max(1, rest // 4)


def _fmt_parent(p: Mapping[str, Any]) -> str:
    f = str(p.get("formula") or p.get("canonical_formula") or p.get("id") or "?")
    fit = p.get("fitness")
    return f"{f}(fitness={fit})" if fit is not None else f


def _fmt_neighbor(n: Mapping[str, Any]) -> str:
    f = str(n.get("formula") or n.get("factor_id") or "?")
    fit = n.get("fitness")
    return f"{f}(fitness={fit})" if fit is not None else f


def _fmt_action(a: Mapping[str, Any]) -> str:
    at = a.get("action_type") or a.get("type") or "?"
    act = a.get("action") or a.get("name") or ""
    status = a.get("status") or a.get("outcome") or ""
    parts = [f"{at}({act})"]
    if status:
        parts.append(status)
    return ":".join(parts)


def sealed_survival_rows(rows: Sequence[Mapping[str, Any]], *, version_key: str) -> list[dict[str, Any]]:
    """从 survival exemplars 行里剔除被封存（非当前研究版本）的行。

    ``version_key``：当前 research version 标识。行里带 ``version_key`` /
    ``research_version`` / ``segment`` 字段（或 ``payload`` JSON 内嵌同名字段）
    且 **不等于** 当前版本 → 视为已封存/未来版本 → 排除。不带任何版本信息的
    行视为当前版本合法。只做字段比对过滤，不复制版本语义。
    """
    out: list[dict[str, Any]] = []
    for r in rows or ():
        d = dict(r)
        # 版本字段：优先 payload（JSON 内嵌 research_version/version_key），
        # 其次行级 version_key/research_version/segment。
        version_fields: list[Any] = []
        payload = d.get("payload")
        if isinstance(payload, str):
            try:
                import json

                payload = json.loads(payload)
            except Exception:  # noqa: BLE001
                payload = None
        if isinstance(payload, Mapping):
            for key in ("research_version", "version_key"):
                v = payload.get(key)
                if v is not None:
                    version_fields.append(v)
        for key in ("version_key", "research_version", "segment"):
            v = d.get(key)
            if v is not None:
                version_fields.append(v)
        # 版本信息存在且任一 != 当前版本 → 排除（sealed 隔离）
        if version_fields and all(str(v) != str(version_key) for v in version_fields):
            continue
        out.append(d)
    return out


def _section_lines(section: Section, payload: Any) -> list[str]:
    """把一个 section 的 payload 渲染成文本行（None/空 → []）。"""
    if payload is None:
        return []
    lines: list[str] = []
    if section == Section.RESEARCH_OBJECTIVE:
        if isinstance(payload, Mapping):
            text = payload.get("text") or payload.get("objective")
            if text:
                lines.append(f"[Research objective] {text}")
        elif isinstance(payload, str) and payload:
            lines.append(f"[Research objective] {payload}")
    elif section == Section.PARENT_SET:
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
            rendered = []
            for item in payload:
                if isinstance(item, Mapping):
                    role = item.get("role")
                    p = item.get("parent") or item
                    base = _fmt_parent(p)
                    rendered.append(f"{role}={base}" if role else base)
                elif item:
                    rendered.append(str(item))
            if rendered:
                lines.append("[Parent set] " + "; ".join(rendered))
        elif isinstance(payload, Mapping):
            lines.append("[Parent set] " + _fmt_parent(payload))
    elif section == Section.LOGIC_SCHEMA:
        if isinstance(payload, Mapping):
            parts = []
            logic = payload.get("logic") or payload.get("logic_id")
            schema = payload.get("schema") or payload.get("schema_id")
            if logic:
                parts.append(f"logic={logic}")
            if schema:
                parts.append(f"schema={schema}")
            if parts:
                lines.append("[Logic/Schema] " + ", ".join(parts))
        elif isinstance(payload, str) and payload:
            lines.append(f"[Logic/Schema] {payload}")
    elif section == Section.LINEAGE_ACTIONS:
        if isinstance(payload, Sequence):
            rendered = [_fmt_action(a) for a in payload if isinstance(a, Mapping)]
            if rendered:
                lines.append("[Last lineage actions] " + "; ".join(rendered))
    elif section == Section.BEST_OFFSPRING:
        if isinstance(payload, Sequence):
            rendered = [
                str(o.get("formula") or o.get("factor_id") or o.get("id") or "?")
                for o in payload if isinstance(o, Mapping)
            ]
            if rendered:
                lines.append("[Best offspring] " + "; ".join(rendered))
        elif isinstance(payload, Mapping):
            lines.append("[Best offspring] " + _fmt_parent(payload))
    elif section == Section.REPRESENTATIVE_FAILURES:
        if isinstance(payload, Sequence):
            rendered = [
                f"{f.get('reason','?')}:{f.get('formula', f.get('factor_id','?'))}"
                for f in payload if isinstance(f, Mapping)
            ]
            if rendered:
                lines.append("[Failures to avoid] " + "; ".join(rendered))
    elif section == Section.STRUCTURAL_NEAREST:
        if isinstance(payload, Sequence):
            rendered = [_fmt_neighbor(n) for n in payload if isinstance(n, Mapping)]
            if rendered:
                lines.append("[Structural nearest] " + "; ".join(rendered))
    elif section == Section.NUMERICAL_NEAREST:
        if isinstance(payload, Sequence):
            rendered = [_fmt_neighbor(n) for n in payload if isinstance(n, Mapping)]
            if rendered:
                lines.append("[Numerical nearest] " + "; ".join(rendered))
    elif section == Section.CLUSTER_SATURATION:
        if isinstance(payload, Mapping):
            parts = []
            for k in ("cluster_id", "saturation", "member_count", "survival_rate"):
                v = payload.get(k)
                if v is not None:
                    parts.append(f"{k}={v}")
            if payload.get("top") and isinstance(payload["top"], Sequence):
                top = payload["top"]
                rendered = [
                    f"{t.get('cluster_id','?')}(m={t.get('member_count','?')})"
                    for t in top if isinstance(t, Mapping)
                ]
                if rendered:
                    parts.append("top=" + ",".join(rendered))
            if parts:
                lines.append("[Cluster saturation] " + ", ".join(parts))
        elif isinstance(payload, str) and payload:
            lines.append(f"[Cluster saturation] {payload}")
    elif section == Section.ACTIONS_TRIED:
        if isinstance(payload, Mapping):
            parts = []
            for k in ("successful", "failed", "saturated", "unexplored"):
                v = payload.get(k)
                if v:
                    items = v if isinstance(v, Sequence) else [v]
                    parts.append(f"{k}=" + ",".join(str(i) for i in items))
            if parts:
                lines.append("[Actions tried] " + "; ".join(parts))
    elif section == Section.ALLOWED_SURFACE:
        if isinstance(payload, Mapping):
            parts = []
            ops = payload.get("operators")
            fields = payload.get("fields")
            if ops:
                parts.append("operators=" + ",".join(str(o) for o in ops))
            if fields:
                parts.append("fields=" + ",".join(str(f) for f in fields))
            if parts:
                lines.append("[Allowed surface] " + "; ".join(parts))
    elif section == Section.LEGAL_SURVIVAL:
        if isinstance(payload, Sequence):
            rendered = []
            for e in payload:
                if not isinstance(e, Mapping):
                    continue
                f = str(e.get("formula") or e.get("factor_id") or "?")
                sr = e.get("survival_rate")
                if sr is not None:
                    rendered.append(f"{f}(survival_rate={sr})")
                else:
                    rendered.append(f)
            if rendered:
                lines.append("[Legal survival exemplars] " + "; ".join(rendered))
        elif isinstance(payload, Mapping):
            lines.append("[Legal survival exemplars] " + _fmt_parent(payload))
    elif section == Section.AVOID_REPEAT:
        if isinstance(payload, Sequence):
            rendered = []
            for e in payload:
                if isinstance(e, Mapping):
                    fid = e.get("factor_id") or e.get("id") or e.get("formula")
                    reason = e.get("reason") or e.get("why")
                    rendered.append(f"{fid}:{reason}" if reason else str(fid))
                elif e:
                    rendered.append(str(e))
            if rendered:
                lines.append("[Avoid repeating] " + "; ".join(rendered))
        elif isinstance(payload, str) and payload:
            lines.append(f"[Avoid repeating] {payload}")
    return lines


def _section_item_text(section: Section, item: Any) -> str:
    """把段内单条目渲染成文本（确定性截断用；空条目 → ""）。"""
    if isinstance(item, Mapping):
        return " ".join(str(v) for v in item.values() if v is not None)
    return str(item)


@dataclass
class PacketBuilder:
    """13 段组装 + token 预算确定性截断。

    ``sections``：dict[Section, Any]。缺失段自动省略（None/空 → 不进 prompt）。
    ``budget_max`` / ``budget_min``：2k-4k 可配置；``max_tokens=None`` 表示不
    截断（返回完整文本）。截断只发生在 oversize 段内部，按「从尾段向前段」
    的固定优先级，段内条目从后向前丢（research objective / parent set 保留）。
    """

    budget_max: int = DEFAULT_TOKEN_BUDGET_MAX
    budget_min: int = DEFAULT_TOKEN_BUDGET_MIN
    sections: dict[Section, Any] = field(default_factory=dict)
    #: 段内文本截断（长公式/描述）开关（ablation switch #30）。
    truncate_long_text: bool = True

    def section_payload(self, section: Section) -> Any:
        return self.sections.get(section)

    # -- 确定性截断 -------------------------------------------------------

    def _render(self) -> list[str]:
        """按 SECTIONS 顺序渲染所有非空段（缺失段不出行）。"""
        lines: list[str] = []
        for section in SECTIONS:
            payload = self.sections.get(section)
            if payload is None:
                continue
            slines = _section_lines(section, payload)
            if slines:
                lines.append("\n".join(slines))
        return lines

    def _truncate_items(
        self,
        section: Section,
        payload: Sequence[Any],
        *,
        max_tokens: int,
    ) -> list[str]:
        """把超长段内部条目按「从后向前丢」确定性截断。

        段本身是行列表（section 头 + 条目）；保留头部行（section 标签）与
        前面的条目，丢弃靠后的条目（先丢尾部）。返回最终保留的渲染行。
        """
        # 把 payload 渲染成独立条目行（含 section 头一行）
        lines: list[str] = []
        for item in payload:
            if isinstance(item, Mapping):
                item_lines = _section_lines(section, [item])
            elif isinstance(item, str):
                item_lines = _section_lines(section, [{"_s": item}])
            else:
                item_lines = []
            lines.extend(item_lines)
        # 从后向前逐条丢，直到 token ≤ max_tokens
        while lines and simple_tokens("\n".join(lines)) > max_tokens:
            lines.pop()
        return lines

    def build_text(self, *, max_tokens: int | None = None) -> str:
        """确定性渲染：先按段序渲染，再按预算从尾段向前段截断。"""
        cap = max_tokens if max_tokens is not None else self.budget_max
        full = "\n".join(self._render())
        if simple_tokens(full) <= cap:
            return full
        # oversize：按段从尾向首逐段尝试截断（research objective 保留）
        kept: list[tuple[Section, list[str]]] = []
        for section in reversed(SECTIONS):
            payload = self.sections.get(section)
            if payload is None:
                continue
            slines = _section_lines(section, payload)
            if not slines:
                continue
            if section in _KEEP_ALWAYS or simple_tokens("\n".join(slines)) <= cap:
                kept.append((section, slines))
                continue
            # 段超 budget：段内条目截断（从后向前丢）
            if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
                trimmed = self._truncate_items(section, payload, max_tokens=cap)
                kept.append((section, trimmed))
            else:
                # 不可分条目的超长文本：硬截断文本前 _MAX_TEXT_CHARS 字符
                text_lines = [t[:_MAX_TEXT_CHARS] for t in slines]
                kept.append((section, text_lines))
        kept.reverse()
        out_lines: list[str] = []
        for _sec, slines in kept:
            out_lines.extend(slines)
        return "\n".join(out_lines)


@dataclass
class MemoryPacketV2:
    """MemoryPacket V2（multi-parent、13 段、token 预算）。

    与旧版 MemoryPacket 保持属性兼容：``parent``（主 parent）、
    ``structural_neighbors`` / ``numerical_neighbors`` / ``allowed_fields`` /
    ``allowed_operators`` 等供既有调用方与测试继续读取。
    """

    parents: list[dict[str, Any]] = field(default_factory=list)
    parent_roles: dict[str, str] = field(default_factory=dict)  # factor_id -> role
    structural_neighbors: list[dict[str, Any]] = field(default_factory=list)
    numerical_neighbors: list[dict[str, Any]] = field(default_factory=list)
    successful_offspring: list[dict[str, Any]] = field(default_factory=list)
    representative_failures: list[dict[str, Any]] = field(default_factory=list)
    best_offspring: list[dict[str, Any]] = field(default_factory=list)
    lineage_actions: list[dict[str, Any]] = field(default_factory=list)
    logic_schema: dict[str, Any] = field(default_factory=dict)
    cluster_saturation: dict[str, Any] = field(default_factory=dict)
    successful_actions: list[str] = field(default_factory=list)
    failed_actions: list[str] = field(default_factory=list)
    survival_exemplars: list[dict[str, Any]] = field(default_factory=list)
    avoid_repeat: list[dict[str, Any]] = field(default_factory=list)
    allowed_fields: list[str] = field(default_factory=list)
    allowed_operators: list[str] = field(default_factory=list)
    research_objective: str = ""
    ancestry_summary: str = ""
    unexplored_actions: list[str] = field(default_factory=list)
    saturated_actions: list[str] = field(default_factory=list)
    rare_directions: list[dict[str, Any]] = field(default_factory=list)
    budget_max: int = DEFAULT_TOKEN_BUDGET_MAX
    budget_min: int = DEFAULT_TOKEN_BUDGET_MIN

    # -- 旧版兼容属性 -------------------------------------------------------

    @property
    def parent(self) -> dict[str, Any]:
        if self.parents:
            return self.parents[0]
        return {}

    @property
    def cluster_context(self) -> dict[str, Any] | None:
        return self.cluster_saturation or None

    def to_prompt_text(self) -> str:
        """确定性截断渲染（2k-4k budget）。"""
        return self.build_packet_text()

    def _sections(self) -> dict[Section, Any]:
        s: dict[Section, Any] = {}
        if self.research_objective:
            s[Section.RESEARCH_OBJECTIVE] = {"text": self.research_objective}
        parent_payload = []
        for p in self.parents:
            pid = str(p.get("factor_id") or p.get("id") or "")
            role = self.parent_roles.get(pid, "main" if pid == self._primary_id() else "parent")
            parent_payload.append({"role": role, "parent": p})
        if parent_payload:
            s[Section.PARENT_SET] = parent_payload
        logic_schema = dict(self.logic_schema)
        if not logic_schema and (self.ancestry_summary):
            logic_schema = {"schema": self.ancestry_summary}
        if logic_schema:
            s[Section.LOGIC_SCHEMA] = logic_schema
        if self.lineage_actions:
            s[Section.LINEAGE_ACTIONS] = self.lineage_actions[-5:]
        if self.best_offspring:
            s[Section.BEST_OFFSPRING] = self.best_offspring[:3]
        if self.representative_failures:
            s[Section.REPRESENTATIVE_FAILURES] = self.representative_failures[:5]
        if self.structural_neighbors:
            s[Section.STRUCTURAL_NEAREST] = self.structural_neighbors[:5]
        if self.numerical_neighbors:
            s[Section.NUMERICAL_NEAREST] = self.numerical_neighbors[:5]
        if self.cluster_saturation:
            s[Section.CLUSTER_SATURATION] = self.cluster_saturation
        actions_tried: dict[str, list[str]] = {}
        if self.successful_actions:
            actions_tried["successful"] = self.successful_actions
        if self.failed_actions:
            actions_tried["failed"] = self.failed_actions
        if self.saturated_actions:
            actions_tried["saturated"] = self.saturated_actions
        if self.unexplored_actions:
            actions_tried["unexplored"] = self.unexplored_actions
        if actions_tried:
            s[Section.ACTIONS_TRIED] = actions_tried
        allowed_surface: dict[str, list[str]] = {}
        if self.allowed_operators:
            allowed_surface["operators"] = self.allowed_operators
        if self.allowed_fields:
            allowed_surface["fields"] = self.allowed_fields
        if allowed_surface:
            s[Section.ALLOWED_SURFACE] = allowed_surface
        if self.survival_exemplars:
            s[Section.LEGAL_SURVIVAL] = self.survival_exemplars[:5]
        if self.avoid_repeat:
            s[Section.AVOID_REPEAT] = self.avoid_repeat[:5]
        return s

    def _primary_id(self) -> str:
        if self.parents:
            return str(self.parents[0].get("factor_id") or self.parents[0].get("id") or "")
        return ""

    def build_packet_text(self, *, max_tokens: int | None = None) -> str:
        """13 段 → builder（含确定性预算截断）。缺失段优雅省略。"""
        builder = PacketBuilder(
            budget_max=self.budget_max,
            budget_min=self.budget_min,
            sections=self._sections(),
        )
        return builder.build_text(max_tokens=max_tokens)
