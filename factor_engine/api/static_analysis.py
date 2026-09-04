# -*- coding: utf-8 -*-
"""R61-FI-011/012 — 因子定义静态分析清单（field usage / operator usage /
existing-treatment lineage / lookback / complexity）+ 算子语义元数据桥。

公共入口
--------
``analyze_factor_definition(dsl_or_expr, *, surface="daily") ->
FactorStaticAnalysisArtifact``

接受 canonical DSL 字符串或已解析的 ``Expr``，返回冻结清单对象。本模块是
**只读、无副作用** 的静态分析层：

- 复用 ``api.dsl_parser.parse_expr`` 做语法解析（不重写 parser）；
- 复用 ``ir.analyzer.Analyzer`` 做字段引用 / lookback / ts-cs 推导（不重写 lowerer）；
- 复用 ``identity.canonicalizer.canonicalize`` / ``get_factor_identity`` 做
  canonical DSL 与哈希（不重写身份层）；
- 算子名一律先经 ``cleaned_operators.registry.OperatorRegistry.resolve_canonical``
  归一到 canonical 再统计；
- **不 import factor_preprocess**；existing-treatment lineage 由本模块内嵌的
  FE 侧语义桥（canonical/alias → treatment lineage）派生；
- **不修改 OperatorRegistry / OperatorMetadata** —— 语义元数据作为本模块的
  可选通用映射提供，供 FO/FA 消费（stage_hint / causality_class 从 registry
  catalog 的既有 ``category`` 与 ``tags`` 派生）。

清单字段（与计划 §6 对齐）
--------------------------
``FactorStaticAnalysisArtifact``:
    factor_definition_id  : 用 identity canonical_ast_hash 派生（``F_`` + 前 11 位
                            base32，与 ``identity.hasher.derive_factor_id`` 一致）。
    canonical_dsl_hash    : 64 位 SHA-256（= FactorIdentity.canonical_ast_hash）。
    field_usages          : tuple[FieldUsage, ...]
    operator_usages       : tuple[OperatorUsage, ...]
    existing_treatment_semantic_ids : tuple[str, ...]（已存在的处理语义，
                            按算子在公式中出现的顺序保留，去重）。
    max_lookback          : int（来自 Analyzer.history_requirement / AnalysisResult）。
    complexity            : dict（nodes/depth/operators/fields/literals…）。
    timing_flags          : dict（has_time_series_op / has_cross_section_op /
                            has_group_cross_section_op / requires_full_history）。
    content_hash          : 对清单载荷稳定序列化的 SHA-256。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple, Union

from factor_engine.expr.base import Expr
from factor_engine.expr.cleaned_call import CleanedCall
from factor_engine.expr.column import ColumnRef
from factor_engine.expr.field import FieldRef
from factor_engine.expr.literal import Literal as _LiteralNode

# 允许跨进程稳定导入（本模块不 import factor_preprocess）。
_ANALYSIS_NS = "factor_static_analysis_v1"


def _sha256(data: str) -> str:
    return hashlib.sha256(
        (_ANALYSIS_NS + data).encode("utf-8"), usedforsecurity=False
    ).hexdigest()


# ---------------------------------------------------------------------------
# 公开冻结契约
# ---------------------------------------------------------------------------

from typing import Literal as _TLiteral
DirectOrDerived = _TLiteral["direct", "derived"]


@dataclass(frozen=True)
class FieldUsage:
    """公式对单个 catalog 字段的引用统计。

    ``canonical_field_id`` 为 ``FieldSpec.field_id``（如
    ``StockDailyBarAdj.close``）；同名但不同表/市场的字段不会塌缩。
    ``dataset_id`` / ``domain`` / ``role`` 来自 ``FieldSpec``。
    ``occurrence_count`` 是 AST 中出现的次数（含嵌套/重复）。
    ``direct_or_derived``：出现在原始算子面板输入位置为 derived；出现在
    显式 ``field()`` 叶子位置（最内层直接数据引用）为 direct。
    ``timing_class`` 来自字段的 temporal role（``FieldSpec.role`` /
    ``knowledge_time_column``），不可得时为 None。
    """

    canonical_field_id: str
    dataset_id: str | None = None
    table: str | None = None
    domain: str | None = None
    role: str | None = None
    occurrence_count: int = 1
    direct_or_derived: DirectOrDerived = "derived"
    timing_class: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_field_id": self.canonical_field_id,
            "dataset_id": self.dataset_id,
            "table": self.table,
            "domain": self.domain,
            "role": self.role,
            "occurrence_count": self.occurrence_count,
            "direct_or_derived": self.direct_or_derived,
            "timing_class": self.timing_class,
        }


@dataclass(frozen=True)
class OperatorUsage:
    """公式对单个 canonical 算子的引用统计。

    ``operator_id`` 为 canonical 名（别名一律先归一），``operator_version``
    取 registry catalog 的 ``semantic_version``（缺省 "1.0"）。
    ``stage_hint`` / ``causality_class`` 由 FE 侧语义桥派生（见模块级
    ``semantic_stage_hint_of`` / ``causality_class_of``）；不可得时为 None。
    ``axis_effect`` 来自 ``ir.types.axis_effect_contract_for``（如
    ``cross_section`` / ``time_series`` / ``elementwise``），是 operator 的
    ``has_ts``/``has_cs`` 权威契约。
    """

    operator_id: str
    operator_version: str = "1.0"
    occurrence_count: int = 1
    stage_hint: str | None = None
    causality_class: str | None = None
    axis_effect: str | None = None
    category: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "operator_version": self.operator_version,
            "occurrence_count": self.occurrence_count,
            "stage_hint": self.stage_hint,
            "causality_class": self.causality_class,
            "axis_effect": self.axis_effect,
            "category": self.category,
        }


#: 语义桥支持的处理 lineage 值。与 FP ``treatment_lineage`` 的语义 ID 对齐，
#: 但本模块不 import FP —— 只复用其稳定命名约定（``WINSOR:...`` /
#: ``CS_RANK:...`` / ``ZSCORE:...`` / ``SMOOTH:...`` / ``NEUTRAL:...`` /
#: ``INDUSTRY_NEUTRAL:...`` / ``SIZE_NEUTRAL:...`` / ``DUAL_NEUTRAL:...``）。
#: 本清单的 ``existing_treatment_semantic_ids`` 由这些值派生（去重、保序）。
TreatmentSemanticId = _TLiteral[
    "CS_RANK:pct",
    "ZSCORE:cs",
    "WINSOR:cs",
    "SMOOTH:ewma",
    "SMOOTH:trailing_sma",
    "SMOOTH:median",
    "NEUTRAL:cs",
    "INDUSTRY_NEUTRAL:sw_l1",
    "SIZE_NEUTRAL:log_mktcap",
    "DUAL_NEUTRAL:industry_size",
]


@dataclass(frozen=True)
class FactorStaticAnalysisArtifact:
    """因子定义的静态分析清单（R61-FI-011/012）。

    ``complexity`` 为 ``Mapping[str, int | float | str]``：
        nodes            — AST 节点总数（算子调用 + 字段 + 字面量，与 identity
                            口径一致）；
        depth            — AST 最大深度（根为 1）；
        operators        — 去重 canonical 算子数；
        fields           — 去重 canonical 字段数；
        literals         — 数字/字符串字面量总数；
        max_arity        — 单个算子调用的最大位置参数个数。

    ``timing_flags``：
        has_time_series_op / has_cross_section_op /
        has_group_cross_section_op / requires_full_history。
    """

    factor_definition_id: str
    canonical_dsl: str
    canonical_dsl_hash: str
    field_usages: Tuple[FieldUsage, ...]
    operator_usages: Tuple[OperatorUsage, ...]
    existing_treatment_semantic_ids: Tuple[str, ...]
    max_lookback: int
    complexity: Mapping[str, int | float | str]
    timing_flags: Mapping[str, bool]
    content_hash: str
    surface: str = "daily"

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_definition_id": self.factor_definition_id,
            "canonical_dsl": self.canonical_dsl,
            "canonical_dsl_hash": self.canonical_dsl_hash,
            "field_usages": [u.to_dict() for u in self.field_usages],
            "operator_usages": [u.to_dict() for u in self.operator_usages],
            "existing_treatment_semantic_ids": list(self.existing_treatment_semantic_ids),
            "max_lookback": self.max_lookback,
            "complexity": dict(self.complexity),
            "timing_flags": dict(self.timing_flags),
            "content_hash": self.content_hash,
            "surface": self.surface,
        }

    def content_payload(self) -> dict[str, Any]:
        payload = dict(self.to_dict())
        payload.pop("content_hash", None)
        return payload


# ---------------------------------------------------------------------------
# FE 侧 treatment-semantic 语义桥（R61-FI-012）
#
# 硬约束：本模块不得 import factor_preprocess；语义桥由 cleaned_operators 的
# alias 面（canonical / _aliases / registry alias）推导。stage_hint /
# causality_class 作为 *通用可选* 元数据提供给 FO/FA，不触碰 registry。
# ---------------------------------------------------------------------------

#: canonical/alias → treatment lineage semantic id（与 FP 命名约定对齐）。
#: 数值上排名后取绝对偏离的 ``abs(cs_rank(x) - c)`` 形式由
#: ``abs`` + ``subtract`` + 常量 ``c`` 组合表达，不单独占用一个 lineage。
_LINEAGE_MAP: Mapping[str, str] = {
    # Rank / representation
    "rank": "CS_RANK:pct",
    "cs_rank": "CS_RANK:pct",
    # Z-score / scaling
    "zscore": "ZSCORE:cs",
    "cs_zscore": "ZSCORE:cs",
    "cs_standardize": "ZSCORE:cs",
    "standardize": "ZSCORE:cs",
    # Winsor / clip (clamp/cap/bound → clip canonical)
    "winsorize": "WINSOR:cs",
    "clip": "WINSOR:cs",
    "cap": "WINSOR:cs",
    "bound": "WINSOR:cs",
    "clamp": "WINSOR:cs",
    "saturate": "WINSOR:cs",
    "winsorize_mean": "WINSOR:cs",
    # EWMA / EMA temporal smoothing
    "ts_ema": "SMOOTH:ewma",
    "ts_ewma": "SMOOTH:ewma",
    "ema": "SMOOTH:ewma",
    "ewma": "SMOOTH:ewma",
    "ewm_mean": "SMOOTH:ewma",
    # FP 的 trailing_sma 语义对应 FE 的 ts_mean；但 ts_mean 在因子公式中通常
    # 是特征构造（非预处理平滑），保守不自动标记为 SMOOTH 处理。仅当它作为
    # 显式平滑输入由调用方自行判断。ts_median 同理映射 SMOOTH:median。
    "trailing_sma": "SMOOTH:trailing_sma",
    "ts_median": "SMOOTH:median",
    # Neutralization
    "cs_neutralize": "NEUTRAL:cs",
    "neutralize": "NEUTRAL:cs",
    "cs_demean": "DEMEAN:cs",
    "group_neutralize": "INDUSTRY_NEUTRAL:sw_l1",
    "industry_neutralize": "INDUSTRY_NEUTRAL:sw_l1",
    "ind_neutralize": "INDUSTRY_NEUTRAL:sw_l1",
    "industry_neutral": "INDUSTRY_NEUTRAL:sw_l1",
    "size_neutralize": "SIZE_NEUTRAL:log_mktcap",
    "size_neutral": "SIZE_NEUTRAL:log_mktcap",
    "market_cap_neutralize": "SIZE_NEUTRAL:log_mktcap",
    "cap_neutralize": "SIZE_NEUTRAL:log_mktcap",
    "industry_size_neutralize": "DUAL_NEUTRAL:industry_size",
    "dual_neutral": "DUAL_NEUTRAL:industry_size",
    "cs_regression": "NEUTRAL:cs",
    "cs_resid": "NEUTRAL:cs",
}

#: stage_hint（公式构造阶段角色；与 FP TransformStage 大致对应但不 import FP）。
_STAGE_HINT_BY_SEMANTIC: Mapping[str, str] = {
    "CS_RANK:pct": "representation",
    "ZSCORE:cs": "representation",
    "DEMEAN:cs": "representation",
    "SCALE:cs": "representation",
    "WINSOR:cs": "outlier",
    "SMOOTH:ewma": "temporal",
    "SMOOTH:trailing_sma": "temporal",
    "SMOOTH:median": "temporal",
    "NEUTRAL:cs": "neutralization",
    "INDUSTRY_NEUTRAL:sw_l1": "neutralization",
    "SIZE_NEUTRAL:log_mktcap": "neutralization",
    "DUAL_NEUTRAL:industry_size": "neutralization",
    "FILL:forward": "missingness",
    "FILL:freshness_aware": "missingness",
    "EVENT_DECAY:short_halflife": "temporal",
}

#: causality_class —— 该算子对面板的因果/时序立场。
#: causal/rolling 时序算子（只用过去/当前窗口）→ "causal"；
#: 截面（按日横截面）→ "cross_sectional"；纯元素级/代数 → "elementwise"。
_CAUSALITY_CLASS_CATEGORIES_TS = frozenset({
    "time_series", "technical_signal", "price_volume", "price_volume_extension",
    "ohlc_volatility", "candle_pattern", "intraday_microstructure", "signal",
    "price_structure", "chart_pattern", "fundamental_period",
})


def resolve_canonical_operator(name: str) -> str:
    """把 DSL 名（canonical 或 alias）解析为 canonical 算子名。

    未知名抛 KeyError（fail-closed；与 registry resolve_canonical_strict 同语义）。
    """
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    return OperatorRegistry.resolve_canonical_strict(name)


def semantic_stage_hint_of(name: str) -> str | None:
    """算子 → stage_hint（通用可选元数据，缺省 None）。"""
    canonical = _soft_canonical(name)
    semantic = _LINEAGE_MAP.get(canonical) or _LINEAGE_MAP.get(name)
    if semantic is None:
        return None
    return _STAGE_HINT_BY_SEMANTIC.get(semantic)


def _soft_canonical(name: str) -> str:
    """别名 → canonical 的容错解析：未知/未实现算子原样返回（不抛错）。"""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    try:
        return OperatorRegistry.resolve_canonical(name)
    except Exception:
        return str(name)


def causality_class_of(name: str) -> str | None:
    """算子 → causality_class（通用可选元数据，缺省 None）。

    截面/分组中性化与 rank/zscore/scale/normalize/winsorize 视为
    ``cross_sectional``；ts_ 前缀或时序 category 视为 ``causal``；
    其余（元素级代数/数学）视为 ``elementwise``。
    """
    canonical = _soft_canonical(name)
    if canonical in _LINEAGE_MAP:
        # 语义桥里的截面处理算子优先归 cross_sectional
        if _STAGE_HINT_BY_SEMANTIC.get(_LINEAGE_MAP[canonical]) == "neutralization":
            return "cross_sectional"
        if canonical in {"rank", "zscore", "cs_demean", "cs_quantile", "scale", "normalize",
                          "winsorize", "group_rank", "group_zscore", "group_winsorize"}:
            return "cross_sectional"
    if str(canonical).startswith(("ts_", "expanding_")):
        return "causal"
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    op = OperatorRegistry.get(canonical)
    category = str(getattr(getattr(op, "metadata", None), "category", "") or "")
    if category in _CAUSALITY_CLASS_CATEGORIES_TS:
        return "causal"
    # group_* / cs_* 前缀的截面算子（如 group_rank / group_zscore）
    if str(canonical).startswith(("group_", "cs_")) and canonical in {
        "cs_rank", "cs_zscore", "cs_demean", "cs_quantile", "cs_standardize",
        "cs_pct_rank", "cs_scale", "cs_fill_mean", "cs_neutralize",
        "group_rank", "group_zscore", "group_winsorize", "group_demean",
        "group_neutralize", "group_normalize", "group_std", "group_mean",
    }:
        return "cross_sectional"
    if category in {"cross_sectional", "cross_sectional_regression", "group_neutralization"}:
        return "cross_sectional"
    return "elementwise"


def existing_treatment_semantic_id_of(name: str) -> str | None:
    """解析 DSL 名对应的 treatment lineage semantic id（未知返回 None）。

    别名先经 registry 归一；本函数对未知名 **不抛错** —— 便于调用方判断
    “无既有处理”。
    """
    canonical = _soft_canonical(name)
    return _LINEAGE_MAP.get(canonical) or _LINEAGE_MAP.get(name)


# ---------------------------------------------------------------------------
# 实现
# ---------------------------------------------------------------------------


def _coerce_expr(dsl_or_expr: str | Expr, *, surface: str) -> Expr:
    """接受 DSL 字符串或已解析 Expr（含 canonical_dsl JSON 文本）。"""
    from factor_engine.expr.base import Expr as _Expr

    if isinstance(dsl_or_expr, _Expr):
        return dsl_or_expr
    from factor_engine.api.dsl_parser import parse_expr

    return parse_expr(str(dsl_or_expr), surface=surface)


def _iter_nodes(node: Expr):
    """深度优先遍历 Expr 树（含当前节点）。"""
    yield node
    if isinstance(node, CleanedCall):
        for child in node.args:
            yield from _iter_nodes(child)


def _canonical_dsl_of(node: Expr) -> str:
    """把一个 Expr（可能缺少 catalog_hash）重写成 DSL 文本。

    仅用于 Analyzer 前修复 catalog 绑定的回退；普通 parse 路径不经过这里。
    """
    from factor_engine.identity.serializer import canonical_ast_text

    return canonical_ast_text(node)


def _rebind_field_catalog_hashes(node: Expr, *, surface: str) -> Expr:
    """把缺少 catalog_hash 的 FieldRef 叶子重新绑定到当前 A-share catalog。

    用于接受 canonical_dsl JSON / payload 反序列化 Expr 的幂等兼容输入：
    JSON 载荷不含 catalog_hash，但 Analyzer 校验叶子 hash 必须匹配活动
    catalog。这里逐叶子用 ``fields`` 解析器补回 field_id/catalog_hash，
    不改变表达式结构。返回一棵等价但叶子已绑定 catalog 的 Expr。
    """
    from factor_engine.expr.field import FieldRef as _FR
    from factor_engine.fields.market_registry import MULTI_MARKET_FIELD_REGISTRY
    from factor_engine.market.context import ASHARE_CONTEXT

    def rebuild(n: Expr) -> Expr:
        if isinstance(n, CleanedCall):
            return CleanedCall(
                op=n.op,
                args=tuple(rebuild(a) for a in n.args),
                kwargs=n.kwargs,
            )
        if isinstance(n, _FR):
            if n.catalog_hash:
                return n
            from factor_engine.fields.resolver import resolve_market_field

            resolved = None
            # 镜像 api/columns.field 的裸名解析：table=None 的严格解析先失败时，
            # 行情 anchor 名回退到后复权权威表 StockDailyBarAdj。
            if n.table:
                resolved = resolve_market_field(
                    n.canonical_name or n.name, ASHARE_CONTEXT,
                    table=n.table, strict=False,
                )
            if resolved is None or resolved.spec is None:
                resolved = resolve_market_field(
                    n.canonical_name or n.name, ASHARE_CONTEXT, strict=False
                )
            if resolved is None or resolved.spec is None:
                resolved = resolve_market_field(
                    n.canonical_name or n.name, ASHARE_CONTEXT,
                    table="StockDailyBarAdj", strict=False,
                )
            if resolved is not None and resolved.spec is not None:
                spec = resolved.spec
                registry = MULTI_MARKET_FIELD_REGISTRY.registry_for("ashare")
                return _FR(
                    name=n.name,
                    field_id=str(spec.field_id),
                    canonical_name=str(spec.name),
                    table=str(spec.table),
                    source_name=str(spec.source_name),
                    catalog_hash=registry.catalog_hash(),
                )
            # 无法绑定（未知字段）保留原样，Analyzer 会照常 fail-closed
            return n
        return n

    return rebuild(node)


def _field_id_of(ref: FieldRef | ColumnRef) -> str | None:
    """返回字段 canonical id；纯 ColumnRef（无 catalog 绑定）时回退名字。"""
    if isinstance(ref, FieldRef):
        return str(ref.field_id) if ref.field_id else ref.canonical_name or ref.name
    if isinstance(ref, ColumnRef):
        return str(ref.name)
    return None


def _timing_class_of(field_id: str, spec: Any) -> str | None:
    """从 FieldSpec 解析字段的 temporal role / timing class。"""
    if spec is None:
        return None
    role = str(getattr(spec, "role", "") or "")
    if role == "time":
        return "time"
    if role in {"group_key", "identifier", "instrument"}:
        return "panel"
    if role in {"knowledge_time", "effective_time", "period_id", "ingestion_time"}:
        return "pit_metadata"
    available = getattr(spec, "available_at", None)
    if available:
        return str(available)
    return None


#: 直接数据引用 = 叶子位置裸 ``close`` / 显式 ``field(...)``（非算子入参位置）。
#: 更细：若某算子显式声明 ``panel_params`` / ``input_fields``，其对应位置视为
#: derived（算子的面板输入）；否则叶子位置一律 direct。静态层从简：叶子位置
#: (parent 为算子的面板入参) 计 derived，其余（公式根叶子 / 未包裹调用）计
#: direct。字段出现在 ts/cs 算子内部时在 AST 里就是叶子，计入 derived 更接近
#: “被处理过”。为保守且可测，默认按 “出现在任何算子调用内部 = derived”。
#: 实现：某叶子字段如果其父节点是 CleanedCall（即确实被算子处理）则 derived，
#: 否则 direct。
def _collect_field_usages(
    root: Expr, specs: Mapping[str, Any]
) -> tuple[tuple[FieldUsage, ...], dict[str, int]]:
    """收集字段引用统计（field_id 键控，occurrence 累加）。"""
    counts: dict[str, int] = {}
    fields_seen: dict[str, Any] = {}
    for node in _iter_nodes(root):
        if isinstance(node, (FieldRef, ColumnRef)):
            fid = _field_id_of(node)
            if fid is None:
                continue
            counts[fid] = counts.get(fid, 0) + 1
            fields_seen.setdefault(fid, node)
    # 判定 direct / derived：字段若被任意算子包裹 → derived；否则 direct。
    derived_ids: set[str] = set()
    for node in _iter_nodes(root):
        if isinstance(node, CleanedCall):
            for arg in node.args:
                if isinstance(arg, (FieldRef, ColumnRef)):
                    derived_ids.add(_field_id_of(arg) or "")
    usages: list[FieldUsage] = []
    for fid in sorted(counts):
        node = fields_seen[fid]
        spec = specs.get(fid) or specs.get(getattr(node, "canonical_name", None))
        usages.append(
            FieldUsage(
                canonical_field_id=fid,
                dataset_id=str(getattr(spec, "dataset", "") or "") or None,
                table=str(getattr(node, "table", "") or "") or None,
                domain=str(getattr(spec, "domain", "") or "") or None,
                role=str(getattr(spec, "role", "") or "") or None,
                occurrence_count=counts[fid],
                direct_or_derived="derived" if fid in derived_ids else "direct",
                timing_class=_timing_class_of(fid, spec),
            )
        )
    return tuple(usages), counts


def _operator_canonical_counts(root: Expr) -> tuple[dict[str, int], list[str]]:
    """收集 canonical 算子出现次数 + 按出现顺序的 canonical 列表。"""
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    counts: dict[str, int] = {}
    ordered: list[str] = []
    for node in _iter_nodes(root):
        if not isinstance(node, CleanedCall):
            continue
        try:
            canonical = OperatorRegistry.resolve_canonical(node.op)
        except Exception:
            canonical = node.op
        counts[canonical] = counts.get(canonical, 0) + 1
        ordered.append(canonical)
    return counts, ordered


def _max_arity(root: Expr) -> int:
    return max(
        (len(node.args) for node in _iter_nodes(root) if isinstance(node, CleanedCall)),
        default=0,
    )


def _iter_ir_ops(ir: Any):
    """遍历 IR 树算子（AnalysisResult.ir）。"""
    if ir is None:
        return
    yield ir
    for child in getattr(ir, "inputs", ()) or ():
        yield from _iter_ir_ops(child)


def _precise_timing_flags(ir: Any, base: dict[str, bool]) -> dict[str, bool]:
    """按 AxisEffectContract 精确派生 group_cross_section 标志。"""
    from factor_engine.ir.types import AxisEffectKind, axis_effect_contract_for

    has_group = False
    for node in _iter_ir_ops(ir):
        contract = axis_effect_contract_for(str(node.op))
        if contract is not None and contract.kind is AxisEffectKind.GROUP_CROSS_SECTION:
            has_group = True
    flags = dict(base)
    flags["has_group_cross_section_op"] = has_group
    return flags


def _content_hash(artifact: FactorStaticAnalysisArtifact) -> str:
    payload = json.dumps(
        artifact.content_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return _sha256(payload)


def analyze_factor_definition(
    dsl_or_expr: str | Expr,
    *,
    surface: str = "daily",
    include_canonical_dsl: bool = True,
) -> FactorStaticAnalysisArtifact:
    """分析一个因子定义并返回静态清单（R61-FI-011/012 公共入口）。

    Parameters
    ----------
    dsl_or_expr : str | Expr
        FE DSL 公式文本（如 ``rank(ts_mean(close, 5))``）或已解析的 ``Expr``。
        传入以 ``{`` 开头的 canonical_dsl JSON 文本亦被接受（幂等兼容）。
    surface : str
        解析 surface（默认 ``daily``）。
    include_canonical_dsl : bool
        是否在清单中保留 canonical DSL JSON 文本（默认 True；content_hash
        不受影响，永远基于载荷）。

    Returns
    -------
    FactorStaticAnalysisArtifact
        冻结静态分析清单。

    Raises
    ------
    DSLParseError / FactorIdentityError / KeyError
        语法无效或引用了未注册算子时抛错（fail-closed）。

    Notes
    -----
    - 本函数 **不物化数据、不执行算子** —— 纯 parse/analyze 层。
    - ``field_usages`` 以 ``FieldSpec.field_id``（registry 权威）键控。
    - ``max_lookback`` 取 ``Analyzer.history_requirement`` 的行数（P0-03 单一权威）；
      不可行时回退到 ``AnalysisResult.lookback``。
    """
    from factor_engine.identity.canonicalizer import canonicalize
    from factor_engine.identity.hasher import derive_factor_id

    expr = _coerce_expr(dsl_or_expr, surface=surface)

    # 1. canonical AST + canonical DSL
    canonical_ast = canonicalize(expr)

    # 1b. Analyzer 需要 FieldRef 带 catalog_hash；canonicalize 不会剥离该属性
    #     （parse_expr 产生的叶子带 hash）。若入参是 _payload_to_expr 之类
    #     丢失 hash 的 Expr，则重建带 catalog 绑定的叶子：最内层 field 叶子
    #     无法从 canonical JSON 还原 catalog_hash，因此改为按 field_id/name
    #     重新 resolve。实现上用一条受控文本路径：把 canonical JSON 反序列化
    #     的 FieldRef 逐个经 fields.resolve_market_field 绑定回 catalog。
    _needs_hash_backfill = False
    for _node in _iter_nodes(canonical_ast):
        if isinstance(_node, FieldRef) and not _node.catalog_hash:
            _needs_hash_backfill = True
            break
    if _needs_hash_backfill and isinstance(dsl_or_expr, Expr):
        canonical_ast = _rebind_field_catalog_hashes(canonical_ast, surface=surface)

    from factor_engine.identity.serializer import canonical_ast_text

    canonical_dsl = canonical_ast_text(canonical_ast)

    # 2. analyzer（字段引用 / lookback / ts-cs / full-history）
    from factor_engine.ir.analyzer import Analyzer

    analysis = Analyzer().lower(canonical_ast)

    # 3. identity canonical hash（复用 canonical_ast_hash；与 identity 完全一致）
    from factor_engine.identity import get_factor_identity

    ident = get_factor_identity(canonical_dsl)
    canonical_ast_hash = ident.canonical_ast_hash
    factor_definition_id = derive_factor_id(canonical_ast_hash)

    # 4. field usages（以 FieldSpec 元数据增强）
    specs: dict[str, Any] = {}
    for raw_name, spec in (analysis.referenced_fields or {}).items():
        specs[str(getattr(spec, "field_id", "") or raw_name)] = spec
        specs.setdefault(str(getattr(spec, "name", "") or raw_name), spec)
    field_usages, field_counts = _collect_field_usages(canonical_ast, specs)

    # 5. operator usages（canonical 归一 + 语义桥）
    operator_counts, ordered_canonicals = _operator_canonical_counts(canonical_ast)
    from factor_engine.cleaned_operators.registry import OperatorRegistry
    from factor_engine.ir.types import axis_effect_contract_for

    operator_usages: list[OperatorUsage] = []
    for canonical in sorted(operator_counts):
        entry = OperatorRegistry.catalog().get(canonical, {})
        axis = axis_effect_contract_for(canonical)
        operator_usages.append(
            OperatorUsage(
                operator_id=canonical,
                operator_version=str(entry.get("semantic_version") or "1.0"),
                occurrence_count=operator_counts[canonical],
                stage_hint=semantic_stage_hint_of(canonical),
                causality_class=causality_class_of(canonical),
                axis_effect=axis.kind.value if axis is not None else None,
                category=str(entry.get("category") or "") or None,
            )
        )

    # 6. existing-treatment lineage（按 AST 出现顺序，去重）
    lineage_seen: list[str] = []
    for canonical in ordered_canonicals:
        sid = _LINEAGE_MAP.get(canonical)
        if sid is not None and sid not in lineage_seen:
            lineage_seen.append(sid)

    # 7. lookback（P0-03 单一权威 = history_requirement.rows）
    history_req = getattr(analysis, "history_requirement", None)
    if history_req is not None and not bool(getattr(history_req, "is_full_history", False)):
        max_lookback = max(0, int(getattr(history_req, "rows", 0) or 0))
    else:
        max_lookback = max(0, int(getattr(analysis, "lookback", 0) or 0))

    # 8. complexity
    complexity: dict[str, int | float | str] = {
        "nodes": ident.complexity,
        "depth": ident.depth,
        "operators": len(operator_counts),
        "fields": len(field_counts),
        "literals": sum(
            1 for node in _iter_nodes(canonical_ast) if isinstance(node, _LiteralNode)
        ),
        "max_arity": _max_arity(canonical_ast),
    }

    # 9. timing flags
    base_flags = {
        "has_time_series_op": bool(getattr(analysis, "has_ts_op", False)),
        "has_cross_section_op": bool(getattr(analysis, "has_cs_op", False)),
        "has_group_cross_section_op": False,
        "requires_full_history": bool(getattr(analysis, "requires_full_history", False)),
    }
    timing_flags = _precise_timing_flags(getattr(analysis, "ir", None), base_flags)

    artifact = FactorStaticAnalysisArtifact(
        factor_definition_id=factor_definition_id,
        canonical_dsl=canonical_dsl if include_canonical_dsl else "",
        canonical_dsl_hash=canonical_ast_hash,
        field_usages=tuple(field_usages),
        operator_usages=tuple(operator_usages),
        existing_treatment_semantic_ids=tuple(lineage_seen),
        max_lookback=max_lookback,
        complexity=complexity,
        timing_flags=timing_flags,
        content_hash="",  # 占位：下两行用内容哈希回填
        surface=surface,
    )
    content_hash = _content_hash(artifact)
    return FactorStaticAnalysisArtifact(
        factor_definition_id=artifact.factor_definition_id,
        canonical_dsl=artifact.canonical_dsl,
        canonical_dsl_hash=artifact.canonical_dsl_hash,
        field_usages=artifact.field_usages,
        operator_usages=artifact.operator_usages,
        existing_treatment_semantic_ids=artifact.existing_treatment_semantic_ids,
        max_lookback=artifact.max_lookback,
        complexity=artifact.complexity,
        timing_flags=artifact.timing_flags,
        content_hash=content_hash,
        surface=artifact.surface,
    )


__all__ = [
    "DirectOrDerived",
    "FactorStaticAnalysisArtifact",
    "FieldUsage",
    "OperatorUsage",
    "TreatmentSemanticId",
    "analyze_factor_definition",
    "causality_class_of",
    "existing_treatment_semantic_id_of",
    "resolve_canonical_operator",
    "semantic_stage_hint_of",
]
