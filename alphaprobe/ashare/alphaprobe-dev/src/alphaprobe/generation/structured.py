"""结构化 generation（任务书 §27 / §37 / R61 Task7-A9）。

LLM 输出结构化 action JSON（如 ``{"action": "transform", "op": "zscore",
"target": "parent_0", "params": {...}}`` 或 ``{"action": "combine",
"parents": [...], "weights": [...]}``），经 action schema 校验 → **FE typed AST
transform**（``factor_engine.api.ast_transform``）把 parent 因子 AST 按 action
变换成新 Expr → serialize 回 DSL 文本 → FE validate/identity。

设计约束（R61 Task7/A9, Part E 边界）：
- **算子语义一律来自 FE registry**：arity / 参数角色 / window role / surface /
  输入类型（condition 必须 bool）全部由 ``factor_engine.api.ast_transform``
  读取；本模块**不再维护**任何本地 op 分类表（``_TRANSFORM_UNARY_OPS`` /
  ``_WINDOW_OPS`` / ``_BINARY_OPS`` / ``_CONDITIONAL_OPS`` 已删除）。
- 生产路径**不经过字符串 ``_wrap()``**：action 意图由 FE typed transform 执行
  （带类型校验 + fail-closed）。仅 ``OFFLINE_TEST``（显式 ``_LEGACY_STRING_PATH
  = True``）保留最小字符串兼容面，绝不作为生产 fallback。
- 最终产物仍是 FE 可 parse 的 DSL 文本 + FE identity 可计算（reparse →
  canonical_ast_hash / signal_equivalence_id 一致是硬验收）。
- 向后兼容：``normalize_llm_output`` 仍把旧文本输出（§31 candidates JSON）映射
  为 identity action（该路径只回传原公式，不重建算子语义）。
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# legacy 字符串路径开关（OFFLINE_TEST-only；生产默认 False）
# ---------------------------------------------------------------------------

#: 显式 OFFLINE_TEST-only：True 时才允许字符串拼接 legacy 路径（仅测试/离线），
#: 绝不作为生产 fallback（R61 Task7/A9：生产 AST 变异走 FE typed transform）。
_LEGACY_STRING_PATH = False


def _ast_transform() -> Any:
    """加载 FE typed AST transform 公共 API（懒加载）。

    Raises
    ------
    RuntimeError
        FE transform API 不可用（fail-closed：生产路径拒绝而不是字符串降级）。
    """
    from alphaprobe.fe_bridge.paths import ensure_factor_engine_importable

    ensure_factor_engine_importable()
    import factor_engine.api as fe_api

    return fe_api


# ---------------------------------------------------------------------------
# FE operator surface（运行时校验白名单）
# ---------------------------------------------------------------------------

#: 内置保守白名单：仅 OFFLINE_TEST（FE 不可 import 且 _LEGACY_STRING_PATH=True）
#: 时才使用；生产一律以 FE registry 表面为准。
_BUILTIN_OP_WHITELIST: frozenset[str] = frozenset(
    {
        "zscore", "rank", "add", "multiply", "subtract", "divide", "neg", "abs",
        "log", "sqrt", "sign", "ts_mean", "ts_std", "ts_rank", "ts_sum",
        "ts_max", "ts_min", "ts_median", "ts_delta", "ts_corr", "ts_cov",
        "ts_pct", "ts_skew", "ts_kurt", "ts_quantile", "ts_product", "ts_var",
        "ts_decay_linear", "ts_regression_slope", "ts_time_slope", "ts_topk_sum",
        "cs_rank", "cs_zscore", "cs_demean", "scale", "power", "clip",
        "winsorize", "delay", "delta", "ema", "beta", "rolling_beta",
        "percentile", "corr", "cov", "rankcorr", "rank_corr", "if_else",
        "where", "iif", "coalesce", "fillna", "gt", "lt", "ge", "le", "eq",
        "ne", "and_", "or_", "not_", "is_nan", "is_infinite", "is_null",
        "is_not_null", "cap_neutralize", "industry_neutralize", "ind_neutralize",
        "neutralize", "market_cap_neutralize", "size_neutralize",
    }
)

#: 默认 action 允许的 op 白名单（保守动作面；运行时再与 FE registry 表面求交）。
DEFAULT_ACTION_OP_WHITELIST: frozenset[str] = frozenset(
    {
        "zscore", "rank", "add", "multiply", "subtract", "divide", "neg", "abs",
        "log", "sqrt", "sign", "ts_mean", "ts_std", "ts_rank", "ts_sum",
        "ts_max", "ts_min", "ts_median", "ts_delta", "ts_corr", "ts_cov",
        "ts_pct", "ts_skew", "ts_kurt", "ts_quantile", "ts_product", "ts_var",
        "ts_decay_linear", "ts_regression_slope", "ts_time_slope", "ts_topk_sum",
        "cs_rank", "cs_zscore", "cs_demean", "scale", "power", "clip",
        "winsorize", "delay", "delta", "ema", "beta", "rolling_beta",
        "percentile", "corr", "cov", "rankcorr", "rank_corr", "if_else",
        "where", "iif", "coalesce", "fillna", "gt", "lt", "ge", "le", "eq",
        "ne", "and_", "or_", "not_", "is_nan", "is_infinite", "is_null",
        "is_not_null", "cap_neutralize", "industry_neutralize", "ind_neutralize",
        "neutralize", "market_cap_neutralize", "size_neutralize",
    }
)

_FE_SURFACE_CACHE: frozenset[str] | None = None
_FE_SURFACE_TRIED = False


def fe_operator_surface(*, surface: str = "daily") -> frozenset[str]:
    """FE 已注册算子表面（daily allowlist 的 key 集合）。

    FE 可 import 时从 ``factor_engine.api.operator_registry.build_dsl_allowlist``
    读取（只读消费，不修改 FE）；不可 import 时退化为内置保守白名单（仅
    OFFLINE_TEST；生产路径由调用方决定 fail-closed）。
    结果缓存（首次调用约 10-35s 加载 cleaned_operators 注册表）。
    """
    global _FE_SURFACE_CACHE, _FE_SURFACE_TRIED
    if _FE_SURFACE_CACHE is not None:
        return _FE_SURFACE_CACHE
    if _FE_SURFACE_TRIED:
        return _BUILTIN_OP_WHITELIST
    _FE_SURFACE_TRIED = True
    try:
        from alphaprobe.fe_bridge.paths import ensure_factor_engine_importable

        ensure_factor_engine_importable()
        from factor_engine.api.operator_registry import build_dsl_allowlist

        allow = build_dsl_allowlist(surface=str(surface or "daily"))
        _FE_SURFACE_CACHE = frozenset(str(k) for k in allow.keys())
        return _FE_SURFACE_CACHE
    except Exception as exc:  # noqa: BLE001 - FE 不可用退化为内置保守白名单
        logger.warning("fe_operator_surface unavailable, using builtin whitelist: %s", exc)
        _FE_SURFACE_CACHE = _BUILTIN_OP_WHITELIST
        return _FE_SURFACE_CACHE


# ---------------------------------------------------------------------------
# action schema
# ---------------------------------------------------------------------------

#: 支持的 action 类型
ACTION_TRANSFORM = "transform"
ACTION_COMBINE = "combine"
ACTION_IDENTITY = "identity"


@dataclass
class ActionValidation:
    """action 校验结果。``ok`` 为 False 时 ``reason`` 记录拒绝原因。"""

    ok: bool
    reason: str = ""
    action_type: str = ""
    op: str = ""


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(v) for v in value]
    return []


def _as_float_list(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, Sequence):
        out: list[float] = []
        for v in value:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                continue
        return out
    return []


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


#: 条件算子（action 侧白名单；FE 侧 `where/if_else` 的 condition 槽由 registry
#: 签名校验必须 bool——此处只做轻量 action 级检查，语义校验在 FE transform）。
_CONDITIONAL_OPS: frozenset[str] = frozenset(
    {"if_else", "where", "iif", "coalesce", "fillna"}
)


def validate_action(
    action: Mapping[str, Any],
    *,
    op_whitelist: frozenset[str] | None = None,
    surface: str = "daily",
) -> ActionValidation:
    """校验 action JSON 是否符合 schema。

    - ``action`` 必须是 dict，含 ``action`` 字段（transform/combine/identity）；
    - ``op`` 必须在 op 白名单内（默认 ``DEFAULT_ACTION_OP_WHITELIST`` ∩ FE 表面）；
    - 参数按 action 类型做**轻量**检查（combine ≥2 parents 且 weights 长度匹配；
      conditional op 需 cond；window 需正整数）；**类型级语义校验**（condition
      必须 bool、window 只改声明槽等）统一在 FE typed transform 执行。
    """
    if not isinstance(action, Mapping):
        return ActionValidation(False, "action must be a JSON object")
    a = dict(action)
    action_type = str(a.get("action") or "").strip().lower()
    if action_type == ACTION_IDENTITY:
        return ActionValidation(True, action_type=action_type)
    if action_type not in (ACTION_TRANSFORM, ACTION_COMBINE):
        return ActionValidation(
            False, f"unsupported action type: {action_type!r}", action_type=action_type
        )

    op = str(a.get("op") or "").strip()
    if not op:
        return ActionValidation(False, "missing op", action_type=action_type)
    # op 白名单：默认集 ∩ FE 表面（运行时校验，禁止发明 FE 没有的算子语义）
    if op_whitelist is None:
        surface_set = fe_operator_surface(surface=surface)
        op_whitelist = DEFAULT_ACTION_OP_WHITELIST & surface_set
    if op not in op_whitelist:
        return ActionValidation(
            False,
            f"op not in whitelist: {op!r}",
            action_type=action_type,
            op=op,
        )

    if action_type == ACTION_COMBINE:
        parents = _as_str_list(a.get("parents"))
        if len(parents) < 2:
            return ActionValidation(
                False, "combine requires >=2 parents", action_type=action_type, op=op
            )
        weights = _as_float_list(a.get("weights"))
        if weights and len(weights) != len(parents):
            return ActionValidation(
                False,
                f"weights length {len(weights)} != parents length {len(parents)}",
                action_type=action_type,
                op=op,
            )
        return ActionValidation(True, action_type=action_type, op=op)

    # transform（轻量参数检查）
    if op in _CONDITIONAL_OPS:
        cond = a.get("cond")
        if cond is None:
            return ActionValidation(
                False, f"conditional op {op!r} requires cond", action_type=action_type, op=op
            )
        return ActionValidation(True, action_type=action_type, op=op)
    params = a.get("params")
    window = (
        _as_int(params.get("window"))
        if isinstance(params, Mapping) and "window" in params
        else _as_int(a.get("window"))
    )
    if window is not None and window <= 0:
        return ActionValidation(
            False, f"op {op!r} requires positive window", action_type=action_type, op=op
        )
    second = a.get("second")
    if second is None and _needs_second_operand(op):
        return ActionValidation(
            False, f"binary op {op!r} requires second operand", action_type=action_type, op=op
        )
    return ActionValidation(True, action_type=action_type, op=op)


#: 需要 second 操作数的算子（二元比较/组合对；语义权威仍在 FE registry——
#: 这里只做 action JSON 缺失参数提示，不做 arity 决策）。
_BINARY_SECOND_OPS: frozenset[str] = frozenset(
    {"gt", "lt", "ge", "le", "eq", "ne", "and_", "or_", "subtract", "divide",
     "ts_corr", "ts_cov", "corr", "cov", "beta", "rolling_beta"}
)


def _needs_second_operand(op: str) -> bool:
    return op in _BINARY_SECOND_OPS


# ---------------------------------------------------------------------------
# AST Transform 构造器：action → 新公式文本（FE typed transform 权威）
# ---------------------------------------------------------------------------


def _parent_formula(parent: Mapping[str, Any]) -> str:
    return str(parent.get("formula") or parent.get("canonical_formula") or "").strip()


def _parent_id(parent: Mapping[str, Any]) -> str:
    return str(parent.get("factor_id") or parent.get("id") or "")


def build_formula(
    action: Mapping[str, Any],
    parents: Sequence[Mapping[str, Any]],
    *,
    op_whitelist: frozenset[str] | None = None,
    surface: str = "daily",
) -> tuple[str | None, str]:
    """把 action 应用到 parents，构造新公式文本（FE typed AST transform）。

    Returns (formula, reason)。成功时 reason=""；失败时 formula=None 且 reason
    记录原因（非白名单 op / 参数缺失 / parent 缺失 / FE 类型校验失败等）。
    """
    if not isinstance(action, Mapping):
        return None, "action must be a JSON object"
    a = dict(action)
    action_type = str(a.get("action") or "").strip().lower()
    if action_type == ACTION_IDENTITY:
        if not parents:
            return None, "identity action requires a parent"
        # 旧文本兼容：identity action 可携带原公式（normalize_llm_output 从 §31
        # candidates 提取）；否则用第一个 parent 的公式。该路径只回传原公式，
        # 不重建算子语义。
        formula = str(a.get("formula") or "").strip()
        if formula:
            return formula, ""
        return _parent_formula(parents[0]), ""

    v = validate_action(a, op_whitelist=op_whitelist, surface=surface)
    if not v.ok:
        return None, v.reason

    try:
        fe_api = _ast_transform()
    except Exception as exc:  # noqa: BLE001
        if _LEGACY_STRING_PATH:
            # OFFLINE_TEST-only legacy 路径：显式开关才允许字符串拼接
            logger.warning("legacy string path (OFFLINE_TEST-only): %s", exc)
            return _build_formula_legacy(a, parents, v.op, surface=surface)
        return None, f"FE AST transform unavailable (no production fallback): {exc}"

    if action_type == ACTION_COMBINE:
        return _build_combine_ast(fe_api, a, parents, v.op)
    return _build_transform_ast(fe_api, a, parents, v.op)


def _resolve_parent_ref(
    ref: str, parents: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    by_id = {_parent_id(p): p for p in parents}
    by_idx: dict[str, Mapping[str, Any]] = {}
    for i, p in enumerate(parents):
        by_idx[str(i)] = p
        by_idx[f"parent_{i}"] = p
    return by_id.get(ref) or by_idx.get(ref)


# ---------------------------------------------------------------------------
# FE typed AST 构造路径（生产）
# ---------------------------------------------------------------------------


def _build_combine_ast(
    fe_api: Any,
    a: Mapping[str, Any],
    parents: Sequence[Mapping[str, Any]],
    op: str,
) -> tuple[str | None, str]:
    parent_refs = _as_str_list(a.get("parents"))
    if len(parent_refs) < 2:
        return None, "combine requires >=2 parents"
    chosen: list[Mapping[str, Any]] = []
    for ref in parent_refs:
        p = _resolve_parent_ref(ref, parents)
        if p is None:
            return None, f"combine parent ref not found: {ref!r}"
        chosen.append(p)
    formulas = [_parent_formula(p) for p in chosen]
    if any(not f for f in formulas):
        return None, "combine parent missing formula"
    weights = _as_float_list(a.get("weights"))
    try:
        if weights and len(weights) == len(formulas):
            # 加权和：add(multiply(w0, f0), multiply(w1, f1), ...)
            terms = []
            for w, f in zip(weights, formulas):
                if op not in {"add", "multiply"}:
                    # 加权和只对可折叠标量组合算子成立；其他 combine 算子
                    # 不接受 weights（保守拒绝）
                    return (
                        None,
                        f"combine op {op!r} does not accept weights (only add/multiply)",
                    )
                # 权重是标量：作为 Literal 传入 ast_call（不能把 "0.5" 当公式 parse）
                from factor_engine.expr.literal import Literal

                mul = fe_api.ast_call("multiply", (Literal(float(w)), f), {})
                terms.append(mul)
            node = fe_api.ast_call("add", tuple(terms), {})
            return fe_api.ast_to_dsl_text(node), ""
        # 无 weights → 等权 op（按算子声明的 panel 数量校验）
        node = fe_api.ast_call(op, tuple(formulas), {})
        return fe_api.ast_to_dsl_text(node), ""
    except Exception as exc:  # noqa: BLE001 - FE transform 校验失败即拒绝
        return None, f"FE transform rejected combine: {exc}"


def _build_transform_ast(
    fe_api: Any,
    a: Mapping[str, Any],
    parents: Sequence[Mapping[str, Any]],
    op: str,
) -> tuple[str | None, str]:
    target = str(a.get("target") or "parent_0")
    parent = _resolve_parent_ref(target, parents)
    if parent is None:
        return None, f"transform target not found: {target!r}"
    f = _parent_formula(parent)
    if not f:
        return None, "transform parent missing formula"

    params = a.get("params")
    params = dict(params) if isinstance(params, Mapping) else {}
    window = (
        params.get("window")
        if "window" in params
        else a.get("window")
    )
    second = a.get("second")
    cond = a.get("cond")
    otherwise = a.get("otherwise", "0.0")

    try:
        semantics = fe_api.ast_resolve_op_semantics(op)
        canon = semantics.canonical
        window_names = tuple(semantics.window_names)
        # WINDOW op：带 window param → 只改声明的 WINDOW 槽
        if window_names and window is not None:
            w = int(window)
            params_out = dict(params)
            params_out[window_names[0]] = w
            node = fe_api.ast_call(canon, (f,), params_out)
            return fe_api.ast_to_dsl_text(node), ""
        # 双 panel 对算子：second 是额外 panel
        pair_panels = len(tuple(semantics.panel_positions)) >= 2
        if second is not None and pair_panels:
            node = fe_api.ast_call(canon, (f, str(second)), {w_k: w for w_k, w in params.items() if w_k in window_names})
            return fe_api.ast_to_dsl_text(node), ""
        # 条件算子：cond 作为第一个面板输入（FE 签名要求 bool）
        if op in _CONDITIONAL_OPS:
            if cond is None:
                return None, f"conditional op {op!r} requires cond"
            node = fe_api.ast_call(canon, (str(cond), f, str(otherwise)), {})
            return fe_api.ast_to_dsl_text(node), ""
        # 其余：一元/普通二元（second 可选作面板；无 second 则单面板）
        if second is not None:
            node = fe_api.ast_call(canon, (f, str(second)), params)
        else:
            node = fe_api.ast_call(canon, (f,), params)
        return fe_api.ast_to_dsl_text(node), ""
    except Exception as exc:  # noqa: BLE001 - FE transform 校验失败即拒绝
        return None, f"FE transform rejected {op!r}: {exc}"


# ---------------------------------------------------------------------------
# legacy 字符串拼接路径（OFFLINE_TEST-only，显式开关，不用于生产）
# ---------------------------------------------------------------------------


def _wrap(op: str, *args: str) -> str:
    """legacy 字符串拼接（仅 OFFLINE_TEST；生产已删除对它的依赖）。"""
    return f"{op}({', '.join(args)})"


def _fmt_num(x: float) -> str:
    if float(x).is_integer():
        return str(int(x))
    return repr(float(x))


def _build_formula_legacy(
    a: Mapping[str, Any],
    parents: Sequence[Mapping[str, Any]],
    op: str,
    *,
    surface: str = "daily",
) -> tuple[str | None, str]:
    """legacy 字符串路径（显式 OFFLINE_TEST-only；不构成生产 fallback）。

    NOTE: 保留局部最小 op 分类表仅为离线/测试兼容；生产一律走
    ``build_formula`` 的 FE typed transform 分支（本函数永不从生产调用）。
    """
    # 本地最小 legacy 分类（仅 offline 使用；与 FE registry 对拍由
    # tests/test_v31_ast_generation.py 保证白名单 op 都可 parse）。
    legacy_unary = frozenset(
        {"zscore", "rank", "neg", "abs", "log", "sqrt", "sign", "scale",
         "cs_rank", "cs_zscore", "cs_demean", "clip", "winsorize", "delay",
         "delta", "is_nan", "is_infinite", "is_null", "is_not_null",
         "cap_neutralize", "industry_neutralize", "ind_neutralize", "neutralize",
         "market_cap_neutralize", "size_neutralize"}
    )
    legacy_window = frozenset(
        {"ts_mean", "ts_std", "ts_rank", "ts_sum", "ts_max", "ts_min",
         "ts_median", "ts_delta", "ts_corr", "ts_cov", "ts_pct", "ts_skew",
         "ts_kurt", "ts_quantile", "ts_product", "ts_var", "ts_decay_linear",
         "ts_regression_slope", "ts_time_slope", "ts_topk_sum", "ema", "beta",
         "rolling_beta", "percentile"}
    )
    legacy_cond = frozenset({"if_else", "where", "iif", "coalesce", "fillna"})
    if a.get("action") == ACTION_COMBINE:
        parent_refs = _as_str_list(a.get("parents"))
        formulas = []
        for ref in parent_refs:
            p = _resolve_parent_ref(ref, parents)
            if p is None:
                return None, f"combine parent ref not found: {ref!r}"
            formulas.append(_parent_formula(p))
        if any(not f for f in formulas):
            return None, "combine parent missing formula"
        weights = _as_float_list(a.get("weights"))
        if weights and len(weights) == len(formulas):
            terms = [_wrap("multiply", _fmt_num(w), f) for w, f in zip(weights, formulas)]
            return _wrap("add", *terms), ""
        return _wrap("add", *formulas), ""

    target = str(a.get("target") or "parent_0")
    parent = _resolve_parent_ref(target, parents)
    if parent is None:
        return None, f"transform target not found: {target!r}"
    f = _parent_formula(parent)
    if not f:
        return None, "transform parent missing formula"
    params = a.get("params")
    params = dict(params) if isinstance(params, Mapping) else {}
    if op in legacy_unary:
        return _wrap(op, f), ""
    if op in legacy_window:
        window = _as_int(params.get("window") if "window" in params else a.get("window"))
        if window is None or window <= 0:
            return None, f"window op {op!r} requires positive window"
        if op in ("ts_corr", "ts_cov", "corr", "cov", "beta", "rolling_beta"):
            second = a.get("second")
            if second is None:
                return None, f"pair op {op!r} requires second operand"
            return _wrap(op, f, str(second), str(window)), ""
        return _wrap(op, f, str(window)), ""
    if op in _BINARY_SECOND_OPS:
        second = a.get("second")
        if second is None:
            return None, f"binary op {op!r} requires second operand"
        return _wrap(op, f, str(second)), ""
    if op in legacy_cond:
        cond = a.get("cond")
        otherwise = a.get("otherwise", "0.0")
        if cond is None:
            return None, f"conditional op {op!r} requires cond"
        return _wrap(op, str(cond), f, str(otherwise)), ""
    return None, f"op {op!r} not supported for transform"


# ---------------------------------------------------------------------------
# StructuredGenerator：action → 候选池（FE validate 过滤）
# ---------------------------------------------------------------------------


@dataclass
class StructuredCandidate:
    """结构化 generation 产物：公式 + lineage + 校验诊断。"""

    formula: str
    parent_ids: list[str] = field(default_factory=list)
    action: dict[str, Any] = field(default_factory=dict)
    action_type: str = "REFINE"
    explanation: str = ""
    hypothesis: str = ""
    schema_tags: dict[str, str] = field(default_factory=dict)
    fe_valid: bool = True
    fe_message: str = ""


@dataclass
class GenerationResult:
    """一次结构化 generation 的结果。"""

    candidates: list[StructuredCandidate] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    #: 每个候选的 lineage（parent ids 列表）
    lineages: list[list[str]] = field(default_factory=list)


@dataclass
class StructuredGenerator:
    """action JSON → FE typed AST transform → FE validate → 候选池。

    Parameters
    ----------
    validator : Callable[[str], tuple[bool, str]] | None
        FE validate 注入（默认 ``FactorEngineAdapter().validate``）。测试可注入
        假 validator 验证「FE 校验失败不进候选池」。
    op_whitelist : frozenset[str] | None
        op 白名单覆盖（默认 ``DEFAULT_ACTION_OP_WHITELIST`` ∩ FE 表面）。
    surface : str
        FE 校验/白名单的算子面（默认 daily）。
    """

    validator: Callable[[str], tuple[bool, str]] | None = None
    op_whitelist: frozenset[str] | None = None
    surface: str = "daily"

    def __post_init__(self) -> None:
        if self.validator is None:
            from alphaprobe.integration.factor_engine_adapter import FactorEngineAdapter

            self.validator = FactorEngineAdapter().validate

    def _validate_formula(self, formula: str) -> tuple[bool, str]:
        try:
            return self.validator(formula)
        except Exception as exc:  # noqa: BLE001 - validator 异常视为校验失败
            return False, f"validator error: {exc}"

    def generate(
        self,
        actions: Sequence[Mapping[str, Any]],
        parents: Sequence[Mapping[str, Any]],
        *,
        default_action_type: str = "REFINE",
    ) -> GenerationResult:
        """把 action 列表应用到 parents，产出候选池。

        每个 action：build_formula（FE typed AST transform）→ FE validate →
        通过则进候选池（记录 lineage = 该 action 引用的 parent ids）；失败则
        记录 rejected（含原因）。
        """
        result = GenerationResult()
        parent_list = list(parents or [])
        for action in actions:
            formula, reason = build_formula(
                action, parent_list, op_whitelist=self.op_whitelist, surface=self.surface
            )
            if formula is None:
                result.rejected.append(
                    {"action": dict(action), "reason": reason, "stage": "build"}
                )
                continue
            ok, msg = self._validate_formula(formula)
            if not ok:
                result.rejected.append(
                    {
                        "action": dict(action),
                        "reason": msg,
                        "stage": "fe_validate",
                        "formula": formula,
                    }
                )
                continue
            parent_ids = self._lineage_of(action, parent_list)
            result.candidates.append(
                StructuredCandidate(
                    formula=formula,
                    parent_ids=parent_ids,
                    action=dict(action),
                    action_type=str(action.get("action_type") or default_action_type),
                    explanation=str(action.get("explanation") or ""),
                    hypothesis=str(action.get("hypothesis") or ""),
                    schema_tags=dict(action.get("schema_tags") or {}),
                    fe_valid=True,
                    fe_message=msg,
                )
            )
            result.lineages.append(parent_ids)
        return result

    @staticmethod
    def _lineage_of(
        action: Mapping[str, Any], parents: Sequence[Mapping[str, Any]]
    ) -> list[str]:
        """候选的 lineage = action 引用的 parent ids（去重保序）。"""
        a = dict(action)
        action_type = str(a.get("action") or "").strip().lower()
        refs: list[str] = []
        if action_type == ACTION_COMBINE:
            refs = _as_str_list(a.get("parents"))
        elif action_type == ACTION_TRANSFORM:
            target = str(a.get("target") or "parent_0")
            refs = [target]
        elif action_type == ACTION_IDENTITY:
            refs = ["parent_0"]
        by_id = {_parent_id(p): p for p in parents}
        by_idx: dict[str, str] = {}
        for i, p in enumerate(parents):
            by_idx[str(i)] = _parent_id(p)
            by_idx[f"parent_{i}"] = _parent_id(p)
        out: list[str] = []
        for ref in refs:
            pid = by_id.get(ref, {}).get("factor_id") or by_id.get(ref, {}).get("id") or by_idx.get(ref, "")
            if pid and pid not in out:
                out.append(pid)
        return out


# ---------------------------------------------------------------------------
# llm_fn 契约升级：结构化对象 + 旧文本向后兼容
# ---------------------------------------------------------------------------

def normalize_llm_output(
    output: Any,
    *,
    parents: Sequence[Mapping[str, Any]] | None = None,
    default_action_type: str = "REFINE",
) -> list[dict[str, Any]]:
    """把 llm_fn 输出归一化为 action dict 列表。

    - 结构化对象（dict 含 ``action`` 字段，或 list[dict]）→ 原样返回；
    - 旧文本输出（§31 candidates JSON 字符串）→ 映射为 identity action
      （formula 原样保留），并打日志（向后兼容路径）。
    - 其他（None/空）→ 空列表。
    """
    if output is None:
        return []
    if isinstance(output, str):
        # 旧文本路径：尝试解析 §31 candidates JSON
        try:
            obj = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            logger.warning("normalize_llm_output: text not JSON, treating as empty")
            return []
        items = obj.get("candidates") if isinstance(obj, dict) else None
        if not isinstance(items, list):
            return []
        logger.info("normalize_llm_output: legacy text output -> identity actions")
        out: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            formula = str(item.get("formula") or "").strip()
            if not formula:
                continue
            out.append(
                {
                    "action": ACTION_IDENTITY,
                    "formula": formula,
                    "action_type": str(item.get("action_type") or default_action_type),
                    "explanation": str(item.get("explanation") or ""),
                    "hypothesis": str(item.get("hypothesis") or ""),
                    "parent_ids": [str(p) for p in (item.get("parent_ids") or []) if p],
                    "schema_tags": dict(item.get("schema_tags") or {}),
                }
            )
        return out
    if isinstance(output, Mapping):
        if "action" in output:
            return [dict(output)]
        # 可能是 {"candidates": [...]} 或 {"actions": [...]} 包装
        for key in ("candidates", "actions"):
            items = output.get(key)
            if isinstance(items, list):
                return [dict(i) for i in items if isinstance(i, Mapping)]
        return []
    if isinstance(output, Sequence):
        return [dict(i) for i in output if isinstance(i, Mapping)]
    return []


# ---------------------------------------------------------------------------
# 确定性 stub llm_fn（可种子复现，产出 action JSON）
# ---------------------------------------------------------------------------

def make_structured_stub_llm_fn(
    *,
    rng: Any | None = None,
    seed: int = 0,
    forced_actions: Sequence[Mapping[str, Any]] | None = None,
) -> Callable[[str, str, str], Any]:
    """构造确定性、可种子复现的 action JSON stub llm_fn。

    返回的 llm_fn 契约升级为返回结构化对象（dict 含 ``action`` 字段），
    与 ``normalize_llm_output`` 兼容。``forced_actions`` 非空时固定返回这些
    action（测试用）；否则从 parents 与 action 规则化生成 action JSON。
    绝不发起任何网络 / 模型调用。
    """
    rng = rng or random.Random(seed)

    def _llm_fn(system_prompt: str, user_prompt: str, model_class: str) -> Any:
        if forced_actions:
            return {"actions": list(forced_actions)}
        # 从 user_prompt 的 parent 行提取公式（"## Parent factor(s)" 之后 "- f :: desc"）
        parents: list[dict[str, Any]] = []
        for line in user_prompt.splitlines():
            line = line.strip()
            if not line.startswith("- "):
                continue
            body = line[2:]
            f = body.split("::", 1)[0].strip()
            if f and f != "?":
                parents.append({"formula": f})
        if not parents:
            return {"actions": []}
        out: list[dict[str, Any]] = []
        for i, p in enumerate(parents):
            base = str(p.get("formula") or "")
            if not base:
                continue
            # 确定性：同 seed 同输出
            for k in range(2):
                op = _stub_op(rng, i, k)
                out.append(
                    {
                        "action": "transform",
                        "op": op,
                        "target": f"parent_{i}",
                        "params": {"window": 20} if _stub_window_op(op) else {},
                        "action_type": "REFINE",
                        "explanation": f"structured-stub transform #{k}",
                        "hypothesis": "确定性 stub：action JSON 变换",
                        "schema_tags": {},
                    }
                )
        return {"actions": out}

    return _llm_fn


def _stub_window_op(op: str) -> bool:
    """stub 是否给该 op 带 window 参数（离线确定性；仅影响 stub 生成）。"""
    return op in {
        "ts_mean", "ts_std", "ts_rank", "ts_sum", "ts_max", "ts_min",
        "ts_median", "ts_delta", "ts_pct", "ts_skew", "ts_kurt",
        "ts_quantile", "ts_product", "ts_var", "ts_decay_linear",
        "ts_regression_slope", "ts_time_slope", "ts_topk_sum", "ema",
        "beta", "rolling_beta", "percentile",
    }


def _stub_op(rng: Any, i: int, k: int) -> str:
    """确定性选 op：同 (seed, i, k) 同 op（rng 已种子化，保证可种子复现）。"""
    pool = ("zscore", "rank", "ts_mean", "ts_std", "neg", "abs", "ts_rank", "ts_sum")
    # 用 rng 的确定性序列选 op：同 seed → 同序列 → 同 op；不同 seed → 不同 op。
    return pool[rng.randrange(len(pool))]


__all__ = [
    "ACTION_COMBINE",
    "ACTION_IDENTITY",
    "ACTION_TRANSFORM",
    "ActionValidation",
    "DEFAULT_ACTION_OP_WHITELIST",
    "GenerationResult",
    "StructuredCandidate",
    "StructuredGenerator",
    "build_formula",
    "fe_operator_surface",
    "make_structured_stub_llm_fn",
    "normalize_llm_output",
    "validate_action",
]
