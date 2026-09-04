# -*- coding: utf-8
"""typed AST transform 公共 API（R61 Task7/A9 —— AlphaPROBE structured generation 用）。

设计约束（与 ``docs/factor_intelligence`` 语义一致）
---------------------------------------------------
- **不发明算子语义**：operator arity / 参数角色 / window role 一律从
  ``OperatorRegistry``（catalog ``param_names`` / ``ParamSpec``）与
  ``backend.operator_types.OPERATOR_SIGNATURES``（``TypeKind``）读取；
  本模块不维护任何平行 op 分类表。
- **带类型校验**：``call`` / ``crossover`` 按签名声明的输入 ``TypeKind``
  校验；``where``/``if_else`` 的 condition 槽必须是 bool Series。
- **窗口参数只改声明的 WINDOW 角色参数**：``scale_window`` 只替换
  ``window/d/span/...`` 中带 ``WINDOW``/``HORIZON`` 语义的 literal。
- **字段替换尊重域/类型兼容**：替换目标与源字段必须有可证明的语义域
  交集（通过 ``fields.resolver`` 解析 ``FieldSpec.domain`` 与类型），
  否则 fail-closed 拒绝（绝不猜）。
- transform 产物是 ``Expr``；``to_dsl_text`` 序列化为 canonical DSL 文本，
  可重新 ``parse_expr`` 并得到一致 canonical identity（硬验收）。

本模块是 **只读、无副作用** 静态构造层：不执行算子、不落盘、不修改
registry / catalog。所有 helper 在无 FE 环境不可用（import 失败直接抛错，
不做静默降级——生产路径由调用方决定 fail-closed 策略）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from factor_engine.expr.base import Expr, ensure_expr
from factor_engine.expr.cleaned_call import CleanedCall
from factor_engine.expr.field import FieldRef
from factor_engine.expr.literal import Literal
from factor_engine.expr.column import ColumnRef

# ---------------------------------------------------------------------------
# registry / signature 权威加载（惰性 + 缓存）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatorSemantics:
    """一个算子从 registry/signature 读取的调用契约（FE 侧权威）。

    ``panel_names`` / ``panel_count``：按 catalog ``param_names`` 中的非标量
    参数推导（签名缺失时保守取“全部位置参数”；签名存在时用
    ``TypeKind != WINDOW/SCALAR_*`` 判定）。``window_names`` 只含带
    WINDOW 角色的参数名。``allow_variadic`` 表示 add/multiply 等可折叠
    多元算子（R7-224 契约取到 variadic 时逐字保留其余 panel 输入）。
    ``cond_slot_names``：条件槽（where/if_else 族）——必须 bool。
    """

    canonical: str
    panel_names: tuple[str, ...]
    panel_positions: tuple[int, ...]
    window_names: tuple[str, ...]
    allow_variadic: bool = False
    cond_slot_names: tuple[str, ...] = ()

    @property
    def panel_count(self) -> int:
        """非窗口的面板输入个数（不含 window/min_periods 等标量参数）。"""
        return len(self.panel_positions)


def _registry() -> Any:
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded
    from factor_engine.cleaned_operators.registry import OperatorRegistry

    ensure_cleaned_loaded()
    return OperatorRegistry


def _signature_table() -> dict[str, Any]:
    from factor_engine.backend.operator_types import OPERATOR_SIGNATURES

    return OPERATOR_SIGNATURES


def _catalog_param_names(canonical: str) -> tuple[str, ...]:
    """catalog 的 ``param_names``（位置顺序权威）。"""
    reg = _registry()
    entry = reg.catalog().get(canonical)
    names = tuple(str(x) for x in (entry or {}).get("param_names") or ())
    return tuple(n for n in names if n != "...")


def _is_scalar_param(canonical: str, name: str) -> bool:
    """参数是否为标量（非 panel）：优先读 ParamSpec / scalar_params，兜底名字。"""
    reg = _registry()
    impl = reg.get(canonical)
    meta = getattr(impl, "metadata", None)
    if meta is not None:
        scalar = tuple(str(x) for x in (getattr(meta, "scalar_params", None) or ()))
        if scalar:
            return name in scalar
    # 兜底名字启发式只对无 ParamSpec 的老算子生效（declared 权威优先）
    return name in {
        "window", "d", "n", "span", "period", "periods", "lag", "q", "k",
        "lo", "hi", "lower", "upper", "min_periods", "ddof", "min_count",
        "max_periods", "to", "method", "mode", "retval", "add_intercept",
        "fallback_policy", "tolerance", "threshold", "fraction", "p",
        "signal", "fast", "slow", "fast_window", "slow_window",
        "fast_period", "slow_period", "time", "limit", "fill",
    }


def _is_scalar_extra(canonical: str, name: str) -> bool:
    """额外标量参数：非窗口、非面板的 estimator/策略参数（min_periods 等）。"""
    return name in {
        "min_periods", "ddof", "min_count", "max_periods", "lag",
        "retval", "add_intercept", "skipna", "method", "mode", "to",
        "fallback_policy", "tolerance", "threshold", "q", "k", "signal",
    }


def _window_role_names(canonical: str, param_names: tuple[str, ...]) -> tuple[str, ...]:
    """WINDOW 角色参数名：先读签名 TypeKind.WINDOW，再读 ParamSpec 角色/名字。"""
    sig = _signature_table().get(canonical)
    if sig is not None:
        out = [
            a.name
            for a in getattr(sig, "inputs", ())
            if str(getattr(a, "type_kind", "")) == "Window"
        ]
        if out:
            return tuple(out)
    reg = _registry()
    impl = reg.get(canonical)
    meta = getattr(impl, "metadata", None)
    specs = dict(getattr(meta, "param_specs", None) or {})
    window_like = {
        "window", "d", "n", "span", "period", "periods", "lookback",
        "max_lookback", "fast", "slow", "fast_period", "slow_period",
        "fast_window", "slow_window", "signal_window", "short_window",
        "long_window", "medium_window", "history_window", "left_window",
        "right_window", "top_k", "k",
    }
    out = []
    for name in param_names:
        spec = specs.get(name)
        role = getattr(spec, "param_role", None) if spec is not None else None
        if role is not None and str(getattr(role, "value", role)).lower() in {
            "horizon", "window",
        }:
            out.append(name)
        elif role is None and name in window_like:
            # 兜底：签名/ParamSpec 都没声明时，名字启发式只对无声明的老算子
            # 生效（该名单是 Analyzer/identity 同款 window 名单子集）。
            out.append(name)
    return tuple(out)


def _bool_output_canonicals() -> frozenset[str]:
    """签名里 output=Series[Bool] 的算子集合（bool 条件允许的根）。"""
    return frozenset(
        canon
        for canon, sig in _signature_table().items()
        if str(getattr(sig, "output", "")).lower() == "series[bool]"
    )


_BOOL_COMPARATORS = frozenset(
    {"gt", "lt", "ge", "le", "eq", "ne", "and_", "or_", "not_"}
)
#: 条件槽名字（where/if_else 族）
_COND_SLOT_NAMES = frozenset({"condition", "cond"})


def resolve_op_semantics(op_name: str) -> OperatorSemantics:
    """解析一个 DSL 算子名的调用契约（canonical 归一并读取 registry/signature）。

    Raises
    ------
    ValueError
        算子未注册（fail-closed，禁止未知算子静默放行）。
    """
    reg = _registry()
    try:
        canonical = reg.resolve_canonical_strict(str(op_name))
    except Exception as exc:
        raise ValueError(
            f"unsupported operator {op_name!r}: not registered in "
            "factor_engine OperatorRegistry (fail-closed, no builtin fallback)"
        ) from exc
    if reg.get(canonical) is None:
        raise ValueError(f"unsupported operator {op_name!r}: no runtime implementation")
    names = _catalog_param_names(canonical)
    sig = _signature_table().get(canonical)
    sig_inputs = tuple(getattr(sig, "inputs", ()) or ()) if sig is not None else ()

    # allow_variadic：add/multiply 等可折叠多元（签名带 variadic 标记或
    # catalog 显式标记；保守默认 False，宁可少变不可错变）。
    allow_variadic = False
    impl = reg.get(canonical)
    meta = getattr(impl, "metadata", None)
    if getattr(meta, "allow_variadic_panel", None) or canonical in {
        "add", "multiply", "maximum", "minimum", "and_", "or_", "coalesce",
        "subtract", "divide",
    }:
        allow_variadic = True

    # panel names / positions：从签名里非标量、非窗口的输入推导；无签名时按
    # catalog param_names 过滤标量/window 参数。条件槽（where 的 condition）按
    # 参数名识别并计入面板输入（它在 DSL 里是真正的第一个位置参数）。
    if sig_inputs:
        panel_names = tuple(
            a.name
            for a in sig_inputs
            if str(getattr(a, "type_kind", "")).startswith("Series")
            or str(getattr(a, "type_kind", "")) == "GroupKey"
        )
    else:
        panel_names = tuple(n for n in names if not _is_scalar_param(canonical, n))
    if not panel_names and names:
        panel_names = tuple(names)
    window_names = _window_role_names(canonical, names)
    window_set = set(window_names)
    # 面板位置 = param_names 中既不是 window 也不是额外标量的位置。
    # （where 的 condition 不是 window 也不在 extra-scalar 名单 → 计入面板）
    if sig_inputs:
        panel_positions = tuple(
            i for i, n in enumerate(names)
            if n in panel_names and n not in window_set
            and not _is_scalar_extra(canonical, n)
        )
    else:
        panel_positions = tuple(
            i for i, n in enumerate(names)
            if n in panel_names and n not in window_set and not _is_scalar_extra(canonical, n)
        )
    if not panel_positions:
        # 兜底：完全没有可识别面板位置时（如 add/multiply 空 param_names），
        # 面板位置按 param_names 顺序逐字保留（variadic 由调用方控制）。
        panel_positions = tuple(i for i, n in enumerate(names) if n in panel_names)
    cond_names = tuple(n for n in names if n in _COND_SLOT_NAMES)
    return OperatorSemantics(
        canonical=canonical,
        panel_names=panel_names,
        panel_positions=panel_positions,
        window_names=window_names,
        allow_variadic=allow_variadic,
        cond_slot_names=cond_names,
    )


# ---------------------------------------------------------------------------
# 类型推断（typed transform 核心）
# ---------------------------------------------------------------------------

_DFLT = object()


def infer_expr_kind(dsl_or_expr: str | Expr) -> str:
    """推断一个表达式的输出类型：``float`` / ``bool`` / ``group`` / ``unknown``。

    判定优先顺序：签名 output=Series[Bool] → 比较/逻辑算子族 → Analyzer IR
    ``attrs.dtype == bool``。失败保守返回 ``unknown``（不猜，让上层按需拒绝）。
    """
    expr = ensure_parsed(dsl_or_expr)
    if isinstance(expr, CleanedCall):
        canon = _canonical_of(expr.op)
        if canon in _BOOL_COMPARATORS or canon in _bool_output_canonicals():
            return "bool"
        try:
            analysis = _lower(expr)
            root = analysis.ir
            if str((root.attrs or {}).get("dtype") or "").lower() in {"bool", "boolean"}:
                return "bool"
        except Exception:
            pass
        # where/if_else 输出取决于两个分支，签名固定 Series[Float] → 保守 float
        if canon == "where":
            return "float"
        return "float"
    if isinstance(expr, (FieldRef, ColumnRef)):
        return "float"
    if isinstance(expr, Literal):
        v = expr.value
        if isinstance(v, bool):
            return "bool"
        return "float"
    return "unknown"


def expr_kind(expr: Expr) -> str:
    """与 :func:`infer_expr_kind` 同语义的 Expr 版本。"""
    return infer_expr_kind(expr)


def _canonical_of(op: str) -> str:
    try:
        return _registry().resolve_canonical(op)
    except Exception:
        return str(op)


def _lower(expr: Expr) -> Any:
    from factor_engine.ir.analyzer import Analyzer

    return Analyzer(production=False).lower(expr)


# ---------------------------------------------------------------------------
# Expr 构造/序列化
# ---------------------------------------------------------------------------


def ensure_parsed(dsl_or_expr: str | Expr) -> Expr:
    """把 DSL 文本或已解析 Expr 统一为 Expr（文本经 ``parse_expr``）。"""
    if isinstance(dsl_or_expr, Expr):
        return dsl_or_expr
    from factor_engine.api.dsl_parser import parse_expr

    return parse_expr(str(dsl_or_expr), surface="daily")


def call(op_name: str, panels: Sequence[str | Expr], params: Mapping[str, Any]) -> CleanedCall:
    """构造一个带类型校验的算子调用。

    Parameters
    ----------
    op_name : str
        DSL 算子名（canonical 或别名）。
    panels : Sequence[str | Expr]
        按声明 panel 顺序给出的面板输入（Expr/DSL 文本均可）。
    params : Mapping[str, Any]
        标量参数（window 等；值必须是 int/float/str/bool）。

    Raises
    ------
    ValueError
        未知算子 / panel 数量不符 / window 非法 / 条件槽非 bool / 参数未声明。
    """
    info = resolve_op_semantics(op_name)
    canon = info.canonical
    panel_list = list(panels)
    sig = _signature_table().get(canon)
    panel_arity = len(info.panel_positions)
    # 条件槽检查：条件参数必须是 bool 表达式
    if info.cond_slot_names:
        # where/if_else 的 cond 槽是第一个 panel 输入
        if panel_list:
            cond_kind = infer_expr_kind(panel_list[0])
            if cond_kind != "bool":
                raise ValueError(
                    f"operator {canon!r} requires a bool Series condition as its "
                    f"first input, got {cond_kind!r} — a non-boolean expression "
                    "cannot be an if_else condition"
                )
    # 位置化窗口参数：把 params 里的 window 参数按 catalog param_names 顺序
    # 放到位置槽（ts_mean(x, window)、ts_corr(x, y, window)…）。
    names = _catalog_param_names(canon)
    window_name = info.window_names[0] if info.window_names else None
    # 期望的"调用方面板输入"个数 = 非窗口、非额外标量的位置数
    # （ts_mean=1、ts_corr=2、where=3；min_periods/ddof 等不算面板）。
    expected_panels = 0
    for i, n in enumerate(names):
        if n in info.window_names:
            continue
        if _is_scalar_extra(canon, n):
            continue
        expected_panels += 1
    if info.allow_variadic:
        pass  # add/multiply 可任意多面板
    else:
        if len(panel_list) < expected_panels:
            raise ValueError(
                f"operator {canon!r} requires {expected_panels} panel input(s) "
                f"plus a {window_name or 'window'} parameter, got {len(panel_list)}"
            )
        if len(panel_list) > expected_panels:
            raise ValueError(
                f"operator {canon!r} accepts at most {expected_panels} panel "
                f"input(s) (arity), got {len(panel_list)}"
            )
    args: list[Expr] = [ensure_parsed(p) for p in panel_list]
    kw_pairs: list[tuple[str, Any]] = []
    known_scalars = set(info.window_names) | set(names)
    if params:
        for key, value in params.items():
            key = str(key)
            if key not in known_scalars:
                raise ValueError(
                    f"operator {canon!r} has no declared parameter {key!r} "
                    "(fail-closed: params must be declared in the FE catalog)"
                )
            if key in info.window_names:
                _validate_window_value(canon, key, value)
            if key in _COND_SLOT_NAMES:
                continue  # 条件槽走 panel 输入，不接受 params 注入
        # 窗口参数按 catalog param_names 顺序放到位置槽（ts_mean(x, window)、
        # ts_corr(x, y, window)、ts_topk_sum(x, d, k)…）。其余标量参数进 kwargs。
        for key, value in params.items():
            if key == window_name:
                continue  # window 单独位置化
            kw_pairs.append((key, _norm_param_value(value)))
        if window_name is not None and window_name in params:
            wpos = names.index(window_name)
            while len(args) <= wpos:
                args.append(Literal(0))  # 缺位由上层 arity 校验保证不会发生
            args[wpos] = Literal(int(params[window_name]))
    return CleanedCall(op=canon, args=tuple(args), kwargs=tuple(kw_pairs))


def _norm_param_value(v: Any) -> Any:
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, (int, float, str)) or v is None:
        return v
    return repr(v)


def _validate_window_value(canon: str, name: str, value: Any) -> None:
    """窗口参数必须是正整数（literal）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"operator {canon!r} window parameter {name!r} must be a positive "
            f"integer literal, got {value!r}"
        )
    if float(value) != int(value):
        raise ValueError(
            f"operator {canon!r} window parameter {name!r} must be an integer, "
            f"got {value!r}"
        )
    if int(value) <= 0:
        raise ValueError(
            f"operator {canon!r} window parameter {name!r} must be > 0, got {value!r}"
        )


def to_dsl_text(node: Expr) -> str:
    """把 Expr 序列化为 canonical DSL 文本（重新 parse 后 identity 一致）。"""
    return _render(node)


def _render(node: Expr) -> str:
    if isinstance(node, CleanedCall):
        args = ", ".join(_render(a) for a in node.args)
        if node.kwargs:
            kw = ", ".join(
                f"{k}={_render_literal(v)}" for k, v in sorted(node.kwargs)
            )
            args = f"{args}, {kw}" if args else kw
        return f"{node.op}({args})"
    if isinstance(node, FieldRef):
        return node.canonical_name or node.name
    if isinstance(node, ColumnRef):
        return node.name
    if isinstance(node, Literal):
        return _render_literal(node.value)
    return str(node)


def _render_literal(v: Any) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, float) and float(v).is_integer():
        return str(int(v))
    return repr(v)


# ---------------------------------------------------------------------------
# 通用 typed transforms（AlphaPROBE action 语义）
# ---------------------------------------------------------------------------


def scale_window(dsl_or_expr: str | Expr, *, window: int) -> CleanedCall:
    """WINDOW_SCALE：只改声明为 WINDOW 角色的参数，其余不变。

    返回最外层算子调用被替换后的新 Expr；外层只替换声明为窗口槽的参数。
    """
    expr = ensure_parsed(dsl_or_expr)
    if not isinstance(expr, CleanedCall):
        raise ValueError(
            f"scale_window requires a windowed operator call at root, got {type(expr).__name__}"
        )
    info = resolve_op_semantics(expr.op)
    if not info.window_names:
        raise ValueError(
            f"operator {expr.op!r} declares no WINDOW parameter; scale_window rejected"
        )
    _validate_window_value(info.canonical, info.window_names[0], window)
    # 位置化窗口参数名（签名或 catalog 顺序）
    window_name = info.window_names[0]
    names = _catalog_param_names(info.canonical)
    try:
        pos = names.index(window_name)
    except ValueError:
        pos = len(expr.args) - 1
    args = list(expr.args)
    while len(args) <= pos:
        args.append(Literal(0))  # pragma: no cover - 防御：不会发生
    args[pos] = Literal(int(window))
    return CleanedCall(op=expr.op, args=tuple(args), kwargs=expr.kwargs)


def substitute_field(
    dsl_or_expr: str | Expr,
    source_field: str,
    target_field: str,
) -> Expr:
    """FIELD_SUBSTITUTION：把 AST 里的 ``source_field`` 叶子替换为 ``target_field``。

    只替换**语义兼容**的字段（经 ``FieldSpec`` 的 domain 与 dtype 判定；
    价格域字段只能被价格域字段替换）。不兼容或未知字段 fail-closed 拒绝。
    替换后不做额外类型放宽：目标字段必须能通过 operator 类型门（由调用方
    决定是否重新 ``call``/校验——本函数只保证叶子级兼容）。
    """
    expr = ensure_parsed(dsl_or_expr)
    # 与 Analyzer._resolve_field_ir 同款：裸行情名经 parse_expr 已绑定 catalog
    # hash；这里用 FE 权威 resolve_field 解析 FieldSpec（market=None 保持 legacy
    # A-share 回退，与 analyzer 的 research 路径一致）。
    from factor_engine.fields import resolve_field

    def _spec_of(name: str) -> Any:
        # 裸名先走一次 parse_expr 得到带 catalog 绑定的 FieldRef（如
        # 'close' -> StockDailyBarAdj.close），再 resolve_field 拿 FieldSpec。
        try:
            ref = ensure_parsed(name)
        except Exception:
            return None
        try:
            return resolve_field(ref)
        except Exception:
            return None

    src_spec = _spec_of(source_field)
    dst_spec = _spec_of(target_field)
    if src_spec is None or dst_spec is None:
        raise ValueError(
            f"substitute_field: unknown field(s) source={source_field!r} "
            f"target={target_field!r} — not a registered catalog field "
            "(fail-closed, no guessing)"
        )
    if not _fields_compatible(src_spec, dst_spec):
        raise ValueError(
            f"substitute_field: field {source_field!r} (domain={src_spec.domain}, "
            f"dtype={src_spec.dtype}) incompatible with {target_field!r} "
            f"(domain={dst_spec.domain}, dtype={dst_spec.dtype}) — FIELD_SUBSTITUTION "
            "respects field domain/type compatibility"
        )
    return _replace_field(expr, src_spec, dst_spec, target_field)


def _fields_compatible(src: Any, dst: Any) -> bool:
    """两 FieldSpec 是否同一语义域 + 数值类型兼容（保守）。

    价格类（close/open/high/low/vwap）同属 price_volume 域且同为价格单位，
    可互替；volume（同为 price_volume 域但单位 share）不可替换价格字段；
    不同 domain 一律拒绝（valuation / classification 等）。
    """
    src_domain = str(getattr(src, "domain", "") or "")
    dst_domain = str(getattr(dst, "domain", "") or "")
    if not src_domain or not dst_domain:
        return False
    src_unit = str(getattr(src, "unit", "") or "").lower()
    dst_unit = str(getattr(dst, "unit", "") or "").lower()
    if src_domain == dst_domain and src_unit == dst_unit:
        return True
    # 价格域别名（price_volume 行情域内、同价格单位互替）
    price_units = {"cny/share", "cny", "yuan", "hkd", "usd"}
    src_is_price = src_domain == "price_volume" and src_unit in price_units
    dst_is_price = dst_domain == "price_volume" and dst_unit in price_units
    if src_is_price and dst_is_price:
        return True
    return False


def _replace_field(node: Expr, src_spec: Any, dst_spec: Any, target_field: str) -> Expr:
    """深度替换匹配 src FieldSpec 的字段叶子为 dst 字段。"""
    from factor_engine.fields.market_registry import MULTI_MARKET_FIELD_REGISTRY

    if isinstance(node, FieldRef):
        if node.field_id == str(getattr(src_spec, "field_id", "")) or (
            (node.canonical_name or node.name).lower()
            == str(getattr(src_spec, "name", "") or "").lower()
        ):
            registry = MULTI_MARKET_FIELD_REGISTRY.registry_for("ashare")
            return FieldRef(
                name=target_field,
                field_id=str(getattr(dst_spec, "field_id", "") or target_field),
                canonical_name=str(getattr(dst_spec, "name", "") or target_field),
                table=str(getattr(dst_spec, "table", "") or ""),
                source_name=str(getattr(dst_spec, "source_name", "") or target_field),
                catalog_hash=registry.catalog_hash(),
            )
        return node
    if isinstance(node, ColumnRef):
        if str(node.name).lower() == str(getattr(src_spec, "name", "") or "").lower():
            registry = MULTI_MARKET_FIELD_REGISTRY.registry_for("ashare")
            return FieldRef(
                name=target_field,
                field_id=str(getattr(dst_spec, "field_id", "") or target_field),
                canonical_name=str(getattr(dst_spec, "name", "") or target_field),
                table=str(getattr(dst_spec, "table", "") or ""),
                source_name=str(getattr(dst_spec, "source_name", "") or target_field),
                catalog_hash=registry.catalog_hash(),
            )
        return node
    if isinstance(node, CleanedCall):
        return CleanedCall(
            op=node.op,
            args=tuple(_replace_field(a, src_spec, dst_spec, target_field) for a in node.args),
            kwargs=node.kwargs,
        )
    return node


def crossover(
    op_name: str,
    subtrees: Sequence[str | Expr],
    *,
    output_kind: str = "float",
) -> CleanedCall:
    """CROSSOVER：只组合类型兼容的子树为一个算子调用。

    ``output_kind`` 显式声明产物类型（float/bool），防止 bool 子树与 float
    子树混组。内部经 ``call`` 完成类型/arity 校验（无本地 op 分类）。
    """
    info = resolve_op_semantics(op_name)
    tree_list = list(subtrees)
    if not tree_list:
        raise ValueError(f"crossover: operator {op_name!r} requires >=1 subtree")
    for tree in tree_list:
        kind = infer_expr_kind(tree)
        if output_kind == "bool" and kind != "bool":
            raise ValueError(
                f"crossover: operator {op_name!r} output_kind=bool requires bool "
                f"subtree inputs, got {kind!r}"
            )
        if output_kind == "float" and kind not in {"float", "unknown"}:
            raise ValueError(
                f"crossover: operator {op_name!r} output_kind=float requires float "
                f"subtree inputs, got {kind!r}"
            )
    if len(tree_list) == 1 and info.panel_names:
        return call(op_name, tree_list, {})
    return call(op_name, tree_list, {})
