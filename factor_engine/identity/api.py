# -*- coding: utf-8 -*-
"""身份计算 API（唯一公共入口）。

``get_factor_identity(formula, *, surface="daily") -> FactorIdentity``
"""

from __future__ import annotations

from typing import Tuple

from factor_engine.expr.cleaned_call import CleanedCall
from factor_engine.expr.field import FieldRef
from factor_engine.expr.literal import Literal
from factor_engine.expr.column import ColumnRef
from factor_engine.expr.base import Expr

from . import hasher
from . import serializer
from .models import FactorIdentity, FactorIdentityError
from .canonicalizer import canonicalize
from .sign_normalizer import compute_signal_equivalence_id, detect_orientation
from .parameter_family import parameter_family_id, parameter_family_text
from .subtree import compute_merkle_root, compute_subtree_hashes
from .operator_meta import get_operator_meta

_NS = "factor_identity_v1"


def _parse(formula: str, *, surface: str) -> Expr:
    """解析公式为 AST。失败抛 FactorIdentityError（带原文与原因）。

    解析前先触发 cleaned_operators 注册表加载（首次调用约 10-35s），
    之后 DSL 解析走 warm 路径（约 1-3ms）。
    """
    from factor_engine.api.dsl_parser import parse_expr

    try:
        return parse_expr(str(formula), surface=surface)
    except Exception as exc:  # noqa: BLE001 - 统一包装
        raise FactorIdentityError(
            str(formula),
            f"DSL parse failed ({type(exc).__name__}): {exc}",
        ) from exc


def _coerce_input(formula: str) -> Expr | None:
    """幂等兼容：接受 ``Expr`` 实例或 canonical_dsl JSON 文本。

    - 传入 ``Expr``（CleanedCall/FieldRef/...）直接使用；
    - 传入以 ``{`` 开头的 JSON（本包产出的 canonical_dsl）尝试反序列化；
    - 其他字符串返回 None（走 DSL 解析）。
    """
    from factor_engine.expr.base import Expr as _Expr

    if isinstance(formula, _Expr):
        return formula
    if isinstance(formula, str) and formula.lstrip().startswith("{"):
        try:
            return _payload_to_expr(formula)
        except Exception:
            return None
    return None


def _payload_to_expr(text: str) -> Expr:
    """把 canonical_ast_payload JSON 反序列化为 Expr 树。"""
    import json

    from factor_engine.expr.cleaned_call import CleanedCall
    from factor_engine.expr.column import ColumnRef
    from factor_engine.expr.field import FieldRef
    from factor_engine.expr.literal import Literal

    def build(payload: dict) -> Expr:
        kind = payload.get("kind")
        if kind == "field":
            return FieldRef(
                name=payload.get("canonical_name", ""),
                field_id=payload.get("field_id", ""),
                canonical_name=payload.get("canonical_name", ""),
                table=payload.get("table", ""),
                source_name=payload.get("source_name", ""),
                catalog_hash=payload.get("catalog_hash", ""),
            )
        if kind == "column":
            return ColumnRef(name=payload.get("name", ""))
        if kind == "literal":
            return Literal(value=payload.get("value"))
        if kind == "call":
            return CleanedCall(
                op=payload.get("op", ""),
                args=tuple(build(a) for a in payload.get("args", [])),
                kwargs=tuple(
                    (str(k), v) for k, v in sorted(payload.get("kwargs", {}).items())
                ),
            )
        raise ValueError(f"unsupported payload kind: {kind!r}")

    return build(json.loads(text))


_registry_loaded = False


def _ensure_registry_loaded() -> None:
    """一次性加载 cleaned_operators 注册表（线程安全，幂等）。

    解析路径依赖 build_dsl_allowlist，而 allowlist 构建会加载全部算子
    （冷启动约 10-35s）。这里显式触发，让**身份计算本身**的耗时
    收敛到 warm 路径（<50ms 量级）。
    """
    global _registry_loaded
    if _registry_loaded:
        return
    from factor_engine.backend.cleaned_bridge import ensure_cleaned_loaded

    ensure_cleaned_loaded()
    _registry_loaded = True


def _field_signature(node: Expr) -> str:
    """字段签名：收集全部字段的规范名（排序、去重），hash(namespace + 文本)。"""
    fields: set[str] = set()

    def walk(n: Expr) -> None:
        if isinstance(n, FieldRef):
            fields.add(n.field_id)
        elif isinstance(n, CleanedCall):
            for c in n.args:
                walk(c)

    walk(node)
    text = "\x1f".join(sorted(fields))
    return hasher.compute_hash("field_signature\x00" + text)


def _operator_signature(node: Expr) -> str:
    """算子签名：收集全部算子名（排序、去重），hash(namespace + 文本)。"""
    ops: set[str] = set()

    def walk(n: Expr) -> None:
        if isinstance(n, CleanedCall):
            ops.add(n.op)
            for c in n.args:
                walk(c)

    walk(node)
    text = "\x1f".join(sorted(ops))
    return hasher.compute_hash("operator_signature\x00" + text)


def _complexity(node: Expr) -> int:
    """AST 节点总数（算子调用 + 字段/列引用 + 字面量）。"""
    count = 1
    if isinstance(node, CleanedCall):
        count += sum(_complexity(c) for c in node.args)
    return count


def _depth(node: Expr) -> int:
    """AST 最大深度（根为 1）。"""
    if isinstance(node, CleanedCall):
        child_depths = [_depth(c) for c in node.args]
        return 1 + max(child_depths) if child_depths else 1
    return 1


def _lookback(node: Expr) -> int:
    """最大 lookback：WINDOW/LAG 角色的最大数字参数（启发式）。

    遍历 WINDOW 角色的算子调用，取其数字字面量参数的最大值。
    """
    lookback = 0

    def walk(n: Expr) -> None:
        nonlocal lookback
        if isinstance(n, CleanedCall):
            meta = get_operator_meta(n.op)
            role = meta.role if meta else "OTHER"
            for arg in n.args:
                if role in ("WINDOW", "LAG") and isinstance(arg, Literal):
                    v = arg.value
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        lookback = max(lookback, int(v))
                walk(arg)
        elif isinstance(n, (FieldRef, ColumnRef, Literal)):
            return

    walk(node)
    return lookback


def get_factor_identity(
    formula: str,
    *,
    surface: str = "daily",
    family_threshold: bool = True,
) -> FactorIdentity:
    """计算公式的无状态因子身份。

    Parameters
    ----------
    formula : str
        FE DSL 公式文本。
    surface : str
        解析 surface（默认 daily）。
    family_threshold : bool
        THRESHOLD 角色参数是否 family 化（默认 True，可配置）。

    Returns
    -------
    FactorIdentity
        冻结身份对象。

    Raises
    ------
    FactorIdentityError
        解析失败时抛出（带原文与原因）。
    """
    _ensure_registry_loaded()
    # 幂等兼容：canonical_dsl（JSON 文本）或任意 Expr 实例传入时，
    # 先反序列化为 Expr 再计算（不经过 DSL 语法解析）。
    raw_expr = _coerce_input(formula)
    if raw_expr is None:
        raw_expr = _parse(formula, surface=surface)

    # 1. 规范化 AST（§7 安全化简）
    canonical_ast = canonicalize(raw_expr)

    # 2. canonical DSL 文本（稳定序列化）
    canonical_dsl = serializer.canonical_ast_text(canonical_ast)

    # 3. canonical_ast_hash = sha256(ns + canonical_text)
    canonical_ast_hash = hasher.compute_hash(canonical_dsl)

    # 4. sign / orientation
    orientation = detect_orientation(canonical_ast)
    signal_equivalence_id = compute_signal_equivalence_id(canonical_ast)

    # 5. parameter family
    from .parameter_family import ParameterFamilyConfig

    fam_config = ParameterFamilyConfig(family_threshold=family_threshold)
    fam_id = parameter_family_id(canonical_ast, config=fam_config)

    # 6. subtree hashes
    subtree_hashes = compute_subtree_hashes(canonical_ast)
    merkle_root = compute_merkle_root(canonical_ast)

    # 7. signatures / metrics
    field_signature = _field_signature(canonical_ast)
    operator_signature = _operator_signature(canonical_ast)
    complexity = _complexity(canonical_ast)
    depth = _depth(canonical_ast)
    lookback = _lookback(canonical_ast)

    return FactorIdentity(
        canonical_dsl=canonical_dsl,
        canonical_ast_hash=canonical_ast_hash,
        signal_equivalence_id=signal_equivalence_id,
        parameter_family_id=fam_id,
        orientation=orientation,
        subtree_hashes=subtree_hashes,
        field_signature=field_signature,
        operator_signature=operator_signature,
        complexity=complexity,
        depth=depth,
        lookback=lookback,
        identity_version="factor_identity_v1",
        operator_semantics_version="v1",
    )
