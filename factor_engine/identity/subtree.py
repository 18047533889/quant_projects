# -*- coding: utf-8 -*-
"""Merkle 子树哈希（§14/§15）。

每节点 H = sha256("factor_identity_v1" + canonical_ast_text(node_subtree))，
其中 ``canonical_ast_text`` 是该节点为根的整棵子树的稳定序列化
（基于 ``factor_engine.expr.canonical``），children 以其规范文本内嵌进父节点
序列化，因此父节点 H 隐式绑定全部后代。

- root == canonical_ast_hash（顶层节点子树即整棵 AST）
- ``subtree_hashes`` = 全部节点 H 去重（后序），用于子树级去重/指纹比对

为何不逐层拼接 ``node_payload + children_hashes``：任务 §14 要求 root ==
canonical_ast_hash（SHA-256 of 命名空间 + 稳定序列化）。要让两者严格相等，
根哈希必须等于 ``sha256(ns + canonical_ast_text(整棵))``；若父节点只拼
"自身 payload + 子哈希"，根值无法与 canonical_ast_hash 重合。本实现以
``canonical_ast_text(subtree)`` 作为每节点的规范化 payload（子节点以规范文本
内嵌，等价于 hash 依赖链），既满足 §15 的 Merkle 语义（父依赖全部后代），
又满足 §12/§13 的统一哈希口径。
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


def _node_hash(node: Expr) -> str:
    """单个节点的哈希 = sha256(ns + 该子树 canonical 文本)。"""
    text = serializer.canonical_ast_text(node)
    return hasher.compute_hash(text)


def _merkle(node: Expr) -> Tuple[str, Tuple[str, ...]]:
    """后序遍历：返回 (节点哈希, 该子树全部节点哈希列表[后序])。"""
    child_subtrees: list[str] = []
    if isinstance(node, CleanedCall):
        for child in node.args:
            _, child_subtree = _merkle(child)
            child_subtrees.extend(child_subtree)

    node_h = _node_hash(node)
    child_subtrees.append(node_h)
    return node_h, tuple(child_subtrees)


def compute_merkle_root(node: Expr) -> str:
    """计算 Merkle 根哈希（= canonical_ast_hash）。"""
    root, _ = _merkle(node)
    return root


def compute_subtree_hashes(node: Expr) -> Tuple[str, ...]:
    """计算全部子树哈希（去重，保持后序遍历顺序）。"""
    _, subtrees = _merkle(node)
    seen = set()
    deduped: list[str] = []
    for s in subtrees:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return tuple(deduped)